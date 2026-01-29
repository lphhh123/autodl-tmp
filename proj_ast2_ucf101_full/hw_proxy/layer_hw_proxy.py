"""Layer hardware proxy wrapper following v5.4.

Backends:
- NEW tabular .pt ckpts (preferred if present):
    proxy_ms.pt
    proxy_peak_mem_mb.pt
    proxy_energy_mj.pt
- LEGACY .pth (fallback):
    latency_proxy.pth
    mem_proxy.pth
    power_proxy.pth

Outputs (both backends):
  - lat_ms, mem_mb, power_w

Hard requirements:
  - torch path must be differentiable (for stable_hw hw_grad smoke)
  - outputs clamped to non-negative
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .layer_proxy_model import LayerProxyModel

# If proxy_retrain is inside project root, allow absolute import
try:
    from proxy_retrain.proxy_utils import MLPRegressor
except Exception:
    # fallback: relative if someone moved it under hw_proxy/
    from ..proxy_retrain.proxy_utils import MLPRegressor  # type: ignore


LAYER_TYPES = {"patch_embed": 0, "attn": 1, "mlp": 2, "other": 3}
INV_LAYER_TYPES = {v: k for k, v in LAYER_TYPES.items()}


# ---------------------------
# Legacy v5.4 feature builder (kept for .pth fallback)
# ---------------------------
def build_layer_features(layer_cfg: Dict, device_cfg: Dict) -> np.ndarray:
    layer_type_idx = int(layer_cfg.get("layer_type", 3))
    flops = float(layer_cfg.get("flops", 0.0))
    bytes_ = float(layer_cfg.get("bytes", 0.0))
    embed_dim = float(layer_cfg.get("embed_dim", 0))
    num_heads = float(layer_cfg.get("num_heads", 1))
    mlp_ratio = float(layer_cfg.get("mlp_ratio", 4.0))
    seq_len = float(layer_cfg.get("seq_len", 0))
    precision = float(layer_cfg.get("precision", 1))

    peak_flops_tflops = device_cfg.get("peak_flops_tflops", None)
    if peak_flops_tflops is not None:
        peak_flops = float(peak_flops_tflops) * 1e12
    else:
        peak_flops = float(device_cfg.get("peak_flops", 1.0))

    peak_bw_gbps = device_cfg.get("peak_bw_gbps", None)
    if peak_bw_gbps is not None:
        peak_bw = float(peak_bw_gbps) * 1e9
    else:
        peak_bw = float(device_cfg.get("peak_bw", 1.0))

    vec = [
        math.log10(flops + 1.0),
        math.log10(bytes_ + 1.0),
        math.log10(peak_flops + 1.0),
        math.log10(peak_bw + 1.0),
    ]
    one_hot = [0.0] * 4
    one_hot[layer_type_idx] = 1.0
    vec.extend(one_hot)
    vec.extend(
        [
            embed_dim / 1024.0,
            num_heads / 16.0,
            mlp_ratio / 4.0,
            seq_len / 1024.0,
            precision,
        ]
    )
    return np.array(vec, dtype=np.float32)


def _to_t(x, device=None, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(float(x), device=device, dtype=dtype)


def build_layer_features_torch(layer_cfg: Dict, device_cfg: Dict, device=None) -> torch.Tensor:
    layer_type_idx = int(layer_cfg.get("layer_type", 3))
    flops = _to_t(layer_cfg.get("flops", 0.0), device=device)
    bytes_ = _to_t(layer_cfg.get("bytes", 0.0), device=device)
    embed_dim = _to_t(layer_cfg.get("embed_dim", 0.0), device=device)
    num_heads = _to_t(layer_cfg.get("num_heads", 1.0), device=device)
    mlp_ratio = _to_t(layer_cfg.get("mlp_ratio", 4.0), device=device)
    seq_len = _to_t(layer_cfg.get("seq_len", 0.0), device=device)
    precision = _to_t(layer_cfg.get("precision", 1.0), device=device)

    if device_cfg.get("peak_flops_tflops", None) is not None:
        peak_flops = _to_t(device_cfg["peak_flops_tflops"], device=device) * 1e12
    else:
        peak_flops = _to_t(device_cfg.get("peak_flops", 1.0), device=device)

    if device_cfg.get("peak_bw_gbps", None) is not None:
        peak_bw = _to_t(device_cfg["peak_bw_gbps"], device=device) * 1e9
    else:
        peak_bw = _to_t(device_cfg.get("peak_bw", 1.0), device=device)

    v0 = torch.log10(flops + 1.0)
    v1 = torch.log10(bytes_ + 1.0)
    v2 = torch.log10(peak_flops + 1.0)
    v3 = torch.log10(peak_bw + 1.0)

    one_hot = F.one_hot(torch.tensor(layer_type_idx, device=device), num_classes=4).to(torch.float32)

    tail = torch.stack(
        [
            embed_dim / 1024.0,
            num_heads / 16.0,
            mlp_ratio / 4.0,
            seq_len / 1024.0,
            precision,
        ],
        dim=0,
    )
    return torch.cat([torch.stack([v0, v1, v2, v3], dim=0), one_hot, tail], dim=0)


# ---------------------------
# NEW tabular .pt bundle (differentiable torch inference)
# ---------------------------
class _TabularBundle:
    def __init__(self, ckpt_path: Path):
        bundle = torch.load(str(ckpt_path), map_location="cpu")
        self.num_cols: List[str] = list(bundle["num_cols"])
        self.cat_cols: List[str] = list(bundle["cat_cols"])
        self.cat_vocab: Dict[str, List[str]] = dict(bundle["cat_vocab"])
        self.stats: Dict[str, Dict[str, float]] = dict(bundle.get("stats", {}))
        self.target_mode: str = str(bundle.get("target_mode", "linear"))
        self.input_dim: int = int(bundle["input_dim"])
        hidden = list(bundle.get("model_hidden", [256, 256, 128]))
        dropout = float(bundle.get("dropout", 0.1))

        self.model = MLPRegressor(input_dim=self.input_dim, hidden=hidden, dropout=dropout)
        self.model.load_state_dict(bundle["model_state"], strict=True)
        self.model.eval()

        # cat: build strict value->idx
        self._cat_to_index: Dict[str, Dict[str, int]] = {}
        for c in self.cat_cols:
            vs = self.cat_vocab.get(c, [])
            self._cat_to_index[c] = {v: i for i, v in enumerate(vs)}

        # numeric stats
        mu = [float(self.stats.get("num_mean", {}).get(c, 0.0)) for c in self.num_cols]
        sd = [float(self.stats.get("num_std", {}).get(c, 1.0)) for c in self.num_cols]
        sd = [x if abs(x) > 1e-12 else 1.0 for x in sd]
        self._mu = torch.tensor(mu, dtype=torch.float32)
        self._sd = torch.tensor(sd, dtype=torch.float32)

    def _encode_cat_onehot(self, col: str, values: List[str], device: torch.device) -> torch.Tensor:
        v2i = self._cat_to_index[col]
        n_cls = len(v2i)
        idx = []
        for v in values:
            if v not in v2i:
                raise ValueError(f"[ProxyVocabMismatch] col={col} value={v!r} not in vocab (size={n_cls})")
            idx.append(v2i[v])
        idx_t = torch.tensor(idx, device=device, dtype=torch.long)
        return F.one_hot(idx_t, num_classes=n_cls).to(torch.float32)

    def build_x(self, rows: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
        # numeric (support tensors)
        num_parts = []
        for c in self.num_cols:
            vals = []
            for r in rows:
                v = r[c]
                if torch.is_tensor(v):
                    vals.append(v.to(device=device, dtype=torch.float32))
                else:
                    vals.append(torch.tensor(float(v), device=device, dtype=torch.float32))
            num_parts.append(torch.stack(vals, dim=0))
        x_num = torch.stack(num_parts, dim=1) if num_parts else torch.zeros((len(rows), 0), device=device)

        mu = self._mu.to(device=device)
        sd = self._sd.to(device=device)
        x_num = (x_num - mu) / sd

        # cat
        cat_parts = []
        for c in self.cat_cols:
            cat_parts.append(self._encode_cat_onehot(c, [str(r[c]) for r in rows], device=device))
        x_cat = torch.cat(cat_parts, dim=1) if cat_parts else torch.zeros((len(rows), 0), device=device)

        x = torch.cat([x_num, x_cat], dim=1)
        if int(x.shape[1]) != int(self.input_dim):
            raise RuntimeError(f"[ProxyInputDimMismatch] got {int(x.shape[1])} expected {self.input_dim}")
        return x

    def predict(self, rows: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
        self.model = self.model.to(device)
        x = self.build_x(rows, device=device)
        # NOTE:
        # - MLPRegressor.forward() already does squeeze(-1) and returns shape (B,)
        # - DO NOT squeeze again here; batch_size=1 would become 0-dim scalar.
        y = self.model(x)

        # Make sure output is always 1D: (B,)
        y = y.reshape(-1)
        if self.target_mode.lower() == "log":
            # exp(log(x+eps)) - eps ; keep strictly positive
            y = torch.exp(y) - 1e-6
            y = torch.clamp(y, min=1e-6)
        return torch.clamp(y, min=0.0)


class _Tabular3Proxy:
    """ms <-> peak_mem fixed-point + energy->power."""

    def __init__(self, ckpt_dir: Path):
        ms_pt = ckpt_dir / "proxy_ms.pt"
        mem_pt = ckpt_dir / "proxy_peak_mem_mb.pt"
        eng_pt = ckpt_dir / "proxy_energy_mj.pt"
        if not (ms_pt.is_file() and mem_pt.is_file() and eng_pt.is_file()):
            raise FileNotFoundError(
                f"[ProxyMissing] expected in {ckpt_dir}: proxy_ms.pt, proxy_peak_mem_mb.pt, proxy_energy_mj.pt"
            )
        self.ms = _TabularBundle(ms_pt)
        self.mem = _TabularBundle(mem_pt)
        self.eng = _TabularBundle(eng_pt)

    @staticmethod
    def _layer_type_str(cfg: Dict[str, Any]) -> str:
        if "layer_kind" in cfg:
            return str(cfg["layer_kind"])
        idx = int(cfg.get("layer_type", 3))
        return INV_LAYER_TYPES.get(idx, "other")

    @staticmethod
    def _prec_str(cfg: Dict[str, Any]) -> str:
        p = float(cfg.get("precision", 1.0))
        return "fp16" if p >= 1.0 else "fp32"

    def _base_row(self, cfg: Dict[str, Any], run_ctx: Dict[str, Any]) -> Dict[str, Any]:
        keep_ratio = float(cfg.get("keep_ratio", 1.0))
        L_patch = float(cfg.get("L_patch", cfg.get("seq_len", 0.0)))
        L_eff = float(math.floor(L_patch * keep_ratio) + 1.0)

        embed_dim = float(cfg.get("embed_dim", run_ctx.get("embed_dim", 0.0)))
        num_heads = float(cfg.get("num_heads", run_ctx.get("num_heads", 1.0)))
        mlp_ratio = float(cfg.get("mlp_ratio", run_ctx.get("mlp_ratio", 4.0)))
        head_dim = float(embed_dim) / max(1.0, float(num_heads))

        return {
            "device": str(cfg.get("device", run_ctx.get("device", "unknown"))),
            "cfg": str(cfg.get("cfg", run_ctx.get("cfg", "default"))),
            "img": float(run_ctx.get("img", 224)),
            "bs": float(run_ctx.get("bs", 1)),
            "keep_ratio": keep_ratio,
            "L_patch": L_patch,
            "L_eff": L_eff,
            "status": "ok",
            "depth": float(run_ctx.get("depth", 0)),
            "embed_dim": float(embed_dim),
            "num_heads": float(num_heads),
            "mlp_ratio": float(mlp_ratio),
            "complexity_ratio": float(cfg.get("complexity_ratio", 1.0)),
            "head_dim": float(head_dim),
            "tp_world_size": float(run_ctx.get("tp_world_size", 1)),
            # categorical still in ckpt? keep if present
            "layer_type": self._layer_type_str(cfg),
            "prec": self._prec_str(cfg),
        }

    def predict_all_torch(
        self,
        layers_cfg: List[Dict[str, Any]],
        default_device_cfg: Dict[str, Any],
        run_ctx: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # ---- Fixed-point stabilizer priors ----
        # Proxy training data typically has ms in ~[0.05ms, <10ms] range (device dependent).
        # If ms0 is far below training distribution, mem(ms)->ms(mem) fixed-point can diverge.
        min_ms_prior = float(run_ctx.get("proxy_ms_prior_min", 0.05))
        max_ms_prior = float(run_ctx.get("proxy_ms_prior_max", 100.0))

        # ---- Per-row physical memory caps (MB) ----
        # Clamp predicted peak_mem_mb to device memory size to avoid impossible values (e.g., > 24GB).
        mem_caps = []
        for cfg in layers_cfg:
            dc = cfg.get("device_cfg", default_device_cfg) or {}
            if "mem_gb" in dc and dc["mem_gb"] is not None:
                cap_mb = float(dc["mem_gb"]) * 1024.0
            else:
                cap_mb = float(dc.get("mem_size_mb", dc.get("mem_mb", 24000.0)))
            mem_caps.append(_to_t(cap_mb, device=device))
        mem_cap_t = torch.stack(mem_caps, dim=0).reshape(-1)

        rows_base = []
        for cfg in layers_cfg:
            rows_base.append(self._base_row(cfg, run_ctx))

        # init ms using roofline estimate (keeps dependence on eff_specs tensors)
        ms0 = []
        min_ms_t = _to_t(min_ms_prior, device=device)
        max_ms_t = _to_t(max_ms_prior, device=device)
        for cfg in layers_cfg:
            dc = cfg.get("device_cfg", default_device_cfg) or {}
            flops = _to_t(cfg.get("flops", 0.0), device=device)
            bytes_ = _to_t(cfg.get("bytes", 0.0), device=device)
            peak_flops = _to_t(dc.get("peak_flops", 1e12), device=device)
            peak_bw = _to_t(dc.get("peak_bw", 0.0), device=device)
            t_comp = flops / (peak_flops + 1e-9)
            t_mem = torch.zeros_like(t_comp)
            if torch.any(peak_bw > 0):
                t_mem = bytes_ / (peak_bw + 1e-9)
            ms_est = (t_comp + t_mem) * 1e3
            # IMPORTANT: clamp ms0 into a reasonable prior range to stabilize fixed-point iteration
            ms0.append(torch.clamp(ms_est, min=min_ms_t, max=max_ms_t))
        ms = torch.stack(ms0, dim=0).reshape(-1)

        # fixed-point: mem(ms) -> ms(mem) x2
        for _ in range(2):
            # ms -> mem
            rows_mem = []
            for i, base in enumerate(rows_base):
                r = dict(base)
                r["ms"] = ms[i]
                rows_mem.append(r)
            mem_mb = self.mem.predict(rows_mem, device=device).reshape(-1)

            # Clamp mem into [0, device_mem_cap]
            mem_mb = torch.clamp(mem_mb, min=0.0)
            mem_mb = torch.minimum(mem_mb, mem_cap_t)

            # mem -> ms
            rows_ms = []
            for i, base in enumerate(rows_base):
                r = dict(base)
                r["peak_mem_mb"] = mem_mb[i]
                rows_ms.append(r)
            ms = self.ms.predict(rows_ms, device=device).reshape(-1)
            # Clamp ms into the same reasonable prior range
            ms = torch.clamp(ms, min=min_ms_t, max=max_ms_t)

        # energy -> power
        rows_eng = []
        for i, base in enumerate(rows_base):
            dc = layers_cfg[i].get("device_cfg", default_device_cfg) or {}
            avg_power_w = _to_t(dc.get("tdp_w", 200.0), device=device)
            r = dict(base)
            r["avg_power_w"] = avg_power_w
            r["ms_event"] = ms[i]
            r["runs"] = float(run_ctx.get("runs", 10))
            r["warmup"] = float(run_ctx.get("warmup", 5))
            rows_eng.append(r)
        energy_mj = self.eng.predict(rows_eng, device=device).reshape(-1)

        power_w = energy_mj * 1000.0 / (ms + 1e-6)
        power_w = torch.clamp(power_w, min=0.0)
        return torch.clamp(ms, min=0.0), torch.clamp(mem_mb, min=0.0), power_w

    @torch.no_grad()
    def predict_all_numpy(
        self,
        layers_cfg: List[Dict[str, Any]],
        default_device_cfg: Dict[str, Any],
        run_ctx: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        dev = torch.device("cpu")
        ms, mem, pw = self.predict_all_torch(layers_cfg, default_device_cfg, run_ctx, device=dev)
        return {
            "lat_ms": ms.detach().cpu().numpy(),
            "mem_mb": mem.detach().cpu().numpy(),
            "power_w": pw.detach().cpu().numpy(),
        }


# ---------------------------
# Main proxy wrapper
# ---------------------------
class LayerHwProxy:
    def __init__(self, device_name: str, gpu_yaml: str, weight_dir: str, run_ctx: Optional[Dict[str, Any]] = None):
        with open(gpu_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if isinstance(data, dict) and "chip_types" in data:
            self.gpu_cfg = {entry["name"]: entry for entry in data["chip_types"]}
        else:
            self.gpu_cfg = data or {}

        self.device_name = str(device_name)
        self.weight_dir = Path(weight_dir)
        self.run_ctx = dict(run_ctx or {})
        self.run_ctx.setdefault("img", 224)
        self.run_ctx.setdefault("bs", 1)
        self.run_ctx.setdefault("depth", 0)
        self.run_ctx.setdefault("embed_dim", 0)
        self.run_ctx.setdefault("num_heads", 1)
        self.run_ctx.setdefault("mlp_ratio", 4.0)
        self.run_ctx.setdefault("tp_world_size", 1)
        self.run_ctx.setdefault("runs", 10)
        self.run_ctx.setdefault("warmup", 5)
        self.run_ctx.setdefault("device", self.device_name)
        self.run_ctx.setdefault("cfg", "default")
        self._warned_no_layers_cfg = False

        # Detect NEW .pt ckpts (prefer per-device subdir if present)
        ckpt_dir = None
        device_dir = self.weight_dir / self.device_name
        if (device_dir / "proxy_ms.pt").is_file():
            ckpt_dir = device_dir
        elif (self.weight_dir / "proxy_ms.pt").is_file():
            ckpt_dir = self.weight_dir

        self._tabular: Optional[_Tabular3Proxy] = None
        self.lat_model = None
        self.mem_model = None
        self.power_model = None

        if ckpt_dir is not None:
            self._tabular = _Tabular3Proxy(ckpt_dir)
            ckpt_files = ["proxy_ms.pt", "proxy_peak_mem_mb.pt", "proxy_energy_mj.pt"]
            found = [name for name in ckpt_files if (ckpt_dir / name).is_file()]
            print(
                "[LayerHwProxy][ckpt] "
                f"device={self.device_name} ckpt_dir={ckpt_dir} files={found}"
            )
        else:
            # legacy .pth
            pth_dir = self.weight_dir
            if (pth_dir / "latency_proxy.pth").is_file():
                pass
            elif (self.weight_dir / self.device_name / "latency_proxy.pth").is_file():
                pth_dir = self.weight_dir / self.device_name

            in_dim = 4 + 4 + 5
            self.lat_model = self._load_model(pth_dir / "latency_proxy.pth", in_dim)
            self.mem_model = self._load_model(pth_dir / "mem_proxy.pth", in_dim)
            self.power_model = self._load_model(pth_dir / "power_proxy.pth", in_dim)
            print(
                "[LayerHwProxy][ckpt] "
                f"device={self.device_name} legacy_dir={pth_dir} files=[latency_proxy.pth, mem_proxy.pth, power_proxy.pth]"
            )

    def _load_model(self, path: Path, in_dim: int) -> LayerProxyModel:
        if not path.is_file():
            raise RuntimeError(f"[ProxyMissingLegacy] missing legacy proxy weight: {path}")
        model = LayerProxyModel(in_dim)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model

    def predict_layers_batch(self, layers_cfg: List[Dict]) -> Dict[str, np.ndarray]:
        default_device_cfg = self.gpu_cfg.get(self.device_name, {})
        if self._tabular is not None:
            pred = self._tabular.predict_all_numpy(layers_cfg, default_device_cfg, self.run_ctx)
            pred["lat_ms"] = np.maximum(pred["lat_ms"], 0.0)
            pred["mem_mb"] = np.maximum(pred["mem_mb"], 0.0)
            pred["power_w"] = np.maximum(pred["power_w"], 0.0)
            return pred

        feats = np.stack(
            [build_layer_features(cfg, cfg.get("device_cfg", default_device_cfg)) for cfg in layers_cfg],
            axis=0,
        )
        x = torch.tensor(feats, dtype=torch.float32)
        lat = self.lat_model(x).squeeze(-1).detach().cpu().numpy()
        mem = self.mem_model(x).squeeze(-1).detach().cpu().numpy()
        power = self.power_model(x).squeeze(-1).detach().cpu().numpy()
        pred = {"lat_ms": lat, "mem_mb": mem, "power_w": power}
        pred["lat_ms"] = np.maximum(pred["lat_ms"], 0.0)
        pred["mem_mb"] = np.maximum(pred["mem_mb"], 0.0)
        pred["power_w"] = np.maximum(pred["power_w"], 0.0)
        return pred

    def predict_layers_batch_torch(
        self,
        layers_cfg: List[Dict],
        device: torch.device | None = None,
    ) -> Dict[str, torch.Tensor]:
        default_device_cfg = self.gpu_cfg.get(self.device_name, {})
        dev = device if device is not None else torch.device("cpu")

        if self._tabular is not None:
            ms, mem, pw = self._tabular.predict_all_torch(layers_cfg, default_device_cfg, self.run_ctx, device=dev)
            return {"lat_ms": ms, "mem_mb": mem, "power_w": pw}

        feats = []
        for cfg in layers_cfg:
            dc = cfg.get("device_cfg", default_device_cfg)
            feats.append(build_layer_features_torch(cfg, dc, device=dev))
        x = torch.stack(feats, dim=0).to(torch.float32)

        self.lat_model = self.lat_model.to(dev)
        self.mem_model = self.mem_model.to(dev)
        self.power_model = self.power_model.to(dev)

        lat = self.lat_model(x).squeeze(-1)
        mem = self.mem_model(x).squeeze(-1)
        power = self.power_model(x).squeeze(-1)

        lat = torch.clamp(lat, min=0.0)
        mem = torch.clamp(mem, min=0.0)
        power = torch.clamp(power, min=0.0)
        return {"lat_ms": lat, "mem_mb": mem, "power_w": power}

    def predict_from_model_info(
        self,
        model_info: Dict,
        token_keep: float = 1.0,
        head_keep: float = 1.0,
        ch_keep: float = 1.0,
        block_keep: float = 1.0,
    ) -> Dict[str, float]:
        layers_cfg = model_info.get("layers_cfg") or model_info.get("layer_configs")
        if layers_cfg:
            pred = self.predict_layers_batch(layers_cfg)
            lat_ms = float(np.sum(pred["lat_ms"]))
            mem_mb = float(np.max(pred["mem_mb"])) if pred["mem_mb"].size > 0 else 0.0
            energy_mj = float(np.sum(pred["power_w"] * pred["lat_ms"])) / 1000.0 if pred["power_w"].size > 0 else 0.0
            return {"latency_ms": lat_ms, "mem_mb": mem_mb, "energy_mj": energy_mj}

        base_latency = float(model_info.get("latency_ms_ref", 1.0))
        base_mem = float(model_info.get("mem_mb_ref", 1.0))
        base_energy = float(model_info.get("energy_mj_ref", 1.0))
        if not self._warned_no_layers_cfg:
            print(
                "[LayerHwProxy][warn] model_info missing layers_cfg; using ref_* fallback (lat/mem/energy may look constant)."
            )
            self._warned_no_layers_cfg = True
        scale = max(1e-6, token_keep * head_keep * ch_keep * block_keep)
        return {
            "latency_ms": base_latency * scale,
            "mem_mb": base_mem * max(1e-6, ch_keep),
            "energy_mj": base_energy * scale,
        }
