"""Layer hardware proxy wrapper (v5.4) — use NEW tabular .pt proxies only.

This file replaces legacy .pth proxies:
  - latency_proxy.pth / mem_proxy.pth / power_proxy.pth

with NEW ckpts:
  - proxy_ms.pt
  - proxy_peak_mem_mb.pt
  - proxy_energy_mj.pt

Outputs:
  - lat_ms, mem_mb, power_w (power derived from energy/latency)

Key requirements:
  - torch path must be differentiable (for stable_hw hw_grad smoke)
  - clamp outputs to non-negative (prevent negative reward)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from proxy_retrain.proxy_utils import MLPRegressor


# ---------------------------
# Helpers: device token mapping
# ---------------------------
def _device_token(device_name: str) -> str:
    # RTX3090_FP16 -> RTX3090, RTX2080Ti_FP16 -> RTX2080Ti
    s = str(device_name)
    s = s.replace("_FP16", "").replace("_FP32", "").replace("_fp16", "").replace("_fp32", "")
    # normalize 2080ti naming
    if "2080" in s and ("ti" in s.lower()) and ("Ti" not in s):
        s = s.replace("2080ti", "2080Ti").replace("2080TI", "2080Ti")
    return s


def _prec_token(precision: Any) -> str:
    # project uses precision int: 1=>fp16, 0=>fp32
    try:
        p = float(precision)
        return "fp16" if p >= 1.0 else "fp32"
    except Exception:
        return str(precision)


def _layer_type_token(layer_cfg: Dict[str, Any]) -> str:
    # prefer explicit string
    if "layer_kind" in layer_cfg:
        return str(layer_cfg["layer_kind"])
    # fallback from numeric id (legacy)
    idx = int(layer_cfg.get("layer_type", 3))
    return {0: "patch_embed", 1: "attn", 2: "mlp"}.get(idx, "other")


def _cfg_token(depth: int, embed_dim: int, num_heads: int, mlp_ratio: float) -> str:
    # follow proxy_retrain dataset format: vith_gridH_d{d}_e{e}_h{h}_r{r}
    r = str(float(mlp_ratio)).replace(".", "_")
    return f"vith_gridH_d{int(depth)}_e{int(embed_dim)}_h{int(num_heads)}_r{r}"


# ---------------------------
# Tabular PT bundle (torch differentiable)
# ---------------------------
class _TabularBundle:
    def __init__(self, ckpt_path: Path):
        bundle = torch.load(str(ckpt_path), map_location="cpu")
        self.num_cols: List[str] = list(bundle["num_cols"])
        self.cat_cols: List[str] = list(bundle["cat_cols"])
        self.cat_vocab: Dict[str, List[str]] = dict(bundle["cat_vocab"])
        self.stats: Dict[str, Dict[str, float]] = dict(bundle["stats"])
        self.target_mode: str = str(bundle.get("target_mode", "linear"))
        self.target_col: str = str(bundle.get("target_col", ""))

        self.input_dim = int(bundle["input_dim"])
        hidden = list(bundle.get("model_hidden", [256, 256, 128]))
        dropout = float(bundle.get("dropout", 0.1))

        self.model = MLPRegressor(input_dim=self.input_dim, hidden=hidden, dropout=dropout)
        self.model.load_state_dict(bundle["model_state"], strict=True)
        self.model.eval()

        # build categorical value->index maps (strict)
        self._cat_to_index: Dict[str, Dict[str, int]] = {}
        for c in self.cat_cols:
            vs = self.cat_vocab.get(c, [])
            self._cat_to_index[c] = {v: i for i, v in enumerate(vs)}

        # cache mean/std tensors for numeric
        mu = [float(self.stats["num_mean"].get(c, 0.0)) for c in self.num_cols]
        sd = [float(self.stats["num_std"].get(c, 1.0)) for c in self.num_cols]
        sd = [x if abs(x) > 1e-12 else 1.0 for x in sd]
        self._mu = torch.tensor(mu, dtype=torch.float32)
        self._sd = torch.tensor(sd, dtype=torch.float32)

    def _encode_cat_onehot(self, col: str, values: List[str], device: torch.device) -> torch.Tensor:
        v2i = self._cat_to_index[col]
        n_cls = len(v2i)
        idx = []
        for v in values:
            if v not in v2i:
                # strict fail-fast: no silent fallback
                raise ValueError(f"[ProxyVocabMismatch] col={col} value={v!r} not in vocab (size={n_cls})")
            idx.append(v2i[v])
        idx_t = torch.tensor(idx, device=device, dtype=torch.long)
        return F.one_hot(idx_t, num_classes=n_cls).to(torch.float32)

    def build_x_torch(self, rows: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
        # numeric: stack per column (support torch tensors)
        num_parts = []
        for c in self.num_cols:
            vals = []
            for r in rows:
                v = r[c]
                if torch.is_tensor(v):
                    vals.append(v.to(device=device, dtype=torch.float32))
                else:
                    vals.append(torch.tensor(float(v), device=device, dtype=torch.float32))
            num_parts.append(torch.stack(vals, dim=0))  # [N]
        x_num = torch.stack(num_parts, dim=1) if num_parts else torch.zeros((len(rows), 0), device=device)

        # standardize
        mu = self._mu.to(device=device)
        sd = self._sd.to(device=device)
        x_num = (x_num - mu) / sd

        # categorical: concat one-hots
        cat_parts = []
        for c in self.cat_cols:
            vals = [str(r[c]) for r in rows]
            cat_parts.append(self._encode_cat_onehot(c, vals, device=device))
        x_cat = torch.cat(cat_parts, dim=1) if cat_parts else torch.zeros((len(rows), 0), device=device)

        x = torch.cat([x_num, x_cat], dim=1)
        if int(x.shape[1]) != int(self.input_dim):
            raise RuntimeError(f"[ProxyInputDimMismatch] got {int(x.shape[1])} expected {self.input_dim}")
        return x

    def predict_torch(self, rows: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
        self.model = self.model.to(device)
        x = self.build_x_torch(rows, device=device)
        y = self.model(x)  # [N]
        if self.target_mode.lower() == "log":
            # inverse of log(x+eps): exp(y)-eps
            y = torch.exp(y) - 1e-6
            y = torch.clamp(y, min=1e-6)
        # guardrail: non-negative
        y = torch.clamp(y, min=0.0)
        return y


class _Tabular3Proxy:
    """3-head proxy via 3 independent ckpts: ms, peak_mem_mb, energy_mj."""

    def __init__(self, ckpt_dir: Path):
        ms_pt = ckpt_dir / "proxy_ms.pt"
        mem_pt = ckpt_dir / "proxy_peak_mem_mb.pt"
        eng_pt = ckpt_dir / "proxy_energy_mj.pt"
        if not (ms_pt.is_file() and mem_pt.is_file() and eng_pt.is_file()):
            raise FileNotFoundError(
                f"[ProxyMissing] expected in {ckpt_dir}:\n"
                f"  proxy_ms.pt / proxy_peak_mem_mb.pt / proxy_energy_mj.pt"
            )
        self.ms = _TabularBundle(ms_pt)
        self.mem = _TabularBundle(mem_pt)
        self.eng = _TabularBundle(eng_pt)

        # cache cfg vocab for strict check (no silent drift)
        self.cfg_vocab = set(self.ms.cat_vocab.get("cfg", []))

    def predict_all_torch(
        self,
        layers_cfg: List[Dict[str, Any]],
        device_name: str,
        default_device_cfg: Dict[str, Any],
        run_ctx: Dict[str, Any],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Build base rows (shared fields)
        rows_base: List[Dict[str, Any]] = []
        for cfg in layers_cfg:
            dc = cfg.get("device_cfg", default_device_cfg) or {}
            depth = int(cfg.get("depth", run_ctx.get("depth", 0)))
            embed_dim = int(cfg.get("embed_dim", run_ctx.get("embed_dim", 0)))
            num_heads = int(cfg.get("num_heads", run_ctx.get("num_heads", 1)))
            mlp_ratio = float(cfg.get("mlp_ratio", run_ctx.get("mlp_ratio", 4.0)))
            cfg_tag = _cfg_token(depth, embed_dim, num_heads, mlp_ratio)
            if cfg_tag not in self.cfg_vocab:
                # strict: fail-fast (avoid silent semantic drift)
                raise ValueError(
                    f"[ProxyCfgOOV] cfg={cfg_tag} not in proxy vocab. "
                    f"Your model (depth/embed/heads/mlp_ratio) must match the proxy training set."
                )

            keep_ratio = float(cfg.get("keep_ratio", 1.0))
            L_patch = float(cfg.get("seq_len", 0.0))  # VideoViT uses spatial patches count
            # L_eff = floor(L_patch*keep_ratio) + 1 (matches your CSV)
            L_eff = float(math.floor(L_patch * keep_ratio) + 1.0)

            row = {
                "device": _device_token(device_name),
                "cfg": cfg_tag,
                "layer_type": _layer_type_token(cfg),
                "prec": _prec_token(cfg.get("precision", 1.0)),
                "img": float(run_ctx.get("img", 224)),
                "bs": float(run_ctx.get("bs", 1)),
                "keep_ratio": float(keep_ratio),
                "L_patch": float(L_patch),
                "L_eff": float(L_eff),
                "status": "ok",
                "depth": float(depth),
                "embed_dim": float(embed_dim),
                "num_heads": float(num_heads),
                "mlp_ratio": float(mlp_ratio),
                "complexity_ratio": float(cfg.get("complexity_ratio", 1.0)),
                "head_dim": float(embed_dim) / max(1.0, float(num_heads)),
                "tp_world_size": float(run_ctx.get("tp_world_size", 1)),
            }
            rows_base.append(row)

        # init ms using roofline-like estimate (keeps gradient via peak_flops/peak_bw in torch path)
        def _as_t(v, default=0.0):
            if torch.is_tensor(v):
                return v.to(device=device, dtype=torch.float32)
            return torch.tensor(float(v if v is not None else default), device=device, dtype=torch.float32)

        ms0 = []
        for cfg in layers_cfg:
            dc = cfg.get("device_cfg", default_device_cfg) or {}
            flops = float(cfg.get("flops", 0.0))
            bytes_ = float(cfg.get("bytes", 0.0))
            peak_flops = _as_t(dc.get("peak_flops", 1e12), default=1e12)
            peak_bw = _as_t(dc.get("peak_bw", 0.0), default=0.0)
            t_comp = torch.tensor(flops, device=device) / (peak_flops + 1e-9)
            t_mem = torch.tensor(0.0, device=device)
            if float(bytes_) > 0.0 and torch.any(peak_bw > 0):
                t_mem = torch.tensor(bytes_, device=device) / (peak_bw + 1e-9)
            ms_est = (t_comp + t_mem) * 1e3
            ms0.append(torch.clamp(ms_est, min=1e-4))
        ms = torch.stack(ms0, dim=0)  # [N]

        # fixed-point: mem(ms) -> ms(mem), repeat twice
        for _ in range(2):
            rows_mem = []
            for i, base in enumerate(rows_base):
                r = dict(base)
                r["ms"] = ms[i]
                rows_mem.append(r)
            mem_mb = self.mem.predict_torch(rows_mem, device=device)

            rows_ms = []
            for i, base in enumerate(rows_base):
                r = dict(base)
                r["peak_mem_mb"] = mem_mb[i]
                rows_ms.append(r)
            ms = self.ms.predict_torch(rows_ms, device=device)

        # energy proxy: needs avg_power_w, ms_event, runs, warmup
        rows_eng = []
        for i, base in enumerate(rows_base):
            dc = layers_cfg[i].get("device_cfg", default_device_cfg) or {}
            avg_power_w = _as_t(dc.get("tdp_w", 200.0), default=200.0)
            r = dict(base)
            r["avg_power_w"] = avg_power_w
            r["ms_event"] = ms[i]
            r["runs"] = float(run_ctx.get("runs", 10))
            r["warmup"] = float(run_ctx.get("warmup", 5))
            rows_eng.append(r)
        energy_mj = self.eng.predict_torch(rows_eng, device=device)

        power_w = energy_mj * 1000.0 / (ms + 1e-6)
        power_w = torch.clamp(power_w, min=0.0)
        return ms, mem_mb, power_w

    @torch.no_grad()
    def predict_all_numpy(
        self,
        layers_cfg: List[Dict[str, Any]],
        device_name: str,
        default_device_cfg: Dict[str, Any],
        run_ctx: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        dev = torch.device("cpu")
        ms, mem, pw = self.predict_all_torch(layers_cfg, device_name, default_device_cfg, run_ctx, device=dev)
        return {
            "lat_ms": ms.detach().cpu().numpy(),
            "mem_mb": mem.detach().cpu().numpy(),
            "power_w": pw.detach().cpu().numpy(),
        }


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
        # provide safe defaults (CSV shows bs=1 is common; img default from cfg)
        self.run_ctx.setdefault("img", 224)
        self.run_ctx.setdefault("bs", 1)
        self.run_ctx.setdefault("tp_world_size", 1)
        self.run_ctx.setdefault("runs", 10)
        self.run_ctx.setdefault("warmup", 5)

        # NEW: always use tabular proxies
        self._tab = _Tabular3Proxy(self.weight_dir)

    def predict_layers_batch(self, layers_cfg: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        default_device_cfg = self.gpu_cfg.get(self.device_name, {})
        pred = self._tab.predict_all_numpy(layers_cfg, self.device_name, default_device_cfg, self.run_ctx)
        # v5.4: non-negative guard
        pred["lat_ms"] = np.maximum(pred["lat_ms"], 0.0)
        pred["mem_mb"] = np.maximum(pred["mem_mb"], 0.0)
        pred["power_w"] = np.maximum(pred["power_w"], 0.0)
        return pred

    def predict_layers_batch_torch(
        self,
        layers_cfg: List[Dict[str, Any]],
        device: torch.device | None = None,
    ) -> Dict[str, torch.Tensor]:
        default_device_cfg = self.gpu_cfg.get(self.device_name, {})
        dev = device if device is not None else torch.device("cpu")
        ms, mem, pw = self._tab.predict_all_torch(layers_cfg, self.device_name, default_device_cfg, self.run_ctx, device=dev)
        # v5.4: guardrail
        ms = torch.clamp(ms, min=0.0)
        mem = torch.clamp(mem, min=0.0)
        pw = torch.clamp(pw, min=0.0)
        return {"lat_ms": ms, "mem_mb": mem, "power_w": pw}

    def predict_from_model_info(
        self,
        model_info: Dict[str, Any],
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

        # fallback: simple scaling
        base_latency = float(model_info.get("latency_ms_ref", 1.0))
        base_mem = float(model_info.get("mem_mb_ref", 1.0))
        base_energy = float(model_info.get("energy_mj_ref", 1.0))
        scale = max(1e-6, token_keep * head_keep * ch_keep * block_keep)
        return {
            "latency_ms": base_latency * scale,
            "mem_mb": base_mem * max(1e-6, ch_keep),
            "energy_mj": base_energy * scale,
        }
