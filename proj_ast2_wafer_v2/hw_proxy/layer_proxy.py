
import os
import math
from typing import Dict, Any

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants and basic FLOPs / Bytes estimation utilities
# These mirror the logic used when training the layerwise ms/mem/power proxies.
# ---------------------------------------------------------------------------

PATCH_SIZE = 16
IN_CHANS = 3


def gemm_flops_bytes(M: int, K: int, N: int, bpe: int):
    """FLOPs and Bytes for a dense GEMM: [M, K] x [K, N]."""
    flops = 2.0 * M * K * N
    bytes_ = bpe * (M * K + K * N + M * N)
    return flops, bytes_


def attn_core_flops_bytes(L: int, dmodel: int, dk: int, dv: int, heads: int, bpe: int):
    """Approximate FLOPs / Bytes for attention core (softmax(QK^T)V)."""
    H = heads if heads is not None else max(1, dmodel // dk)

    # FLOPs ~ 2 * H * L^2 * (dk + dv)
    flops = 2.0 * H * (L ** 2) * (dk + dv)

    # Bytes 近似：Q/K/V + 输出 + 注意力矩阵
    bytes_qkv = 3.0 * L * H * dk
    bytes_out = 1.0 * L * H * dv
    bytes_attn = 2.0 * (L ** 2) * H
    bytes_ = bpe * (bytes_qkv + bytes_out + bytes_attn)
    return flops, bytes_


def flops_bytes_layer(meta: Dict[str, Any], bpe: int):
    """Estimate FLOPs / Bytes for a single ViT layer.

    meta keys expected (per your profiling dataset scripts):
      - layer_type: 'patch_embed' / 'attn' / 'mlp'
      - bs, embed_dim, num_heads, mlp_ratio, L_patch, L_eff
    """
    layer_type = meta["layer_type"]
    bs = meta["bs"]
    embed_dim = meta["embed_dim"]
    num_heads = max(1, meta["num_heads"])
    mlp_ratio = meta["mlp_ratio"]
    L_patch = meta["L_patch"]
    L_eff = meta["L_eff"]

    if layer_type == "patch_embed":
        # conv/linear → [B, 3, H, W] -> [B, L_patch, C]
        M = L_patch * bs
        K = IN_CHANS * PATCH_SIZE * PATCH_SIZE
        N = embed_dim
        F, B = gemm_flops_bytes(M, K, N, bpe)

    elif layer_type == "attn":
        head_dim = embed_dim // num_heads
        dk = dv = head_dim

        # QKV GEMM
        M_qkv = L_eff * bs
        K_qkv = embed_dim
        N_qkv = 3 * embed_dim
        F_qkv, B_qkv = gemm_flops_bytes(M_qkv, K_qkv, N_qkv, bpe)

        # 注意力核心（per head）
        F_attn, B_attn = attn_core_flops_bytes(
            L=L_eff, dmodel=embed_dim, dk=dk, dv=dv,
            heads=num_heads, bpe=bpe
        )
        F_attn *= bs
        B_attn *= bs

        # 输出投影
        M_proj = L_eff * bs
        K_proj = embed_dim
        N_proj = embed_dim
        F_proj, B_proj = gemm_flops_bytes(M_proj, K_proj, N_proj, bpe)

        F = F_qkv + F_attn + F_proj
        B = B_qkv + B_attn + B_proj

    else:  # mlp
        d_ff = int(embed_dim * mlp_ratio)
        M = L_eff * bs
        F1, B1 = gemm_flops_bytes(M, embed_dim, d_ff, bpe)
        F2, B2 = gemm_flops_bytes(M, d_ff, embed_dim, bpe)
        F = F1 + F2
        B = B1 + B2

    return F, B


def _encode_layer_type(layer_type: str):
    """One-hot encode layer type: patch_embed / attn / mlp."""
    if layer_type == "patch_embed":
        return [1.0, 0.0, 0.0]
    elif layer_type == "attn":
        return [0.0, 1.0, 0.0]
    else:
        return [0.0, 0.0, 1.0]


def build_layer_features(layer_meta: Dict[str, Any], gpu_spec: Dict[str, Any]):
    """Build input feature vector for the proxies (ms/mem/power).

    This is consistent with the build_features() function in your
    train_layerwise_proxy.py and train_layerwise_power_proxy.py scripts.

    We assume the proxies are trained for a fixed precision (typically fp16),
    so we fix bpe=2 here. If you later train separate fp32 proxies, you can
    extend this function to select bpe based on a 'prec' field in layer_meta.
    """
    # Precision: assume fp16 for now
    bpe = 2

    eta_comp = 0.75
    eta_bw = 0.75

    # --------------- basic per-layer stats ---------------
    F_L, B_L = flops_bytes_layer(layer_meta, bpe)

    peak_flops = float(gpu_spec.get("peak_flops", 1.0))
    peak_bw = float(gpu_spec.get("peak_bw", 1.0))

    t_comp = F_L / (eta_comp * peak_flops + 1e-9)
    t_bw = B_L / (eta_bw * peak_bw + 1e-9)
    t_roof = max(t_comp, t_bw)

    layer_type = layer_meta["layer_type"]
    lt = _encode_layer_type(layer_type)

    img = layer_meta.get("img", 224)
    bs = layer_meta.get("bs", 1)
    keep = layer_meta.get("keep_ratio", 1.0)
    L_patch = layer_meta.get("L_patch", layer_meta.get("L_eff", 196))
    L_eff = layer_meta.get("L_eff", L_patch)

    embed_dim = layer_meta["embed_dim"]
    num_heads = max(1, layer_meta["num_heads"])
    mlp_ratio = layer_meta["mlp_ratio"]
    head_dim = layer_meta.get("head_dim", embed_dim // num_heads)
    complexity_ratio = layer_meta.get("complexity_ratio", 1.0)

    x = [
        # 1) layer FLOPs / Bytes
        math.log10(F_L + 1e-9),
        math.log10(B_L + 1e-9),

        # 2) roofline 下界
        math.log10(t_roof + 1e-9),
        t_comp / (t_roof + 1e-9),
        t_bw / (t_roof + 1e-9),

        # 3) 输入规模 & token 信息
        img / 224.0,
        bs / 64.0,
        keep,
        L_patch / 256.0,
        L_eff / 256.0,

        # 4) 结构超参（相对 ViT-H）
        embed_dim / 1280.0,
        num_heads / 16.0,
        head_dim / 128.0,
        mlp_ratio / 4.0,
        complexity_ratio,

        # 5) GPU 峰值参数
        math.log10(peak_flops + 1e-9),
        math.log10(peak_bw + 1e-9),

        # 6) layer 类型 one-hot
        *lt,
    ]

    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # [1, D]
    return x_tensor

class MLPRegressor(nn.Module):
    """Same architecture as in train_layerwise_proxy.py (for ms/mem)."""

    def __init__(self, in_dim: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.ReLU())
            d = hidden
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PowerProxyMLP(nn.Module):
    """Same architecture as in train_layerwise_power_proxy.py."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

class LayerHwProxy:
    """Layer-wise hardware oracle using your trained ms/mem/power proxies.

    This class is instantiated with:
      - gpu_yaml_path: mapping from chip_name -> {peak_flops, peak_bw, ...}
      - proxy_weight_dir: directory containing weight files like:
            layer_proxy_ms_2080ti.pt
            layer_proxy_mem_2080ti.pt
            layer_proxy_power_dyn_2080ti.pt
        (and similarly for 3090 / 4090)
      - device: torch device to run the tiny MLPs on

    It exposes:
      - predict_layer(chip_name, layer_meta) -> dict with
            {ms, peak_mem_mb, avg_power_w, energy_mj}
    """

    def __init__(self, gpu_yaml_path: str, proxy_weight_dir: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.gpu_specs = self._load_gpu_specs(gpu_yaml_path)
        self.proxy_weight_dir = proxy_weight_dir

        # cache { (chip_name, metric) : model }
        self._models: Dict[str, nn.Module] = {}

    # ---------------- GPU spec loader ---------------- #

    def _load_gpu_specs(self, yaml_path: str) -> Dict[str, Dict[str, Any]]:
        import yaml

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"gpu_data.yaml should be a dict mapping name->spec, got {type(data)}")

        gpu_specs = {}
        for name, spec in data.items():
            if not isinstance(spec, dict):
                continue
            gpu_specs[name] = spec
        return gpu_specs

    # ---------------- Proxy weight key helper ---------------- #

    def _build_model_key(self, chip_name: str, metric: str) -> str:
        """Return the filename for the proxy weights for a given chip & metric.

        We follow the naming convention you described:
          - layer_proxy_ms_2080ti.pt
          - layer_proxy_mem_2080ti.pt
          - layer_proxy_power_dyn_2080ti.pt
        and just replace the GPU name part.
        """
        if metric == "ms":
            fname = f"layer_proxy_ms_{chip_name}.pt"
        elif metric == "mem":
            fname = f"layer_proxy_mem_{chip_name}.pt"
        elif metric == "power":
            fname = f"layer_proxy_power_dyn_{chip_name}.pt"
        else:
            raise ValueError(f"Unknown metric type: {metric}")
        return os.path.join(self.proxy_weight_dir, fname)

    # ---------------- Model loader ---------------- #

    def _load_model(self, chip_name: str, metric: str) -> nn.Module:
        key = f"{chip_name}:{metric}"
        if key in self._models:
            return self._models[key]

        weight_path = self._build_model_key(chip_name, metric)
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Proxy weight file not found: {weight_path}")

        # Feature dimension is fixed by build_layer_features: 20
        in_dim = 20

        if metric in ("ms", "mem"):
            model = MLPRegressor(in_dim=in_dim, hidden=128, depth=3)
        elif metric == "power":
            model = PowerProxyMLP(in_dim=in_dim, hidden=128)
        else:
            raise ValueError(f"Unknown metric type: {metric}")

        state_dict = torch.load(weight_path, map_location=self.device)
        # In your training code you saved state_dict directly
        if isinstance(state_dict, dict):
            model.load_state_dict(state_dict)
        else:
            # fallback: if somehow an nn.Module was saved
            model = state_dict

        model.to(self.device)
        model.eval()
        self._models[key] = model
        return model

    # ---------------- Single-metric forward ---------------- #

    def _forward_proxy(self, chip_name: str, metric: str, x: torch.Tensor) -> float:
        """Run one of the proxies and post-process back to original scale.

        For ms/mem, your training used log-target regression:
          y_proc = log(y + eps)
        so at inference we must apply exp and subtract eps.

        For power, you regressed the raw avg_power_w directly.
        """
        model = self._load_model(chip_name, metric)
        x = x.to(self.device)

        with torch.no_grad():
            y = model(x).view(-1)[0].item()

        if metric in ("ms", "mem"):
            eps = 1e-3
            y = math.exp(y) - eps
            if y < 0.0:
                y = 0.0
        return float(y)

    # ---------------- Public API ---------------- #

    def predict_layer(self, chip_name: str, layer_meta: Dict[str, Any]) -> Dict[str, float]:
        """Predict (ms, mem, power, energy) for a single layer on a given chip.

        Args:
            chip_name: must match a key in configs/gpu_data.yaml AND the suffix
                       of your proxy weight files (e.g. '2080ti', '3090', '4090').
            layer_meta: dict with at least:
                layer_type, bs, img, keep_ratio, L_patch, L_eff,
                embed_dim, num_heads, mlp_ratio, complexity_ratio, head_dim.

        Returns:
            dict with keys: ms, peak_mem_mb, avg_power_w, energy_mj
        """
        if chip_name not in self.gpu_specs:
            raise KeyError(f"Unknown chip_name '{chip_name}' for hardware proxy; "
                           f"please add it to configs/gpu_data.yaml")

        gpu_spec = self.gpu_specs[chip_name]
        x = build_layer_features(layer_meta, gpu_spec)

        ms = self._forward_proxy(chip_name, "ms", x)
        mem = self._forward_proxy(chip_name, "mem", x)
        power = self._forward_proxy(chip_name, "power", x)

        # energy_mj ≈ power[W] * time[ms]
        energy_mj = power * ms

        return {
            "ms": ms,
            "peak_mem_mb": mem,
            "avg_power_w": power,
            "energy_mj": energy_mj,
        }
