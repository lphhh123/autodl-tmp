
import math
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml


# ---------- FLOPs / Bytes estimators (aligned with your training scripts) ----------

PATCH_SIZE = 16
IN_CHANS = 3


def gemm_flops_bytes(M, K, N, bpe):
    flops = 2.0 * M * K * N
    bytes_ = bpe * (M * K + K * N + M * N)
    return flops, bytes_


def attn_core_flops_bytes(L, dmodel, dk, dv, heads, bpe):
    H = heads if heads is not None else max(1, dmodel // dk)
    flops = 2.0 * H * (L ** 2) * (dk + dv)

    bytes_qkv = 3.0 * L * H * dk
    bytes_out = 1.0 * L * H * dv
    bytes_attn = 2.0 * (L ** 2) * H
    bytes_ = bpe * (bytes_qkv + bytes_out + bytes_attn)
    return flops, bytes_


def flops_bytes_layer(row: Dict[str, Any], bpe: int) -> Tuple[float, float]:
    layer_type = row["layer_type"]
    bs = row["bs"]
    embed_dim = row["embed_dim"]
    num_heads = max(1, row["num_heads"])
    mlp_ratio = row["mlp_ratio"]
    L_patch = row["L_patch"]
    L_eff = row["L_eff"]

    if layer_type == "patch_embed":
        M = L_patch * bs
        K = IN_CHANS * PATCH_SIZE * PATCH_SIZE
        N = embed_dim
        F, B = gemm_flops_bytes(M, K, N, bpe)

    elif layer_type == "attn":
        head_dim = embed_dim // num_heads
        dk = dv = head_dim

        M_qkv = L_eff * bs
        K_qkv = embed_dim
        N_qkv = 3 * embed_dim
        F_qkv, B_qkv = gemm_flops_bytes(M_qkv, K_qkv, N_qkv, bpe)

        F_attn, B_attn = attn_core_flops_bytes(
            L=L_eff, dmodel=embed_dim, dk=dk, dv=dv, heads=num_heads, bpe=bpe
        )
        F_attn *= bs
        B_attn *= bs

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


# ---------- GPU meta loading ----------


def load_gpu_data(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    chip_map = {}
    for chip in data.get("chiplets", []):
        chip_map[chip["name"]] = chip

    defaults = data.get("defaults", {})
    return chip_map, defaults


# ---------- MLP regressors (same as training script) ----------


class MLPRegressor(nn.Module):
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


# ---------- Feature builder (for ms/mem/power) ----------


def build_features_single(row: Dict[str, Any], chip_map: Dict[str, Any], defaults: Dict[str, Any]):
    layer_type = row["layer_type"]
    prec = row.get("prec", "fp16").lower()
    if prec == "fp16":
        prec_tag = "FP16"
        bpe = 2
    else:
        prec_tag = "FP32"
        bpe = 4

    device_name = row.get("device", None)
    if device_name is None:
        raise ValueError("row must contain 'device' field for LayerHwProxy")

    chip_key = f"{device_name}_{prec_tag}"
    chip = chip_map.get(chip_key, None)
    if chip is None:
        raise KeyError(f"Unknown chip {chip_key}; please check gpu_data.yaml")

    peak_flops = float(chip["peak_flops"])
    peak_bw = float(chip["peak_bw"])

    eta_comp_default = float(defaults.get("eta_comp", 0.75))
    eta_bw_default = float(defaults.get("eta_bw", 0.75))
    eta_comp = eta_comp_default
    eta_bw = eta_bw_default

    F_L, B_L = flops_bytes_layer(row, bpe)

    t_comp = F_L / (eta_comp * peak_flops + 1e-9)
    t_bw = B_L / (eta_bw * peak_bw + 1e-9)
    t_roof = max(t_comp, t_bw)

    if layer_type == "patch_embed":
        lt = [1.0, 0.0, 0.0]
    elif layer_type == "attn":
        lt = [0.0, 1.0, 0.0]
    else:
        lt = [0.0, 0.0, 1.0]

    img = row["img"]
    bs = row["bs"]
    keep = row["keep_ratio"]
    L_patch = row["L_patch"]
    L_eff = row["L_eff"]
    embed_dim = row["embed_dim"]
    num_heads = max(1, row["num_heads"])
    mlp_ratio = row["mlp_ratio"]
    head_dim = row["head_dim"]
    complexity_ratio = row.get("complexity_ratio", 1.0)

    x = [
        math.log10(F_L + 1e-9),
        math.log10(B_L + 1e-9),
        math.log10(t_roof + 1e-9),
        t_comp / (t_roof + 1e-9),
        t_bw / (t_roof + 1e-9),
        img / 224.0,
        bs / 64.0,
        keep,
        L_patch / 256.0,
        L_eff / 256.0,
        embed_dim / 1280.0,
        num_heads / 16.0,
        head_dim / 128.0,
        mlp_ratio / 4.0,
        complexity_ratio,
        math.log10(peak_flops + 1e-9),
        math.log10(peak_bw + 1e-9),
        *lt,
    ]
    return np.asarray(x, dtype=np.float32)


# ---------- High-level LayerHwProxy ----------


class LayerHwProxy:
    """Wrap ms/mem/power proxy models for a specific GPU.

    This class assumes the proxy weights are already trained and stored as:
      - layer_proxy_ms_{device}.pt
      - layer_proxy_mem_{device}.pt
      - layer_proxy_power_dyn_{device}.pt  (or layer_proxy_power_{device}.pt)

    The device string should match the 'device' column in your layer dataset
    (e.g., 'RTX2080Ti', 'RTX3090', 'RTX4090' or simplified '2080ti', '3090', '4090'
    depending on how gpu_data.yaml and the dataset were written).
    """

    def __init__(self, device_name: str, gpu_yaml: str, weight_dir: str = "proxy_weights"):
        self.device_name = device_name
        self.weight_dir = weight_dir

        self.chip_map, self.defaults = load_gpu_data(gpu_yaml)

        # Infer feature dimension from a dummy row (this is a bit hacky but fine here)
        dummy_row = dict(
            layer_type="attn",
            bs=1,
            img=224,
            keep_ratio=1.0,
            L_patch=256,
            L_eff=256,
            depth=0,
            embed_dim=768,
            num_heads=12,
            mlp_ratio=4.0,
            complexity_ratio=1.0,
            head_dim=64,
            tp_world_size=1,
            prec="fp16",
            device=self.device_name,
        )
        x = build_features_single(dummy_row, self.chip_map, self.defaults)
        in_dim = int(x.shape[0])

        self.ms_model = MLPRegressor(in_dim=in_dim)
        self.mem_model = MLPRegressor(in_dim=in_dim)
        self.power_model = PowerProxyMLP(in_dim=in_dim)

        # Load weights if available; otherwise keep random (for dummy run).
        self._load_weights()

    def _load_weights(self):
        import os

        ms_path = os.path.join(self.weight_dir, f"layer_proxy_ms_{self.device_name}.pt")
        mem_path = os.path.join(self.weight_dir, f"layer_proxy_mem_{self.device_name}.pt")
        power_dyn_path = os.path.join(
            self.weight_dir, f"layer_proxy_power_dyn_{self.device_name}.pt"
        )
        power_path = os.path.join(self.weight_dir, f"layer_proxy_power_{self.device_name}.pt")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ms_model.to(device)
        self.mem_model.to(device)
        self.power_model.to(device)

        if os.path.exists(ms_path):
            self.ms_model.load_state_dict(torch.load(ms_path, map_location=device))
            print(f"[LayerHwProxy] device={self.device_name}, ms_model=OK")
        else:
            print(f"[LayerHwProxy] device={self.device_name}, ms_model=RANDOM (no weight file)")
        if os.path.exists(mem_path):
            self.mem_model.load_state_dict(torch.load(mem_path, map_location=device))
            print(f"[LayerHwProxy] device={self.device_name}, mem_model=OK")
        else:
            print(f"[LayerHwProxy] device={self.device_name}, mem_model=RANDOM (no weight file)")
        if os.path.exists(power_dyn_path):
            self.power_model.load_state_dict(torch.load(power_dyn_path, map_location=device))
            print(f"[LayerHwProxy] device={self.device_name}, power_model=OK (dyn)")
        elif os.path.exists(power_path):
            self.power_model.load_state_dict(torch.load(power_path, map_location=device))
            print(f"[LayerHwProxy] device={self.device_name}, power_model=OK")
        else:
            print(f"[LayerHwProxy] device={self.device_name}, power_model=RANDOM (no weight file)")

        self.device = device
        self.ms_model.eval()
        self.mem_model.eval()
        self.power_model.eval()

    def predict(self, layer_row: Dict[str, Any]) -> Dict[str, float]:
        """Predict (ms, mem_mb, power_w) for one layer row.

        layer_row must already contain all meta fields, but 'device' will be
        overwritten by this proxy's device_name.
        """
        row = dict(layer_row)
        row["device"] = self.device_name
        x = build_features_single(row, self.chip_map, self.defaults)
        xt = torch.from_numpy(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            ms = self.ms_model(xt).item()
            mem = self.mem_model(xt).item()
            power = self.power_model(xt).item()

        return dict(ms=max(ms, 0.01), mem_mb=max(mem, 1.0), power_w=max(power, 1.0))
