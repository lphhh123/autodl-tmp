"""Layer hardware proxy wrapper following SPEC_version_c.md."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import yaml

from .layer_proxy_model import LayerProxyModel


LAYER_TYPES = {"patch_embed": 0, "attn": 1, "mlp": 2, "other": 3}


# SPEC §6.1

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


class LayerHwProxy:
    def __init__(self, device_name: str, gpu_yaml: str, weight_dir: str):
        with open(gpu_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # support chip_types list or dict keyed by name
        if isinstance(data, dict) and "chip_types" in data:
            self.gpu_cfg = {entry["name"]: entry for entry in data["chip_types"]}
        else:
            self.gpu_cfg = data
        self.device_name = device_name
        self.weight_dir = Path(weight_dir)
        in_dim = 4 + 4 + 5  # log feats + one-hot + normalized dims
        self.lat_model = self._load_model(self.weight_dir / "latency_proxy.pth", in_dim)
        self.mem_model = self._load_model(self.weight_dir / "mem_proxy.pth", in_dim)
        self.power_model = self._load_model(self.weight_dir / "power_proxy.pth", in_dim)

    def _load_model(self, path: Path, in_dim: int) -> LayerProxyModel:
        model = LayerProxyModel(in_dim)
        if path.is_file():
            model.load_state_dict(torch.load(path, map_location="cpu"))
        return model

    def predict_layers_batch(self, layers_cfg: List[Dict]) -> Dict[str, np.ndarray]:
        default_device_cfg = self.gpu_cfg.get(self.device_name, {})
        feats = np.stack(
            [
                build_layer_features(cfg, cfg.get("device_cfg", default_device_cfg))
                for cfg in layers_cfg
            ],
            axis=0,
        )
        x = torch.tensor(feats, dtype=torch.float32)
        lat = self.lat_model(x).squeeze(-1).detach().cpu().numpy()
        mem = self.mem_model(x).squeeze(-1).detach().cpu().numpy()
        power = self.power_model(x).squeeze(-1).detach().cpu().numpy()
        return {"lat_ms": lat, "mem_mb": mem, "power_w": power}
