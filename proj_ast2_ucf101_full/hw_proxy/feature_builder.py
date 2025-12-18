"""Feature builders for hardware proxy models."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class LayerConfig:
    layer_type: str
    depth: int
    embed_dim: int
    num_heads: int
    mlp_ratio: float
    seq_len: int
    batch_size: int
    precision: str = "fp16"
    keep_ratio: float = 1.0


@dataclass
class DeviceSpec:
    name: str
    peak_flops: float
    peak_bw: float
    mem_size_gb: float
    area_mm2: float
    tdp_watt: float
    energy_per_bit_pj: float = 10.0
    mem_bandwidth: float | None = None


PRECISION_FLAG = {"fp16": 0.0, "fp32": 1.0}


def build_features(layer_cfg: LayerConfig, device_spec: DeviceSpec) -> torch.Tensor:
    flops = layer_cfg.seq_len * layer_cfg.embed_dim * layer_cfg.embed_dim * layer_cfg.mlp_ratio
    bytes_ = layer_cfg.seq_len * layer_cfg.embed_dim * 4
    feats = [
        math.log10(flops + 1e-9),
        math.log10(bytes_ + 1e-9),
        layer_cfg.depth,
        layer_cfg.embed_dim,
        layer_cfg.num_heads,
        layer_cfg.mlp_ratio,
        layer_cfg.seq_len,
        PRECISION_FLAG.get(layer_cfg.precision.lower(), 0.0),
        math.log10(device_spec.peak_flops + 1e-9),
        math.log10(device_spec.peak_bw + 1e-9),
        math.log10(device_spec.mem_size_gb + 1e-9),
    ]
    return torch.tensor(feats, dtype=torch.float32)


def to_device_spec(entry: Dict) -> DeviceSpec:
    return DeviceSpec(
        name=entry["name"],
        peak_flops=float(entry["peak_flops"]),
        peak_bw=float(entry.get("peak_bw", entry.get("mem_bandwidth", 0.0))),
        mem_size_gb=float(entry.get("mem_size_gb", 0.0)),
        area_mm2=float(entry.get("area_mm2", 0.0)),
        tdp_watt=float(entry.get("tdp_watt", 0.0)),
        energy_per_bit_pj=float(entry.get("energy_per_bit_pj", 10.0)),
        mem_bandwidth=float(entry.get("mem_bandwidth", entry.get("peak_bw", 0.0))),
    )
