"""Segment-to-chip mapping solver."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from hw_proxy.feature_builder import DeviceSpec
from hw_proxy.layer_hw_proxy import LayerHwProxy
from .segmenter import Segment


class MappingSolver:
    def __init__(self, hw_proxy: LayerHwProxy) -> None:
        self.hw_proxy = hw_proxy

    def build_cost_matrix(self, segments: List[Segment], devices: List[DeviceSpec]):
        cost = []
        for seg in segments:
            row = []
            for dev in devices:
                row.append({"lat_ms": seg.flops / max(1e-6, dev.peak_flops) * 1e3, "mem_mb": seg.bytes})
            cost.append(row)
        return cost

    def solve_mapping(self, segments: List[Segment], devices: List[DeviceSpec], layout=None, strategy: str = "proxy_dp"):
        mapping: Dict[int, int] = {}
        device_times = {i: 0.0 for i in range(len(devices))}
        for seg in segments:
            best_slot = 0
            best_time = float("inf")
            for i, dev in enumerate(devices):
                est_time = device_times[i] + seg.flops / max(1e-6, dev.peak_flops)
                if est_time < best_time:
                    best_time = est_time
                    best_slot = i
            mapping[seg.id] = best_slot
            device_times[best_slot] += best_time
        total_latency = max(device_times.values()) if device_times else 0.0
        comm_time = 0.0
        return mapping, device_times, total_latency, comm_time
