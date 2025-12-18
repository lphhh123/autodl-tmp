"""Continuous wafer layout model."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from hw_proxy.feature_builder import DeviceSpec


class WaferLayout(nn.Module):
    def __init__(self, wafer_radius_mm: float, devices: List[DeviceSpec]):
        super().__init__()
        self.wafer_radius = wafer_radius_mm
        self.num_slots = len(devices)
        self.positions = nn.Parameter(torch.zeros(self.num_slots, 2))

    def forward(self) -> torch.Tensor:
        return self.positions


def layout_boundary_penalty(positions: torch.Tensor, wafer_radius: float) -> torch.Tensor:
    dist = positions.norm(dim=1)
    return torch.clamp(dist - wafer_radius, min=0).pow(2).mean()


def layout_overlap_penalty(positions: torch.Tensor, device_areas: List[float], margin: float = 1.5) -> torch.Tensor:
    penalty = torch.zeros((), device=positions.device)
    for i in range(len(device_areas)):
        for j in range(i + 1, len(device_areas)):
            min_dist = margin * ((device_areas[i] + device_areas[j]) ** 0.5)
            dist = (positions[i] - positions[j]).norm()
            penalty = penalty + torch.clamp(min_dist - dist, min=0) ** 2
    return penalty / max(1, len(device_areas))


def layout_comm_cost(positions: torch.Tensor, comm_graph: List[Tuple[int, int, float]], energy_per_bit: float) -> torch.Tensor:
    cost = torch.zeros((), device=positions.device)
    for src, dst, traffic in comm_graph:
        dist = (positions[src] - positions[dst]).norm()
        cost = cost + traffic * dist * energy_per_bit
    return cost


def layout_thermal_penalty(positions: torch.Tensor, powers: List[float], kernel_cfg: dict) -> torch.Tensor:
    ambient = kernel_cfg.get("ambient", 40.0)
    threshold = kernel_cfg.get("threshold", 95.0)
    sigma = kernel_cfg.get("sigma", 5.0)
    scale = kernel_cfg.get("scale", 0.1)
    temp = torch.zeros((), device=positions.device) + ambient
    for idx, power in enumerate(powers):
        dist = (positions - positions[idx]).pow(2).sum(dim=1)
        kernel = torch.exp(-dist / (2 * sigma ** 2))
        temp = temp + scale * power * kernel.sum()
    return torch.clamp(temp - threshold, min=0).mean()
