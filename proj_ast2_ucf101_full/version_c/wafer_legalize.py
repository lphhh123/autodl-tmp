"""Layout legalization utilities."""
from __future__ import annotations

from typing import List

import torch


def legalize_layout(positions: torch.Tensor, device_areas: List[float], wafer_radius: float) -> torch.Tensor:
    pos = positions.clone()
    dist = pos.norm(dim=1, keepdim=True)
    scale = torch.clamp(wafer_radius / (dist + 1e-6), max=1.0)
    pos = pos * scale
    for i in range(len(device_areas)):
        for j in range(i + 1, len(device_areas)):
            diff = pos[j] - pos[i]
            d = diff.norm()
            min_d = (device_areas[i] + device_areas[j]) ** 0.25
            if d < min_d:
                push = (min_d - d) * diff / (d + 1e-6)
                pos[j] = pos[j] + push
                pos[i] = pos[i] - push
    return pos
