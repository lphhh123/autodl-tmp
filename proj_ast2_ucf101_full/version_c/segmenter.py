"""ViT segment utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch.nn as nn


@dataclass
class Segment:
    id: int
    layer_indices: List[int]
    flops: float
    bytes: float


class ViTSegmenter:
    def __init__(self, backbone_config, strategy: str = "uniform_block_group", group_size: int = 2) -> None:
        self.backbone_config = backbone_config
        self.strategy = strategy
        self.group_size = group_size

    def build_segments(self, model: nn.Module) -> List[Segment]:
        segments: List[Segment] = []
        depth = len(getattr(model, "blocks", []))
        gid = 0
        for start in range(0, depth, self.group_size):
            idx = list(range(start, min(start + self.group_size, depth)))
            segments.append(Segment(id=gid, layer_indices=idx, flops=float(len(idx)), bytes=float(len(idx))))
            gid += 1
        return segments
