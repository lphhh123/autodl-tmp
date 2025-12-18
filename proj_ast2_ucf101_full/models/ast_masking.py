"""Voronoi-style spatial partitioning and masking utilities."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class VoronoiSpatialPartitioner(nn.Module):
    """Approximate Voronoi partition using a fixed grid.

    The implementation keeps the interface open for future learned centroids
    while providing deterministic region ids for current experiments.
    """

    def __init__(self, num_regions: int) -> None:
        super().__init__()
        self.num_regions = num_regions

    def forward(self, coords: torch.Tensor, entropy_spatial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assign patches to coarse regions and compute region weights.

        Parameters
        ----------
        coords: torch.Tensor
            Normalized patch coordinates ``[N, 2]``.
        entropy_spatial: torch.Tensor
            Per-patch entropy ``[B, N]``.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            ``region_ids`` of shape ``[N]`` and ``region_weights`` of shape
            ``[B, num_regions]``.
        """
        n = coords.shape[0]
        side = int(self.num_regions ** 0.5)
        side = max(1, side)
        # grid assignment
        xs = (coords[:, 0] * side).long().clamp(0, side - 1)
        ys = (coords[:, 1] * side).long().clamp(0, side - 1)
        region_ids = ys * side + xs
        region_ids = region_ids.clamp(max=self.num_regions - 1)

        region_weights = []
        for rid in range(self.num_regions):
            mask = region_ids == rid
            if mask.sum() == 0:
                region_weights.append(entropy_spatial.new_zeros(entropy_spatial.shape[0]))
            else:
                weight = (entropy_spatial[:, mask]).mean(dim=1)
                region_weights.append(weight)
        region_weights = torch.stack(region_weights, dim=1)
        return region_ids, region_weights
