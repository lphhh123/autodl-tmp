"""Region construction utilities (ring + sector partitioning).

Implements the region-aware partitioning described in the v4.3.2 spec:
the wafer is split into concentric rings and angular sectors, each with a
capacity derived from the number of sites it contains. The mapping from site
to region is deterministic and reused across the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Region:
    region_id: int
    ring_idx: int
    sector_idx: int
    centroid_xy_mm: Tuple[float, float]
    site_ids: List[int]
    capacity: int


def build_regions(
    sites_xy_mm: np.ndarray,
    wafer_radius_mm: float,
    ring_edges_ratio: List[float],
    sectors_per_ring: List[int],
    capacity_ratio: float = 1.0,
) -> tuple[List[Region], np.ndarray]:
    """Partition sites into ring/sector regions.

    Args:
        sites_xy_mm: [Ns,2] site coordinates.
        wafer_radius_mm: wafer radius.
        ring_edges_ratio: monotonic list of ring edges normalized by radius.
        sectors_per_ring: number of angular sectors per ring (len = num_rings-1).
        capacity_ratio: optional capacity multiplier (<=1 to tighten).

    Returns:
        regions: list of Region objects.
        site_to_region: [Ns] array mapping site id -> region id.
    """

    if len(ring_edges_ratio) < 2:
        raise ValueError("ring_edges_ratio must contain at least 2 edges")
    if len(sectors_per_ring) != len(ring_edges_ratio) - 1:
        raise ValueError("sectors_per_ring length must be num_rings-1")

    Ns = sites_xy_mm.shape[0]
    ring_edges = np.asarray(ring_edges_ratio, dtype=np.float64) * float(wafer_radius_mm)
    regions: List[Region] = []
    site_to_region = -np.ones(Ns, dtype=np.int64)

    region_id = 0
    radii = np.linalg.norm(sites_xy_mm, axis=1)
    angles = np.arctan2(sites_xy_mm[:, 1], sites_xy_mm[:, 0])  # [-pi, pi]
    # shift to [0, 2pi)
    angles = (angles + 2 * np.pi) % (2 * np.pi)

    for ring_idx in range(len(ring_edges) - 1):
        inner_r = ring_edges[ring_idx]
        outer_r = ring_edges[ring_idx + 1]
        in_ring = (radii >= inner_r) & (radii <= outer_r + 1e-9)
        sector_count = sectors_per_ring[ring_idx]
        for sector_idx in range(sector_count):
            theta0 = (2 * np.pi / sector_count) * sector_idx
            theta1 = theta0 + 2 * np.pi / sector_count
            in_sector = (angles >= theta0) & (angles < theta1)
            mask = in_ring & in_sector
            site_ids = np.nonzero(mask)[0].tolist()
            if not site_ids:
                centroid = (0.0, 0.0)
            else:
                centroid_xy = np.mean(sites_xy_mm[site_ids], axis=0)
                centroid = (float(centroid_xy[0]), float(centroid_xy[1]))
            capacity = max(1, int(len(site_ids) * float(capacity_ratio)))
            regions.append(
                Region(
                    region_id=region_id,
                    ring_idx=ring_idx,
                    sector_idx=sector_idx,
                    centroid_xy_mm=centroid,
                    site_ids=site_ids,
                    capacity=capacity,
                )
            )
            for sid in site_ids:
                site_to_region[sid] = region_id
            region_id += 1

    if np.any(site_to_region < 0):
        raise RuntimeError("Some sites were not assigned to a region")

    return regions, site_to_region

