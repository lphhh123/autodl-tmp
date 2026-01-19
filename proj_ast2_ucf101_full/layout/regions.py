"""Region construction for ring/sector partitioning (SPEC v5.4 §8.3.1)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np


@dataclass
class Region:
    region_id: int
    ring_idx: int
    sector_idx: int
    site_ids: List[int]
    capacity: int
    centroid_xy_mm: np.ndarray
    ring_score: float


def build_regions(
    sites_xy_mm: np.ndarray,
    wafer_radius_mm: float,
    ring_edges_ratio: Sequence[float],
    sectors_per_ring: Sequence[int],
    ring_score: Sequence[float],
    capacity_ratio: float = 1.0,
) -> Tuple[List[Region], np.ndarray]:
    """Construct regions and site mapping.

    The ring_edges_ratio must include 0 and 1.0 and be strictly increasing.
    """

    if len(ring_edges_ratio) < 2:
        raise ValueError("ring_edges_ratio must have at least two entries")
    if len(sectors_per_ring) != len(ring_edges_ratio) - 1:
        raise ValueError("sectors_per_ring length must equal len(ring_edges_ratio)-1")
    if len(ring_score) < len(sectors_per_ring):
        raise ValueError("ring_score must provide score for each ring")

    rings_mm = [r * float(wafer_radius_mm) for r in ring_edges_ratio]
    site_to_region = np.full((sites_xy_mm.shape[0],), -1, dtype=int)
    regions: List[Region] = []
    region_id = 0
    # Pre-compute sector boundaries per ring
    for ring_idx, (r_in, r_out) in enumerate(zip(rings_mm[:-1], rings_mm[1:])):
        num_sector = sectors_per_ring[ring_idx]
        sector_edges = np.linspace(-math.pi, math.pi, num_sector + 1)
        for sector_idx in range(num_sector):
            site_ids: List[int] = []
            angle_lo, angle_hi = sector_edges[sector_idx], sector_edges[sector_idx + 1]
            for sid, (x, y) in enumerate(sites_xy_mm):
                if site_to_region[sid] != -1:
                    continue
                r = math.sqrt(float(x) ** 2 + float(y) ** 2)
                if r < r_in or r > r_out:
                    continue
                ang = math.atan2(float(y), float(x))
                if ang < angle_lo or ang > angle_hi:
                    continue
                site_ids.append(sid)
            if not site_ids:
                centroid = np.zeros(2, dtype=np.float32)
            else:
                centroid = sites_xy_mm[site_ids].mean(axis=0)
            capacity = int(len(site_ids) * capacity_ratio)
            capacity = max(capacity, len(site_ids)) if capacity_ratio >= 1.0 else capacity
            regions.append(
                Region(
                    region_id=region_id,
                    ring_idx=ring_idx,
                    sector_idx=sector_idx,
                    site_ids=site_ids,
                    capacity=capacity,
                    centroid_xy_mm=centroid.astype(np.float32),
                    ring_score=ring_score[ring_idx] if ring_idx < len(ring_score) else 1.0,
                )
            )
            for sid in site_ids:
                site_to_region[sid] = region_id
            region_id += 1
    if np.any(site_to_region < 0):
        # Fallback: assign remaining to nearest centroid-less region (should be rare)
        unassigned = np.where(site_to_region < 0)[0]
        for sid in unassigned:
            # assign to region with closest centroid
            d_min = float("inf")
            best_rid = 0
            for reg in regions:
                dist = np.linalg.norm(sites_xy_mm[sid] - reg.centroid_xy_mm)
                if dist < d_min:
                    d_min = dist
                    best_rid = reg.region_id
            site_to_region[sid] = best_rid
            regions[best_rid].site_ids.append(int(sid))
            regions[best_rid].capacity = max(regions[best_rid].capacity, len(regions[best_rid].site_ids))
    return regions, site_to_region
