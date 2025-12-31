"""Cluster expansion: assign slots to concrete sites within regions."""

from __future__ import annotations

from typing import List

import numpy as np

from .coarsen import Cluster
from .regions import Region


def expand_clusters_to_sites(
    clusters: List[Cluster],
    cluster_to_region: np.ndarray,
    regions: List[Region],
    sites_xy_mm: np.ndarray,
    traffic_sym: np.ndarray,
    intra_refine_steps: int = 0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Assign each slot in a cluster to a site within its region."""

    if rng is None:
        rng = np.random.default_rng()

    Ns = sites_xy_mm.shape[0]
    S = traffic_sym.shape[0]
    assign = -np.ones(S, dtype=np.int64)
    occupied: set[int] = set()

    # initial placement: deterministic site ordering per region
    for c in clusters:
        rid = int(cluster_to_region[c.cluster_id])
        region_sites = [s for s in regions[rid].site_ids if s not in occupied]
        if len(region_sites) < len(c.slots):
            region_sites = region_sites + [s for s in range(Ns) if s not in occupied]
        region_sites = region_sites[: len(c.slots)]
        for slot, site_id in zip(c.slots, region_sites):
            assign[slot] = site_id
            occupied.add(site_id)

    # fill remaining slots if any
    free_sites = [s for s in range(Ns) if s not in occupied]
    for s in range(S):
        if assign[s] < 0:
            if not free_sites:
                raise RuntimeError("No free sites left to assign")
            site_id = free_sites.pop(0)
            assign[s] = site_id
            occupied.add(site_id)

    # intra-cluster swap refinement
    for _ in range(intra_refine_steps):
        c_idx = int(rng.integers(0, len(clusters)))
        slots = clusters[c_idx].slots
        if len(slots) < 2:
            continue
        i, j = rng.choice(slots, size=2, replace=False)
        # evaluate delta in L_comm within cluster
        def comm_cost():
            cost = 0.0
            for a in slots:
                for b in slots:
                    if a >= b:
                        continue
                    sa = assign[a]
                    sb = assign[b]
                    dist = float(np.linalg.norm(sites_xy_mm[sa] - sites_xy_mm[sb]))
                    cost += float(traffic_sym[a, b]) * dist
            return cost

        cur_cost = comm_cost()
        new_assign = assign.copy()
        new_assign[i], new_assign[j] = new_assign[j], new_assign[i]
        new_cost = 0.0
        for a in slots:
            for b in slots:
                if a >= b:
                    continue
                sa = new_assign[a]
                sb = new_assign[b]
                dist = float(np.linalg.norm(sites_xy_mm[sa] - sites_xy_mm[sb]))
                new_cost += float(traffic_sym[a, b]) * dist
        if new_cost < cur_cost:
            assign = new_assign

    return assign

