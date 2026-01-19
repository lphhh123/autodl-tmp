"""Expand cluster-to-region plan into slot-to-site assignment (SPEC v5.4 §8.4)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from layout.coarsen import Cluster
from layout.evaluator import LayoutState, LayoutEvaluator
from layout.regions import Region


def _intra_cluster_cost(cluster: Cluster, assign: np.ndarray, traffic_sym: np.ndarray, sites_xy: np.ndarray) -> float:
    slots = cluster.slots
    total = 0.0
    for i, si in enumerate(slots):
        for sj in slots[i + 1 :]:
            dist = float(np.linalg.norm(sites_xy[assign[si]] - sites_xy[assign[sj]]))
            total += float(traffic_sym[si, sj]) * dist
    return total


def expand_clusters(
    clusters: List[Cluster],
    cluster_to_region: List[int],
    regions: List[Region],
    base_assign: np.ndarray,
    traffic_sym: np.ndarray,
    sites_xy: np.ndarray,
    intra_refine_steps: int = 0,
) -> np.ndarray:
    assign = base_assign.copy()
    # track free sites per region
    free_sites: Dict[int, List[int]] = {reg.region_id: reg.site_ids.copy() for reg in regions}
    for cluster_idx, cluster in enumerate(clusters):
        rid = cluster_to_region[cluster_idx]
        candidates = free_sites.get(rid, [])
        if len(candidates) < len(cluster.slots):
            # fallback: pool all remaining
            candidates = [s for v in free_sites.values() for s in v]
        if len(candidates) < len(cluster.slots):
            raise RuntimeError("Not enough candidate sites during expansion")
        chosen = candidates[: len(cluster.slots)]
        free_sites[rid] = candidates[len(cluster.slots) :]
        for slot, site in zip(cluster.slots, chosen):
            assign[slot] = site

        if intra_refine_steps > 0 and len(cluster.slots) > 1:
            best_cost = _intra_cluster_cost(cluster, assign, traffic_sym, sites_xy)
            for _ in range(intra_refine_steps):
                # simple swap within cluster
                a, b = np.random.choice(cluster.slots, 2, replace=False)
                assign[a], assign[b] = assign[b], assign[a]
                new_cost = _intra_cluster_cost(cluster, assign, traffic_sym, sites_xy)
                if new_cost <= best_cost:
                    best_cost = new_cost
                else:
                    assign[a], assign[b] = assign[b], assign[a]
    # fill remaining slots using any free sites (deterministic order)
    remaining_sites = [s for v in free_sites.values() for s in v]
    for idx in range(len(assign)):
        if assign[idx] < 0 and remaining_sites:
            assign[idx] = remaining_sites.pop(0)
    return assign
