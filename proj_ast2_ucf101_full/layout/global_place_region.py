"""Cluster-to-region placement (SPEC v4.3.2 §8.3.2)."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import numpy as np

from layout.coarsen import Cluster
from layout.regions import Region


def _placement_cost(
    cluster_to_region: List[int],
    clusters: List[Cluster],
    regions: List[Region],
    W: np.ndarray,
    lambda_graph: float,
    lambda_ring: float,
    lambda_cap: float,
) -> float:
    total = 0.0
    for i, ci in enumerate(clusters):
        ri = regions[cluster_to_region[i]]
        total += lambda_ring * ci.tdp_sum * ri.ring_score
        # capacity violation (quadratic)
        violation = max(0, len(ci.slots) - ri.capacity)
        total += lambda_cap * (violation ** 2)
        for j in range(i + 1, len(clusters)):
            w = W[i, j]
            if w <= 0:
                continue
            rj = regions[cluster_to_region[j]]
            dist = float(np.linalg.norm(ri.centroid_xy_mm - rj.centroid_xy_mm))
            total += lambda_graph * w * dist
    return total


def assign_clusters_to_regions(
    clusters: List[Cluster],
    regions: List[Region],
    W: np.ndarray,
    lambda_graph: float,
    lambda_ring: float,
    lambda_cap: float,
    refine_cfg: Dict,
) -> Tuple[List[int], float]:
    # greedy init by tdp descending
    cluster_order = sorted(range(len(clusters)), key=lambda i: clusters[i].tdp_sum, reverse=True)
    cluster_to_region = [-1 for _ in clusters]
    for cid in cluster_order:
        best_r = 0
        best_cost = float("inf")
        for rid, reg in enumerate(regions):
            if len(reg.site_ids) == 0:
                continue
            cluster_to_region[cid] = rid
            cost = _placement_cost(cluster_to_region, clusters, regions, W, lambda_graph, lambda_ring, lambda_cap)
            if cost < best_cost:
                best_cost = cost
                best_r = rid
        cluster_to_region[cid] = best_r

    if refine_cfg.get("enabled", False):
        steps = int(refine_cfg.get("steps", 0))
        T = float(refine_cfg.get("sa_T0", 1.0))
        alpha = float(refine_cfg.get("sa_alpha", 0.99))
        current_cost = _placement_cost(cluster_to_region, clusters, regions, W, lambda_graph, lambda_ring, lambda_cap)
        for _ in range(steps):
            a, b = random.sample(range(len(clusters)), 2)
            cluster_to_region[a], cluster_to_region[b] = cluster_to_region[b], cluster_to_region[a]
            new_cost = _placement_cost(cluster_to_region, clusters, regions, W, lambda_graph, lambda_ring, lambda_cap)
            delta = new_cost - current_cost
            if delta < 0 or math.exp(-delta / max(T, 1e-6)) > random.random():
                current_cost = new_cost
            else:
                cluster_to_region[a], cluster_to_region[b] = cluster_to_region[b], cluster_to_region[a]
            T *= alpha
    final_cost = _placement_cost(cluster_to_region, clusters, regions, W, lambda_graph, lambda_ring, lambda_cap)
    return cluster_to_region, final_cost
