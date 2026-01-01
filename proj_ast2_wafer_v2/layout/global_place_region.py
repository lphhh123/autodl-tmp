"""Cluster-to-region assignment with greedy + optional SA refine."""

from __future__ import annotations

import math
from typing import List

import numpy as np

from .coarsen import Cluster
from .regions import Region


def _objective(
    cluster_to_region: np.ndarray,
    clusters: List[Cluster],
    regions: List[Region],
    W_cluster: np.ndarray,
    ring_score: List[float],
    lambda_graph: float,
    lambda_ring: float,
    lambda_cap: float,
) -> float:
    obj = 0.0
    for c1 in range(len(clusters)):
        for c2 in range(c1 + 1, len(clusters)):
            r1 = regions[int(cluster_to_region[c1])]
            r2 = regions[int(cluster_to_region[c2])]
            d = np.linalg.norm(np.asarray(r1.centroid_xy_mm) - np.asarray(r2.centroid_xy_mm))
            obj += lambda_graph * float(W_cluster[c1, c2]) * d
    for cid, cl in enumerate(clusters):
        r_idx = int(cluster_to_region[cid])
        r = regions[r_idx]
        ring_penalty = ring_score[min(r.ring_idx, len(ring_score) - 1)]
        obj += lambda_ring * float(cl.tdp_sum) * ring_penalty
        violation = max(0, len(cl.slots) - r.capacity)
        obj += lambda_cap * violation
    return obj


def assign_clusters_to_regions(
    clusters: List[Cluster],
    regions: List[Region],
    W_cluster: np.ndarray,
    lambda_graph: float,
    lambda_ring: float,
    lambda_cap: float,
    ring_score: List[float],
    refine_steps: int = 0,
    sa_T0: float = 1.0,
    sa_alpha: float = 0.995,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Greedy assignment with light SA refinement."""

    if rng is None:
        rng = np.random.default_rng()

    C = len(clusters)
    cluster_to_region = -np.ones(C, dtype=np.int64)

    # greedy tdp-descending placement
    order = sorted(range(C), key=lambda i: clusters[i].tdp_sum, reverse=True)
    for cid in order:
        best_region = None
        best_obj = math.inf
        for rid, region in enumerate(regions):
            if len(clusters[cid].slots) > region.capacity:
                continue
            cluster_to_region[cid] = rid
            obj = _objective(cluster_to_region, clusters, regions, W_cluster, ring_score, lambda_graph, lambda_ring, lambda_cap)
            if obj < best_obj:
                best_obj = obj
                best_region = rid
        if best_region is None:
            # fall back to least filled region
            best_region = int(np.argmin([r.capacity for r in regions]))
        cluster_to_region[cid] = best_region

    # SA refine
    cur_obj = _objective(cluster_to_region, clusters, regions, W_cluster, ring_score, lambda_graph, lambda_ring, lambda_cap)
    best_assign = cluster_to_region.copy()
    best_obj = cur_obj
    for step in range(refine_steps):
        cid = int(rng.integers(0, C))
        rid = int(rng.integers(0, len(regions)))
        cand = cluster_to_region.copy()
        cand[cid] = rid
        cand_obj = _objective(cand, clusters, regions, W_cluster, ring_score, lambda_graph, lambda_ring, lambda_cap)
        delta = cand_obj - cur_obj
        T = sa_T0 * (sa_alpha ** step)
        accept = delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9))
        if accept:
            cluster_to_region = cand
            cur_obj = cand_obj
            if cand_obj < best_obj:
                best_obj = cand_obj
                best_assign = cand

    return best_assign

