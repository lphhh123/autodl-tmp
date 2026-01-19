"""Greedy traffic-based clustering (SPEC v5.4 §8.2)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Cluster:
    cluster_id: int
    slots: List[int]
    tdp_sum: float


def _cluster_pair_weight(cluster_a: Cluster, cluster_b: Cluster, traffic_sym: np.ndarray) -> float:
    total = 0.0
    for i in cluster_a.slots:
        for j in cluster_b.slots:
            total += float(traffic_sym[i, j])
    return total


def coarsen_traffic(
    traffic_sym: np.ndarray, slot_tdp: np.ndarray, target_num_clusters: int, min_merge_traffic: float
) -> Tuple[List[Cluster], np.ndarray]:
    """Greedy agglomerative clustering.

    Returns clusters list and cluster weight matrix W_cluster.
    """

    clusters: List[Cluster] = [Cluster(cluster_id=i, slots=[i], tdp_sum=float(slot_tdp[i])) for i in range(traffic_sym.shape[0])]
    next_id = len(clusters)
    while len(clusters) > target_num_clusters:
        # compute pair weights
        best_pair = None
        best_w = -1.0
        for a_idx, ca in enumerate(clusters):
            for b_idx in range(a_idx + 1, len(clusters)):
                cb = clusters[b_idx]
                w = _cluster_pair_weight(ca, cb, traffic_sym)
                if w > best_w:
                    best_w = w
                    best_pair = (a_idx, b_idx)
        if best_pair is None or best_w < min_merge_traffic:
            break
        a_idx, b_idx = best_pair
        ca, cb = clusters[a_idx], clusters[b_idx]
        merged = Cluster(cluster_id=next_id, slots=ca.slots + cb.slots, tdp_sum=ca.tdp_sum + cb.tdp_sum)
        next_id += 1
        # rebuild cluster list without a_idx/b_idx
        clusters = [c for idx, c in enumerate(clusters) if idx not in best_pair]
        clusters.append(merged)

    C = len(clusters)
    W = np.zeros((C, C), dtype=float)
    for i, ci in enumerate(clusters):
        for j in range(i + 1, C):
            cj = clusters[j]
            W[i, j] = _cluster_pair_weight(ci, cj, traffic_sym)
            W[j, i] = W[i, j]
    return clusters, W
