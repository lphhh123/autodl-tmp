"""Traffic-based clustering (agglomerative) for layout coarsening."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Cluster:
    cluster_id: int
    slots: List[int]
    tdp_sum: float


def coarsen_traffic(
    T_sym: np.ndarray,
    chip_tdp_w: np.ndarray,
    slot_mask_used: np.ndarray | None,
    target_num_clusters: int,
    min_merge_traffic: float,
) -> Tuple[List[Cluster], np.ndarray]:
    """Greedy agglomerative merge using symmetric traffic weights."""

    S = T_sym.shape[0]
    if slot_mask_used is None:
        slot_mask_used = np.ones(S, dtype=bool)

    clusters: List[Cluster] = []
    for s in range(S):
        if not slot_mask_used[s]:
            continue
        clusters.append(Cluster(cluster_id=len(clusters), slots=[s], tdp_sum=float(chip_tdp_w[s])))

    # compute initial cluster weight matrix
    def build_weight_matrix(cls: List[Cluster]) -> np.ndarray:
        C = len(cls)
        W = np.zeros((C, C), dtype=np.float64)
        for i in range(C):
            for j in range(i + 1, C):
                w = 0.0
                for a in cls[i].slots:
                    for b in cls[j].slots:
                        w += float(T_sym[a, b])
                W[i, j] = W[j, i] = w
        return W

    def merge_clusters(cls: List[Cluster], idx_a: int, idx_b: int) -> List[Cluster]:
        new_slots = cls[idx_a].slots + cls[idx_b].slots
        new_cluster = Cluster(cluster_id=len(cls), slots=new_slots, tdp_sum=float(sum(chip_tdp_w[new_slots])))
        new_list = [c for k, c in enumerate(cls) if k not in (idx_a, idx_b)]
        new_list.append(new_cluster)
        # reassign ids
        for i, c in enumerate(new_list):
            c.cluster_id = i
        return new_list

    W = build_weight_matrix(clusters)
    while len(clusters) > target_num_clusters:
        C = len(clusters)
        if C <= 1:
            break
        # find max weight pair
        max_w = -1.0
        max_pair = None
        for i in range(C):
            for j in range(i + 1, C):
                if W[i, j] > max_w:
                    max_w = float(W[i, j])
                    max_pair = (i, j)
        if max_pair is None or max_w < float(min_merge_traffic):
            break
        clusters = merge_clusters(clusters, max_pair[0], max_pair[1])
        W = build_weight_matrix(clusters)

    C = len(clusters)
    W_cluster = np.zeros((C, C), dtype=np.float64)
    for i in range(C):
        for j in range(i + 1, C):
            w = 0.0
            for a in clusters[i].slots:
                for b in clusters[j].slots:
                    w += float(T_sym[a, b])
            W_cluster[i, j] = W_cluster[j, i] = w

    return clusters, W_cluster

