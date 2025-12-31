"""Site-based wafer layout utilities aligned with the v4.3.2 spec.

This module moves away from the previous continuous layout parameterization
and instead works with discrete site assignments (`assign[s] -> site_id`).
It provides:

* seed initialization strategies (grid baseline + traffic-aware greedy)
* a lightweight micro-placement simulated annealing kernel (swap/relocate)
* helper utilities to trigger micro placement when the mapping changes

The implementation intentionally mirrors the spec sections 4 and 7 to ensure
the training-time layout generation matches the offline EDA pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .evaluator import LayoutEvaluator, LayoutState
from .sites import build_sites


@dataclass
class MicroPlaceStats:
    steps: int
    accepted: int
    start_scalar: float
    end_scalar: float
    trigger: str


def grid_baseline_assign(S: int, Ns: int) -> np.ndarray:
    """Deterministic grid baseline: fill slots with ascending site ids."""

    if Ns < S:
        raise ValueError(f"Need at least {S} sites, got {Ns}")
    assign = np.arange(S, dtype=np.int64)
    return assign


def _topk_pairs_from_matrix(mat: np.ndarray, k: int) -> list[Tuple[int, int, float]]:
    """Return top-k upper-triangular (i<j) pairs with their weight."""

    assert mat.shape[0] == mat.shape[1]
    S = mat.shape[0]
    pairs: list[Tuple[int, int, float]] = []
    for i in range(S):
        for j in range(i + 1, S):
            w = float(mat[i, j])
            if w > 0:
                pairs.append((i, j, w))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]


def _nearest_site_pairs(sites_xy_mm: np.ndarray, M: int) -> list[Tuple[int, int, float]]:
    """Return the closest M site pairs (i<j) by Euclidean distance."""

    Ns = sites_xy_mm.shape[0]
    pairs: list[Tuple[int, int, float]] = []
    for i in range(Ns):
        for j in range(i + 1, Ns):
            dist = float(np.linalg.norm(sites_xy_mm[i] - sites_xy_mm[j]))
            pairs.append((i, j, dist))
    pairs.sort(key=lambda x: x[2])
    return pairs[:M]


def traffic_aware_greedy_assign(
    traffic_bytes_sym: np.ndarray,
    sites_xy_mm: np.ndarray,
    topk_pairs: int = 8,
    nearest_site_pairs: int = 64,
) -> np.ndarray:
    """Greedy seed: map hot traffic pairs to the closest site pairs.

    Mirrors spec §7.1: pick the hottest communication pairs and place them on
    the shortest site pairs without conflicts, then fill remaining slots.
    """

    S = traffic_bytes_sym.shape[0]
    Ns = sites_xy_mm.shape[0]
    if Ns < S:
        raise ValueError(f"Need at least {S} sites, got {Ns}")

    assign = -np.ones(S, dtype=np.int64)
    used_sites: set[int] = set()

    hot_pairs = _topk_pairs_from_matrix(traffic_bytes_sym, topk_pairs)
    close_site_pairs = _nearest_site_pairs(sites_xy_mm, nearest_site_pairs)

    site_pair_iter = iter(close_site_pairs)
    for (i, j, _) in hot_pairs:
        target_pair = None
        # find next available close site pair with no conflicts
        for (sa, sb, _) in site_pair_iter:
            if sa in used_sites or sb in used_sites:
                continue
            target_pair = (sa, sb)
            break
        if target_pair is None:
            break
        assign[i], assign[j] = target_pair
        used_sites.update(target_pair)

    # fill remaining slots with the smallest unused site ids
    free_sites = [s for s in range(Ns) if s not in used_sites]
    free_iter = iter(free_sites)
    for s in range(S):
        if assign[s] >= 0:
            continue
        try:
            assign[s] = next(free_iter)
        except StopIteration:
            raise RuntimeError("Ran out of free sites when filling greedy seed")
    return assign


def _evaluate_assign(
    assign: np.ndarray,
    evaluator: LayoutEvaluator,
    sites_xy_mm: np.ndarray,
    chip_tdp_w: np.ndarray,
    traffic_bytes: np.ndarray,
    meta: Optional[Dict] = None,
) -> Dict:
    st = LayoutState(
        S=assign.shape[0],
        Ns=sites_xy_mm.shape[0],
        wafer_radius_mm=0.0,  # unused by evaluator beyond penalty check
        sites_xy_mm=sites_xy_mm,
        assign=assign,
        chip_tdp_w=chip_tdp_w,
        traffic_bytes=traffic_bytes,
        meta=meta or {},
    )
    return evaluator.evaluate(st)


def should_trigger_micro_place(
    prev_mapping: Optional[np.ndarray],
    mapping: np.ndarray,
    changed_ratio_threshold: float,
    steps_since_last: int,
    min_steps_between_triggers: int,
) -> bool:
    if prev_mapping is None:
        return True
    changed = np.mean(prev_mapping != mapping)
    return changed >= changed_ratio_threshold and steps_since_last >= min_steps_between_triggers


def simulated_annealing_micro_place(
    assign: np.ndarray,
    evaluator: LayoutEvaluator,
    sites_xy_mm: np.ndarray,
    chip_tdp_w: np.ndarray,
    traffic_bytes: np.ndarray,
    steps: int = 80,
    T0: float = 1.0,
    alpha: float = 0.995,
    same_region_bias: float = 0.8,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, MicroPlaceStats]:
    """Lightweight simulated annealing with swap/relocate moves.

    Region awareness is abstracted via `same_region_bias`: relocates prefer to
    stay within the same implicit neighborhood by sampling nearby empty sites
    with higher probability.
    """

    if rng is None:
        rng = np.random.default_rng()

    S = assign.shape[0]
    Ns = sites_xy_mm.shape[0]
    best_assign = assign.copy()
    cur_assign = assign.copy()

    def eval_assign(a: np.ndarray) -> float:
        res = _evaluate_assign(a, evaluator, sites_xy_mm, chip_tdp_w, traffic_bytes)
        return float(res["total_scalar"])

    cur_cost = eval_assign(cur_assign)
    best_cost = cur_cost
    start_cost = cur_cost
    accepted = 0

    def propose_swap(a: np.ndarray) -> np.ndarray:
        i, j = rng.choice(S, size=2, replace=False)
        new_a = a.copy()
        new_a[i], new_a[j] = new_a[j], new_a[i]
        return new_a

    def propose_relocate(a: np.ndarray) -> np.ndarray:
        occupied = set(int(x) for x in a.tolist())
        free = [s for s in range(Ns) if s not in occupied]
        if not free:
            return a
        target_slot = int(rng.integers(0, S))
        if rng.random() < same_region_bias:
            # prefer closer sites
            dists = [
                (fid, float(np.linalg.norm(sites_xy_mm[fid] - sites_xy_mm[int(a[target_slot])])))
                for fid in free
            ]
            dists.sort(key=lambda x: x[1])
            candidate_site = dists[0][0]
        else:
            candidate_site = int(rng.choice(free))
        new_a = a.copy()
        new_a[target_slot] = candidate_site
        return new_a

    for step in range(steps):
        move = propose_swap if rng.random() < 0.5 else propose_relocate
        cand_assign = move(cur_assign)
        cand_cost = eval_assign(cand_assign)
        delta = cand_cost - cur_cost
        T = T0 * (alpha ** step)
        accept = delta < 0 or rng.random() < math.exp(-delta / max(T, 1e-9))
        if accept:
            cur_assign = cand_assign
            cur_cost = cand_cost
            accepted += 1
            if cand_cost < best_cost:
                best_cost = cand_cost
                best_assign = cand_assign

    stats = MicroPlaceStats(
        steps=steps,
        accepted=accepted,
        start_scalar=float(start_cost),
        end_scalar=float(cur_cost),
        trigger="mapping_change",
    )
    return best_assign, stats


def build_layout_inputs(
    S: int,
    wafer_radius_mm: float,
    chip_max_width_mm: float,
    chip_max_height_mm: float,
    margin_mm: float,
    traffic_matrix: np.ndarray,
    chip_tdp_w: np.ndarray,
    sigma_mm: float,
    scalar_weights: Dict[str, float],
    seed: int = 0,
) -> Dict:
    """Utility to generate baseline + seed assignments and evaluations.

    Returns a dictionary ready to be embedded into `layout_input.json`.
    """

    sites_xy = build_sites(
        wafer_radius_mm=wafer_radius_mm,
        chip_max_width_mm=chip_max_width_mm,
        chip_max_height_mm=chip_max_height_mm,
        margin_mm=margin_mm,
        method="square_grid_in_circle",
        grid_pitch_mm=None,
        seed=seed,
    )
    Ns = sites_xy.shape[0]
    assign_grid = grid_baseline_assign(S, Ns)

    # First pass to get baseline normalization from grid
    temp_eval = LayoutEvaluator(
        sigma_mm=sigma_mm,
        baseline={"L_comm_baseline": 1.0, "L_therm_baseline": 1.0},
        scalar_w=scalar_weights,
    )
    grid_raw = temp_eval.evaluate(
        LayoutState(
            S=S,
            Ns=Ns,
            wafer_radius_mm=wafer_radius_mm,
            sites_xy_mm=sites_xy,
            assign=assign_grid,
            chip_tdp_w=chip_tdp_w,
            traffic_bytes=traffic_matrix,
            meta={"stage": "baseline"},
        )
    )
    evaluator = LayoutEvaluator(
        sigma_mm=sigma_mm,
        baseline={
            "L_comm_baseline": grid_raw["L_comm"],
            "L_therm_baseline": grid_raw["L_therm"],
        },
        scalar_w=scalar_weights,
    )

    assign_seed = traffic_aware_greedy_assign(traffic_matrix + traffic_matrix.T, sites_xy)
    eval_grid = evaluator.evaluate(
        LayoutState(
            S=S,
            Ns=Ns,
            wafer_radius_mm=wafer_radius_mm,
            sites_xy_mm=sites_xy,
            assign=assign_grid,
            chip_tdp_w=chip_tdp_w,
            traffic_bytes=traffic_matrix,
            meta={"stage": "baseline"},
        )
    )
    eval_seed = evaluator.evaluate(
        LayoutState(
            S=S,
            Ns=Ns,
            wafer_radius_mm=wafer_radius_mm,
            sites_xy_mm=sites_xy,
            assign=assign_seed,
            chip_tdp_w=chip_tdp_w,
            traffic_bytes=traffic_matrix,
            meta={"stage": "seed"},
        )
    )

    assign_micro, micro_stats = simulated_annealing_micro_place(
        assign_seed,
        evaluator,
        sites_xy,
        chip_tdp_w,
        traffic_matrix,
    )
    eval_micro = evaluator.evaluate(
        LayoutState(
            S=S,
            Ns=Ns,
            wafer_radius_mm=wafer_radius_mm,
            sites_xy_mm=sites_xy,
            assign=assign_micro,
            chip_tdp_w=chip_tdp_w,
            traffic_bytes=traffic_matrix,
            meta={"stage": "micro_place"},
        )
    )

    return {
        "sites_xy_mm": sites_xy,
        "assign_grid": assign_grid,
        "assign_seed": assign_seed,
        "assign_micro": assign_micro,
        "eval_grid": eval_grid,
        "eval_seed": eval_seed,
        "eval_micro": eval_micro,
        "micro_place_stats": micro_stats,
        "baseline": {
            "L_comm_baseline": grid_raw["L_comm"],
            "L_therm_baseline": grid_raw["L_therm"],
        },
    }

