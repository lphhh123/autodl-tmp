"""Fast *analytic* delta evaluation for permutation-safe layout moves.

Why:
  The current candidate_pool and macro execution often call LayoutEvaluator.evaluate()
  many times per step (each evaluate is O(S^2)). Under an eval-call budget, this
  becomes a fixed tax that reduces the number of effective search iterations.

This module provides O(S) delta estimators for swap / relocate (permutation-safe).
They are *exact* for L_comm/L_therm for those moves, and avoid calling evaluate().

We ignore penalty deltas because the supported operators are permutation-safe
(no duplicates) and do not change the set of sites in a way that violates boundary
constraints (sites are pre-defined within wafer). Extend if you add non-permutation ops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np


@dataclass
class ObjectiveParams:
    sigma_mm: float
    L_comm_baseline: float
    L_therm_baseline: float
    w_comm: float
    w_therm: float


def _safe_norm2(xy: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(xy * xy, axis=1))


def delta_terms_swap(
    assign: np.ndarray,
    i: int,
    j: int,
    sites_xy_mm: np.ndarray,
    traffic_sym: np.ndarray,
    chip_tdp_w: np.ndarray,
    sigma_mm: float,
) -> Tuple[float, float]:
    """Exact delta (raw) for L_comm and L_therm under a swap of slots i<->j. O(S)."""
    a = np.asarray(assign, dtype=int)
    S = int(a.shape[0])
    i = int(i)
    j = int(j)
    if i == j or i < 0 or j < 0 or i >= S or j >= S:
        return 0.0, 0.0

    pos = np.asarray(sites_xy_mm, dtype=np.float64)[a]
    old_i = pos[i]
    old_j = pos[j]

    mask = np.ones(S, dtype=bool)
    mask[i] = False
    mask[j] = False
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return 0.0, 0.0

    pk = pos[idx]

    d_old_i = _safe_norm2(pk - old_i[None, :])
    d_new_i = _safe_norm2(pk - old_j[None, :])
    d_old_j = _safe_norm2(pk - old_j[None, :])
    d_new_j = _safe_norm2(pk - old_i[None, :])

    wi = np.asarray(traffic_sym, dtype=np.float64)[i, idx]
    wj = np.asarray(traffic_sym, dtype=np.float64)[j, idx]
    dL_comm = float(np.sum(wi * (d_new_i - d_old_i) + wj * (d_new_j - d_old_j)))

    sigma = float(max(1e-9, sigma_mm))
    ti = float(np.asarray(chip_tdp_w, dtype=np.float64)[i])
    tj = float(np.asarray(chip_tdp_w, dtype=np.float64)[j])
    tk = np.asarray(chip_tdp_w, dtype=np.float64)[idx]
    e_old_i = np.exp(-d_old_i / sigma)
    e_new_i = np.exp(-d_new_i / sigma)
    e_old_j = np.exp(-d_old_j / sigma)
    e_new_j = np.exp(-d_new_j / sigma)
    dL_therm = float(np.sum(ti * tk * (e_new_i - e_old_i) + tj * tk * (e_new_j - e_old_j)))
    return dL_comm, dL_therm


def delta_terms_move_one(
    assign: np.ndarray,
    i: int,
    new_site_id: int,
    sites_xy_mm: np.ndarray,
    traffic_sym: np.ndarray,
    chip_tdp_w: np.ndarray,
    sigma_mm: float,
) -> Tuple[float, float]:
    """Exact delta (raw) for moving slot i to new_site_id.

    If new_site_id is occupied by j, this reduces to swap(i,j).
    If new_site_id is empty, only terms involving i change.
    """
    a = np.asarray(assign, dtype=int)
    S = int(a.shape[0])
    i = int(i)
    if i < 0 or i >= S:
        return 0.0, 0.0
    new_site_id = int(new_site_id)

    occ = np.where(a == new_site_id)[0]
    if occ.size > 0:
        j = int(occ[0])
        return delta_terms_swap(a, i, j, sites_xy_mm, traffic_sym, chip_tdp_w, sigma_mm)

    pos = np.asarray(sites_xy_mm, dtype=np.float64)[a]
    old_i = pos[i]
    new_i = np.asarray(sites_xy_mm, dtype=np.float64)[new_site_id]

    mask = np.ones(S, dtype=bool)
    mask[i] = False
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return 0.0, 0.0

    pk = pos[idx]
    d_old = _safe_norm2(pk - old_i[None, :])
    d_new = _safe_norm2(pk - new_i[None, :])

    w = np.asarray(traffic_sym, dtype=np.float64)[i, idx]
    dL_comm = float(np.sum(w * (d_new - d_old)))

    sigma = float(max(1e-9, sigma_mm))
    ti = float(np.asarray(chip_tdp_w, dtype=np.float64)[i])
    tk = np.asarray(chip_tdp_w, dtype=np.float64)[idx]
    dL_therm = float(np.sum(ti * tk * (np.exp(-d_new / sigma) - np.exp(-d_old / sigma))))
    return dL_comm, dL_therm


def estimate_delta_norm_and_total(
    dL_comm: float,
    dL_therm: float,
    obj: ObjectiveParams,
) -> Tuple[float, float, float]:
    """Convert raw deltas to normalized deltas and scalar total delta."""
    denom_c = float(obj.L_comm_baseline) + 1e-9
    denom_t = float(obj.L_therm_baseline) + 1e-9
    d_comm_norm = float(dL_comm) / denom_c
    d_therm_norm = float(dL_therm) / denom_t
    d_total = float(obj.w_comm) * d_comm_norm + float(obj.w_therm) * d_therm_norm
    return d_comm_norm, d_therm_norm, d_total


def estimate_action_delta(
    assign: np.ndarray,
    action: Dict[str, Any],
    sites_xy_mm: np.ndarray,
    traffic_sym: np.ndarray,
    chip_tdp_w: np.ndarray,
    obj: ObjectiveParams,
) -> Dict[str, float]:
    """Estimate delta for an action without calling evaluator.

    Supported ops: swap / relocate.
    Returns: {d_comm_norm, d_therm_norm, d_total}.
    """
    op = str(action.get("op", ""))
    if op == "swap":
        i = int(action.get("i", -1))
        j = int(action.get("j", -1))
        dLc, dLt = delta_terms_swap(assign, i, j, sites_xy_mm, traffic_sym, chip_tdp_w, obj.sigma_mm)
    elif op == "relocate":
        i = int(action.get("i", -1))
        to_site = int(action.get("site_id", -1))
        dLc, dLt = delta_terms_move_one(assign, i, to_site, sites_xy_mm, traffic_sym, chip_tdp_w, obj.sigma_mm)
    else:
        dLc, dLt = 0.0, 0.0

    dcn, dtn, dt = estimate_delta_norm_and_total(dLc, dLt, obj)
    return {"d_comm_norm": float(dcn), "d_therm_norm": float(dtn), "d_total": float(dt)}

