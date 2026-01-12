from __future__ import annotations

import math
from typing import Dict, List

import numpy as np


def _compute_comm(pos: np.ndarray, traffic_bytes: np.ndarray) -> float:
    t_sym = traffic_bytes + traffic_bytes.T
    total = 0.0
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            total += float(t_sym[i, j]) * dist
    return total


def _compute_therm(pos: np.ndarray, chip_tdp_w: np.ndarray, sigma_mm: float) -> float:
    total = 0.0
    for i in range(pos.shape[0]):
        for j in range(i + 1, pos.shape[0]):
            dist = float(np.linalg.norm(pos[i] - pos[j]))
            if dist <= 1e-9:
                continue
            total += float(chip_tdp_w[i] * chip_tdp_w[j]) * math.exp(-dist / sigma_mm)
    return total


def evaluate_layout(instance_data: dict, assign: List[int], obj_cfg: Dict | None = None) -> Dict[str, float]:
    objective_cfg = obj_cfg or instance_data.get("objective_cfg", {}) or {}
    scalar = objective_cfg.get("scalar_weights", objective_cfg) if isinstance(objective_cfg, dict) else {}
    baseline = instance_data.get("baseline", {}) or {}

    wafer = instance_data.get("wafer", {})
    slots = instance_data.get("slots", {})
    sites = instance_data.get("sites", {})

    sites_xy = instance_data.get("sites_xy")
    if sites_xy is None:
        sites_xy = sites.get("sites_xy", [])
    sites_xy = np.asarray(sites_xy, dtype=float)
    S = int(slots.get("S", len(slots.get("tdp", []) or [])))
    Ns = int(sites.get("Ns", len(sites_xy)))

    traffic = np.asarray(instance_data.get("mapping", {}).get("traffic_matrix", []), dtype=float)
    if traffic.size == 0:
        traffic = np.zeros((S, S), dtype=float)
    chip_tdp = np.asarray(slots.get("tdp", []), dtype=float)

    assign_arr = np.asarray(assign, dtype=int)
    if assign_arr.shape[0] != S:
        raise ValueError(f"assign length {assign_arr.shape[0]} != S={S}")
    bad = assign_arr[(assign_arr < 0) | (assign_arr >= Ns)]
    if bad.size > 0:
        raise ValueError(f"assign contains invalid site_id(s): {np.unique(bad)[:8].tolist()} (Ns={Ns})")

    pos = sites_xy[assign_arr]
    dup_count = len(assign_arr) - len(np.unique(assign_arr))
    penalty_duplicate = float(dup_count) ** 2 if dup_count > 0 else 0.0
    boundary_overflow = np.linalg.norm(pos, axis=1) - float(wafer.get("radius_mm", 0.0))
    penalty_boundary = float(np.sum(np.maximum(boundary_overflow, 0.0) ** 2))

    sigma_mm = float(objective_cfg.get("sigma_mm", 20.0))
    L_comm = _compute_comm(pos, traffic)
    L_therm = _compute_therm(pos, chip_tdp, sigma_mm)
    comm_norm = L_comm / (float(baseline.get("L_comm", 1.0)) + 1e-9)
    therm_norm = L_therm / (float(baseline.get("L_therm", 1.0)) + 1e-9)

    total = (
        float(scalar.get("w_comm", 1.0)) * comm_norm
        + float(scalar.get("w_therm", 1.0)) * therm_norm
        + float(scalar.get("w_penalty", 1.0)) * (penalty_duplicate + penalty_boundary)
    )
    return {
        "L_comm": L_comm,
        "L_therm": L_therm,
        "comm_norm": comm_norm,
        "therm_norm": therm_norm,
        "penalty": {"duplicate": penalty_duplicate, "boundary": penalty_boundary},
        "total_scalar": total,
    }
