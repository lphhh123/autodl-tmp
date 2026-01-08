"""Problem state helpers for wafer layout."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _top_k_pairs(values: List[Tuple[int, int, float]], k: int) -> List[Tuple[int, int, float]]:
    values.sort(key=lambda x: x[2], reverse=True)
    return values[:k]


def get_instance_problem_state(instance_data: dict) -> dict:
    sites_xy = np.asarray(instance_data["sites"]["sites_xy"], dtype=float)
    radii = np.linalg.norm(sites_xy, axis=1)
    traffic = np.asarray(instance_data["mapping"]["traffic_matrix"], dtype=float)
    pairs = []
    S = int(instance_data["slots"]["S"])
    Ns = int(instance_data["sites"].get("Ns", len(sites_xy)))
    for i in range(S):
        for j in range(i + 1, S):
            pairs.append((i, j, float(traffic[i, j])))
    return {
        "S": int(instance_data["slots"]["S"]),
        "Ns": Ns,
        "wafer_radius_mm": float(instance_data["wafer"]["radius_mm"]),
        "objective_cfg": instance_data.get("objective_cfg", {}),
        "sites_xy": sites_xy.tolist(),
        "sites_radius_stats": {
            "min": float(radii.min()) if radii.size else 0.0,
            "max": float(radii.max()) if radii.size else 0.0,
            "mean": float(radii.mean()) if radii.size else 0.0,
        },
        "traffic_topk": _top_k_pairs(pairs, 10),
    }


def get_solution_problem_state(instance_data: dict, current_solution) -> dict:
    assign = list(getattr(current_solution, "assign", []) or [])
    assign_arr = np.asarray(assign, dtype=int)
    sites_xy = np.asarray(instance_data.get("sites", {}).get("sites_xy", []), dtype=float)
    Ns = int(instance_data.get("sites", {}).get("Ns", len(sites_xy)))
    radius_mean = 0.0
    if sites_xy.size and assign_arr.size:
        valid = assign_arr[(assign_arr >= 0) & (assign_arr < len(sites_xy))]
        if valid.size:
            radius_mean = float(np.linalg.norm(sites_xy[valid], axis=1).mean())
    return {
        "assign": assign,
        "free_sites_count": int(Ns - len(set(assign))),
        "duplicate_count": int(len(assign) - len(set(assign))),
        "assign_radius_mean": radius_mean,
    }


def get_observation_problem_state(problem_state: dict) -> dict:
    return {
        "instance": problem_state.get("instance", {}),
        "solution": problem_state.get("solution", {}),
    }
