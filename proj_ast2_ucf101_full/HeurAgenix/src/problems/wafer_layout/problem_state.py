"""Problem state helpers for wafer layout."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _top_k_pairs(values: List[Tuple[int, int, float]], k: int) -> List[Tuple[int, int, float]]:
    values.sort(key=lambda x: x[2], reverse=True)
    return values[:k]


def get_instance_problem_state(instance_data: dict, **kwargs) -> dict:
    sites_xy = np.asarray(instance_data["sites"]["sites_xy"], dtype=float)
    radii = np.linalg.norm(sites_xy, axis=1)
    traffic = np.asarray(instance_data["mapping"]["traffic_matrix"], dtype=float)
    pairs = []
    S = int(instance_data["slots"]["S"])
    for i in range(S):
        for j in range(i + 1, S):
            pairs.append((i, j, float(traffic[i, j])))
    return {
        "S": int(instance_data["slots"]["S"]),
        "Ns": int(instance_data["sites"]["Ns"]),
        "wafer_radius_mm": float(instance_data["wafer"]["radius_mm"]),
        "objective_cfg": instance_data.get("objective_cfg", {}),
        "sites_radius_stats": {
            "min": float(radii.min()) if radii.size else 0.0,
            "max": float(radii.max()) if radii.size else 0.0,
            "mean": float(radii.mean()) if radii.size else 0.0,
        },
        "traffic_topk": _top_k_pairs(pairs, 10),
    }


def get_solution_problem_state(problem_state: dict, current_solution=None, **kwargs) -> dict:
    solution = problem_state.get("solution", {})
    eval_out = problem_state.get("eval", {})
    assign = solution.get("assign", [])
    if current_solution is not None and hasattr(current_solution, "assign"):
        assign = current_solution.assign
    assign_arr = np.asarray(assign, dtype=int)
    return {
        "total_scalar": eval_out.get("total_scalar", 0.0),
        "comm_norm": eval_out.get("comm_norm", 0.0),
        "therm_norm": eval_out.get("therm_norm", 0.0),
        "free_sites_count": int(problem_state.get("Ns", 0) - len(set(assign))),
        "duplicate_count": int(len(assign) - len(set(assign))),
        "assign_radius_mean": float(assign_arr.mean()) if assign_arr.size else 0.0,
    }


def get_observation_problem_state(problem_state: dict, **kwargs) -> dict:
    return {
        "instance": problem_state.get("instance", {}),
        "solution": problem_state.get("solution", {}),
    }
