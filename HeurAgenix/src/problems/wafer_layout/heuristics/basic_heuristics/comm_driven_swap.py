from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np

from src.problems.wafer_layout.components import NoopOperator, SwapOperator


def _get_traffic_matrix(instance_data: Dict) -> np.ndarray:
    mapping = instance_data.get("mapping", {}) or {}
    traffic = mapping.get("traffic_matrix", mapping.get("traffic_bytes", []))
    arr = np.asarray(traffic, dtype=float)
    if arr.ndim != 2:
        return np.zeros((0, 0), dtype=float)
    return arr


def comm_driven_swap(problem_state: Dict, algorithm_data: Dict, **kwargs) -> Tuple[SwapOperator, Dict]:
    helper = problem_state.get("helper_function", {}) or {}
    rng = helper.get("rng", random)
    instance_data = problem_state["instance_data"]
    solution = problem_state["current_solution"]

    assign = list(solution.assign)
    S = len(assign)
    if S < 2:
        return NoopOperator(), {}

    traffic = _get_traffic_matrix(instance_data)
    if traffic.shape[0] != S:
        i, j = rng.sample(range(S), 2)
        return SwapOperator(i, j), {"reason": "random_fallback"}

    t_sym = traffic + traffic.T
    best_pair = None
    best_score = -1.0
    for i in range(S):
        for j in range(i + 1, S):
            score = float(t_sym[i, j])
            if score > best_score:
                best_score = score
                best_pair = (i, j)
    if best_pair is None:
        i, j = rng.sample(range(S), 2)
        return SwapOperator(i, j), {"reason": "random_fallback"}
    return SwapOperator(best_pair[0], best_pair[1]), {"traffic": float(best_score)}
