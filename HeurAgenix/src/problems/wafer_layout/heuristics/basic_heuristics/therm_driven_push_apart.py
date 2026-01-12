from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np

from src.problems.wafer_layout.components import NoopOperator, SwapOperator


def therm_driven_push_apart(problem_state: Dict, algorithm_data: Dict, **kwargs) -> Tuple[SwapOperator, Dict]:
    instance_data = problem_state["instance_data"]
    helper = problem_state.get("helper_function", {}) or {}
    rng = helper.get("rng", random)
    solution = problem_state["current_solution"]

    assign = list(solution.assign)
    S = len(assign)
    if S < 2:
        return NoopOperator(), {}

    slots = instance_data.get("slots", {}) or {}
    slot_tdp = np.asarray(slots.get("tdp", slots.get("slot_tdp_w", [])), dtype=float)
    sites = instance_data.get("sites", {}) or {}
    sites_xy = np.asarray(sites.get("sites_xy", []), dtype=float)

    if slot_tdp.shape[0] != S or sites_xy.shape[0] == 0:
        i, j = rng.sample(range(S), 2)
        return SwapOperator(i, j), {"reason": "random_fallback"}

    hot_idx = int(np.argmax(slot_tdp))
    hot_site = assign[hot_idx]
    if hot_site < 0 or hot_site >= sites_xy.shape[0]:
        i, j = rng.sample(range(S), 2)
        return SwapOperator(i, j), {"reason": "random_fallback"}

    pos = sites_xy[assign]
    dists = np.linalg.norm(pos - pos[hot_idx], axis=1)
    far_idx = int(np.argmax(dists))
    if far_idx == hot_idx:
        i, j = rng.sample(range(S), 2)
        return SwapOperator(i, j), {"reason": "random_fallback"}

    return SwapOperator(hot_idx, far_idx), {"hot_idx": hot_idx, "far_idx": far_idx}
