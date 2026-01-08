from __future__ import annotations

import random
from typing import Dict, Tuple

from problems.wafer_layout.components import SwapSlots


def best_swap_by_total_delta(problem_state: Dict, algorithm_data: Dict, **kwargs) -> Tuple[SwapSlots, Dict]:
    env = algorithm_data["env"]
    rng: random.Random = algorithm_data["rng"]
    solution = problem_state["solution"]
    best_op = SwapSlots(0, 0)
    best_delta = float("inf")
    S = solution.S
    candidates = [(i, j) for i in range(S) for j in range(i + 1, S)]
    rng.shuffle(candidates)
    for i, j in candidates[: min(50, len(candidates))]:
        op = SwapSlots(i, j)
        new_solution = op.run(solution)
        delta = env.get_key_value(new_solution) - env.get_key_value(solution)
        if delta < best_delta:
            best_delta = delta
            best_op = op
    return best_op, {"delta": best_delta}
