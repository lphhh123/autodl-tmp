from __future__ import annotations

import random
from typing import Dict, Tuple

from src.problems.wafer_layout.components import NoopOperator, SwapOperator


def random_swap(problem_state: Dict, algorithm_data: Dict, **kwargs) -> Tuple[SwapOperator, Dict]:
    helper = problem_state.get("helper_function", {}) or {}
    rng = helper.get("rng", random)
    solution = problem_state["current_solution"]

    assign = list(solution.assign)
    S = len(assign)
    if S < 2:
        return NoopOperator(), {}

    i, j = rng.sample(range(S), 2)
    return SwapOperator(i, j), {}
