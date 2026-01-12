from __future__ import annotations

import random
from typing import Dict, Tuple

from src.problems.wafer_layout.components import NoopOperator, SwapOperator


def best_swap_by_total_delta(problem_state: Dict, algorithm_data: Dict, **kwargs) -> Tuple[SwapOperator, Dict]:
    helper = problem_state.get("helper_function", {}) or {}
    rng = helper.get("rng", random)
    eval_assign = helper.get("eval_assign")
    solution = problem_state["current_solution"]

    assign = list(solution.assign)
    S = len(assign)
    if S < 2 or eval_assign is None:
        return NoopOperator(), {"delta": 0.0}

    current_score = float(eval_assign(assign)["total_scalar"])
    pairs = [(i, j) for i in range(S) for j in range(i + 1, S)]
    rng.shuffle(pairs)

    best_delta = 0.0
    best_op: SwapOperator | NoopOperator = SwapOperator(0, 1)
    for i, j in pairs[: min(50, len(pairs))]:
        cand = assign.copy()
        cand[i], cand[j] = cand[j], cand[i]
        cand_score = float(eval_assign(cand)["total_scalar"])
        delta = cand_score - current_score
        if delta < best_delta:
            best_delta = delta
            best_op = SwapOperator(i, j)

    return best_op, {"delta": float(best_delta)}
