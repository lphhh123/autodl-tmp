from __future__ import annotations

import random
from typing import Dict, Tuple

from src.problems.wafer_layout.components import NoopOperator, RelocateOperator


def best_relocate_to_free_by_total_delta(problem_state: Dict, algorithm_data: Dict, **kwargs) -> Tuple[RelocateOperator, Dict]:
    instance_data = problem_state["instance_data"]
    helper = problem_state.get("helper_function", {}) or {}
    rng = helper.get("rng", random)
    eval_assign = helper.get("eval_assign")
    solution = problem_state["current_solution"]

    assign = list(solution.assign)
    S = len(assign)
    Ns = int(instance_data.get("sites", {}).get("Ns", len(instance_data.get("sites", {}).get("sites_xy", []))))
    if S == 0 or Ns == 0 or eval_assign is None:
        return NoopOperator(), {"delta": 0.0}

    current_score = float(eval_assign(assign)["total_scalar"])
    used = set(assign)
    free_sites = [i for i in range(Ns) if i not in used]
    if not free_sites:
        return NoopOperator(), {"delta": 0.0}

    candidates = [(i, site_id) for i in range(S) for site_id in free_sites]
    rng.shuffle(candidates)

    best_delta = 0.0
    best_op: RelocateOperator | NoopOperator = RelocateOperator(0, free_sites[0])
    for i, site_id in candidates[: min(50, len(candidates))]:
        cand = assign.copy()
        cand[i] = site_id
        cand_score = float(eval_assign(cand)["total_scalar"])
        delta = cand_score - current_score
        if delta < best_delta:
            best_delta = delta
            best_op = RelocateOperator(i, site_id)

    return best_op, {"delta": float(best_delta)}
