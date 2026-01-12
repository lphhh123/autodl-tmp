from __future__ import annotations

import random
from typing import Dict, Tuple

from src.problems.wafer_layout.components import RandomKickOperator


def region_shuffle_small(problem_state: Dict, algorithm_data: Dict, **kwargs) -> Tuple[RandomKickOperator, Dict]:
    instance_data = problem_state["instance_data"]
    helper = problem_state.get("helper_function", {}) or {}
    rng = helper.get("rng", random)
    solution = problem_state["current_solution"]

    assign = list(solution.assign)
    S = len(assign)
    Ns = int(instance_data.get("sites", {}).get("Ns", len(instance_data.get("sites", {}).get("sites_xy", []))))
    if S == 0 or Ns == 0:
        return RandomKickOperator([], []), {"k": 0}

    k = min(3, S)
    idxs = rng.sample(range(S), k)
    site_ids = [rng.randrange(Ns) for _ in range(k)]
    return RandomKickOperator(idxs, site_ids), {"k": k}
