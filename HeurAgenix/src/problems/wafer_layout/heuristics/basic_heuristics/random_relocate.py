from __future__ import annotations

import random
from typing import Dict, Tuple

from src.problems.wafer_layout.components import NoopOperator, RelocateOperator


def random_relocate(problem_state: Dict, algorithm_data: Dict, **kwargs) -> Tuple[RelocateOperator, Dict]:
    instance_data = problem_state["instance_data"]
    helper = problem_state.get("helper_function", {}) or {}
    rng = helper.get("rng", random)
    solution = problem_state["current_solution"]

    assign = list(solution.assign)
    S = len(assign)
    Ns = int(instance_data.get("sites", {}).get("Ns", len(instance_data.get("sites", {}).get("sites_xy", []))))
    if S == 0 or Ns == 0:
        return NoopOperator(), {}

    used = set(assign)
    free_sites = [i for i in range(Ns) if i not in used]
    i = rng.randrange(S)
    if free_sites:
        site_id = rng.choice(free_sites)
    else:
        site_id = rng.randrange(Ns)
    return RelocateOperator(i, site_id), {"free_site": bool(free_sites)}
