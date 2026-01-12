from __future__ import annotations

import random
from typing import Dict, Tuple

from src.problems.wafer_layout.components import RelocateSlot


def best_relocate_to_free_by_total_delta(problem_state: Dict, algorithm_data: Dict, **kwargs) -> Tuple[RelocateSlot, Dict]:
    env = algorithm_data["env"]
    rng: random.Random = algorithm_data["rng"]
    solution = problem_state["current_solution"]
    used = set(solution.assign)
    free_sites = [s for s in range(solution.Ns) if s not in used]
    if not free_sites:
        free_sites = list(range(solution.Ns))
    rng.shuffle(free_sites)
    best_op = RelocateSlot(0, free_sites[0])
    best_delta = float("inf")
    for i in range(solution.S):
        for site in free_sites[: min(30, len(free_sites))]:
            op = RelocateSlot(i, site)
            new_solution = op.run(solution)
            delta = env.get_key_value(new_solution) - env.get_key_value(solution)
            if delta < best_delta:
                best_delta = delta
                best_op = op
    return best_op, {"delta": best_delta}
