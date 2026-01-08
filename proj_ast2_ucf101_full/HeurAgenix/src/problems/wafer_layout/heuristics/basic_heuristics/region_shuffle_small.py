from __future__ import annotations

import random
from typing import Dict, Tuple

from problems.wafer_layout.components import RandomKick, SwapSlots


def region_shuffle_small(problem_state: Dict, algorithm_data: Dict, **kwargs) -> Tuple[RandomKick, Dict]:
    rng: random.Random = algorithm_data["rng"]
    solution = problem_state["current_solution"]
    slots = list(range(solution.S))
    rng.shuffle(slots)
    ops = []
    for i in range(0, min(4, len(slots) - 1), 2):
        ops.append(SwapSlots(slots[i], slots[i + 1]))
    return RandomKick(ops), {"n_ops": len(ops)}
