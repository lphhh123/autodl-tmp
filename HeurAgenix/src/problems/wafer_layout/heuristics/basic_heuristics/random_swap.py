from __future__ import annotations

import random
from typing import Dict, Tuple

from src.problems.wafer_layout.components import SwapSlots


def random_swap(problem_state: Dict, algorithm_data: Dict, **kwargs) -> Tuple[SwapSlots, Dict]:
    rng: random.Random = algorithm_data["rng"]
    solution = problem_state["current_solution"]
    i, j = rng.sample(range(solution.S), 2)
    return SwapSlots(i, j), {"pair": (i, j)}
