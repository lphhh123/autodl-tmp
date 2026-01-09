from __future__ import annotations

import random
from typing import Dict, Tuple

from src.problems.wafer_layout.components import RelocateSlot


def random_relocate(problem_state: Dict, algorithm_data: Dict, **kwargs) -> Tuple[RelocateSlot, Dict]:
    rng: random.Random = algorithm_data["rng"]
    solution = problem_state["current_solution"]
    i = rng.randrange(solution.S)
    site = rng.randrange(solution.Ns)
    return RelocateSlot(i, site), {"slot": i, "site": site}
