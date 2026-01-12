import random
from typing import Tuple, Dict
from src.problems.base.components import BaseOperator
from src.problems.wafer_layout.components import RelocateOperator


def relocate_one(problem_state: dict, algorithm_data: dict, **kwargs) -> Tuple[BaseOperator, Dict]:
    """
    Relocate one random slot to a (preferably empty) site.

    Args:
        problem_state: dict.
        algorithm_data: dict.
    """
    ins = problem_state["instance_problem_state"]
    sol = problem_state["solution_problem_state"]
    S = int(ins["S"])
    Ns = int(ins["Ns"])
    assign = sol["assign"]
    used = set(assign)
    empty = [s for s in range(Ns) if s not in used]

    i = random.randrange(0, S)
    if empty:
        site_id = random.choice(empty)
    else:
        site_id = random.randrange(0, Ns)
    return RelocateOperator(i, site_id), algorithm_data
