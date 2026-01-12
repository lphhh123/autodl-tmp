import random
from typing import Tuple, Dict
from src.problems.base.components import BaseOperator
from src.problems.wafer_layout.components import SwapOperator


def swap_two(problem_state: dict, algorithm_data: dict, **kwargs) -> Tuple[BaseOperator, Dict]:
    """
    Swap two random slots' assigned sites.

    Args:
        problem_state: dict containing observation + current solution summary.
        algorithm_data: dict for hyper-parameters (unused here).
    """
    S = int(problem_state["instance_problem_state"]["S"])
    i = random.randrange(0, S)
    j = random.randrange(0, S)
    while j == i and S > 1:
        j = random.randrange(0, S)
    return SwapOperator(i, j), algorithm_data
