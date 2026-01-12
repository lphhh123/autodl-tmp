from typing import Tuple, Dict
from src.problems.base.components import BaseOperator
from src.problems.wafer_layout.components import NoopOperator


def noop(problem_state: dict, algorithm_data: dict, **kwargs) -> Tuple[BaseOperator, Dict]:
    """
    Do nothing.

    Args:
        problem_state: dict.
        algorithm_data: dict.
    """
    return NoopOperator(), algorithm_data
