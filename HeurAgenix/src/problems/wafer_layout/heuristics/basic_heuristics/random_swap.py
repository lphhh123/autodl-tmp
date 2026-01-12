import numpy as np

from ...components import SwapOperator


def random_swap(problem_state: dict, algorithm_data: dict, **kwargs):
    """Randomly swap two chiplets."""
    sol = problem_state["current_solution"]
    s_count = int(getattr(sol, "assign").shape[0])
    i, j = np.random.randint(0, s_count), np.random.randint(0, s_count)
    return SwapOperator(i, j), {}
