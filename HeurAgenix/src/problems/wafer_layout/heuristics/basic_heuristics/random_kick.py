import random
from typing import Tuple, Dict
from src.problems.base.components import BaseOperator
from src.problems.wafer_layout.components import RandomKickOperator


def random_kick(problem_state: dict, algorithm_data: dict, **kwargs) -> Tuple[BaseOperator, Dict]:
    """
    Randomly reassign k slots to random sites (kick move for escaping local minima).

    Args:
        problem_state: dict.
        algorithm_data: dict. You can pass k via algorithm_data.get("k", 3).
    """
    ins = problem_state["instance_problem_state"]
    S = int(ins["S"])
    Ns = int(ins["Ns"])
    k = int(algorithm_data.get("k", 3))
    k = max(1, min(k, S))

    idxs = random.sample(range(S), k) if S >= k else [random.randrange(0, S) for _ in range(k)]
    site_ids = [random.randrange(0, Ns) for _ in range(k)]
    return RandomKickOperator(idxs, site_ids), algorithm_data
