import numpy as np

from ...components import ClusterMoveOperator


def cluster_move_3(problem_state: dict, algorithm_data: dict, **kwargs):
    """Pick 3 chiplets and reassign them to 3 random sites."""
    inst = problem_state["instance_data"]
    s_count = int(inst["S"])
    ns_count = int(inst["Ns"])
    idx = np.random.choice(s_count, size=min(3, s_count), replace=False)
    sites = np.random.choice(ns_count, size=idx.size, replace=False if ns_count >= idx.size else True)
    return ClusterMoveOperator(
        idx.tolist(),
        [int(x) for x in sites.tolist()],
    ), {"k": int(idx.size)}
