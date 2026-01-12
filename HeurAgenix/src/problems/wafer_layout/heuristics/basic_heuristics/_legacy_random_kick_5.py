import numpy as np

from ...components import RandomKickOperator


def random_kick_5(problem_state: dict, algorithm_data: dict, **kwargs):
    """Apply 5 random swap/relocate moves as a kick."""
    inst = problem_state["instance_data"]
    s_count = int(inst["S"])
    ns_count = int(inst["Ns"])
    moves = []
    for _ in range(5):
        if np.random.rand() < 0.5:
            i = int(np.random.randint(0, s_count))
            j = int(np.random.randint(0, s_count))
            moves.append({"type": "swap", "i": i, "j": j})
        else:
            i = int(np.random.randint(0, s_count))
            site = int(np.random.randint(0, ns_count))
            moves.append({"type": "relocate", "i": i, "site_id": site})
    return RandomKickOperator(moves), {"k": 5}
