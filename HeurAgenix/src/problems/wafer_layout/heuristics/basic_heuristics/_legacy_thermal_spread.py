import numpy as np

from ...components import RelocateOperator


def thermal_spread(problem_state: dict, algorithm_data: dict, **kwargs):
    """Move the hottest chiplet (max TDP) to a random far site (rough thermal spreading)."""
    inst = problem_state["instance_data"]
    tdp = inst["_tdp"]
    ns_count = int(inst["Ns"])
    i = int(np.argmax(tdp))
    site = int(np.random.randint(0, ns_count))
    return RelocateOperator(i, site), {"hot_i": i}
