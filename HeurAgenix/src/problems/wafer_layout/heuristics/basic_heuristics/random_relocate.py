import numpy as np

from ...components import RelocateOperator


def random_relocate(problem_state: dict, algorithm_data: dict, **kwargs):
    """Randomly relocate one chiplet to a random site."""
    inst = problem_state["instance_data"]
    s_count = int(inst["S"])
    ns_count = int(inst["Ns"])
    i = int(np.random.randint(0, s_count))
    site = int(np.random.randint(0, ns_count))
    return RelocateOperator(i, site), {}
