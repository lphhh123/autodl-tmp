import numpy as np

from ...components import RelocateOperator


def hotspot_relocate(problem_state: dict, algorithm_data: dict, **kwargs):
    """Relocate a chiplet with highest traffic degree to a random site."""
    inst = problem_state["instance_data"]
    traffic = inst["_traffic"]
    s_count = int(inst["S"])
    ns_count = int(inst["Ns"])
    deg = traffic.sum(axis=1)
    i = int(np.argmax(deg))
    site = int(np.random.randint(0, ns_count))
    return RelocateOperator(i, site), {"hotspot_i": i}
