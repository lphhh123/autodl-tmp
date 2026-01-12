import numpy as np

from ...components import RelocateOperator


def greedy_relocate_k(problem_state: dict, algorithm_data: dict, k: int = 8, **kwargs):
    """Sample K relocations and pick the best."""
    inst = problem_state["instance_data"]
    sol = problem_state["current_solution"]
    s_count = int(inst["S"])
    ns_count = int(inst["Ns"])
    eval_assign = problem_state.get("eval_assign", None)

    assign0 = sol.assign.copy()
    best = None
    best_val = float(problem_state.get("total_scalar", 1e30))

    for _ in range(int(k)):
        i = int(np.random.randint(0, s_count))
        site = int(np.random.randint(0, ns_count))
        cand = assign0.copy()
        cand[i] = site
        if eval_assign is not None:
            val = float(eval_assign(cand)["total_scalar"])
        else:
            val = 0.0
        if best is None or val < best_val:
            best, best_val = (i, site), val

    if best is None:
        best = (0, int(assign0[0]))
    return RelocateOperator(best[0], best[1]), {"picked_from_k": int(k)}
