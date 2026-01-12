import numpy as np


def get_instance_problem_state(instance_data: dict) -> dict:
    return {
        "S": int(instance_data.get("S", 1)),
        "Ns": int(instance_data.get("Ns", 1)),
        "weights": instance_data.get("weights", {}),
        "sigma": float(instance_data.get("sigma", 1.0)),
    }


def get_solution_problem_state(instance_data: dict, current_solution) -> dict:
    assign = np.asarray(current_solution.assign, dtype=np.int64)
    uniq = len(np.unique(assign))
    dup = int(assign.size - uniq)
    return {
        "unique_sites": uniq,
        "duplicate_sites": dup,
        "assign_head": assign[: min(16, assign.size)].tolist(),
    }


def get_observation_problem_state(problem_state: dict) -> dict:
    inst = problem_state.get("instance_problem_state", {})
    sol = problem_state.get("solution_problem_state", {})
    return {
        "instance": inst,
        "solution": sol,
        "key_value": float(problem_state.get("key_value", 0.0)),
    }
