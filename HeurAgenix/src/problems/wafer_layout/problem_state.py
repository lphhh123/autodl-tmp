import numpy as np


def get_instance_problem_state(instance_data: dict) -> dict:
    s_count = int(instance_data.get("S", 1))
    ns_count = int(instance_data.get("Ns", 1))
    w = instance_data.get("weights", {})
    sigma = float(instance_data.get("sigma", 1.0))
    return {
        "S": s_count,
        "Ns": ns_count,
        "wafer_radius": float(instance_data.get("wafer", {}).get("radius", 1.0)),
        "weights": {
            "w_comm": float(w.get("w_comm", 0.5)),
            "w_therm": float(w.get("w_therm", 0.5)),
        },
        "sigma": sigma,
    }


def get_solution_problem_state(instance_data: dict, current_solution) -> dict:
    assign = np.asarray(getattr(current_solution, "assign", []), dtype=np.int64)
    uniq = int(len(set(assign.tolist()))) if assign.size > 0 else 0
    dup = int(assign.size - uniq)
    return {
        "unique_sites": uniq,
        "dup_sites": dup,
        "assign_head": [int(x) for x in assign.tolist()[: min(16, assign.size)]],
    }


def get_observation_problem_state(problem_state: dict) -> dict:
    inst = problem_state.get("instance_problem_state", {})
    sol = problem_state.get("solution_problem_state", {})
    key_item = problem_state.get("key_item", "total_scalar")
    key_val = problem_state.get("key_value", None)
    return {
        "instance": inst,
        "solution": sol,
        "key_item": key_item,
        "key_value": key_val,
    }


def attach_key_value(problem_state: dict, key_value: float):
    problem_state["key_value"] = float(key_value)
    return problem_state
