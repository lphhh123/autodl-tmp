from __future__ import annotations


def get_instance_problem_state(instance_data: dict) -> dict:
    slots = instance_data.get("slots", {})
    sites = instance_data.get("sites", {})
    objective_cfg = instance_data.get("objective_cfg", {}) or {}
    scalar_weights = objective_cfg.get("scalar_weights", objective_cfg) if isinstance(objective_cfg, dict) else {}
    return {
        "S": int(slots.get("S", len(slots.get("tdp", slots.get("slot_tdp_w", []))))),
        "Ns": int(len(sites.get("sites_xy", []))),
        "radius_mm": float(instance_data.get("wafer", {}).get("radius_mm", 0.0)),
        "objective_weights": dict(scalar_weights),
    }


def _evaluate_solution(instance_data: dict, assign: list[int]) -> dict:
    try:
        from src.problems.wafer_layout.evaluator_copy import evaluate_layout
    except Exception:  # noqa: BLE001
        return {"total_scalar": 0.0, "comm_norm": 0.0, "therm_norm": 0.0}
    return evaluate_layout(instance_data, assign)


def get_solution_problem_state(instance_data: dict, solution: dict) -> dict:
    assign = list(solution["assign"])
    metrics = _evaluate_solution(instance_data, assign)
    dup = len(assign) - len(set(assign))
    return {
        "total_scalar": float(metrics.get("total_scalar", 0.0)),
        "comm_norm": float(metrics.get("comm_norm", 0.0)),
        "therm_norm": float(metrics.get("therm_norm", 0.0)),
        "assign_summary": {"duplicates": int(dup)},
    }


def get_observation_problem_state(problem_state: dict) -> dict:
    ins = problem_state.get("instance", problem_state.get("instance_problem_state", {}))
    sol = problem_state.get("solution", problem_state.get("solution_problem_state", {}))
    return {
        "summary": {
            "total_scalar": sol.get("total_scalar"),
            "comm_norm": sol.get("comm_norm"),
            "therm_norm": sol.get("therm_norm"),
        },
        "assign_summary": sol.get("assign_summary"),
        "weights": ins.get("objective_weights"),
    }
