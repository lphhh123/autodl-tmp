from __future__ import annotations
from collections import Counter


def _sum_matrix(matrix) -> float:
    return float(sum(sum(row) for row in matrix)) if matrix else 0.0


def get_instance_problem_state(instance_data: dict) -> dict:
    sites_xy = instance_data.get("sites", {}).get("sites_xy", [])
    slots = instance_data.get("slots", {})
    mapping = instance_data.get("mapping", {})
    traffic = mapping.get("traffic_bytes", mapping.get("traffic_matrix", []))
    slot_tdp = slots.get("slot_tdp_w", slots.get("tdp", []))
    S = int(slots.get("S", len(slot_tdp)))
    return {
        "S": S,
        "Ns": int(len(sites_xy)),
        "wafer_radius_mm": float(instance_data.get("wafer", {}).get("radius_mm", 0.0)),
        "traffic_sum": _sum_matrix(traffic),
        "tdp_sum": float(sum(slot_tdp)) if slot_tdp else 0.0,
    }


def get_solution_problem_state(instance_data: dict, current_solution) -> dict:
    assign = list(current_solution.assign)
    cnt = Counter(assign)
    dup = sum(v - 1 for v in cnt.values() if v > 1)
    used = len(cnt)
    Ns = int(len(instance_data.get("sites", {}).get("sites_xy", [])))
    empty = Ns - used
    return {
        "assign": assign,
        "used_sites": used,
        "empty_sites": empty,
        "duplicate_count": int(dup),
    }


def get_observation_problem_state(problem_state: dict) -> dict:
    ins = problem_state.get("instance_problem_state", {})
    sol = problem_state.get("solution_problem_state", {})
    return {
        "S": ins.get("S"),
        "Ns": ins.get("Ns"),
        "used_sites": sol.get("used_sites"),
        "empty_sites": sol.get("empty_sites"),
        "duplicate_count": sol.get("duplicate_count"),
    }
