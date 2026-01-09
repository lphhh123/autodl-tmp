"""Problem state helpers for wafer layout."""
from __future__ import annotations

from typing import Dict


def _infer_slot_count(instance_data: dict) -> int:
    slots = instance_data.get("slots", [])
    if isinstance(slots, dict):
        return int(slots.get("S", len(slots.get("tdp", []) or [])))
    return int(len(slots))


def _infer_site_count(instance_data: dict) -> int:
    sites_xy = instance_data.get("sites_xy")
    if sites_xy is None:
        sites_xy = instance_data.get("sites", {}).get("sites_xy", [])
    return int(instance_data.get("sites", {}).get("Ns", len(sites_xy or [])))


def get_instance_problem_state(instance_data: dict) -> dict:
    obj = instance_data.get("objective_cfg", {}) or {}
    baseline = instance_data.get("baseline", {}) or {}
    scalar = obj.get("scalar_weights", obj) if isinstance(obj, dict) else {}
    return {
        "num_slots": int(_infer_slot_count(instance_data)),
        "num_sites": int(_infer_site_count(instance_data)),
        "wafer_radius": float(instance_data.get("wafer", {}).get("radius_mm", obj.get("wafer_radius", 1.0))),
        "w_comm": float(scalar.get("w_comm", 1.0)),
        "w_therm": float(scalar.get("w_therm", 1.0)),
        "baseline_comm": float(baseline.get("L_comm", obj.get("comm_baseline", 1.0))),
        "baseline_therm": float(baseline.get("L_therm", obj.get("therm_baseline", 1.0))),
    }


def get_solution_problem_state(instance_data: dict, solution) -> dict:
    assign = getattr(solution, "assign", None)
    if assign is None:
        assign = solution.get("assign", [])

    n = len(assign)
    num_placed = sum(1 for a in assign if int(a) >= 0)
    uniq_sites = len({int(a) for a in assign if int(a) >= 0})
    dup = max(0, num_placed - uniq_sites)

    return {
        "num_slots": int(n),
        "num_placed": int(num_placed),
        "dup_count": int(dup),
    }


def get_observation_problem_state(problem_state: dict) -> dict:
    return {
        "num_slots": int(problem_state.get("num_slots", 0)),
        "num_placed": int(problem_state.get("num_placed", 0)),
        "dup_count": int(problem_state.get("dup_count", 0)),
    }
