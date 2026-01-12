from __future__ import annotations

import math
from typing import Iterable, List, Tuple


def _topk_pairs(values: Iterable[Tuple[float, int, int]], k: int = 10) -> List[dict]:
    items = sorted(values, key=lambda x: x[0], reverse=True)
    return [{"value": float(v), "i": int(i), "j": int(j)} for v, i, j in items[:k]]


def _traffic_topk(instance_data: dict, k: int = 10) -> List[dict]:
    mapping = instance_data.get("mapping", {}) or {}
    traffic = mapping.get("traffic_matrix", mapping.get("traffic_bytes", []))
    if not traffic:
        return []
    values = []
    for i, row in enumerate(traffic):
        for j in range(i + 1, len(row)):
            v = float(row[j]) + float(traffic[j][i]) if j < len(traffic) and i < len(traffic[j]) else float(row[j])
            if v != 0.0:
                values.append((v, i, j))
    return _topk_pairs(values, k=k)


def _solution_topk(instance_data: dict, assign: list[int], k: int = 10) -> tuple[List[dict], List[dict]]:
    slots = instance_data.get("slots", {}) or {}
    sites = instance_data.get("sites", {}) or {}
    objective_cfg = instance_data.get("objective_cfg", {}) or {}

    sites_xy = instance_data.get("sites_xy", sites.get("sites_xy", []))
    if not sites_xy or not assign:
        return [], []
    traffic = instance_data.get("mapping", {}).get("traffic_matrix", [])
    chip_tdp = slots.get("tdp", slots.get("slot_tdp_w", []))
    sigma_mm = float(objective_cfg.get("sigma_mm", 20.0))

    def _pos(idx: int) -> Tuple[float, float]:
        site_id = int(assign[idx])
        if 0 <= site_id < len(sites_xy):
            x, y = sites_xy[site_id]
            return float(x), float(y)
        return 0.0, 0.0

    comm_values = []
    therm_values = []
    for i in range(len(assign)):
        xi, yi = _pos(i)
        for j in range(i + 1, len(assign)):
            xj, yj = _pos(j)
            dist = math.hypot(xi - xj, yi - yj)
            if traffic and i < len(traffic) and j < len(traffic[i]):
                t_val = float(traffic[i][j]) + float(traffic[j][i]) if j < len(traffic) and i < len(traffic[j]) else float(traffic[i][j])
                if t_val != 0.0:
                    comm_values.append((t_val * dist, i, j))
            if chip_tdp and i < len(chip_tdp) and j < len(chip_tdp) and dist > 1e-9:
                therm_values.append((float(chip_tdp[i]) * float(chip_tdp[j]) * math.exp(-dist / sigma_mm), i, j))

    return _topk_pairs(comm_values, k=k), _topk_pairs(therm_values, k=k)


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
        "traffic_topk": _traffic_topk(instance_data, k=10),
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
    ns = int(instance_data.get("sites", {}).get("Ns", len(instance_data.get("sites", {}).get("sites_xy", []))))
    free_sites = max(0, ns - len(set(assign)))
    comm_topk, therm_topk = _solution_topk(instance_data, assign, k=10)
    return {
        "total_scalar": float(metrics.get("total_scalar", 0.0)),
        "comm_norm": float(metrics.get("comm_norm", 0.0)),
        "therm_norm": float(metrics.get("therm_norm", 0.0)),
        "assign_summary": {"duplicates": int(dup), "free_sites_count": int(free_sites)},
        "comm_topk": comm_topk,
        "therm_topk": therm_topk,
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
        "comm_topk": sol.get("comm_topk", []),
        "therm_topk": sol.get("therm_topk", []),
        "traffic_topk": ins.get("traffic_topk", []),
        "weights": ins.get("objective_weights"),
    }
