from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def evaluate_layout(instance: Dict, assign: np.ndarray) -> Tuple[float, float, float]:
    """
    Return: (total_scalar, comm_norm, therm_norm)
    Must match proj_ast2_ucf101_full/layout/evaluator.py::LayoutEvaluator.evaluate_scalar semantics.
    instance: json dict loaded by HeurAgenix wafer_layout env
    assign : shape [Ns], site_id per chiplet
    """
    if "sites_xy_mm" not in instance or "traffic_bytes" not in instance or "chip_tdp_w" not in instance:
        raise KeyError("instance missing required keys: sites_xy_mm / traffic_bytes / chip_tdp_w")

    sites_xy = np.asarray(instance["sites_xy_mm"], dtype=np.float32)
    traffic = np.asarray(instance["traffic_bytes"], dtype=np.float64)
    tdp = np.asarray(instance["chip_tdp_w"], dtype=np.float64)

    Ns = int(instance.get("Ns", len(assign)))
    S = int(instance.get("S", sites_xy.shape[0]))
    if assign.shape[0] != Ns:
        raise ValueError(f"assign length mismatch: got {assign.shape[0]}, expected Ns={Ns}")
    if sites_xy.shape[0] != S:
        raise ValueError(f"sites_xy_mm mismatch: got {sites_xy.shape[0]}, expected S={S}")
    if traffic.shape[0] != Ns or traffic.shape[1] != Ns:
        raise ValueError(f"traffic_bytes shape mismatch: got {traffic.shape}, expected ({Ns},{Ns})")
    if tdp.shape[0] != Ns:
        raise ValueError(f"chip_tdp_w length mismatch: got {tdp.shape[0]}, expected Ns={Ns}")

    scalar_w = instance.get("scalar_w", {}) or {}
    w_comm = _safe_float(scalar_w.get("w_comm", 1.0), 1.0)
    w_therm = _safe_float(scalar_w.get("w_therm", 1.0), 1.0)

    xy = sites_xy[assign.astype(np.int64)]
    dx = xy[:, 0:1] - xy[:, 0:1].T
    dy = xy[:, 1:2] - xy[:, 1:2].T
    dist = np.sqrt(dx * dx + dy * dy)
    upper = np.triu_indices(Ns, 1)
    comm_cost = float(np.sum(traffic[upper] * dist[upper]))

    sigma = _safe_float(instance.get("sigma_mm", 5.0), 5.0)
    if sigma <= 0:
        sigma = 5.0
    site_heat = np.zeros((S,), dtype=np.float64)
    for i in range(Ns):
        site = int(assign[i])
        if site < 0 or site >= S:
            raise ValueError(f"assign has invalid site_id={site} at i={i}")
        site_heat[site] += float(tdp[i])

    sx = sites_xy[:, 0:1]
    sy = sites_xy[:, 1:2]
    ddx = sx - sx.T
    ddy = sy - sy.T
    d2 = ddx * ddx + ddy * ddy
    K = np.exp(-d2 / (2.0 * sigma * sigma))
    therm_vec = K @ site_heat
    therm_peak = float(np.max(therm_vec))

    baseline = instance.get("baseline", {}) or {}
    base_comm = _safe_float(baseline.get("comm_cost", None), None)
    base_therm = _safe_float(baseline.get("therm_peak", None), None)
    if base_comm is None or base_comm <= 0:
        comm_norm = comm_cost
    else:
        comm_norm = comm_cost / base_comm
    if base_therm is None or base_therm <= 0:
        therm_norm = therm_peak
    else:
        therm_norm = therm_peak / base_therm

    total_scalar = float(w_comm * comm_norm + w_therm * therm_norm)
    return total_scalar, float(comm_norm), float(therm_norm)
