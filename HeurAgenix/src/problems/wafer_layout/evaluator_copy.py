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

    Compatibility:
    - Old flat instance format:
        sites_xy_mm, traffic_bytes, chip_tdp_w
    - Project nested instance format (proj_ast2_ucf101_full layout_input.json style):
        sites.sites_xy, mapping.traffic_matrix, slots.tdp (or slots.slot_tdp_w)
    Baseline normalization:
    - Prefer baseline.comm_cost / baseline.therm_peak (old)
    - Fallback to baseline.L_comm / baseline.L_therm (project)
    Objective weights:
    - Prefer instance.scalar_w (old)
    - Fallback to objective_cfg.scalar_weights (project)
    """

    def _get_path(d: Dict, path: Tuple[str, ...], default=None):
        cur = d
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    # --------- resolve required arrays (support both schemas) ---------
    sites_xy_val = instance.get("sites_xy_mm", None)
    if sites_xy_val is None:
        sites_xy_val = _get_path(instance, ("sites", "sites_xy"), None)
    if sites_xy_val is None:
        sites_xy_val = instance.get("sites_xy", None)

    traffic_val = instance.get("traffic_bytes", None)
    if traffic_val is None:
        traffic_val = _get_path(instance, ("mapping", "traffic_matrix"), None)
    if traffic_val is None:
        traffic_val = _get_path(instance, ("mapping", "traffic_bytes"), None)

    tdp_val = instance.get("chip_tdp_w", None)
    if tdp_val is None:
        tdp_val = _get_path(instance, ("slots", "tdp"), None)
    if tdp_val is None:
        tdp_val = _get_path(instance, ("slots", "slot_tdp_w"), None)

    if sites_xy_val is None or traffic_val is None or tdp_val is None:
        raise KeyError(
            "instance missing required keys. Need either flat keys "
            "(sites_xy_mm/traffic_bytes/chip_tdp_w) OR nested keys "
            "(sites.sites_xy + mapping.traffic_matrix + slots.tdp)."
        )

    sites_xy = np.asarray(sites_xy_val, dtype=np.float32)
    traffic = np.asarray(traffic_val, dtype=np.float64)
    tdp = np.asarray(tdp_val, dtype=np.float64)

    # --------- resolve sizes (project schema uses S=chiplets, Ns=sites) ---------
    # chiplets count
    n_chiplets = None
    slots_S = _get_path(instance, ("slots", "S"), None)
    if slots_S is not None:
        n_chiplets = int(slots_S)
    else:
        # old schema used "Ns" as chiplets count
        n_chiplets = int(instance.get("Ns", len(assign)))

    # sites count
    n_sites = None
    sites_Ns = _get_path(instance, ("sites", "Ns"), None)
    if sites_Ns is not None:
        n_sites = int(sites_Ns)
    else:
        # old schema used "S" as sites count
        n_sites = int(instance.get("S", sites_xy.shape[0]))

    # --------- validate shapes ---------
    if assign.shape[0] != n_chiplets:
        raise ValueError(
            f"assign length mismatch: got {assign.shape[0]}, expected n_chiplets={n_chiplets}"
        )
    if sites_xy.shape[0] != n_sites:
        raise ValueError(f"sites_xy mismatch: got {sites_xy.shape[0]}, expected n_sites={n_sites}")
    if traffic.shape[0] != n_chiplets or traffic.shape[1] != n_chiplets:
        raise ValueError(
            f"traffic shape mismatch: got {traffic.shape}, expected ({n_chiplets},{n_chiplets})"
        )
    if tdp.shape[0] != n_chiplets:
        raise ValueError(
            f"tdp length mismatch: got {tdp.shape[0]}, expected n_chiplets={n_chiplets}"
        )

    # --------- objective weights & sigma ---------
    scalar_w = instance.get("scalar_w", None) or {}
    if not scalar_w:
        obj = instance.get("objective_cfg", None) or instance.get("objective", None) or {}
        scalar_w = obj.get("scalar_weights", None) or {}

    w_comm = _safe_float(scalar_w.get("w_comm", 1.0), 1.0)
    w_therm = _safe_float(scalar_w.get("w_therm", 1.0), 1.0)

    sigma = instance.get("sigma_mm", None)
    if sigma is None:
        obj = instance.get("objective_cfg", None) or instance.get("objective", None) or {}
        sigma = obj.get("sigma_mm", 5.0)
    sigma = _safe_float(sigma, 5.0)
    if sigma <= 0:
        sigma = 5.0

    # --------- compute comm cost (upper triangle) ---------
    xy = sites_xy[assign.astype(np.int64)]
    dx = xy[:, 0:1] - xy[:, 0:1].T
    dy = xy[:, 1:2] - xy[:, 1:2].T
    dist = np.sqrt(dx * dx + dy * dy)
    upper = np.triu_indices(n_chiplets, 1)
    comm_cost = float(np.sum(traffic[upper] * dist[upper]))

    # --------- compute thermal peak via Gaussian kernel over sites ---------
    site_heat = np.zeros((n_sites,), dtype=np.float64)
    for i in range(n_chiplets):
        site = int(assign[i])
        if site < 0 or site >= n_sites:
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

    # --------- baseline normalization (support both schemas) ---------
    baseline = instance.get("baseline", {}) or {}
    # old names
    base_comm = baseline.get("comm_cost", None)
    base_therm = baseline.get("therm_peak", None)
    # project names
    if base_comm is None:
        base_comm = baseline.get("L_comm", None)
    if base_therm is None:
        base_therm = baseline.get("L_therm", None)

    base_comm = _safe_float(base_comm, None)
    base_therm = _safe_float(base_therm, None)

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
