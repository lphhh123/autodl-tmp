from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.stable_hash import stable_hash


def compute_hw_loss(
    cfg,
    hw_proxy,
    model_info: Dict[str, Any],
    stable_hw_cfg=None,
    stable_hw_state: Optional[Dict[str, Any]] = None,
    segments=None,
    mapping=None,
    mapping_sig: Optional[str] = None,
    segments_sig: Optional[str] = None,
    eff_specs: Optional[Dict[str, torch.Tensor]] = None,
    layout_positions: Optional[torch.Tensor] = None,
    mapping_solver=None,
    wafer_layout=None,
    alpha: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    SPEC-aligned HW loss:
      L_hw = L_hw_norm + L_chip + L_area + L_layout
    Returns:
      (L_hw_total, hw_stats_dict)
    """
    device = None
    for v in (layout_positions, alpha):
        if isinstance(v, torch.Tensor):
            device = v.device
            break
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latency_ms = None
    energy_mj = None
    mem_mb = None
    comm_ms = None

    if segments and mapping is not None and eff_specs is not None and mapping_solver is not None:
        need_grad = False
        if torch.is_grad_enabled():
            if isinstance(alpha, torch.Tensor) and alpha.requires_grad:
                need_grad = True
            if isinstance(layout_positions, torch.Tensor) and layout_positions.requires_grad:
                need_grad = True
            if eff_specs is not None:
                for v in eff_specs.values():
                    if isinstance(v, torch.Tensor) and v.requires_grad:
                        need_grad = True
                        break

        cache = None
        if (not need_grad) and (stable_hw_state is not None) and (mapping_sig is not None):
            cache = stable_hw_state.setdefault("discrete_cache", {}).setdefault("hw_mats", {})

        eff_sig = {}
        if eff_specs is not None:
            eff_sig = {
                k: float(v.detach().mean().cpu())
                for k, v in eff_specs.items()
                if isinstance(v, torch.Tensor)
            }

        key = None
        if cache is not None:
            key = stable_hash(
                {
                    "mapping_sig": mapping_sig,
                    "segments_sig": segments_sig,
                    "eff": eff_sig,
                    "distance_scale": float(getattr(cfg.hw, "distance_scale_ms", 0.0)),
                    "mem_limit_factor": float(getattr(mapping_solver, "mem_limit_factor", 0.9)),
                }
            )

        if cache is not None and key in cache:
            lat_ms_mat = cache[key]["cost_ms"]
            mem_mb_mat = cache[key]["mem_mb"]
            power_w_mat = cache[key]["power_w"]
        else:
            cost = mapping_solver.build_cost_matrix_torch(segments, eff_specs, hw_proxy, device=device)
            lat_ms_mat = cost["lat_ms"]
            mem_mb_mat = cost["mem_mb"]
            power_w_mat = cost["power_w"]

            if cache is not None and key is not None:
                cache[key] = {
                    "cost_ms": lat_ms_mat.detach(),
                    "mem_mb": mem_mb_mat.detach(),
                    "power_w": power_w_mat.detach(),
                }
        K, S = lat_ms_mat.shape

        map_idx = torch.tensor([int(x) for x in mapping[:K]], device=device, dtype=torch.long)
        seg_idx = torch.arange(K, device=device, dtype=torch.long)

        chosen_lat = lat_ms_mat[seg_idx, map_idx]
        chosen_mem = mem_mb_mat[seg_idx, map_idx]
        chosen_power = power_w_mat[seg_idx, map_idx]

        per_slot_time = torch.zeros((S,), device=device, dtype=torch.float32)
        per_slot_time.scatter_add_(0, map_idx, chosen_lat)
        latency_ms = torch.max(per_slot_time)

        per_slot_mem = torch.zeros((S,), device=device, dtype=torch.float32)
        per_slot_mem.scatter_add_(0, map_idx, chosen_mem)
        mem_mb = torch.max(per_slot_mem)

        energy_mj = torch.sum(chosen_power * chosen_lat) / 1000.0

        comm_ms = torch.zeros((), device=device, dtype=torch.float32)
        if layout_positions is not None and "peak_bw" in eff_specs:
            distance_scale_ms = float(getattr(cfg.hw, "distance_scale_ms", 0.0))
            for k in range(min(K - 1, len(segments) - 1)):
                d1 = int(map_idx[k].item())
                d2 = int(map_idx[k + 1].item())
                if d1 == d2:
                    continue
                dist = torch.norm(layout_positions[d1] - layout_positions[d2])
                traffic = float(getattr(segments[k], "traffic_out_bytes", 0.0))
                bw = torch.minimum(eff_specs["peak_bw"][d1], eff_specs["peak_bw"][d2])
                base = (traffic / (bw + 1e-9)) * 1e3
                comm_ms = comm_ms + base + dist * float(distance_scale_ms)
        else:
            comm_ms = torch.zeros((), device=device, dtype=torch.float32)
    else:
        pred = hw_proxy.predict_from_model_info(model_info)
        latency_ms = torch.tensor(float(pred.get("latency_ms", 0.0)), device=device)
        mem_mb = torch.tensor(float(pred.get("mem_mb", 0.0)), device=device)
        energy_mj = torch.tensor(float(pred.get("energy_mj", 0.0)), device=device)
        comm_ms = torch.tensor(0.0, device=device)

    norm_cfg = getattr(stable_hw_cfg, "normalize", None) if stable_hw_cfg else None
    eps = float(getattr(norm_cfg, "eps", 1e-6)) if norm_cfg else 1e-6

    if stable_hw_state is None:
        stable_hw_state = {}
    ref_T = float(stable_hw_state.get("ref_T", float(latency_ms.detach().item())))
    ref_E = float(stable_hw_state.get("ref_E", float(energy_mj.detach().item())))
    ref_M = float(stable_hw_state.get("ref_M", float(mem_mb.detach().item())))
    ref_C = float(stable_hw_state.get("ref_C", float(comm_ms.detach().item())))

    wT = float(getattr(norm_cfg, "wT", 0.2)) if norm_cfg else 0.0
    wE = float(getattr(norm_cfg, "wE", 0.2)) if norm_cfg else 0.0
    wM = float(getattr(norm_cfg, "wM", 0.4)) if norm_cfg else 0.0
    wC = float(getattr(norm_cfg, "wC", 0.2)) if norm_cfg else 0.0

    tT = float(getattr(norm_cfg, "target_ratio_T", 0.9)) if norm_cfg else 1.0
    tE = float(getattr(norm_cfg, "target_ratio_E", 0.9)) if norm_cfg else 1.0
    tM = float(getattr(norm_cfg, "target_ratio_M", 0.9)) if norm_cfg else 1.0
    tC = float(getattr(norm_cfg, "target_ratio_C", 0.9)) if norm_cfg else 1.0

    clip_term_max = float(getattr(norm_cfg, "clip_term_max", 10.0)) if norm_cfg else 10.0
    mem_hinge_only = bool(getattr(norm_cfg, "mem_hinge_only", True)) if norm_cfg else True
    abs_ratio = bool(getattr(norm_cfg, "abs_ratio", False)) if norm_cfg else False

    def _log_ratio(x: torch.Tensor, ref: float) -> torch.Tensor:
        r = x / x.new_tensor(ref)
        if abs_ratio:
            r = torch.abs(r)
        return torch.log(r + eps)

    t = _log_ratio(latency_ms, ref_T)
    e = _log_ratio(energy_mj, ref_E)
    m = _log_ratio(mem_mb, ref_M)
    c = _log_ratio(comm_ms, ref_C)

    if abs_ratio:
        t_term = torch.abs(t)
        e_term = torch.abs(e)
        m_term = torch.abs(m)
        c_term = torch.abs(c)
    else:
        t_term = F.softplus(t - math.log(max(tT, 1e-9)))
        e_term = F.softplus(e - math.log(max(tE, 1e-9)))
        if mem_hinge_only:
            m_term = F.softplus(m)
        else:
            m_term = F.softplus(m - math.log(max(tM, 1e-9)))
        c_term = F.softplus(c - math.log(max(tC, 1e-9)))

    t_term = torch.clamp(t_term, 0.0, clip_term_max)
    e_term = torch.clamp(e_term, 0.0, clip_term_max)
    m_term = torch.clamp(m_term, 0.0, clip_term_max)
    c_term = torch.clamp(c_term, 0.0, clip_term_max)

    L_hw_norm = wT * t_term + wE * e_term + wM * m_term + wC * c_term

    lambda_chip = float(getattr(cfg.hw, "lambda_chip", 0.0))
    L_chip = torch.zeros((), device=device)
    chip_used_expected = None
    if alpha is not None and lambda_chip > 0.0:
        chip_used_prob = 1.0 - alpha[:, -1]
        chip_used_expected = torch.sum(chip_used_prob)
        L_chip = lambda_chip * chip_used_expected

    lambda_area = float(getattr(cfg.hw, "lambda_area", 0.0))
    area_budget_mm2 = float(getattr(cfg.hw, "area_budget_mm2", 0.0))
    L_area = torch.zeros((), device=device)
    area_used_mm2 = None
    if eff_specs is not None and "area_mm2" in eff_specs and lambda_area > 0.0 and area_budget_mm2 > 0.0:
        area_used_mm2 = torch.sum(eff_specs["area_mm2"])
        L_area = lambda_area * F.relu(area_used_mm2 - area_budget_mm2)

    L_layout = torch.zeros((), device=device)
    layout_stats = {}
    if wafer_layout is not None and segments and mapping is not None and eff_specs is not None:
        L_layout, layout_stats = wafer_layout(
            mapping,
            segments,
            eff_specs,
            lambda_boundary=float(getattr(cfg.hw, "lambda_boundary", 0.0)),
            lambda_overlap=float(getattr(cfg.hw, "lambda_overlap", 0.0)),
            lambda_comm=float(getattr(cfg.hw, "lambda_comm_extra", 0.0)),
            lambda_thermal=float(getattr(cfg.hw, "lambda_thermal", 0.0)),
            distance_scale=float(getattr(cfg.hw, "distance_scale_ms", 0.0)),
        )

    L_hw = L_hw_norm + L_chip + L_area + L_layout

    hw_stats = {
        "latency_ms": float(latency_ms.detach().cpu().item()),
        "energy_mj": float(energy_mj.detach().cpu().item()),
        "mem_mb": float(mem_mb.detach().cpu().item()),
        "comm_ms": float(comm_ms.detach().cpu().item()),
        "ref_T": float(ref_T),
        "ref_E": float(ref_E),
        "ref_M": float(ref_M),
        "ref_C": float(ref_C),
        "t_term": float(t_term.detach().cpu().item()),
        "e_term": float(e_term.detach().cpu().item()),
        "m_term": float(m_term.detach().cpu().item()),
        "c_term": float(c_term.detach().cpu().item()),
        "L_hw_norm": float(L_hw_norm.detach().cpu().item()),
        "L_chip": float(L_chip.detach().cpu().item()),
        "L_area": float(L_area.detach().cpu().item()),
        "L_layout": float(L_layout.detach().cpu().item()),
    }
    if chip_used_expected is not None:
        hw_stats["chip_used_expected"] = float(chip_used_expected.detach().cpu().item())
    if area_used_mm2 is not None:
        hw_stats["area_used_mm2"] = float(area_used_mm2.detach().cpu().item())
        hw_stats["area_budget_mm2"] = float(area_budget_mm2)
    if layout_stats:
        hw_stats.update({f"layout_{k}": v for k, v in layout_stats.items()})

    return L_hw, hw_stats
