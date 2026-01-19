from __future__ import annotations

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
      (L_hw_norm, hw_stats_dict)
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
        iso_cfg = getattr(stable_hw_cfg, "discrete_isolation", None) if stable_hw_cfg else None
        force_use_cached_mapping = False
        if iso_cfg is not None:
            if isinstance(iso_cfg, dict):
                force_use_cached_mapping = bool(iso_cfg.get("force_use_cached_mapping", False))
            else:
                force_use_cached_mapping = bool(getattr(iso_cfg, "force_use_cached_mapping", False))

        if force_use_cached_mapping and mapping is not None:
            cost = mapping_solver.build_cost_matrix(segments, eff_specs, hw_proxy)
            fixed = mapping_solver.evaluate_fixed_mapping(
                mapping=mapping,
                segments=segments,
                cost=cost,
                eff_specs=eff_specs,
                layout_positions=layout_positions,
                distance_scale_ms=float(getattr(cfg.hw, "distance_scale_ms", 0.0)),
            )
            latency_ms = torch.tensor(float(fixed.get("total_latency_ms", 0.0)), device=device)
            mem_mb = torch.tensor(float(fixed.get("mem_mb", 0.0)), device=device)
            energy_mj = torch.tensor(float(fixed.get("energy_mj", 0.0)), device=device)
            comm_ms = torch.tensor(float(fixed.get("comm_ms", 0.0)), device=device)
        else:

            # ---- cache policy: ONLY for no-grad paths ----
            use_cached = bool(getattr(iso_cfg, "use_cached_hw_mats", True))
            want_grad = False
            try:
                for v in (eff_specs or {}).values():
                    if hasattr(v, "requires_grad") and bool(v.requires_grad):
                        want_grad = True
                        break
                if torch.is_grad_enabled():
                    want_grad = True
            except Exception:
                want_grad = True

            if want_grad:
                use_cached = False

            cache = None
            if use_cached and (stable_hw_state is not None) and (mapping_sig is not None):
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
                lat_ms_mat = cache[key]["cost_ms"].to(device=device)
                mem_mb_mat = cache[key]["mem_mb"].to(device=device)
                power_w_mat = cache[key]["power_w"].to(device=device)
            else:
                cost = mapping_solver.build_cost_matrix_torch(segments, eff_specs, hw_proxy, device=device)
                lat_ms_mat = cost["lat_ms"]
                mem_mb_mat = cost["mem_mb"]
                power_w_mat = cost["power_w"]

                if use_cached and (cache is not None) and (key is not None) and (not want_grad):
                    cache[key] = {
                        "cost_ms": lat_ms_mat.detach().cpu(),
                        "mem_mb": mem_mb_mat.detach().cpu(),
                        "power_w": power_w_mat.detach().cpu(),
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
    if stable_hw_state is None:
        stable_hw_state = {}
    ref_T = float(stable_hw_state.get("ref_T", float(latency_ms.detach().item())))
    ref_E = float(stable_hw_state.get("ref_E", float(energy_mj.detach().item())))
    ref_M = float(stable_hw_state.get("ref_M", float(mem_mb.detach().item())))
    ref_C = float(stable_hw_state.get("ref_C", float(comm_ms.detach().item())))

    # ---- Non-negative guard: prevent "negative latency reward" ----
    eps_ratio = float(getattr(stable_hw_cfg, "eps_ratio", 1e-9) if stable_hw_cfg is not None else 1e-9)

    # keep raw for logging (may be negative)
    raw_latency_ms = float(latency_ms.detach().item())
    raw_energy_mj = float(energy_mj.detach().item())
    raw_mem_mb = float(mem_mb.detach().item())
    raw_comm_ms = float(comm_ms.detach().item())

    # clamp for optimization / ratios
    latency_ms_pos = torch.clamp(latency_ms, min=eps_ratio)
    energy_mj_pos = torch.clamp(energy_mj, min=eps_ratio)
    mem_mb_pos = torch.clamp(mem_mb, min=eps_ratio)
    comm_ms_pos = torch.clamp(comm_ms, min=eps_ratio)

    # init refs using clamped values (and sanitize any stale negative refs in state)
    def _pos_ref(state_key: str, default_val: float) -> float:
        v = float(stable_hw_state.get(state_key, default_val))
        return max(eps_ratio, v)

    ref_T = _pos_ref("ref_T", float(latency_ms_pos.detach().item()))
    ref_E = _pos_ref("ref_E", float(energy_mj_pos.detach().item()))
    ref_M = _pos_ref("ref_M", float(mem_mb_pos.detach().item()))
    ref_C = _pos_ref("ref_C", float(comm_ms_pos.detach().item()))

    stable_hw_state["ref_T"] = ref_T
    stable_hw_state["ref_E"] = ref_E
    stable_hw_state["ref_M"] = ref_M
    stable_hw_state["ref_C"] = ref_C

    wT = float(getattr(norm_cfg, "wT", 0.2)) if norm_cfg else 0.0
    wE = float(getattr(norm_cfg, "wE", 0.2)) if norm_cfg else 0.0
    wM = float(getattr(norm_cfg, "wM", 0.4)) if norm_cfg else 0.0
    wC = float(getattr(norm_cfg, "wC", 0.2)) if norm_cfg else 0.0

    # ratios / hinge
    rt = latency_ms_pos / ref_T
    re = energy_mj_pos / ref_E
    rm = mem_mb_pos / ref_M
    rc = comm_ms_pos / ref_C

    L_T = torch.clamp(rt - 1.0, min=0.0)
    L_E = torch.clamp(re - 1.0, min=0.0)
    L_M = torch.clamp(rm - 1.0, min=0.0)
    L_C = torch.clamp(rc - 1.0, min=0.0)

    L_hw_norm = wT * L_T + wE * L_E + wM * L_M + wC * L_C

    t_term = L_T
    e_term = L_E
    m_term = L_M
    c_term = L_C

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
        "latency_ms": float(latency_ms_pos.detach().cpu().item()),
        "energy_mj": float(energy_mj_pos.detach().cpu().item()),
        "mem_mb": float(mem_mb_pos.detach().cpu().item()),
        "comm_ms": float(comm_ms_pos.detach().cpu().item()),
        "raw_latency_ms": raw_latency_ms,
        "raw_energy_mj": raw_energy_mj,
        "raw_mem_mb": raw_mem_mb,
        "raw_comm_ms": raw_comm_ms,
        "clamped_latency_ms": float(latency_ms_pos.detach().cpu().item()),
        "clamped_energy_mj": float(energy_mj_pos.detach().cpu().item()),
        "clamped_mem_mb": float(mem_mb_pos.detach().cpu().item()),
        "clamped_comm_ms": float(comm_ms_pos.detach().cpu().item()),
        "ref_latency_ms": float(ref_T),
        "ref_energy_mj": float(ref_E),
        "ref_mem_mb": float(ref_M),
        "ref_comm_ms": float(ref_C),
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

    hw_stats["L_hw_norm"] = float(L_hw_norm.detach().cpu().item())
    hw_stats["L_hw_total"] = float(L_hw.detach().cpu().item())
    return L_hw_norm, hw_stats
