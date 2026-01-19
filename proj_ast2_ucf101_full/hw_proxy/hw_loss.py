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
      (L_hw_total, hw_stats_dict)
    """
    device = None
    for v in (layout_positions, alpha):
        if isinstance(v, torch.Tensor):
            device = v.device
            break
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # v5.4: if eff_specs/alpha need gradients, DO NOT use fixed-eval path that detaches
    want_grad = False
    try:
        want_grad = torch.is_grad_enabled() and (
            (eff_specs is not None and any(getattr(v, "requires_grad", False) for v in eff_specs.values()))
            or (alpha is not None and getattr(alpha, "requires_grad", False))
        )
    except Exception:
        want_grad = False

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

        if force_use_cached_mapping and want_grad:
            print(
                "[WARN] force_use_cached_mapping=True but gradients are needed; "
                "falling back to differentiable torch path."
            )
            force_use_cached_mapping = False

        if (force_use_cached_mapping and (not want_grad)) and mapping is not None:
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

    norm_cfg = getattr(stable_hw_cfg, "normalize", None) if stable_hw_cfg is not None else None
    if stable_hw_state is None:
        stable_hw_state = {}
    ref_T = float(stable_hw_state.get("ref_T", float(getattr(cfg.hw, "latency_ref_ms", 1.0))))
    ref_E = float(stable_hw_state.get("ref_E", float(getattr(cfg.hw, "energy_ref_mj", 1.0))))
    ref_M = float(stable_hw_state.get("ref_M", float(getattr(cfg.hw, "mem_ref_mb", 1.0))))
    ref_C = float(stable_hw_state.get("ref_C", float(getattr(cfg.hw, "comm_ref_ms", 1.0))))

    # ---- Non-negative guard: prevent "negative latency reward" ----
    eps_ratio = float(getattr(stable_hw_cfg, "eps_ratio", 1e-9) if stable_hw_cfg is not None else 1e-9)

    # keep raw for logging (may be negative)
    raw_latency_ms = float(latency_ms.detach().item())
    raw_energy_mj = float(energy_mj.detach().item())
    raw_mem_mb = float(mem_mb.detach().item())
    raw_comm_ms = float(comm_ms.detach().item())

    # v5.4: clamp audit (counts + min raw values)
    clamp_counts = {"T": 0, "E": 0, "M": 0, "C": 0}
    clamp_mins = {"T": None, "E": None, "M": None, "C": None}

    def _audit_clamp(tag: str, x):
        try:
            x_item = float(x.detach().cpu().item())
        except Exception:
            return
        if clamp_mins[tag] is None or x_item < float(clamp_mins[tag]):
            clamp_mins[tag] = x_item
        if x_item < float(eps_ratio):
            clamp_counts[tag] += 1

    _audit_clamp("T", latency_ms)
    _audit_clamp("E", energy_mj)
    _audit_clamp("M", mem_mb)
    _audit_clamp("C", comm_ms)

    # clamp for optimization / ratios
    latency_ms_pos = torch.clamp(latency_ms, min=eps_ratio)
    energy_mj_pos = torch.clamp(energy_mj, min=eps_ratio)
    mem_mb_pos = torch.clamp(mem_mb, min=eps_ratio)
    comm_ms_pos = torch.clamp(comm_ms, min=eps_ratio)

    # init refs using stable state (sanitize any stale negative refs in state)
    def _pos_ref(state_key: str, default_val: float) -> float:
        v = stable_hw_state.get(state_key, None)
        try:
            v = float(v) if v is not None else None
        except Exception:
            v = None
        if v is None or v <= eps_ratio:
            v = float(default_val)
        v = max(eps_ratio, float(v))
        stable_hw_state[state_key] = float(v)
        return float(v)

    ref_T = _pos_ref("ref_T", float(getattr(cfg.hw, "latency_ref_ms", 1.0)))
    ref_E = _pos_ref("ref_E", float(getattr(cfg.hw, "energy_ref_mj", 1.0)))
    ref_M = _pos_ref("ref_M", float(getattr(cfg.hw, "mem_ref_mb", 1.0)))
    ref_C = _pos_ref("ref_C", float(getattr(cfg.hw, "comm_ref_ms", 1.0)))

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

    # ===== stable_hw.normalize: normalize terms by ref and apply hinge/log =====
    norm_cfg = getattr(stable_hw_cfg, "normalize", None) if stable_hw_cfg is not None else None

    # v5.4: 如果 normalize 块存在，默认认为 enabled=True（即使 YAML 里不写 enabled）
    if norm_cfg is None:
        norm_enabled = False
    else:
        norm_enabled = bool(getattr(norm_cfg, "enabled", True))

    # defaults
    mode = "hinge_log_ratio"
    eps = 1e-6
    clip_term_max = 10.0
    mem_hinge_only = True
    abs_ratio = False
    wT = wE = wM = wC = 0.25
    tT = tE = tM = tC = 1.0

    if norm_cfg is not None:
        mode = str(getattr(norm_cfg, "mode", getattr(norm_cfg, "method", mode)))
        eps = float(getattr(norm_cfg, "eps", getattr(norm_cfg, "clip_eps", eps)))
        clip_term_max = float(getattr(norm_cfg, "clip_term_max", clip_term_max))
        mem_hinge_only = bool(getattr(norm_cfg, "mem_hinge_only", mem_hinge_only))
        abs_ratio = bool(getattr(norm_cfg, "abs_ratio", abs_ratio))

        weights = getattr(norm_cfg, "weights", None)
        if isinstance(weights, dict):
            wT = float(weights.get("wT", wT))
            wE = float(weights.get("wE", wE))
            wM = float(weights.get("wM", wM))
            wC = float(weights.get("wC", wC))
        else:
            wT = float(getattr(norm_cfg, "wT", wT))
            wE = float(getattr(norm_cfg, "wE", wE))
            wM = float(getattr(norm_cfg, "wM", wM))
            wC = float(getattr(norm_cfg, "wC", wC))

        ref = getattr(norm_cfg, "ref", None)
        if isinstance(ref, dict):
            tT = float(ref.get("target_ratio_T", tT))
            tE = float(ref.get("target_ratio_E", tE))
            tM = float(ref.get("target_ratio_M", tM))
            tC = float(ref.get("target_ratio_C", tC))
        else:
            tT = float(getattr(norm_cfg, "target_ratio_T", tT))
            tE = float(getattr(norm_cfg, "target_ratio_E", tE))
            tM = float(getattr(norm_cfg, "target_ratio_M", tM))
            tC = float(getattr(norm_cfg, "target_ratio_C", tC))

    # v5.4: 绝对禁止把 Tensor 变成 float/item() —— 会切断计算图 & 造成“配置看似生效实际无效”
    def _as_t(x) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.to(device=latency_ms_pos.device, dtype=latency_ms_pos.dtype)
        return torch.as_tensor(float(x), device=latency_ms_pos.device, dtype=latency_ms_pos.dtype)

    def _term_t(x_pos_t: torch.Tensor, ref_f: float, target_ratio_f: float, do_hinge: bool) -> torch.Tensor:
        # x_pos_t 必须是 Tensor（允许无梯度，但不能变 float）
        x_t = _as_t(x_pos_t)
        ref_t = _as_t(ref_f)
        ratio = (x_t + float(eps)) / (ref_t + float(eps))
        if abs_ratio:
            ratio = ratio.abs()

        mode_local = str(mode)
        if mode_local == "ratio":
            v = ratio
        elif mode_local == "log_ratio":
            v = torch.log(torch.clamp(ratio, min=float(eps)))
        elif mode_local == "hinge_ratio":
            v = torch.clamp(ratio - float(target_ratio_f), min=0.0)
        else:
            # default: hinge_log_ratio (SPEC v5.4)
            tr = max(float(eps), float(target_ratio_f))
            v = torch.log(torch.clamp(ratio / tr, min=float(eps)))
            v = torch.clamp(v, min=0.0)

        if do_hinge:
            v = torch.clamp(v, min=0.0)
        if clip_term_max is not None:
            v = torch.clamp(v, max=float(clip_term_max))
        return v

    if norm_enabled:
        def _ref_from_state(key: str, fallback: float) -> float:
            v = stable_hw_state.get(key, None) if isinstance(stable_hw_state, dict) else None
            try:
                v = float(v) if v is not None else None
            except Exception:
                v = None
            if v is None or v <= eps:
                v = float(fallback)
            stable_hw_state[key] = float(v)
            return float(v)

        T_ref = _ref_from_state("ref_T", float(getattr(cfg.hw, "latency_ref_ms", 1.0)))
        E_ref = _ref_from_state("ref_E", float(getattr(cfg.hw, "energy_ref_mj", 1.0)))
        M_ref = _ref_from_state("ref_M", float(getattr(cfg.hw, "mem_ref_mb", 1.0)))
        C_ref = _ref_from_state("ref_C", float(getattr(cfg.hw, "comm_ref_ms", 1.0)))

        # IMPORTANT: use clamped-positive tensors so negative proxy cannot be rewarded
        latency_term_t = _term_t(latency_ms_pos, T_ref, tT, do_hinge=True)
        energy_term_t = _term_t(energy_mj_pos, E_ref, tE, do_hinge=True)
        mem_term_t = _term_t(mem_mb_pos, M_ref, tM, do_hinge=bool(mem_hinge_only))
        comm_term_t = _term_t(comm_ms_pos, C_ref, tC, do_hinge=True)

        L_hw_norm_t = (float(wT) * latency_term_t) + (float(wE) * energy_term_t) + (float(wM) * mem_term_t) + (
            float(wC) * comm_term_t
        )
    else:
        T_ref = float(getattr(cfg.hw, "latency_ref_ms", 1.0))
        E_ref = float(getattr(cfg.hw, "energy_ref_mj", 1.0))
        M_ref = float(getattr(cfg.hw, "mem_ref_mb", 1.0))
        C_ref = float(getattr(cfg.hw, "comm_ref_ms", 1.0))
        L_hw_norm_t = (
            latency_ms_pos / max(float(eps), T_ref)
            + energy_mj_pos / max(float(eps), E_ref)
            + mem_mb_pos / max(float(eps), M_ref)
            + comm_ms_pos / max(float(eps), C_ref)
        )

    # v5.4: area/layout 必须保持 Tensor（禁止 float(L_area) / float(L_layout)）
    L_hw_total_t = L_hw_norm_t + L_chip + _as_t(L_area) + _as_t(L_layout)

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
        "proxy_clamp_count": int(clamp_counts["T"] + clamp_counts["E"] + clamp_counts["M"] + clamp_counts["C"]),
        "proxy_clamp_min_values": clamp_mins,
    }
    if chip_used_expected is not None:
        hw_stats["chip_used_expected"] = float(chip_used_expected.detach().cpu().item())
    if area_used_mm2 is not None:
        hw_stats["area_used_mm2"] = float(area_used_mm2.detach().cpu().item())
        hw_stats["area_budget_mm2"] = float(area_budget_mm2)
    if layout_stats:
        hw_stats.update({f"layout_{k}": v for k, v in layout_stats.items()})

    hw_stats.update(
        {
            "L_hw_norm": float(L_hw_norm_t.detach().cpu().item()),
            "L_hw_total": float(L_hw_total_t.detach().cpu().item()),
        }
    )

    # ---- v5.4 guardrail: loss terms must be non-negative in value space ----
    assert (
        hw_stats["latency_ms"] >= 0.0
        and hw_stats["energy_mj"] >= 0.0
        and hw_stats["mem_mb"] >= 0.0
        and hw_stats["comm_ms"] >= 0.0
    ), "Non-negative guard failed: HW stats contains negative values."

    return L_hw_total_t, hw_stats
