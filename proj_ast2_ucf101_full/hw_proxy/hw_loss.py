from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import hashlib
import json
import math

import numpy as np
import torch
import torch.nn.functional as F

from utils.stable_hash import stable_hash


def sanitize_latency(latency_ms: float, min_ms: float = 1e-3) -> Tuple[float, float]:
    penalty = 0.0
    if latency_ms is None:
        return min_ms, penalty
    if isinstance(latency_ms, (int, float)):
        if math.isnan(latency_ms) or math.isinf(latency_ms):
            return min_ms, penalty
        # HARD floor: never allow negative latency to become "better"
        if latency_ms < 0:
            penalty = penalty + abs(float(latency_ms)) * 10.0  # strong penalty
            latency_ms = 0.0
        return max(float(latency_ms), float(min_ms)), penalty
    return min_ms, penalty


def _sanitize_latency_ms(x: float, min_ms: float = 1e-3) -> float:
    # No reward for negative/invalid latency: clamp to min_ms
    if x is None:
        return min_ms
    if isinstance(x, (int, float)):
        if math.isnan(x) or math.isinf(x):
            return min_ms
        return max(float(x), float(min_ms))
    # fallback
    return min_ms


def _sanitize_scalar(x: float, fallback: float = 0.0) -> float:
    if x is None:
        return fallback
    if isinstance(x, (int, float)):
        if math.isnan(x) or math.isinf(x):
            return fallback
        return float(x)
    return fallback


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
            use_cached = bool(getattr(iso_cfg, "use_cached_hw_mats", False))
            # NOTE: want_grad is computed at function entry from differentiable inputs.
            # Do NOT override want_grad based on global torch.is_grad_enabled().
            if want_grad:
                use_cached = False

            cache = None
            if use_cached and (stable_hw_state is not None) and (mapping_sig is not None):
                cache = stable_hw_state.setdefault("discrete_cache", {}).setdefault("hw_mats", {})

            def _tensor_sig(t: torch.Tensor) -> dict:
                tt = t.detach().to("cpu", dtype=torch.float32).contiguous()
                h = hashlib.sha1(tt.numpy().tobytes()).hexdigest()
                return {"shape": list(tt.shape), "sha1": h}

            def _effspec_to_hashable(v):
                # NOTE: hash 用于缓存命中；必须包含 tensor + 标量 + 列表等，否则会错用缓存导致跑偏
                if torch.is_tensor(v):
                    if v.numel() == 0:
                        return {"shape": list(v.shape), "sha1": hashlib.sha1(b"").hexdigest()}
                    return _tensor_sig(v)
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, (list, tuple)):
                    # 尽量保留数值列表
                    if all(isinstance(x, (int, float)) for x in v):
                        return [float(x) for x in v]
                    return [str(x) for x in v]
                if isinstance(v, dict):
                    # dict 递归可序列化
                    return {str(kk): _effspec_to_hashable(vv) for kk, vv in v.items()}
                return str(v)

            eff_sig = {str(k): _effspec_to_hashable(v) for k, v in (eff_specs or {}).items()}
            sig_json = json.dumps(eff_sig, sort_keys=True, separators=(",", ":"))
            eff_key = hashlib.md5(sig_json.encode("utf-8")).hexdigest()[:10]

            key = None
            if cache is not None:
                key = stable_hash(
                    {
                        "mapping_sig": mapping_sig,
                        "segments_sig": segments_sig,
                        "eff": eff_key,
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
    hw_cfg = getattr(cfg, "hw", None)
    ref_T = float(stable_hw_state.get("ref_T", float(getattr(hw_cfg, "latency_ref_ms", 1.0))))
    ref_E = float(stable_hw_state.get("ref_E", float(getattr(hw_cfg, "energy_ref_mj", 1.0))))
    ref_M = float(
        stable_hw_state.get(
            "ref_M",
            float(getattr(hw_cfg, "memory_ref_mb", getattr(hw_cfg, "mem_ref_mb", 1.0))),
        )
    )
    ref_C = float(stable_hw_state.get("ref_C", float(getattr(hw_cfg, "comm_ref_ms", 1.0))))

    # ---- Non-negative guard: prevent "negative latency reward" ----
    eps_ratio = float(getattr(stable_hw_cfg, "eps_ratio", 1e-9) if stable_hw_cfg is not None else 1e-9)
    min_latency_ms = float(
        getattr(stable_hw_cfg, "min_latency_ms", getattr(getattr(cfg, "stable_hw", None), "min_latency_ms", 1e-3))
    )

    # keep raw for logging (may be negative)
    raw_latency_ms = float(latency_ms.detach().cpu().item())
    raw_energy_mj = float(energy_mj.detach().cpu().item())
    raw_mem_mb = float(mem_mb.detach().cpu().item())
    raw_comm_ms = float(comm_ms.detach().cpu().item())
    sanitized_latency_ms, latency_penalty = sanitize_latency(raw_latency_ms, min_ms=min_latency_ms)

    def _as_enabled(v, default: bool = True) -> bool:
        if v is None:
            return default
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        if isinstance(v, dict):
            return bool(v.get("enabled", default))
        return bool(getattr(v, "enabled", default))

    stable_hw_cfg = getattr(cfg, "stable_hw", None) if cfg is not None else None
    no_drift_cfg = getattr(stable_hw_cfg, "no_drift", None) if stable_hw_cfg is not None else None
    no_drift_enabled = _as_enabled(no_drift_cfg, default=True)
    stable_hw_state["no_drift_enabled"] = no_drift_enabled

    def _pos_ref(state_key: str, default_val: float, allow_zero: bool = False) -> float:
        # 1) Prefer existing stable ref (critical for NoDrift + LockedRef)
        if state_key in stable_hw_state:
            try:
                ex = float(stable_hw_state[state_key])
            except Exception:
                ex = None
            if ex is not None and math.isfinite(ex) and (ex > 0.0 or (allow_zero and ex >= 0.0)):
                return ex
            if no_drift_enabled:
                raise ValueError(f"[NoDrift] stable_hw_state[{state_key}] exists but invalid: {stable_hw_state[state_key]}")

        # 2) Fallback to cfg default and initialize state (only when missing)
        v = float(default_val)
        if not math.isfinite(v) or (v <= 0.0 and not (allow_zero and v >= 0.0)):
            v = 1.0 if not allow_zero else 0.0
        stable_hw_state[state_key] = v
        return v

    ref_T = _pos_ref("ref_T", getattr(hw_cfg, "latency_ref_ms", 1.0))
    ref_M = _pos_ref("ref_M", getattr(hw_cfg, "memory_ref_mb", getattr(hw_cfg, "mem_ref_mb", 1.0)))
    ref_E = _pos_ref("ref_E", getattr(hw_cfg, "energy_ref_mj", 1.0))
    ref_C = _pos_ref("ref_C", getattr(hw_cfg, "comm_ref_ms", 1.0))
    ref_A = _pos_ref("ref_A", getattr(hw_cfg, "area_ref_mm2", 1.0))
    ref_B = _pos_ref("ref_B", getattr(hw_cfg, "bw_ref_gbps", 1.0))
    ref_P = _pos_ref("ref_P", getattr(hw_cfg, "power_ref_w", 1.0))

    # Keep setdefault (does NOT overwrite), useful if other code expects these keys
    stable_hw_state.setdefault("ref_T", ref_T)
    stable_hw_state.setdefault("ref_M", ref_M)
    stable_hw_state.setdefault("ref_E", ref_E)
    stable_hw_state.setdefault("ref_C", ref_C)
    stable_hw_state.setdefault("ref_A", ref_A)
    stable_hw_state.setdefault("ref_B", ref_B)
    stable_hw_state.setdefault("ref_P", ref_P)

    protect_against_negative_proxy = (
        bool(getattr(getattr(stable_hw_cfg, "normalize", None), "protect_against_negative_proxy", True))
        if stable_hw_cfg is not None
        else True
    )

    ref_T_t = torch.tensor(ref_T, device=device, dtype=torch.float32)
    ref_E_t = torch.tensor(ref_E, device=device, dtype=torch.float32)
    ref_M_t = torch.tensor(ref_M, device=device, dtype=torch.float32)
    ref_C_t = torch.tensor(ref_C, device=device, dtype=torch.float32)

    def _sanitize_no_reward(x: torch.Tensor, ref_t: torch.Tensor) -> torch.Tensor:
        # invalid = negative or non-finite -> replace with ref (NO reward)
        invalid = (x <= 0) | (~torch.isfinite(x))
        y = torch.where(invalid, ref_t, x)
        return y

    # Track invalid clamps for audit/trace
    clamp_counts = {
        "T": int((latency_ms <= 0).sum().detach().cpu().item()),
        "E": int((energy_mj <= 0).sum().detach().cpu().item()),
        "M": int((mem_mb <= 0).sum().detach().cpu().item()),
        "C": int((comm_ms <= 0).sum().detach().cpu().item()),
    }
    clamp_mins = {
        "T": float(latency_ms.min().detach().cpu().item()),
        "E": float(energy_mj.min().detach().cpu().item()),
        "M": float(mem_mb.min().detach().cpu().item()),
        "C": float(comm_ms.min().detach().cpu().item()),
    }

    raw_latency_ms = float(latency_ms.detach().cpu().item())
    raw_energy_mj = float(energy_mj.detach().cpu().item())
    raw_mem_mb = float(mem_mb.detach().cpu().item())
    raw_comm_norm = float(comm_ms.detach().cpu().item())

    def _penalty_from_raw(x: float) -> float:
        # 合同审计：sanitize 发生时必须能追溯 penalty_added（即便 used 被 ref 替代）
        if not math.isfinite(x):
            return 10.0
        if x < 0.0:
            return abs(x) * 10.0
        return 0.0

    penalty_T = _penalty_from_raw(raw_latency_ms)
    penalty_E = _penalty_from_raw(raw_energy_mj)
    penalty_M = _penalty_from_raw(raw_mem_mb)
    penalty_C = _penalty_from_raw(raw_comm_norm)
    extra_penalty = float(penalty_T + penalty_E + penalty_M + penalty_C)

    # sanitize (no reward for invalid; used becomes ref when invalid)
    latency_ms_pos = _sanitize_no_reward(latency_ms, ref_T_t)
    energy_mj_pos = _sanitize_no_reward(energy_mj, ref_E_t)
    mem_mb_pos = _sanitize_no_reward(mem_mb, ref_M_t)
    comm_ms_pos = _sanitize_no_reward(comm_ms, ref_C_t)

    latency_ms_pos = torch.clamp(latency_ms_pos, min=max(float(eps_ratio), float(min_latency_ms)))
    energy_mj_pos = torch.clamp(energy_mj_pos, min=eps_ratio)
    mem_mb_pos = torch.clamp(mem_mb_pos, min=eps_ratio)
    comm_ms_pos = torch.clamp(comm_ms_pos, min=eps_ratio)

    used_latency_ms = float(latency_ms_pos.detach().cpu().item())
    used_energy_mj = float(energy_mj_pos.detach().cpu().item())
    used_mem_mb = float(mem_mb_pos.detach().cpu().item())
    used_comm_norm = float(comm_ms_pos.detach().cpu().item())

    proxy_raw = {
        "latency_ms": raw_latency_ms,
        "energy_mj": raw_energy_mj,
        "mem_mb": raw_mem_mb,
        "comm_norm": raw_comm_norm,
    }
    proxy_used = {
        "latency_ms": used_latency_ms,
        "energy_mj": used_energy_mj,
        "mem_mb": used_mem_mb,
        "comm_norm": used_comm_norm,
    }

    hw_stats: Dict[str, float] = {
        # legacy raw/used (keep)
        "raw_latency_ms": raw_latency_ms,
        "raw_energy_mj": raw_energy_mj,
        "raw_mem_mb": raw_mem_mb,
        "raw_comm_norm": raw_comm_norm,
        "latency_ms": used_latency_ms,
        "energy_mj": used_energy_mj,
        "mem_mb": used_mem_mb,
        "comm_norm": used_comm_norm,

        # v5.4 audit keys (write-stable, used by trainer trace)
        "proxy_raw_latency_ms": raw_latency_ms,
        "proxy_raw_energy_mj": raw_energy_mj,
        "proxy_raw_mem_mb": raw_mem_mb,
        "proxy_raw_comm_norm": raw_comm_norm,
        "proxy_used_latency_ms": used_latency_ms,
        "proxy_used_energy_mj": used_energy_mj,
        "proxy_used_mem_mb": used_mem_mb,
        "proxy_used_comm_norm": used_comm_norm,

        "proxy_penalty_latency_ms": float(penalty_T),
        "proxy_penalty_energy_mj": float(penalty_E),
        "proxy_penalty_mem_mb": float(penalty_M),
        "proxy_penalty_comm_norm": float(penalty_C),
        "proxy_penalty_total": float(extra_penalty),

        "proxy_clamp_count_T": int(clamp_counts["T"]),
        "proxy_clamp_count_E": int(clamp_counts["E"]),
        "proxy_clamp_count_M": int(clamp_counts["M"]),
        "proxy_clamp_count_C": int(clamp_counts["C"]),
        "proxy_clamp_min_T": float(clamp_mins["T"]),
        "proxy_clamp_min_E": float(clamp_mins["E"]),
        "proxy_clamp_min_M": float(clamp_mins["M"]),
        "proxy_clamp_min_C": float(clamp_mins["C"]),

        "proxy_raw": proxy_raw,
        "proxy_used": proxy_used,
        "proxy_extra_penalty": float(extra_penalty),  # backward compat
    }

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

        T_ref = _ref_from_state("ref_T", float(getattr(hw_cfg, "latency_ref_ms", 1.0)))
        E_ref = _ref_from_state("ref_E", float(getattr(hw_cfg, "energy_ref_mj", 1.0)))
        M_ref = _ref_from_state("ref_M", float(getattr(hw_cfg, "memory_ref_mb", getattr(hw_cfg, "mem_ref_mb", 1.0))))
        C_ref = _ref_from_state("ref_C", float(getattr(hw_cfg, "comm_ref_ms", 1.0)))

        # IMPORTANT: use clamped-positive tensors so negative proxy cannot be rewarded
        latency_term_t = _term_t(latency_ms_pos, T_ref, tT, do_hinge=True)
        energy_term_t = _term_t(energy_mj_pos, E_ref, tE, do_hinge=True)
        mem_term_t = _term_t(mem_mb_pos, M_ref, tM, do_hinge=bool(mem_hinge_only))
        comm_term_t = _term_t(comm_ms_pos, C_ref, tC, do_hinge=True)

        L_hw_norm_t = (float(wT) * latency_term_t) + (float(wE) * energy_term_t) + (float(wM) * mem_term_t) + (
            float(wC) * comm_term_t
        )
    else:
        T_ref = float(getattr(hw_cfg, "latency_ref_ms", 1.0))
        E_ref = float(getattr(hw_cfg, "energy_ref_mj", 1.0))
        M_ref = float(getattr(hw_cfg, "memory_ref_mb", getattr(hw_cfg, "mem_ref_mb", 1.0)))
        C_ref = float(getattr(hw_cfg, "comm_ref_ms", 1.0))
        L_hw_norm_t = (
            latency_ms_pos / max(float(eps), T_ref)
            + energy_mj_pos / max(float(eps), E_ref)
            + mem_mb_pos / max(float(eps), M_ref)
            + comm_ms_pos / max(float(eps), C_ref)
        )

    # v5.4: area/layout 必须保持 Tensor（禁止 float(L_area) / float(L_layout)）
    extra_penalty_t = torch.as_tensor(float(extra_penalty), device=device, dtype=latency_ms_pos.dtype)
    L_hw_total_t = L_hw_norm_t + L_chip + _as_t(L_area) + _as_t(L_layout) + extra_penalty_t

    hw_stats.update(
        {
            "comm_ms": float(comm_ms_pos.detach().cpu().item()),
            "raw_comm_ms": raw_comm_norm,
            "latency_ms_sanitized": float(sanitized_latency_ms),
            "comm_ms_sanitized": _sanitize_scalar(raw_comm_norm, fallback=0.0),
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
            "proxy_had_invalid": bool(
                (clamp_counts["T"] + clamp_counts["E"] + clamp_counts["M"] + clamp_counts["C"]) > 0
            ),
        }
    )
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
