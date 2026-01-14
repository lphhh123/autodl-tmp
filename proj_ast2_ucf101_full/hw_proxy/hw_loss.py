from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.config_utils import get_nested
from mapping.mapping_solver import MappingSolver
from hw_proxy.layer_hw_proxy import LayerHwProxy


def _as_tensor(x, device: torch.device, like: Optional[torch.Tensor] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=torch.float32)
    if like is not None:
        return like.new_tensor(float(x))
    return torch.tensor(float(x), device=device, dtype=torch.float32)


def _hinge_log_ratio(x: torch.Tensor, ref: torch.Tensor, target_ratio: float, eps: float) -> torch.Tensor:
    # relu(log((x/ref)/target_ratio))
    t = x.new_tensor(float(target_ratio))
    ratio = x / (ref + eps)
    return F.relu(torch.log((ratio / (t + eps)) + eps))


def _log_ratio(x: torch.Tensor, ref: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.log((x / (ref + eps)) + eps)


def _ratio(x: torch.Tensor, ref: torch.Tensor, eps: float) -> torch.Tensor:
    return x / (ref + eps)


def _clip_term(term: torch.Tensor, clip_term_max: float) -> torch.Tensor:
    if clip_term_max is None:
        return term
    c = float(clip_term_max)
    if c <= 0:
        return term
    return torch.clamp(term, max=c)


def compute_hw_loss(
    cfg: Any,
    hw_proxy: LayerHwProxy,
    model_info: Dict,
    stable_hw_cfg: Optional[Any] = None,
    stable_hw_state: Optional[Dict] = None,
    # -------- Version-C extras (optional) --------
    segments: Optional[list] = None,
    mapping: Optional[list] = None,
    eff_specs: Optional[Dict[str, torch.Tensor]] = None,
    layout_positions: Optional[torch.Tensor] = None,
    mapping_solver: Optional[MappingSolver] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    # ---------------- device ----------------
    if eff_specs is not None and "peak_flops" in eff_specs:
        device = eff_specs["peak_flops"].device
    elif layout_positions is not None:
        device = layout_positions.device
    else:
        device = torch.device("cpu")

    # ---------------- normalize cfg ----------------
    norm_cfg = get_nested(stable_hw_cfg, "normalize", None) if stable_hw_cfg is not None else None

    # weights
    wT = float(get_nested(norm_cfg, "wT", 0.2)) if norm_cfg is not None else 0.2
    wE = float(get_nested(norm_cfg, "wE", 0.2)) if norm_cfg is not None else 0.2
    wM = float(get_nested(norm_cfg, "wM", 0.4)) if norm_cfg is not None else 0.4
    wC = float(get_nested(norm_cfg, "wC", 0.2)) if norm_cfg is not None else 0.2

    # mode / eps / clip
    mode = str(get_nested(norm_cfg, "mode", "hinge_log_ratio")) if norm_cfg is not None else "hinge_log_ratio"
    eps = float(get_nested(norm_cfg, "eps", 1e-9)) if norm_cfg is not None else 1e-9
    clip_term_max = float(get_nested(norm_cfg, "clip_term_max", 10.0)) if norm_cfg is not None else 10.0
    mem_hinge_only = bool(get_nested(norm_cfg, "mem_hinge_only", True)) if norm_cfg is not None else True

    # target ratios (SPEC source of truth)
    tT = float(get_nested(norm_cfg, "target_ratio_T", 0.9)) if norm_cfg is not None else 0.9
    tE = float(get_nested(norm_cfg, "target_ratio_E", 0.9)) if norm_cfg is not None else 0.9
    tM = float(get_nested(norm_cfg, "target_ratio_M", 0.9)) if norm_cfg is not None else 0.9
    tC = float(get_nested(norm_cfg, "target_ratio_C", 0.9)) if norm_cfg is not None else 0.9

    # ---------------- compute base hw metrics ----------------
    hw_latency_ms = _as_tensor(0.0, device)
    hw_energy_mj = _as_tensor(0.0, device)
    hw_mem_mb = _as_tensor(0.0, device)
    hw_comm_ms = _as_tensor(0.0, device)

    # Version-C differentiable path (segments+mapping)
    if segments is not None and mapping is not None and eff_specs is not None and mapping_solver is not None:
        cost = mapping_solver.build_cost_matrix_torch(segments, eff_specs, hw_proxy, device=device)
        lat_ms = cost["lat_ms"]       # [K,S]
        mem_mb = cost["mem_mb"]       # [K,S]
        power_w = cost["power_w"]     # [K,S]

        K, S = lat_ms.shape
        map_idx = torch.tensor([int(x) for x in mapping[:K]], device=device, dtype=torch.long)
        idx = torch.arange(K, device=device)

        seg_lat = lat_ms[idx, map_idx]
        seg_mem = mem_mb[idx, map_idx]
        seg_pow = power_w[idx, map_idx]

        # pipeline latency: max slot load
        slot_load = torch.zeros((S,), device=device, dtype=torch.float32).scatter_add(0, map_idx, seg_lat)
        hw_latency_ms = torch.max(slot_load)

        # peak mem: max per-slot peak
        slot_peak = torch.zeros((S,), device=device, dtype=torch.float32)
        for s in range(S):
            m = seg_mem[map_idx == s]
            if m.numel() > 0:
                slot_peak[s] = torch.max(m)
        hw_mem_mb = torch.max(slot_peak)

        # energy
        hw_energy_mj = torch.sum(seg_pow * seg_lat) / 1000.0

        # comm: traffic + layout_positions
        if layout_positions is not None and "peak_bw" in eff_specs:
            distance_scale_ms = float(get_nested(cfg, "hw.distance_scale_ms", 0.0))

            traffic = mapping_solver.build_traffic_matrix(
                segments, [int(x) for x in mapping[:K]], num_slots=S
            ).to(device=device, dtype=torch.float32)  # [S,S]

            pos = layout_positions.to(device=device, dtype=torch.float32)  # [S,2]
            dist = torch.cdist(pos, pos, p=2)                               # [S,S]

            bw = eff_specs["peak_bw"].to(device=device, dtype=torch.float32)  # [S]
            bw_ij = torch.minimum(bw[:, None], bw[None, :])

            mask = (traffic > 0) & (~torch.eye(S, device=device, dtype=torch.bool))
            base_ms = (traffic / (bw_ij + 1e-9)) * 1e3
            extra_ms = dist * float(distance_scale_ms)
            hw_comm_ms = torch.sum((base_ms + extra_ms) * mask.to(torch.float32))

    else:
        # fallback non-diff path (allowed for non Version-C runs)
        pred = hw_proxy.predict_from_model_info(
            model_info,
            token_keep=float(model_info.get("token_keep", 1.0)),
            head_keep=float(model_info.get("head_keep", 1.0)),
            ch_keep=float(model_info.get("ch_keep", 1.0)),
            block_keep=float(model_info.get("block_keep", 1.0)),
        )
        hw_latency_ms = _as_tensor(pred.get("latency_ms", 0.0), device)
        hw_mem_mb = _as_tensor(pred.get("mem_mb", 0.0), device)
        hw_energy_mj = _as_tensor(pred.get("energy_mj", 0.0), device)

    # ---------------- refs: MUST tolerate None ----------------
    state = stable_hw_state or {}

    def _get_ref(name: str, cur: torch.Tensor) -> torch.Tensor:
        v = state.get(name, None)
        if v is None:
            return cur.detach()  # constant baseline if not initialized yet
        return cur.new_tensor(float(v))

    ref_T_t = _get_ref("ref_T", hw_latency_ms)
    ref_E_t = _get_ref("ref_E", hw_energy_mj)
    ref_M_t = _get_ref("ref_M", hw_mem_mb)
    ref_C_t = _get_ref("ref_C", hw_comm_ms)

    # ---------------- term compute ----------------
    if mode == "ratio":
        term_T = _ratio(hw_latency_ms, ref_T_t, eps)
        term_E = _ratio(hw_energy_mj, ref_E_t, eps)
        term_M = _ratio(hw_mem_mb, ref_M_t, eps)
        term_C = _ratio(hw_comm_ms, ref_C_t, eps)
    elif mode == "log_ratio":
        term_T = _log_ratio(hw_latency_ms, ref_T_t, eps)
        term_E = _log_ratio(hw_energy_mj, ref_E_t, eps)
        term_M = _log_ratio(hw_mem_mb, ref_M_t, eps)
        term_C = _log_ratio(hw_comm_ms, ref_C_t, eps)
    else:
        # default: hinge_log_ratio (SPEC)
        term_T = _hinge_log_ratio(hw_latency_ms, ref_T_t, tT, eps)
        term_E = _hinge_log_ratio(hw_energy_mj, ref_E_t, tE, eps)
        term_C = _hinge_log_ratio(hw_comm_ms, ref_C_t, tC, eps)

        if mem_hinge_only:
            # SPEC: memory uses hinge on ratio, no log
            ratio_M = hw_mem_mb / (ref_M_t + eps)
            term_M = F.relu(ratio_M - hw_mem_mb.new_tensor(float(tM)))
        else:
            term_M = _hinge_log_ratio(hw_mem_mb, ref_M_t, tM, eps)

    # per-term clamp
    term_T = _clip_term(term_T, clip_term_max)
    term_E = _clip_term(term_E, clip_term_max)
    term_M = _clip_term(term_M, clip_term_max)
    term_C = _clip_term(term_C, clip_term_max)

    L_hw = (wT * term_T) + (wE * term_E) + (wM * term_M) + (wC * term_C)

    # ---------------- stats (detach only for logging) ----------------
    stats = {
        "hw_latency_ms": float(hw_latency_ms.detach().item()),
        "hw_energy_mj": float(hw_energy_mj.detach().item()),
        "hw_mem_mb": float(hw_mem_mb.detach().item()),
        "hw_comm_ms": float(hw_comm_ms.detach().item()),
        "ref_T": float(ref_T_t.detach().item()),
        "ref_E": float(ref_E_t.detach().item()),
        "ref_M": float(ref_M_t.detach().item()),
        "ref_C": float(ref_C_t.detach().item()),
        "term_T": float(term_T.detach().item()),
        "term_E": float(term_E.detach().item()),
        "term_M": float(term_M.detach().item()),
        "term_C": float(term_C.detach().item()),
        "target_ratio_T": float(tT),
        "target_ratio_E": float(tE),
        "target_ratio_M": float(tM),
        "target_ratio_C": float(tC),
        "hw_wT": float(wT),
        "hw_wE": float(wE),
        "hw_wM": float(wM),
        "hw_wC": float(wC),
        "hw_norm_mode": str(mode),
        "hw_clip_term_max": float(clip_term_max),
        "hw_mem_hinge_only": bool(mem_hinge_only),
        "refs_inited": bool(state.get("refs_inited", False)),
    }
    return L_hw, stats
