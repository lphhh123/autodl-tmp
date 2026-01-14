from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from utils.config_utils import get_nested
from mapping.mapping_solver import MappingSolver
from hw_proxy.layer_hw_proxy import LayerHwProxy


def _as_t(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(float(x), device=device, dtype=torch.float32)


def _safe_ref(x: float) -> float:
    return float(max(1e-6, x))


def _torch_hinge_log_ratio(
    x: torch.Tensor, ref: float, target: float, eps: float = 1e-9
) -> torch.Tensor:
    """Differentiable hinge on log-ratio: relu(log(x/ref) - log(target))."""
    ref_t = x.new_tensor(float(ref))
    target_t = x.new_tensor(float(target))
    ratio = x / (ref_t + eps)
    return F.relu(torch.log(ratio + eps) - torch.log(target_t + eps))


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
    device = None
    if eff_specs is not None and "peak_flops" in eff_specs:
        device = eff_specs["peak_flops"].device
    elif layout_positions is not None:
        device = layout_positions.device
    else:
        device = torch.device("cpu")

    # ---- weights / normalization refs ----
    norm_cfg = get_nested(stable_hw_cfg, "normalize", None) if stable_hw_cfg is not None else None
    wT = float(get_nested(norm_cfg, "wT", 1.0)) if norm_cfg is not None else 1.0
    wE = float(get_nested(norm_cfg, "wE", 0.0)) if norm_cfg is not None else 0.0
    wM = float(get_nested(norm_cfg, "wM", 0.0)) if norm_cfg is not None else 0.0
    wC = float(get_nested(norm_cfg, "wC", 0.0)) if norm_cfg is not None else 0.0

    mode = str(get_nested(norm_cfg, "mode", "hinge_log_ratio")) if norm_cfg is not None else "hinge_log_ratio"
    eps = float(get_nested(norm_cfg, "eps", 1e-9)) if norm_cfg is not None else 1e-9
    clip_term_max = float(get_nested(norm_cfg, "clip_term_max", 10.0)) if norm_cfg is not None else 10.0
    mem_hinge_only = bool(get_nested(norm_cfg, "mem_hinge_only", True)) if norm_cfg is not None else True

    tT = float(get_nested(norm_cfg, "target_ratio_T", 1.0)) if norm_cfg is not None else 1.0
    tE = float(get_nested(norm_cfg, "target_ratio_E", 1.0)) if norm_cfg is not None else 1.0
    tM = float(get_nested(norm_cfg, "target_ratio_M", 1.0)) if norm_cfg is not None else 1.0
    tC = float(get_nested(norm_cfg, "target_ratio_C", 1.0)) if norm_cfg is not None else 1.0

    # ---- base metrics ----
    hw_latency_ms = _as_t(0.0, device)
    hw_energy_mj = _as_t(0.0, device)
    hw_mem_mb = _as_t(0.0, device)
    hw_comm_ms = _as_t(0.0, device)

    if segments is not None and mapping is not None and eff_specs is not None and mapping_solver is not None:
        # cost matrix from segments+eff_specs
        cost = mapping_solver.build_cost_matrix_torch(segments, eff_specs, hw_proxy, device=device)
        lat_ms = cost["lat_ms"]  # [K,S]
        mem_mb = cost["mem_mb"]
        power_w = cost["power_w"]

        K, S = lat_ms.shape
        map_idx = torch.tensor([int(x) for x in mapping[:K]], device=device, dtype=torch.long)

        idx = torch.arange(K, device=device)
        seg_lat = lat_ms[idx, map_idx]
        seg_mem = mem_mb[idx, map_idx]
        seg_pow = power_w[idx, map_idx]

        # pipeline latency = max slot load
        slot_load = torch.zeros((S,), device=device, dtype=torch.float32)
        slot_load = slot_load.scatter_add(0, map_idx, seg_lat)
        hw_latency_ms = torch.max(slot_load)

        # peak mem = max over slots of max(seg_mem in slot)
        slot_mem = torch.zeros((S,), device=device, dtype=torch.float32)
        for s in range(S):
            m = seg_mem[map_idx == s]
            if m.numel() > 0:
                slot_mem[s] = torch.max(m)
        hw_mem_mb = torch.max(slot_mem)

        # energy = sum(power*time)/1000
        hw_energy_mj = torch.sum(seg_pow * seg_lat) / 1000.0

        # comm: use SxS traffic + layout_positions
        if layout_positions is not None and "peak_bw" in eff_specs:
            distance_scale_ms = float(get_nested(cfg, "hw.distance_scale_ms", 0.0))

            # traffic matrix aligned to slots 0..S-1
            traffic = mapping_solver.build_traffic_matrix(segments, [int(x) for x in mapping[:K]], num_slots=S)  # [S,S]
            traffic = traffic.to(device=device, dtype=torch.float32)

            # pairwise distances [S,S]
            pos = layout_positions.to(device=device, dtype=torch.float32)  # [S,2]
            dist = torch.cdist(pos, pos, p=2)  # [S,S]

            bw = eff_specs["peak_bw"].to(device=device, dtype=torch.float32)  # [S]
            bw_ij = torch.minimum(bw[:, None], bw[None, :])  # [S,S]

            # only i!=j and traffic>0
            mask = (traffic > 0) & (~torch.eye(S, device=device, dtype=torch.bool))
            base = (traffic / (bw_ij + 1e-9)) * 1e3
            extra = dist * float(distance_scale_ms)
            hw_comm_ms = torch.sum((base + extra) * mask.to(torch.float32))

    else:
        # fallback: model_info path
        pred = hw_proxy.predict_from_model_info(
            model_info,
            token_keep=float(model_info.get("token_keep", 1.0)),
            head_keep=float(model_info.get("head_keep", 1.0)),
            ch_keep=float(model_info.get("ch_keep", 1.0)),
            block_keep=float(model_info.get("block_keep", 1.0)),
        )
        hw_latency_ms = _as_t(pred.get("latency_ms", 0.0), device)
        hw_mem_mb = _as_t(pred.get("mem_mb", 0.0), device)
        hw_energy_mj = _as_t(pred.get("energy_mj", 0.0), device)

    state = stable_hw_state or {}
    refs_inited = bool(state.get("refs_inited", False))

    ref_T = float(state.get("ref_T", float(hw_latency_ms.detach().item())))
    ref_E = float(state.get("ref_E", float(hw_energy_mj.detach().item())))
    ref_M = float(state.get("ref_M", float(hw_mem_mb.detach().item())))
    ref_C = float(state.get("ref_C", float(hw_comm_ms.detach().item())))

    target_T = float(getattr(stable_hw_cfg, "target_latency_ratio", 1.30)) if stable_hw_cfg else 1.30
    target_E = float(getattr(stable_hw_cfg, "target_energy_ratio", 1.30)) if stable_hw_cfg else 1.30
    target_M = float(getattr(stable_hw_cfg, "target_mem_ratio", 1.20)) if stable_hw_cfg else 1.20
    target_C = float(getattr(stable_hw_cfg, "target_comm_ratio", 1.50)) if stable_hw_cfg else 1.50

    term_T = _torch_hinge_log_ratio(hw_latency_ms, ref_T, target_T, eps=eps)
    term_E = _torch_hinge_log_ratio(hw_energy_mj, ref_E, target_E, eps=eps)
    term_M = _torch_hinge_log_ratio(hw_mem_mb, ref_M, target_M, eps=eps)
    term_C = _torch_hinge_log_ratio(hw_comm_ms, ref_C, target_C, eps=eps)

    if mem_hinge_only:
        term_M = term_M

    L_hw = wT * term_T + wE * term_E + wM * term_M + wC * term_C

    clamp_max = getattr(stable_hw_cfg, "clamp_hw_term", None) if stable_hw_cfg else None
    if clamp_max is not None:
        L_hw = torch.clamp(L_hw, max=float(clamp_max))

    stats = {
        "hw_latency_ms": float(hw_latency_ms.detach().item()),
        "hw_energy_mj": float(hw_energy_mj.detach().item()),
        "hw_mem_mb": float(hw_mem_mb.detach().item()),
        "hw_comm_ms": float(hw_comm_ms.detach().item()),
        "ref_T": float(ref_T),
        "ref_E": float(ref_E),
        "ref_M": float(ref_M),
        "ref_C": float(ref_C),
        "term_T": float(term_T.detach().item()),
        "term_E": float(term_E.detach().item()),
        "term_M": float(term_M.detach().item()),
        "term_C": float(term_C.detach().item()),
        "target_T": float(target_T),
        "target_E": float(target_E),
        "target_M": float(target_M),
        "target_C": float(target_C),
        "hw_wT": float(wT),
        "hw_wE": float(wE),
        "hw_wM": float(wM),
        "hw_wC": float(wC),
        "hw_norm_mode": str(mode),
        "hw_clip_term_max": float(clip_term_max),
        "hw_mem_hinge_only": bool(mem_hinge_only),
        "refs_inited": bool(refs_inited),
    }
    return L_hw, stats
