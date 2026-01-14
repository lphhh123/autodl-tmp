from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from utils.config_utils import get_nested
from mapping.mapping_solver import MappingSolver
from hw_proxy.layer_hw_proxy import LayerHwProxy


def _as_t(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(float(x), device=device, dtype=torch.float32)


def _safe_ref(x: float) -> float:
    return float(max(1e-6, x))


def _normalize_metric(x: torch.Tensor, ref: float, mode: str, hinge_eps: float = 0.0) -> torch.Tensor:
    """
    ratio: x/ref
    hinge_log_ratio: max(0, log(x/ref) - hinge_eps)
      - hinge_eps 可以用于允许轻微超标而不罚
      - 保证数值稳定：x/ref >= 1e-12
    """
    refv = _safe_ref(ref)
    if mode == "ratio":
        return x / refv
    # default hinge_log_ratio
    r = torch.clamp(x / refv, min=1e-12)
    v = torch.log(r)
    if hinge_eps > 0:
        v = v - float(hinge_eps)
    return torch.clamp(v, min=0.0)


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
    hinge_eps = float(get_nested(norm_cfg, "hinge_eps", 0.0)) if norm_cfg is not None else 0.0

    state = stable_hw_state or {}
    ref_T = float(state.get("ref_T", 1.0))
    ref_E = float(state.get("ref_E", 1.0))
    ref_M = float(state.get("ref_M", 1.0))
    ref_C = float(state.get("ref_C", 1.0))

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

    # normalized objective (SPEC: hinge-log-ratio default)
    Tn = _normalize_metric(hw_latency_ms, ref_T, mode=mode, hinge_eps=hinge_eps)
    En = _normalize_metric(hw_energy_mj, ref_E, mode=mode, hinge_eps=hinge_eps)
    Mn = _normalize_metric(hw_mem_mb, ref_M, mode=mode, hinge_eps=hinge_eps)
    Cn = _normalize_metric(hw_comm_ms, ref_C, mode=mode, hinge_eps=hinge_eps)

    L_hw = _as_t(wT, device) * Tn + _as_t(wE, device) * En + _as_t(wM, device) * Mn + _as_t(wC, device) * Cn

    stats = {
        "hw_latency_ms": float(hw_latency_ms.detach().cpu().item()),
        "hw_energy_mj": float(hw_energy_mj.detach().cpu().item()),
        "hw_mem_mb": float(hw_mem_mb.detach().cpu().item()),
        "hw_comm_ms": float(hw_comm_ms.detach().cpu().item()),
        "hw_norm_T": float(Tn.detach().cpu().item()),
        "hw_norm_E": float(En.detach().cpu().item()),
        "hw_norm_M": float(Mn.detach().cpu().item()),
        "hw_norm_C": float(Cn.detach().cpu().item()),
        "hw_wT": float(wT),
        "hw_wE": float(wE),
        "hw_wM": float(wM),
        "hw_wC": float(wC),
        "hw_norm_mode": str(mode),
        "hw_hinge_eps": float(hinge_eps),
    }
    return L_hw, stats
