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
    """
    Returns:
      L_hw: torch scalar with gradient (if eff_specs/layout_positions are tensors with grad path)
      stats: python floats for logging
    """

    device = None
    if eff_specs is not None and "peak_flops" in eff_specs:
        device = eff_specs["peak_flops"].device
    elif layout_positions is not None:
        device = layout_positions.device
    else:
        device = torch.device("cpu")

    # ---- weights / normalization refs ----
    wT = float(get_nested(stable_hw_cfg, "normalize.wT", 1.0)) if stable_hw_cfg is not None else 1.0
    wE = float(get_nested(stable_hw_cfg, "normalize.wE", 0.0)) if stable_hw_cfg is not None else 0.0
    wM = float(get_nested(stable_hw_cfg, "normalize.wM", 0.0)) if stable_hw_cfg is not None else 0.0
    wC = float(get_nested(stable_hw_cfg, "normalize.wC", 0.0)) if stable_hw_cfg is not None else 0.0

    ref_T = float((stable_hw_state or {}).get("ref_T", 1.0))
    ref_E = float((stable_hw_state or {}).get("ref_E", 1.0))
    ref_M = float((stable_hw_state or {}).get("ref_M", 1.0))
    ref_C = float((stable_hw_state or {}).get("ref_C", 1.0))

    # ---- base: if mapping/eff_specs provided, compute from segments->cost matrix ----
    hw_latency_ms = _as_t(0.0, device)
    hw_energy_mj = _as_t(0.0, device)
    hw_mem_mb = _as_t(0.0, device)
    hw_comm_ms = _as_t(0.0, device)

    if segments is not None and mapping is not None and eff_specs is not None and mapping_solver is not None:
        # Differentiable: cost matrix uses eff_specs tensors, proxy torch path
        cost = mapping_solver.build_cost_matrix_torch(segments, eff_specs, hw_proxy, device=device)
        lat_ms = cost["lat_ms"]  # [K,S]
        mem_mb = cost["mem_mb"]
        power_w = cost["power_w"]

        K, S = lat_ms.shape
        map_idx = torch.tensor([int(x) for x in mapping[:K]], device=device, dtype=torch.long)

        # gather per-seg metrics
        idx = torch.arange(K, device=device)
        seg_lat = lat_ms[idx, map_idx]
        seg_mem = mem_mb[idx, map_idx]
        seg_pow = power_w[idx, map_idx]

        # pipeline latency: "balanced" = max slot load
        slot_load = torch.zeros((S,), device=device, dtype=torch.float32)
        slot_load = slot_load.scatter_add(0, map_idx, seg_lat)
        hw_latency_ms = torch.max(slot_load)

        # memory: peak per slot (approx via scatter_max emulation)
        slot_mem = torch.zeros((S,), device=device, dtype=torch.float32)
        # scatter_max not available on all builds; emulate by loop (S small, acceptable)
        for s in range(S):
            m = seg_mem[map_idx == s]
            if m.numel() > 0:
                slot_mem[s] = torch.max(m)
        hw_mem_mb = torch.max(slot_mem)

        # energy: sum(power * time)/1000
        hw_energy_mj = torch.sum(seg_pow * seg_lat) / 1000.0

        # comm: if layout_positions given, add distance-based comm
        if layout_positions is not None and "peak_bw" in eff_specs:
            distance_scale_ms = float(get_nested(cfg, "hw.distance_scale_ms", 0.0))
            comm = _as_t(0.0, device)
            for k in range(K - 1):
                d1 = int(map_idx[k].item())
                d2 = int(map_idx[k + 1].item())
                if d1 == d2:
                    continue
                p1 = layout_positions[d1]
                p2 = layout_positions[d2]
                dist = torch.norm(p1 - p2)
                traffic = float(getattr(segments[k], "traffic_out_bytes", 0.0))
                bw = torch.minimum(eff_specs["peak_bw"][d1], eff_specs["peak_bw"][d2])
                base = _as_t(traffic, device) / (bw + 1e-9) * 1e3
                comm = comm + base + dist * _as_t(distance_scale_ms, device)
            hw_comm_ms = comm

    else:
        # fallback: use model_info (non-differentiable keep ratios are ok; still return torch)
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

    # normalized objective
    Tn = hw_latency_ms / max(ref_T, 1e-6)
    En = hw_energy_mj / max(ref_E, 1e-6)
    Mn = hw_mem_mb / max(ref_M, 1e-6)
    Cn = hw_comm_ms / max(ref_C, 1e-6)

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
    }
    return L_hw, stats
