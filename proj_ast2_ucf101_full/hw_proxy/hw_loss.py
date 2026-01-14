# proj_ast2_ucf101_full/hw_proxy/hw_loss.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from hw_proxy.layer_hw_proxy import LayerHwProxy
from mapping.mapping_solver import MappingSolver
from mapping.segments import Segment


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _make_fallback_segment(model_info: Optional[Dict[str, Any]] = None) -> List[Segment]:
    # 如果没有细分 segments，则用一个“整体段”兜底
    # 注意：Segment 定义在 mapping/segments.py，字段不足也可容忍（用 getattr）
    if model_info and isinstance(model_info.get("segments"), list) and model_info["segments"]:
        return model_info["segments"]
    seg = Segment(
        id=0,
        layer_ids=[0],
        kind="other",
        flops=_as_float(model_info.get("flops", 0.0) if model_info else 0.0),
        bytes=_as_float(model_info.get("bytes", 0.0) if model_info else 0.0),
        seq_len=int(model_info.get("seq_len", 0) if model_info else 0),
        embed_dim=int(model_info.get("embed_dim", 0) if model_info else 0),
        num_heads=int(model_info.get("num_heads", 0) if model_info else 0),
        mlp_ratio=_as_float(model_info.get("mlp_ratio", 0.0) if model_info else 0.0),
        precision=_as_float(model_info.get("precision", 1.0) if model_info else 1.0),
        traffic_in_bytes=_as_float(model_info.get("traffic_in_bytes", 0.0) if model_info else 0.0),
        traffic_out_bytes=_as_float(model_info.get("traffic_out_bytes", 0.0) if model_info else 0.0),
    )
    return [seg]


def compute_hw_loss_generic(
    *,
    segments: List[Segment],
    eff_specs: Dict[str, torch.Tensor],
    hw_proxy: LayerHwProxy,
    mapping_solver: MappingSolver,
    hw_cfg: Any,
    layout_positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], List[int]]:
    """
    通用 HW loss（single + version_c 可共用）
    - segments: K 段
    - eff_specs: (S,) 的 slot specs
    - 返回：L_hw, hw_stats, mapping
    """
    device = eff_specs["peak_flops"].device
    # 只算一次 cost，避免 solve_mapping 内重复
    cost = mapping_solver.build_cost_matrix(segments, eff_specs, hw_proxy)
    mapping_res = mapping_solver.solve_mapping_from_cost(
        segments=segments,
        eff_specs=eff_specs,
        cost=cost,
        layout_positions=layout_positions,
        strategy=getattr(hw_cfg, "mapping_strategy", "greedy_local"),
        distance_scale_ms=_as_float(getattr(hw_cfg, "distance_scale_ms", 0.0)),
        mem_limit_factor=_as_float(getattr(hw_cfg, "mem_limit_factor", getattr(mapping_solver, "mem_limit_factor", 0.9))),
    )
    mapping = mapping_res["mapping"]
    total_latency_ms = mapping_res["total_latency_ms"].to(device)
    comm_ms = mapping_res["comm_ms"].to(device)

    lat_ms = cost["lat_ms"].to(device)
    mem_mb = cost["mem_mb"].to(device)
    power_w = cost["power_w"].to(device)

    K = len(segments)
    S = eff_specs["peak_flops"].shape[0]

    total_energy_j = lat_ms.new_tensor(0.0)
    mem_usage = torch.zeros(S, device=device)
    for k in range(K):
        d = int(mapping[k])
        total_energy_j = total_energy_j + (lat_ms[k, d] / 1e3) * power_w[k, d]
        mem_usage[d] = torch.maximum(mem_usage[d], mem_mb[k, d])
    peak_mem_mb = mem_usage.max()

    # 基本标量项
    lambda_T = _as_float(getattr(hw_cfg, "lambda_T", 1.0), 1.0)
    lambda_E = _as_float(getattr(hw_cfg, "lambda_E", 0.0), 0.0)
    lambda_mem = _as_float(getattr(hw_cfg, "lambda_mem", 0.0), 0.0)

    L_hw = lambda_T * total_latency_ms + lambda_E * total_energy_j + lambda_mem * peak_mem_mb

    hw_stats = {
        "total_latency_ms": total_latency_ms.detach(),
        "total_energy_j": total_energy_j.detach(),
        "peak_mem_mb": peak_mem_mb.detach(),
        "comm_ms": comm_ms.detach(),
    }
    return L_hw, hw_stats, mapping


def stable_hw_schedule(epoch: int, stable_hw_cfg: Any, state: Dict[str, Any]) -> None:
    sched = getattr(stable_hw_cfg, "lambda_hw_schedule", None)
    if sched is None or not bool(getattr(sched, "enabled", True)):
        state["lambda_hw"] = 0.0
        state["schedule_phase"] = "disabled"
        return
    warmup = int(getattr(sched, "warmup_epochs", 0))
    ramp = int(getattr(sched, "ramp_epochs", 0))
    lam_max = _as_float(getattr(sched, "lambda_hw_max", 0.0), 0.0)
    if epoch < warmup:
        lam = 0.0
        phase = "warmup"
    elif epoch < warmup + ramp:
        prog = (epoch - warmup + 1) / max(1, ramp)
        lam = lam_max * prog
        phase = "ramp"
    else:
        lam = lam_max
        phase = "stabilize"
    state["lambda_hw"] = float(lam)
    state["schedule_phase"] = phase


def apply_accuracy_guard(acc1: float, stable_hw_cfg: Any, state: Dict[str, Any]) -> None:
    guard = getattr(stable_hw_cfg, "accuracy_guard", None)
    if guard is None or not bool(getattr(guard, "enabled", True)):
        return
    baseline = state.get("acc_baseline")
    if baseline is None:
        state["acc_baseline"] = float(acc1)
        baseline = state["acc_baseline"]
    use_ema = bool(getattr(guard, "use_ema", True))
    if use_ema:
        beta = _as_float(getattr(guard, "ema_beta", 0.8), 0.8)
        prev = state.get("acc_ema", float(acc1))
        state["acc_ema"] = float(beta * prev + (1 - beta) * acc1)
        current = state["acc_ema"]
    else:
        current = float(acc1)
    drop = float(baseline - current)
    state["acc_drop"] = drop
    eps = _as_float(getattr(guard, "epsilon_drop", 0.01), 0.01)
    if drop > eps:
        onv = getattr(guard, "on_violate", None)
        scale = _as_float(getattr(onv, "scale_lambda_hw", 0.5) if onv else 0.5, 0.5)
        state["lambda_hw"] = float(state.get("lambda_hw", 0.0) * scale)
        state["violate_streak"] = int(state.get("violate_streak", 0) + 1)
        maxc = int(getattr(onv, "max_consecutive", 3) if onv else 3)
        if state["violate_streak"] >= maxc:
            state["lambda_hw"] = 0.0
    else:
        state["violate_streak"] = 0
