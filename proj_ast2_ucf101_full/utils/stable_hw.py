from __future__ import annotations

from typing import Any, Dict, Tuple

from utils.config_utils import get_nested


def stable_hw_schedule(epoch: int, stable_hw_cfg: Any, state: Dict) -> Tuple[float, str]:
    """
    Returns (lambda_hw, phase) following SPEC:
      warmup -> ramp -> stabilize
    """
    sched = get_nested(stable_hw_cfg, "lambda_hw_schedule", None)
    if sched is None or not bool(get_nested(sched, "enabled", True)):
        state["lambda_hw"] = 0.0
        state["schedule_phase"] = "disabled"
        return 0.0, "disabled"

    warmup = int(get_nested(sched, "warmup_epochs", 0))
    ramp = int(get_nested(sched, "ramp_epochs", 0))
    lam_max = float(get_nested(sched, "lambda_hw_max", 0.0))

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
    return float(lam), phase


def apply_accuracy_guard(acc1: float, stable_hw_cfg: Any, state: Dict) -> None:
    """
    SPEC behavior:
      - baseline = first acc
      - optionally use EMA for current acc
      - if drop > epsilon: scale lambda_hw; if consecutive >= max_consecutive -> lambda_hw = 0
    """
    guard = get_nested(stable_hw_cfg, "accuracy_guard", None)
    if guard is None or not bool(get_nested(guard, "enabled", True)):
        return

    if state.get("acc_baseline") is None:
        state["acc_baseline"] = float(acc1)

    baseline = float(state["acc_baseline"])
    use_ema = bool(get_nested(guard, "use_ema", True))
    if use_ema:
        beta = float(get_nested(guard, "ema_beta", 0.8))
        prev = float(state.get("acc_ema", acc1))
        cur = beta * prev + (1 - beta) * float(acc1)
        state["acc_ema"] = float(cur)
        current = float(cur)
    else:
        current = float(acc1)

    drop = float(baseline - current)
    state["acc_drop"] = float(drop)

    eps = float(get_nested(guard, "epsilon_drop", 0.01))
    if drop > eps:
        onv = get_nested(guard, "on_violate", None)
        scale = float(get_nested(onv, "scale_lambda_hw", 0.5) if onv else 0.5)
        state["lambda_hw"] = float(state.get("lambda_hw", 0.0) * scale)
        state["violate_streak"] = int(state.get("violate_streak", 0) + 1)
        maxc = int(get_nested(onv, "max_consecutive", 3) if onv else 3)
        if state["violate_streak"] >= maxc:
            state["lambda_hw"] = 0.0
    else:
        state["violate_streak"] = 0


def init_or_update_hw_refs_from_stats(
    stable_hw_cfg: Any,
    state: Dict,
    hw_stats: Dict[str, float],
) -> None:
    """
    Initialize ref_T/E/M/C from first observation, then EMA update each epoch/step.
    """
    if hw_stats is None:
        return

    lat = float(hw_stats.get("hw_latency_ms", 0.0))
    eng = float(hw_stats.get("hw_energy_mj", 0.0))
    mem = float(hw_stats.get("hw_mem_mb", 0.0))
    comm = float(hw_stats.get("hw_comm_ms", 0.0))

    # first init
    if not bool(state.get("refs_inited", False)):
        state["ref_T"] = max(1e-6, lat)
        state["ref_E"] = max(1e-6, eng if eng > 0 else 1.0)
        state["ref_M"] = max(1e-6, mem if mem > 0 else 1.0)
        state["ref_C"] = max(1e-6, comm if comm > 0 else 1.0)
        state["refs_inited"] = True
        return

    # EMA update
    norm = get_nested(stable_hw_cfg, "normalize", None)
    beta = float(get_nested(norm, "ema_beta", 0.95)) if norm is not None else 0.95

    state["ref_T"] = beta * float(state.get("ref_T", lat)) + (1 - beta) * lat
    state["ref_E"] = beta * float(state.get("ref_E", eng if eng > 0 else 1.0)) + (1 - beta) * (eng if eng > 0 else 1.0)
    state["ref_M"] = beta * float(state.get("ref_M", mem if mem > 0 else 1.0)) + (1 - beta) * (mem if mem > 0 else 1.0)
    state["ref_C"] = beta * float(state.get("ref_C", comm if comm > 0 else 1.0)) + (1 - beta) * (comm if comm > 0 else 1.0)
