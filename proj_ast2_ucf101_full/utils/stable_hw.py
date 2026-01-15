from __future__ import annotations

from typing import Any, Dict, Tuple

from utils.config_utils import get_nested


def _stable_hw_enabled(stable_hw_cfg) -> bool:
    try:
        return bool(getattr(stable_hw_cfg, "enabled", False))
    except Exception:
        try:
            return bool(stable_hw_cfg.get("enabled", False))
        except Exception:
            return False


def _extract_baseline_refs(baseline_dict: dict) -> tuple[float, float, float]:
    """
    Return (ref_latency_ms, ref_mem_mb, ref_comm_ms) from a baseline_stats.json-like dict.
    Accept multiple key aliases to be robust.
    """

    def pick(keys):
        for k in keys:
            if k in baseline_dict and baseline_dict[k] is not None:
                return float(baseline_dict[k])
        return None

    ref_latency = pick(["ref_latency_ms", "latency_ms", "avg_latency_ms", "mean_latency_ms", "total_latency_ms"])
    ref_mem = pick(["ref_mem_mb", "mem_mb", "avg_mem_mb", "mean_mem_mb", "peak_mem_mb"])
    ref_comm = pick(["ref_comm_ms", "comm_ms", "avg_comm_ms", "mean_comm_ms"])

    if ref_latency is None or ref_mem is None or ref_comm is None:
        raise ValueError(
            f"baseline_stats missing required keys. got keys={sorted(list(baseline_dict.keys()))}. "
            f"need latency({['latency_ms','avg_latency_ms',...]}), mem({['mem_mb','avg_mem_mb',...]}), "
            f"comm({['comm_ms','avg_comm_ms',...]})"
        )
    return ref_latency, ref_mem, ref_comm


def init_hw_refs_from_baseline(
    stable_hw_state: dict,
    stable_hw_cfg,
    baseline_stats_path: str,
) -> dict:
    """
    Initialize stable_hw_state refs from baseline_stats.json.
    No-op if already refs_inited=True.
    """
    if stable_hw_state.get("refs_inited", False):
        return stable_hw_state
    if not _stable_hw_enabled(stable_hw_cfg):
        return stable_hw_state

    import json
    from pathlib import Path

    p = Path(baseline_stats_path)
    if not p.exists():
        raise FileNotFoundError(f"[StableHW] baseline_stats_path not found: {p}")

    d = json.loads(p.read_text(encoding="utf-8"))
    ref_latency, ref_mem, ref_comm = _extract_baseline_refs(d)

    stable_hw_state["ref_latency_ms"] = float(ref_latency)
    stable_hw_state["ref_mem_mb"] = float(ref_mem)
    stable_hw_state["ref_comm_ms"] = float(ref_comm)
    stable_hw_state.setdefault("ref_T", float(ref_latency))
    stable_hw_state.setdefault("ref_M", float(ref_mem))
    stable_hw_state.setdefault("ref_C", float(ref_comm))
    stable_hw_state["refs_inited"] = True
    stable_hw_state["ref_source"] = "baseline_stats"
    stable_hw_state["ref_path"] = str(p)
    return stable_hw_state


def stable_hw_schedule(epoch: int, stable_hw_cfg: Any, state: Dict) -> Tuple[float, str]:
    """
    Returns (lambda_hw, phase) following SPEC:
      warmup -> ramp -> stabilize
    """
    if not _stable_hw_enabled(stable_hw_cfg):
        state["lambda_hw"] = 0.0
        state["schedule_phase"] = "disabled"
        return 0.0, "disabled"

    sched = get_nested(stable_hw_cfg, "lambda_hw_schedule", None)
    if sched is None or not bool(get_nested(sched, "enabled", True)):
        state["lambda_hw"] = 0.0
        state["schedule_phase"] = "schedule_disabled"
        return 0.0, "schedule_disabled"

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

    clamp_min = float(get_nested(sched, "clamp_min", 0.0))
    clamp_max = get_nested(sched, "clamp_max", lam)
    if clamp_max is None:
        clamp_max = lam
    clamp_max = float(clamp_max)
    lam = max(clamp_min, min(clamp_max, lam))

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
    if not _stable_hw_enabled(stable_hw_cfg):
        return

    guard = get_nested(stable_hw_cfg, "accuracy_guard", None)
    if guard is None or not bool(get_nested(guard, "enabled", True)):
        state["guard_active"] = False
        return
    state["guard_active"] = True

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


def update_hw_refs_from_stats(stable_hw_cfg: Any, stable_hw_state: Dict, stats: Dict) -> None:
    """
    Unified ref update:
      - If refs not inited: set ref_* from current stats (dense baseline or first obs)
      - Else: EMA update with beta=stable_hw_cfg.ema_beta
    """
    if stable_hw_state is None:
        return
    if not _stable_hw_enabled(stable_hw_cfg):
        return

    norm_cfg = getattr(stable_hw_cfg, "normalize", None) if stable_hw_cfg else None
    eps = float(getattr(norm_cfg, "eps", 1e-6)) if norm_cfg else 1e-6
    beta = float(getattr(stable_hw_cfg, "ema_beta", 0.9)) if stable_hw_cfg else 0.9

    refs_inited = bool(stable_hw_state.get("refs_inited", False))

    def _f(key: str, default: float = 0.0) -> float:
        v = stats.get(key, default)
        try:
            import torch

            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item())
        except Exception:
            pass
        return float(v)

    cur_T = _f("latency_ms", 0.0)
    cur_E = _f("energy_mj", 0.0)
    cur_M = _f("mem_mb", 0.0)
    cur_C = _f("comm_ms", 0.0)

    if not refs_inited:
        stable_hw_state["ref_T"] = max(eps, cur_T)
        stable_hw_state["ref_E"] = max(eps, cur_E)
        stable_hw_state["ref_M"] = max(eps, cur_M)
        stable_hw_state["ref_C"] = max(0.0, cur_C)
        stable_hw_state["refs_inited"] = True
        stable_hw_state.setdefault("ref_source", "ema_init")
        return

    stable_hw_state["ref_T"] = beta * float(stable_hw_state.get("ref_T", cur_T)) + (1 - beta) * cur_T
    stable_hw_state["ref_E"] = beta * float(stable_hw_state.get("ref_E", cur_E)) + (1 - beta) * cur_E
    stable_hw_state["ref_M"] = beta * float(stable_hw_state.get("ref_M", cur_M)) + (1 - beta) * cur_M
    stable_hw_state["ref_C"] = beta * float(stable_hw_state.get("ref_C", cur_C)) + (1 - beta) * cur_C
