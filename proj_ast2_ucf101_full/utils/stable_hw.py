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
    SPEC: warmup -> ramp -> stabilize
    Also store scheduled lambda in state['lambda_hw_schedule'] for guard to override.
    """
    if not _stable_hw_enabled(stable_hw_cfg):
        state["lambda_hw_schedule"] = 0.0
        state["lambda_hw"] = 0.0
        state["schedule_phase"] = "disabled"
        return 0.0, "disabled"

    sched = get_nested(stable_hw_cfg, "lambda_hw_schedule", None)
    if sched is None or not bool(get_nested(sched, "enabled", True)):
        state["lambda_hw_schedule"] = 0.0
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
    clamp_max = float(get_nested(sched, "clamp_max", lam))
    lam = max(clamp_min, min(clamp_max, lam))

    state["lambda_hw_schedule"] = float(lam)
    state["schedule_phase"] = phase

    # default lambda_hw follows schedule unless guard forces disable
    if bool(state.get("guard_force_disable_hw", False)):
        state["lambda_hw"] = 0.0
    else:
        state["lambda_hw"] = float(lam)

    return float(state["lambda_hw"]), phase


def apply_accuracy_guard(acc1: float, stable_hw_cfg: Any, state: Dict, epoch: int) -> Dict:
    """
    SPEC intent:
      - baseline_acc: from baseline_stats if provided else during warmup keep best acc
      - current acc: optionally EMA
      - if drop > epsilon_drop: scale lambda_hw; streak; force disable when streak>=max_consecutive
      - optionally freeze rho/pruning updates for K epochs (rho_frozen_until_epoch)
      - force-disable clears automatically when recovered (drop <= epsilon)
    Returns guard_info dict for logging.
    """
    info = {
        "guard_active": False,
        "baseline_acc": None,
        "current_acc": None,
        "acc_drop": None,
        "epsilon_drop": None,
        "violate_streak": int(state.get("violate_streak", 0) or 0),
        "guard_triggered": False,
        "guard_force_disable_hw": bool(state.get("guard_force_disable_hw", False)),
        "rho_frozen_until_epoch": int(state.get("rho_frozen_until_epoch", -1) or -1),
    }

    if not _stable_hw_enabled(stable_hw_cfg):
        return info

    guard = get_nested(stable_hw_cfg, "accuracy_guard", None)
    if guard is None or not bool(get_nested(guard, "enabled", True)):
        state["guard_active"] = False
        return info

    state["guard_active"] = True
    info["guard_active"] = True

    eps = float(get_nested(guard, "epsilon_drop", 0.01))
    info["epsilon_drop"] = eps

    # ---- baseline acc init/update ----
    # Prefer baseline_acc from state if already set (may be loaded externally).
    baseline = state.get("baseline_acc", None)

    # During warmup, keep best acc as baseline if not fixed yet
    phase = str(state.get("schedule_phase", ""))
    if baseline is None:
        baseline = float(acc1)
        state["baseline_acc"] = float(baseline)
        state["baseline_source"] = state.get("baseline_source", "first_seen")

    if phase == "warmup":
        # keep best
        if float(acc1) > float(state.get("baseline_acc", baseline)):
            state["baseline_acc"] = float(acc1)
            baseline = float(acc1)

    baseline = float(state.get("baseline_acc", baseline))
    info["baseline_acc"] = baseline

    # ---- current acc (EMA optional) ----
    use_ema = bool(get_nested(guard, "use_ema", True))
    if use_ema:
        beta = float(get_nested(guard, "ema_beta", 0.8))
        prev = float(state.get("acc1_ema", acc1))
        cur = beta * prev + (1.0 - beta) * float(acc1)
        state["acc1_ema"] = float(cur)
        current = float(cur)
    else:
        current = float(acc1)

    info["current_acc"] = current

    drop = float(baseline - current)
    state["acc_drop"] = float(drop)
    info["acc_drop"] = float(drop)

    onv = get_nested(guard, "on_violate", None)
    scale = float(get_nested(onv, "scale_lambda_hw", 0.5) if onv else 0.5)
    maxc = int(get_nested(onv, "max_consecutive", 3) if onv else 3)
    freeze_k = int(get_nested(guard, "freeze_rho_epochs", 0) or 0)

    # ---- recovered: clear force disable & streak ----
    if drop <= eps:
        state["violate_streak"] = 0
        info["violate_streak"] = 0
        if bool(state.get("guard_force_disable_hw", False)):
            state["guard_force_disable_hw"] = False
        # restore lambda_hw to schedule
        state["lambda_hw"] = float(state.get("lambda_hw_schedule", state.get("lambda_hw", 0.0)))
        info["guard_force_disable_hw"] = bool(state.get("guard_force_disable_hw", False))
        return info

    # ---- violated ----
    info["guard_triggered"] = True
    state["violate_streak"] = int(state.get("violate_streak", 0) + 1)
    info["violate_streak"] = int(state["violate_streak"])

    # scale lambda
    state["lambda_hw"] = float(state.get("lambda_hw", 0.0) * scale)

    # freeze rho/pruning updates
    if freeze_k > 0:
        state["rho_frozen_until_epoch"] = max(int(state.get("rho_frozen_until_epoch", -1)), int(epoch + freeze_k))
        info["rho_frozen_until_epoch"] = int(state["rho_frozen_until_epoch"])

    # force disable hw after consecutive violations
    if state["violate_streak"] >= maxc:
        state["guard_force_disable_hw"] = True
        state["lambda_hw"] = 0.0
        info["guard_force_disable_hw"] = True

    return info


def update_hw_refs_from_stats(stable_hw_cfg: Any, stable_hw_state: Dict, stats: Dict) -> Dict:
    """
    Epoch-end ref update.
    - If ref_source == baseline_stats: never EMA-update.
    - If ref_source == ema: EMA-update with beta=stable_hw.hw_refs_update.ema_beta.
    - Bootstrap if not refs_inited.
    """
    if stable_hw_state is None:
        return stable_hw_state
    if not _stable_hw_enabled(stable_hw_cfg):
        return stable_hw_state

    ref_up = get_nested(stable_hw_cfg, "hw_refs_update", None)
    ref_source = str(get_nested(ref_up, "ref_source", "ema")) if ref_up else "ema"
    beta = float(get_nested(ref_up, "ema_beta", 0.95)) if ref_up else 0.95

    # baseline_stats source => do not update here
    if ref_source == "baseline_stats" and bool(stable_hw_state.get("refs_inited", False)):
        return stable_hw_state

    def _f(key: str, default: float = 0.0) -> float:
        v = stats.get(key, default)
        try:
            import torch
            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item())
        except Exception:
            pass
        return float(v)

    eps = float(get_nested(getattr(stable_hw_cfg, "normalize", None), "eps", 1e-6) or 1e-6)
    cur_T = max(eps, _f("latency_ms", 0.0))
    cur_E = max(eps, _f("energy_mj", 0.0))
    cur_M = max(eps, _f("mem_mb", 0.0))
    cur_C = max(0.0, _f("comm_ms", 0.0))

    if not bool(stable_hw_state.get("refs_inited", False)):
        stable_hw_state["ref_T"] = cur_T
        stable_hw_state["ref_E"] = cur_E
        stable_hw_state["ref_M"] = cur_M
        stable_hw_state["ref_C"] = cur_C
        stable_hw_state["refs_inited"] = True
        stable_hw_state["ref_source"] = stable_hw_state.get("ref_source", "bootstrap_first_obs")
        return stable_hw_state

    stable_hw_state["ref_T"] = beta * float(stable_hw_state.get("ref_T", cur_T)) + (1 - beta) * cur_T
    stable_hw_state["ref_E"] = beta * float(stable_hw_state.get("ref_E", cur_E)) + (1 - beta) * cur_E
    stable_hw_state["ref_M"] = beta * float(stable_hw_state.get("ref_M", cur_M)) + (1 - beta) * cur_M
    stable_hw_state["ref_C"] = beta * float(stable_hw_state.get("ref_C", cur_C)) + (1 - beta) * cur_C
    return stable_hw_state
