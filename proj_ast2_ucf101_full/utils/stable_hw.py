from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)  # OmegaConf-like
    except Exception:
        return getattr(cfg, key, default)


def _get_nested(obj: Any, path: str, default=None):
    cur = obj
    for p in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(p, None)
        else:
            cur = getattr(cur, p, None)
    return default if cur is None else cur


@dataclass
class StableHWState:
    enabled: bool = False

    # schedule output (base) and guard output (effective)
    lambda_hw_base: float = 0.0
    lambda_hw_effective: float = 0.0

    # hard gating to isolate discrete updates (rho/segments) when violated
    allow_discrete_updates: bool = True

    # locked reference
    acc_ref: Optional[float] = None
    acc_ref_source: str = ""  # "baseline_file" | "first_val" | ""

    # last used metric
    acc_used_last: Optional[float] = None

    # recovery FSM
    in_recovery: bool = False
    recovery_start_epoch: int = -1
    recovery_good_epochs: int = 0
    violation_epoch: int = -1

    # logging flags
    metric_mismatch: bool = False
    acc_violation: bool = False
    guard_triggered: bool = False


def _state_get(state: Any, key: str, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def stable_hw_log_fields(st: Any) -> Dict[str, Any]:
    return {
        "stable_hw_enabled": bool(_state_get(st, "enabled", False)),
        "lambda_hw_base": float(_state_get(st, "lambda_hw_base", 0.0) or 0.0),
        "lambda_hw_effective": float(_state_get(st, "lambda_hw_effective", 0.0) or 0.0),
        "allow_discrete_updates": bool(_state_get(st, "allow_discrete_updates", True)),
        "acc_ref": _state_get(st, "acc_ref", None),
        "acc_ref_source": str(_state_get(st, "acc_ref_source", "")),
        "acc_used": _state_get(st, "acc_used", None),
        "acc_used_source": str(_state_get(st, "acc_used_source", "")),
        "guard_mode": str(_state_get(st, "guard_mode", "")),
        "violate_streak": int(_state_get(st, "violate_streak", 0) or 0),
        "recovery_left": int(_state_get(st, "recovery_left", 0) or 0),
        "freeze_schedule_in_recovery": bool(_state_get(st, "freeze_schedule_in_recovery", False)),
        "hw_ref_source": str(_state_get(st, "hw_ref_source", "")),
        "ref_T": _state_get(st, "ref_T", None),
        "ref_E": _state_get(st, "ref_E", None),
        "ref_M": _state_get(st, "ref_M", None),
        "ref_C": _state_get(st, "ref_C", None),
    }


def _load_baseline_acc(baseline_stats_path: str) -> Optional[float]:
    if not baseline_stats_path:
        return None
    p = Path(str(baseline_stats_path))
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    # accept common keys
    for k in ["best_val_acc1", "val_acc1_best", "val_acc1", "acc1", "best_acc1"]:
        if k in data and data[k] is not None:
            try:
                return float(data[k])
            except Exception:
                pass
    return None


def _read_baseline_val_acc1_best(baseline_stats_path: str) -> Optional[float]:
    return _load_baseline_acc(baseline_stats_path)


def stable_hw_schedule(epoch: int, stable_hw_cfg, stable_hw_state: dict) -> float:
    """
    v5.4 NoDoubleScale:
      - schedule produces lambda_hw_base only
      - guard chooses lambda_hw_effective = 0 or base (no extra scaling)
    """
    if stable_hw_state is None:
        stable_hw_state = {}

    enabled = bool(getattr(stable_hw_cfg, "enabled", False)) if stable_hw_cfg is not None else False
    stable_hw_state["enabled"] = bool(enabled)
    if not enabled:
        stable_hw_state["lambda_hw_base"] = float(stable_hw_state.get("lambda_hw_base", 0.0) or 0.0)
        stable_hw_state["lambda_hw_effective"] = float(stable_hw_state.get("lambda_hw_base", 0.0) or 0.0)
        stable_hw_state["allow_discrete_updates"] = True
        return float(stable_hw_state.get("lambda_hw_base", 0.0) or 0.0)

    # v5.4: freeze schedule progression while in RECOVERY if requested
    try:
        guard_mode = stable_hw_state.get("guard_mode", None)
        freeze_flag = bool(stable_hw_state.get("freeze_schedule_in_recovery", False))
        if guard_mode == "RECOVERY" and freeze_flag and ("lambda_hw_base" in stable_hw_state):
            # keep previous schedule_phase/lambda
            return float(stable_hw_state.get("lambda_hw_base", 0.0))
    except Exception:
        pass

    sched = getattr(stable_hw_cfg, "lambda_hw_schedule", None)
    if isinstance(sched, dict):
        sched_enabled = bool(sched.get("enabled", True))
        warmup = int(sched.get("warmup_epochs", 5) or 0)
        ramp = int(sched.get("ramp_epochs", 10) or 0)
        lam_max = float(sched.get("lambda_hw_max", 0.0) or 0.0)
        clamp_min = float(sched.get("clamp_min", 0.0) or 0.0)
        clamp_max = float(sched.get("clamp_max", lam_max) or lam_max)
    else:
        sched_enabled = bool(getattr(sched, "enabled", True)) if sched is not None else True
        warmup = int(getattr(sched, "warmup_epochs", 5)) if sched is not None else 5
        ramp = int(getattr(sched, "ramp_epochs", 10)) if sched is not None else 10
        lam_max = float(getattr(sched, "lambda_hw_max", 0.0) or 0.0) if sched is not None else 0.0
        clamp_min = float(getattr(sched, "clamp_min", 0.0) or 0.0) if sched is not None else 0.0
        clamp_max = float(getattr(sched, "clamp_max", lam_max) or lam_max) if sched is not None else lam_max

    if not sched_enabled:
        stable_hw_state.setdefault("lambda_hw_base", 0.0)
        return float(stable_hw_state.get("lambda_hw_base", 0.0) or 0.0)

    if epoch < warmup:
        lam = 0.0
    else:
        t = epoch - warmup
        if ramp <= 0:
            lam = lam_max
        else:
            lam = lam_max * min(1.0, max(0.0, float(t) / float(ramp)))
    lam = max(float(clamp_min), min(float(clamp_max), float(lam)))
    stable_hw_state["lambda_hw_base"] = float(lam)
    return float(lam)


def _update_warmup_best_value(state: dict, acc: float) -> None:
    """Track best val acc during warmup; stored as warmup_best_val_acc1."""
    if acc is None:
        return
    try:
        acc_f = float(acc)
    except Exception:
        return
    prev = state.get("warmup_best_val_acc1", None)
    if prev is None or acc_f > float(prev):
        state["warmup_best_val_acc1"] = acc_f


def apply_accuracy_guard(
    epoch: int,
    stable_hw_cfg,
    stable_hw_state: dict,
    *,
    val_acc1: float | None,
    train_acc1_ema: float | None,
) -> dict:
    """
    v5.4 Acc-First Hard Gating + LockedAccRef + NoDrift:
      - acc_used := val_acc1 (preferred). If missing, fallback to last seen val or train_ema only if allowed.
      - acc_ref locked: dense_baseline (preferred) OR warmup_best (frozen after warmup).
      - when violated: lambda_hw_effective=0 (or scaled), freeze discrete updates, optionally freeze schedule.
    """
    state = stable_hw_state
    state["epoch"] = int(epoch)

    # ---- read cfg knobs (canonical paths) ----
    guard_cfg = getattr(stable_hw_cfg, "accuracy_guard", None)
    locked_cfg = getattr(stable_hw_cfg, "locked_acc_ref", None)
    controller = None
    if guard_cfg is not None:
        controller = getattr(guard_cfg, "controller", None)
    if controller is None:
        controller = getattr(stable_hw_cfg, "controller", None)  # legacy fallback

    # defaults
    epsilon_drop = float(getattr(guard_cfg, "epsilon_drop", 0.01)) if guard_cfg is not None else 0.01
    allow_train_ema_fallback = bool(getattr(stable_hw_cfg, "allow_train_ema_fallback", True))

    metric_key = "val_acc1"
    if guard_cfg is not None:
        metric_key = str(getattr(guard_cfg, "metric_key", "val_acc1"))

    freeze_schedule_in_recovery = True
    freeze_discrete_in_recovery = True
    cut_hw_loss_on_violate = True
    scale_lambda_hw = 0.0
    recovery_epochs = 1
    recovery_min_epochs = 1
    k_exit = 1
    margin_exit = 0.0

    if controller is not None:
        freeze_schedule_in_recovery = bool(getattr(controller, "freeze_schedule_in_recovery", True))
        freeze_discrete_in_recovery = bool(getattr(controller, "freeze_discrete_updates_in_recovery", True))
        # also accept legacy name
        if hasattr(controller, "freeze_discrete_updates"):
            freeze_discrete_in_recovery = bool(getattr(controller, "freeze_discrete_updates"))
        cut_hw_loss_on_violate = bool(getattr(controller, "cut_hw_loss_on_violate", True))
        scale_lambda_hw = float(getattr(controller, "scale_lambda_hw", 0.0))
        recovery_epochs = int(getattr(controller, "recovery_epochs", 1))
        recovery_min_epochs = int(getattr(controller, "recovery_min_epochs", 1))
        k_exit = int(getattr(controller, "k_exit", 1))
        margin_exit = float(getattr(controller, "margin_exit", 0.0))

    # ---- choose acc_used with explicit source ----
    acc_used = None
    acc_source = "none"

    # 1) val_acc1 now
    if metric_key == "val_acc1" and val_acc1 is not None:
        acc_used = float(val_acc1)
        acc_source = "val"

    # 2) fallback to last val
    if acc_used is None and metric_key == "val_acc1" and state.get("val_acc1_last", None) is not None:
        acc_used = float(state["val_acc1_last"])
        acc_source = "val_last"

    # 3) train ema fallback (only if allowed)
    if acc_used is None and allow_train_ema_fallback and train_acc1_ema is not None:
        acc_used = float(train_acc1_ema)
        acc_source = "train_ema"

    state["acc_used"] = acc_used
    state["acc_used_source"] = acc_source

    # keep last val for next epoch gating
    if val_acc1 is not None:
        state["val_acc1_last"] = float(val_acc1)

    # ---- ensure acc_ref locked (NoDrift) ----
    acc_ref = state.get("acc_ref", None)
    acc_ref_source = state.get("acc_ref_source", None)

    # (a) dense baseline preferred (LockedAccRef)
    prefer_dense = True
    freeze_epoch = int(getattr(locked_cfg, "freeze_epoch", 0)) if locked_cfg is not None else 0
    baseline_stats_path = None
    if locked_cfg is not None:
        prefer_dense = bool(getattr(locked_cfg, "prefer_dense_baseline", True))
        baseline_stats_path = getattr(locked_cfg, "baseline_stats_path", None)
    if baseline_stats_path is None and guard_cfg is not None:
        baseline_stats_path = getattr(guard_cfg, "baseline_stats_path", None)

    if acc_ref is None and prefer_dense and baseline_stats_path:
        v = _read_baseline_val_acc1_best(baseline_stats_path)
        if v is not None:
            acc_ref = float(v)
            acc_ref_source = "dense_baseline"
            state["acc_ref"] = acc_ref
            state["acc_ref_source"] = acc_ref_source

    # (b) warmup_best: track during warmup then freeze after warmup
    if acc_used is not None:
        # warmup tracking always safe
        _update_warmup_best_value(state, acc_used)

    if acc_ref is None:
        # if epoch >= freeze_epoch and we have warmup best, lock it
        if epoch >= int(freeze_epoch) and state.get("warmup_best_val_acc1", None) is not None:
            acc_ref = float(state["warmup_best_val_acc1"])
            acc_ref_source = "warmup_best"
            state["acc_ref"] = acc_ref
            state["acc_ref_source"] = acc_ref_source

    # ---- decide guard mode ----
    guard_mode = state.get("guard_mode", "WARMUP")
    violate = False
    if acc_used is not None and acc_ref is not None:
        violate = (acc_used < (float(acc_ref) - float(epsilon_drop)))

    # streak bookkeeping
    streak = int(state.get("violate_streak", 0))
    if violate:
        streak += 1
    else:
        streak = 0
    state["violate_streak"] = streak

    # ---- transition ----
    if violate:
        guard_mode = "RECOVERY"
        state["recovery_left"] = max(int(recovery_epochs), int(recovery_min_epochs))
    else:
        # exit recovery only after k_exit consecutive safe epochs
        safe_streak = int(state.get("safe_streak", 0))
        safe_streak = safe_streak + 1
        state["safe_streak"] = safe_streak
        if guard_mode == "RECOVERY":
            if safe_streak >= int(k_exit) and (acc_used >= (float(acc_ref) - float(margin_exit))):
                guard_mode = "NORMAL"
        else:
            guard_mode = "NORMAL"

    if guard_mode != "RECOVERY":
        state["recovery_left"] = 0

    state["guard_mode"] = guard_mode

    # ---- apply hard gating to lambda_hw ----
    lambda_hw_base = float(state.get("lambda_hw_base", 0.0))
    lambda_hw_eff = lambda_hw_base

    if guard_mode == "RECOVERY":
        if cut_hw_loss_on_violate:
            lambda_hw_eff = 0.0
        else:
            lambda_hw_eff = float(scale_lambda_hw) * float(lambda_hw_base)

    state["lambda_hw_effective"] = float(lambda_hw_eff)

    # ---- publish discrete update permission (trainer reads this) ----
    allow_discrete_updates = True
    if guard_mode == "RECOVERY" and freeze_discrete_in_recovery:
        allow_discrete_updates = False
    state["allow_discrete_updates"] = bool(allow_discrete_updates)

    # ---- publish schedule freeze hint ----
    state["freeze_schedule_in_recovery"] = bool(freeze_schedule_in_recovery)

    return state


def update_hw_refs_from_stats(stable_hw_cfg, state: dict, hw_stats: dict) -> None:
    if not isinstance(state, dict) or not isinstance(hw_stats, dict):
        return
    norm_cfg = getattr(stable_hw_cfg, "normalize", None) if stable_hw_cfg is not None else None
    beta = float(getattr(norm_cfg, "ref_ema_beta", 0.95)) if norm_cfg is not None else 0.95
    eps = float(getattr(norm_cfg, "eps", 1e-6)) if norm_cfg is not None else 1e-6

    def _update(key: str, value_key: str) -> None:
        raw_val = hw_stats.get(value_key, None)
        if raw_val is None:
            return
        try:
            v = float(raw_val)
        except Exception:
            return
        v = max(float(eps), v)
        prev = state.get(key, None)
        try:
            prev_f = float(prev) if prev is not None else None
        except Exception:
            prev_f = None
        if prev_f is None:
            state[key] = v
        else:
            state[key] = float(beta * prev_f + (1.0 - beta) * v)

    _update("ref_T", "latency_ms")
    _update("ref_E", "energy_mj")
    _update("ref_M", "mem_mb")
    _update("ref_C", "comm_ms")
    state.setdefault("hw_ref_source", "ema")


def init_hw_refs_from_baseline_stats(stable_hw_cfg, state: dict) -> None:
    """Initialize ref_T/E/M/C from dense baseline stats if present."""
    if not isinstance(state, dict):
        return
    guard_cfg = getattr(stable_hw_cfg, "accuracy_guard", None)
    locked_cfg = getattr(stable_hw_cfg, "locked_acc_ref", None)

    path = None
    if locked_cfg is not None:
        path = getattr(locked_cfg, "baseline_stats_path", None)
    if not path and guard_cfg is not None:
        path = getattr(guard_cfg, "baseline_stats_path", None)
    if not path:
        return

    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        return

    def _pick(keys, default=None):
        for k in keys:
            if k in j and j[k] is not None:
                try:
                    return float(j[k])
                except Exception:
                    pass
        return default

    # accept multiple common key names
    T = _pick(["ref_T", "ref_latency_ms", "latency_ref_ms", "ref_latency", "latency_ms"])
    E = _pick(["ref_E", "ref_energy_mj", "energy_ref_mj", "energy_mj"])
    M = _pick(["ref_M", "ref_mem_mb", "mem_ref_mb", "peak_mem_mb", "mem_mb"])
    C = _pick(["ref_C", "ref_comm_ms", "comm_ref_ms", "comm_ms"])

    if T is not None:
        state["ref_T"] = float(max(1e-6, T))
    if E is not None:
        state["ref_E"] = float(max(1e-6, E))
    if M is not None:
        state["ref_M"] = float(max(1e-6, M))
    if C is not None:
        state["ref_C"] = float(max(1e-6, C))

    # lock refs if baseline provided
    if any(v is not None for v in [T, E, M, C]):
        state["hw_ref_source"] = "dense_baseline"
