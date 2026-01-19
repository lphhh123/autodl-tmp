from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------
# helpers
# ---------------------------
def _get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _deepget(cfg: Any, path: str, default=None):
    cur = cfg
    for k in path.split("."):
        cur = _get(cur, k, None)
        if cur is None:
            return default
    return cur


def _load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------
# v5 decision
# ---------------------------
@dataclass
class StableHWDecision:
    guard_mode: str  # "HW_OPT" | "RECOVERY"
    lambda_hw_base: float
    lambda_hw_effective: float
    allow_discrete_updates: bool
    acc_ref: Optional[float]
    acc_used: Optional[float]
    acc_used_source: str  # "val" | "last_val" | "ema_train" | "none"
    schedule_epoch: int
    recovery_epochs_left: int


def _schedule_lambda_hw(schedule_cfg: Any, schedule_epoch: int) -> float:
    enabled = bool(_get(schedule_cfg, "enabled", True))
    if not enabled:
        return 0.0
    lam_max = float(_get(schedule_cfg, "lambda_hw_max", 0.0))
    warmup = int(_get(schedule_cfg, "warmup_epochs", 0))
    ramp = int(_get(schedule_cfg, "ramp_epochs", 0))
    # v5: warmup -> ramp -> plateau
    if lam_max <= 0:
        return 0.0
    if warmup > 0 and schedule_epoch < warmup:
        return 0.0
    if ramp > 0:
        t = schedule_epoch - warmup
        if t < 0:
            t = 0
        if t < ramp:
            return lam_max * float(t + 1) / float(ramp)
    return lam_max


def stable_hw_init_state(cfg: Any) -> Dict[str, Any]:
    state = {
        "guard_mode": "HW_OPT",
        "schedule_epoch": 0,
        "lambda_hw_base": 0.0,
        "lambda_hw_effective": 0.0,
        "allow_discrete_updates": True,
        "acc_ref": None,
        "acc_ref_locked": False,
        "best_warmup_acc": None,
        "acc_ema": None,
        "last_val_acc": None,
        "acc_used": None,
        "acc_used_source": "none",
        "recovery_epochs_left": 0,
        "exit_ok_streak": 0,
    }
    guard = _deepget(cfg, "stable_hw.accuracy_guard", {}) or {}
    baseline_path = str(_get(guard, "baseline_stats_path", "") or "").strip()
    metric_key = str(_get(guard, "metric_key", "val_acc1"))
    state["metric_key"] = metric_key
    if baseline_path:
        js = _load_json(baseline_path)
        cand = None
        if metric_key in js:
            cand = js.get(metric_key)
        if cand is None and "val_acc1_best" in js:
            cand = js.get("val_acc1_best")
        if cand is None and "best" in js and isinstance(js["best"], dict):
            cand = js["best"].get(metric_key)
        if cand is not None:
            state["acc_ref"] = float(cand)
            state["acc_ref_locked"] = True
    return state


def stable_hw_before_epoch(cfg: Any, state: Dict[str, Any]) -> StableHWDecision:
    stable = _get(cfg, "stable_hw", None)
    if stable is None or not bool(_get(stable, "enabled", True)):
        return StableHWDecision(
            guard_mode="HW_OPT",
            lambda_hw_base=0.0,
            lambda_hw_effective=0.0,
            allow_discrete_updates=True,
            acc_ref=state.get("acc_ref"),
            acc_used=state.get("acc_used"),
            acc_used_source=state.get("acc_used_source", "none"),
            schedule_epoch=int(state.get("schedule_epoch", 0)),
            recovery_epochs_left=int(state.get("recovery_epochs_left", 0)),
        )

    controller = _get(stable, "controller", {}) or {}
    schedule_cfg = _get(stable, "lambda_hw_schedule", {}) or {}

    guard_mode = str(state.get("guard_mode", "HW_OPT"))
    schedule_epoch = int(state.get("schedule_epoch", 0))

    freeze_schedule = bool(_get(controller, "freeze_schedule_in_recovery", True))
    if guard_mode != "RECOVERY" or not freeze_schedule:
        lam_base = _schedule_lambda_hw(schedule_cfg, schedule_epoch)
    else:
        lam_base = float(state.get("lambda_hw_base", 0.0))

    lam_eff = lam_base if guard_mode == "HW_OPT" else 0.0
    allow_discrete = bool(state.get("allow_discrete_updates", True)) if guard_mode == "HW_OPT" else False

    state["lambda_hw_base"] = float(lam_base)
    state["lambda_hw_effective"] = float(lam_eff)
    state["allow_discrete_updates"] = bool(allow_discrete)

    return StableHWDecision(
        guard_mode=guard_mode,
        lambda_hw_base=float(lam_base),
        lambda_hw_effective=float(lam_eff),
        allow_discrete_updates=bool(allow_discrete),
        acc_ref=state.get("acc_ref"),
        acc_used=state.get("acc_used"),
        acc_used_source=str(state.get("acc_used_source", "none")),
        schedule_epoch=schedule_epoch,
        recovery_epochs_left=int(state.get("recovery_epochs_left", 0)),
    )


def stable_hw_after_validation(
    cfg: Any,
    state: Dict[str, Any],
    epoch: int,
    val_metrics: Optional[Dict[str, float]] = None,
    train_metrics: Optional[Dict[str, float]] = None,
) -> StableHWDecision:
    stable = _get(cfg, "stable_hw", None)
    if stable is None or not bool(_get(stable, "enabled", True)):
        return stable_hw_before_epoch(cfg, state)

    controller = _get(stable, "controller", {}) or {}
    guard = _get(stable, "accuracy_guard", {}) or {}
    schedule_cfg = _get(stable, "lambda_hw_schedule", {}) or {}

    enabled_guard = bool(_get(guard, "enabled", True))
    metric_key = str(_get(guard, "metric_key", "val_acc1"))
    epsilon_drop = float(_get(guard, "epsilon_drop", 0.01))
    use_ema = bool(_get(guard, "use_ema", True))
    ema_beta = float(_get(guard, "ema_beta", 0.9))

    acc_used = None
    acc_used_source = "none"
    if val_metrics and (metric_key in val_metrics):
        acc_used = float(val_metrics[metric_key])
        acc_used_source = "val"
    elif state.get("last_val_acc") is not None:
        acc_used = float(state["last_val_acc"])
        acc_used_source = "last_val"
    elif train_metrics and (metric_key in train_metrics) and use_ema:
        acc_used = float(train_metrics[metric_key])
        acc_used_source = "ema_train"

    if use_ema and acc_used is not None:
        if state.get("acc_ema") is None:
            state["acc_ema"] = float(acc_used)
        else:
            state["acc_ema"] = float(ema_beta) * float(state["acc_ema"]) + (1.0 - float(ema_beta)) * float(
                acc_used
            )

    if val_metrics and (metric_key in val_metrics):
        state["last_val_acc"] = float(val_metrics[metric_key])

    state["acc_used"] = acc_used
    state["acc_used_source"] = acc_used_source

    if not bool(state.get("acc_ref_locked", False)):
        warmup = int(_get(schedule_cfg, "warmup_epochs", 0))
        if val_metrics and (metric_key in val_metrics):
            cur = float(val_metrics[metric_key])
            best = state.get("best_warmup_acc")
            if best is None or cur > float(best):
                state["best_warmup_acc"] = float(cur)
        if warmup <= 0 or (epoch + 1) >= warmup:
            if state.get("best_warmup_acc") is not None:
                state["acc_ref"] = float(state["best_warmup_acc"])
                state["acc_ref_locked"] = True

    guard_mode = str(state.get("guard_mode", "HW_OPT"))
    allow_discrete = True
    rec_left = int(state.get("recovery_epochs_left", 0))
    exit_ok = int(state.get("exit_ok_streak", 0))
    margin_exit = float(_get(controller, "margin_exit", 0.0))
    k_exit = int(_get(controller, "k_exit", 1))
    rec_min = int(_get(controller, "recovery_min_epochs", 1))

    acc_ref = state.get("acc_ref")
    violate = False
    if enabled_guard and bool(state.get("acc_ref_locked", False)) and acc_ref is not None and acc_used is not None:
        violate = (float(acc_ref) - float(acc_used)) > float(epsilon_drop)

    if guard_mode == "HW_OPT":
        if violate:
            guard_mode = "RECOVERY"
            rec_left = max(rec_left, rec_min)
            exit_ok = 0
        allow_discrete = True
        state["schedule_epoch"] = int(state.get("schedule_epoch", 0)) + 1
    else:
        allow_discrete = False
        if rec_left > 0:
            rec_left -= 1

        ok = False
        if enabled_guard and bool(state.get("acc_ref_locked", False)) and acc_ref is not None and acc_used is not None:
            ok = float(acc_used) >= (float(acc_ref) - float(epsilon_drop) + float(margin_exit))
        exit_ok = (exit_ok + 1) if ok else 0
        if rec_left <= 0 and exit_ok >= max(1, k_exit):
            guard_mode = "HW_OPT"
            allow_discrete = True
            exit_ok = 0
            state["schedule_epoch"] = int(state.get("schedule_epoch", 0)) + 1

    state["guard_mode"] = guard_mode
    state["allow_discrete_updates"] = bool(allow_discrete)
    state["recovery_epochs_left"] = int(rec_left)
    state["exit_ok_streak"] = int(exit_ok)

    return stable_hw_before_epoch(cfg, state)


def stable_hw_log_fields(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "guard_mode": state.get("guard_mode"),
        "lambda_hw_base": state.get("lambda_hw_base"),
        "lambda_hw_effective": state.get("lambda_hw_effective"),
        "allow_discrete_updates": state.get("allow_discrete_updates"),
        "metric_key": state.get("metric_key"),
        "acc_ref": state.get("acc_ref"),
        "acc_ref_locked": state.get("acc_ref_locked"),
        "acc_used": state.get("acc_used"),
        "acc_used_source": state.get("acc_used_source"),
        "acc_ema": state.get("acc_ema"),
        "schedule_epoch": state.get("schedule_epoch"),
        "recovery_epochs_left": state.get("recovery_epochs_left"),
        "exit_ok_streak": state.get("exit_ok_streak"),
    }
