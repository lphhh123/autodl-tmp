from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _get(cfg: Any, k: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(k, default)
    return getattr(cfg, k, default)


def _as_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def _as_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)


def _as_bool(x, default=False) -> bool:
    try:
        return bool(x)
    except Exception:
        return bool(default)


@dataclass
class GuardDecision:
    guard_mode: str  # "HW_OPT" | "RECOVERY"
    lambda_hw_base: float  # schedule(epoch)
    lambda_hw_effective: float  # 0 in RECOVERY
    allow_discrete_updates: bool  # False in RECOVERY
    acc_used: float
    acc_used_source: str  # "val" | "val_last" | "train_ema"
    acc_ref: float
    acc_drop: float
    triggered: bool


def stable_hw_schedule(epoch: int, stable_hw_cfg: Any, state: Dict[str, Any]) -> float:
    """
    v5.1 canonical:
      lambda_hw_base := schedule(epoch)
      NOTE: schedule only returns base; do NOT apply gating here.
    """
    sched = _get(stable_hw_cfg, "lambda_hw_schedule", None) or {}
    warmup = _as_int(_get(sched, "warmup_epochs", 0), 0)
    ramp = _as_int(_get(sched, "ramp_epochs", 0), 0)
    lam_min = _as_float(_get(sched, "lambda_hw_min", 0.0), 0.0)
    lam_max = _as_float(_get(sched, "lambda_hw_max", 0.0), 0.0)
    clamp_min = _get(sched, "clamp_min", None)
    clamp_max = _get(sched, "clamp_max", None)

    if lam_max <= 0:
        phase = "off"
        lam = 0.0
    elif epoch < warmup:
        phase = "warmup"
        lam = lam_min
    elif ramp <= 0:
        phase = "stabilize"
        lam = lam_max
    else:
        t = min(1.0, max(0.0, (epoch - warmup) / float(ramp)))
        phase = "ramp"
        lam = lam_min + t * (lam_max - lam_min)

    if clamp_min is not None:
        lam = max(float(clamp_min), float(lam))
    if clamp_max is not None:
        lam = min(float(clamp_max), float(lam))

    state["lambda_hw_base"] = float(lam)
    if clamp_min is not None:
        state["lambda_hw_clamp_min"] = float(clamp_min)
    if clamp_max is not None:
        state["lambda_hw_clamp_max"] = float(clamp_max)
    state["schedule_phase"] = str(phase)
    return float(lam)


def _locked_acc_ref_init(stable_hw_cfg: Any, state: Dict[str, Any]) -> None:
    """
    v5.1 LockedAccRef:
      - Prefer baseline stats file if provided.
      - Else lock from warmup-best (val) once after warmup window.
    """
    lock = _get(stable_hw_cfg, "locked_acc_ref", None) or {}
    guard_cfg = _get(stable_hw_cfg, "accuracy_guard", None) or {}
    # canonical keys
    baseline_path = _get(lock, "baseline_stats_path", None) or _get(guard_cfg, "baseline_stats_path", None)
    freeze_epoch = _as_int(_get(lock, "freeze_epoch", 0), 0)
    prefer_dense = _as_bool(_get(lock, "prefer_dense_baseline", True), True)

    # already locked
    if _as_bool(state.get("acc_ref_locked", False), False) and state.get("acc_ref") is not None:
        return

    # try from baseline_stats_path
    if baseline_path:
        p = Path(str(baseline_path))
        if p.exists() and p.is_file():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                # accept multiple key names
                cand = None
                for k in ["val_acc1_best", "acc1_best", "best_val_acc1", "best_acc1", "acc_ref"]:
                    if k in obj:
                        cand = obj[k]
                        break
                if cand is not None:
                    state["acc_ref"] = float(cand)
                    state["acc_ref_source"] = "baseline_file"
                    state["acc_ref_locked"] = True
                    state["acc_ref_freeze_epoch"] = freeze_epoch
                    state["acc_ref_prefer_dense"] = bool(prefer_dense)
                    return
            except Exception:
                pass

    # fallback: lock later from warmup-best val
    state.setdefault("acc_ref", None)
    state.setdefault("acc_ref_source", "unset")
    state.setdefault("acc_ref_locked", False)
    state.setdefault("acc_ref_freeze_epoch", freeze_epoch)


def _update_warmup_best(epoch: int, stable_hw_cfg: Any, state: Dict[str, Any], val_acc1: Optional[float]) -> None:
    """
    Track warmup best; lock when epoch >= freeze_epoch if no baseline file.
    """
    lock = _get(stable_hw_cfg, "locked_acc_ref", None) or {}
    freeze_epoch = _as_int(_get(lock, "freeze_epoch", 0), 0)

    if val_acc1 is None:
        return

    best = state.get("warmup_best_val_acc1", None)
    if best is None or float(val_acc1) > float(best):
        state["warmup_best_val_acc1"] = float(val_acc1)

    if not _as_bool(state.get("acc_ref_locked", False), False):
        if epoch >= freeze_epoch:
            # lock from warmup-best
            wbest = state.get("warmup_best_val_acc1", None)
            if wbest is not None:
                state["acc_ref"] = float(wbest)
                state["acc_ref_source"] = "warmup_best"
                state["acc_ref_locked"] = True
                state["acc_ref_freeze_epoch"] = freeze_epoch


def apply_accuracy_guard(
    epoch: int,
    stable_hw_cfg,
    state: Dict[str, Any],
    val_acc1: float | None,
    has_val_this_epoch: bool,
    train_acc1_ema: float | None = None,
) -> None:
    """
    v5.4 Acc-First Hard Gating + LockedAccRef + NoDrift:
      - canonical config: stable_hw.accuracy_guard.controller.*
      - legacy compat: stable_hw.controller.* and root fields
    """

    def _get(obj, key: str, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _get_nested(obj, path: str, default=None):
        cur = obj
        for k in path.split("."):
            if cur is None:
                return default
            if isinstance(cur, dict):
                cur = cur.get(k, None)
            else:
                cur = getattr(cur, k, None)
        return default if cur is None else cur

    guard_cfg = _get_nested(stable_hw_cfg, "accuracy_guard", {}) or {}
    # canonical controller (SPEC)
    ctl_cfg = _get_nested(stable_hw_cfg, "accuracy_guard.controller", None)
    # legacy controller
    legacy_ctl = _get_nested(stable_hw_cfg, "controller", None)
    if isinstance(ctl_cfg, dict) and ctl_cfg:
        controller = ctl_cfg
    else:
        controller = legacy_ctl if isinstance(legacy_ctl, dict) else {}

    # metric selection (canonical priority)
    metric_key = str(controller.get("metric", _get(guard_cfg, "metric_key", "val_acc1")))
    eps_drop = float(controller.get("epsilon_drop", _get(guard_cfg, "epsilon_drop", 0.01)))

    # lock acc_ref config
    lock_cfg = _get_nested(stable_hw_cfg, "locked_acc_ref", {}) or {}
    freeze_epoch = int(lock_cfg.get("freeze_epoch", 0))
    prefer_dense = bool(lock_cfg.get("prefer_dense_baseline", True))
    baseline_stats_path = str(lock_cfg.get("baseline_stats_path", _get(guard_cfg, "baseline_stats_path", "") or ""))

    # schedule + controller knobs
    cut_hw = bool(controller.get("cut_hw_loss_on_violate", _get(stable_hw_cfg, "cut_hw_loss_on_violate", True)))
    freeze_discrete = bool(controller.get("freeze_discrete_updates", _get(stable_hw_cfg, "freeze_discrete_updates", True)))
    recovery_min_epochs = int(
        controller.get("recovery_min_epochs", _get_nested(stable_hw_cfg, "controller.recovery_min_epochs", 1) or 1)
    )
    freeze_schedule_in_recovery = bool(
        controller.get(
            "freeze_schedule_in_recovery", _get_nested(stable_hw_cfg, "controller.freeze_schedule_in_recovery", True)
        )
    )
    margin_exit = float(controller.get("margin_exit", _get_nested(stable_hw_cfg, "controller.margin_exit", 0.0)))
    k_exit = int(controller.get("k_exit", _get_nested(stable_hw_cfg, "controller.k_exit", 1) or 1))

    # track mismatched metric_key for reproducibility audits
    legacy_metric_key = str(_get(guard_cfg, "metric_key", "val_acc1"))
    state["accuracy_guard_metric_key"] = metric_key
    state["accuracy_guard_metric_key_legacy"] = legacy_metric_key
    state["accuracy_guard_metric_mismatch"] = bool(metric_key != legacy_metric_key)

    # choose acc_used
    acc_used = None
    if metric_key == "val_acc1":
        acc_used = float(val_acc1) if val_acc1 is not None else None
    elif metric_key == "train_acc1_ema":
        acc_used = float(train_acc1_ema) if train_acc1_ema is not None else None
    else:
        # unknown -> fall back to val
        acc_used = float(val_acc1) if val_acc1 is not None else None

    # initialize locked acc ref state
    _locked_acc_ref_init(stable_hw_cfg, state)

    # update warmup-best before locking (NoDrift: lock once, never change after)
    if not bool(state.get("acc_ref_locked", False)):
        if acc_used is not None:
            # warmup window: [0, freeze_epoch)
            if epoch < max(0, freeze_epoch):
                _update_warmup_best(state, acc_used)
            # if prefer_dense_baseline and baseline exists: lock immediately
            if prefer_dense and baseline_stats_path:
                # _locked_acc_ref_init already attempted to load; use it if present
                if state.get("acc_ref") is not None:
                    state["acc_ref_locked"] = True
            # lock at freeze_epoch if not already locked
            if (epoch >= max(0, freeze_epoch)) and (not bool(state.get("acc_ref_locked", False))):
                # lock to warmup-best if available else current
                best = state.get("acc_ref_warmup_best", None)
                state["acc_ref"] = float(best) if best is not None else float(acc_used)
                state["acc_ref_locked"] = True

    acc_ref = state.get("acc_ref", None)
    if acc_ref is None and acc_used is not None:
        # ultimate fallback: lock immediately to current
        state["acc_ref"] = float(acc_used)
        state["acc_ref_locked"] = True
        acc_ref = state["acc_ref"]

    # schedule base lambda (must exist even if guard can't run yet)
    lambda_base = float(state.get("lambda_hw_base", state.get("lambda_hw", 0.0)))
    if "lambda_hw_effective" not in state:
        state["lambda_hw_effective"] = lambda_base

    # guard state machine
    mode_prev = str(state.get("guard_mode", "HW_OPT"))
    in_recovery = mode_prev == "RECOVERY"

    # if no accuracy signal, do nothing but keep schedule lambda
    if acc_used is None or acc_ref is None:
        state["guard_mode"] = "RECOVERY" if in_recovery else "HW_OPT"
        state["lambda_hw_effective"] = lambda_base if not in_recovery else float(
            state.get("lambda_hw_effective", lambda_base)
        )
        return

    # determine violation (Acc-First Hard Gating)
    threshold = float(acc_ref) - float(eps_drop)
    violated = bool(float(acc_used) < threshold)

    # recovery exit counter
    if "recovery_ok_streak" not in state:
        state["recovery_ok_streak"] = 0

    if violated:
        state["guard_mode"] = "RECOVERY"
        state["violation_epoch"] = int(epoch)
        state["recovery_epochs_left"] = int(max(1, recovery_min_epochs))
        state["recovery_ok_streak"] = 0

        if cut_hw:
            state["lambda_hw_effective"] = 0.0
        else:
            # if not cutting, at least don't "reward" with negative; keep base
            state["lambda_hw_effective"] = float(lambda_base)

        state["freeze_discrete"] = bool(freeze_discrete)
        state["freeze_schedule"] = bool(freeze_schedule_in_recovery)

        # record debug fields
        state["acc_used"] = float(acc_used)
        state["acc_ref"] = float(acc_ref)
        state["acc_threshold"] = float(threshold)
        return

    # not violated
    if not in_recovery:
        # normal hw-opt
        state["guard_mode"] = "HW_OPT"
        state["lambda_hw_effective"] = float(lambda_base)
        state["freeze_discrete"] = False
        state["freeze_schedule"] = False
        state["acc_used"] = float(acc_used)
        state["acc_ref"] = float(acc_ref)
        state["acc_threshold"] = float(threshold)
        return

    # in recovery but now ok: require min epochs + k_exit streak
    ok_exit = bool(float(acc_used) >= (float(threshold) + float(margin_exit)))
    if ok_exit:
        state["recovery_ok_streak"] = int(state.get("recovery_ok_streak", 0)) + 1
    else:
        state["recovery_ok_streak"] = 0

    # count down recovery epochs
    left = int(state.get("recovery_epochs_left", max(1, recovery_min_epochs)))
    left = max(0, left - 1)
    state["recovery_epochs_left"] = int(left)

    if left == 0 and int(state.get("recovery_ok_streak", 0)) >= max(1, k_exit):
        # exit recovery
        state["guard_mode"] = "HW_OPT"
        state["lambda_hw_effective"] = float(lambda_base)
        state["freeze_discrete"] = False
        state["freeze_schedule"] = False
    else:
        # stay recovery
        if cut_hw:
            state["lambda_hw_effective"] = 0.0
        state["freeze_discrete"] = bool(freeze_discrete)
        state["freeze_schedule"] = bool(freeze_schedule_in_recovery)

    state["acc_used"] = float(acc_used)
    state["acc_ref"] = float(acc_ref)
    state["acc_threshold"] = float(threshold)


# ---- legacy helpers used elsewhere (keep minimal, no drift) ----

def init_hw_refs_from_baseline(state: Dict[str, Any], stable_hw_cfg: Any, baseline_path: str) -> Dict[str, Any]:
    """
    Keep existing behavior, but never overwrite acc_ref once locked.
    """
    state.setdefault("ref_path", baseline_path)
    state.setdefault("refs_inited", True)
    state.setdefault("ref_source", "baseline_path")
    return state


def update_hw_refs_from_stats(stable_hw_cfg: Any, stable_hw_state: Dict[str, Any], epoch_hw_stats: Dict[str, Any]) -> None:
    """
    Update ref_T/ref_E/ref_M/ref_C from epoch stats.
    Accept both legacy keys and v5.4 canonical keys to avoid drift.
    Canonical (v5.4): latency_ms, energy_mj, mem_mb, comm_ms
    Legacy compat    : energy_j, comm_mb
    """
    if stable_hw_cfg is None:
        return
    controller_cfg = getattr(stable_hw_cfg, "controller", None)
    if controller_cfg is not None and not bool(getattr(controller_cfg, "update_hw_refs", True)):
        return

    eps_ratio = float(getattr(stable_hw_cfg, "eps_ratio", 1e-9))

    # prefer canonical
    lat = epoch_hw_stats.get("latency_ms", None)
    en_mj = epoch_hw_stats.get("energy_mj", None)
    mem = epoch_hw_stats.get("mem_mb", None)
    comm = epoch_hw_stats.get("comm_ms", None)

    # legacy fallbacks
    if en_mj is None and "energy_j" in epoch_hw_stats:
        # if someone logs joules, convert to mJ for internal consistency
        try:
            en_mj = float(epoch_hw_stats["energy_j"]) * 1000.0
        except Exception:
            en_mj = None
    if comm is None and "comm_mb" in epoch_hw_stats:
        # legacy comm_mb does not equal comm_ms; only use if user explicitly logs ms elsewhere
        comm = None

    def _set_pos(key: str, v) -> None:
        if v is None:
            return
        try:
            fv = float(v)
        except Exception:
            return
        stable_hw_state[key] = max(eps_ratio, fv)

    _set_pos("ref_T", lat)
    _set_pos("ref_E", en_mj)
    _set_pos("ref_M", mem)
    _set_pos("ref_C", comm)


def stable_hw_log_fields(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "guard_mode": state.get("guard_mode"),
        "lambda_hw_base": state.get("lambda_hw_base"),
        "lambda_hw_effective": state.get("lambda_hw_effective"),
        "allow_discrete_updates": state.get("allow_discrete_updates"),
        "discrete_frozen_init_mapping": state.get("discrete_frozen_init_mapping"),
        "acc_ref": state.get("acc_ref"),
        "acc_ref_locked": state.get("acc_ref_locked"),
        "acc_used": state.get("acc_used"),
        "acc_used_source": state.get("acc_used_source"),
        "acc_drop": state.get("acc_drop"),
        "schedule_phase": state.get("schedule_phase"),
        "violate_streak": state.get("violate_streak"),
    }
