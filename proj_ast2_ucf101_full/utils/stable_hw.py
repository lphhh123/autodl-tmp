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


def stable_hw_log_fields(st: StableHWState) -> Dict[str, Any]:
    return {
        "stable_hw_enabled": bool(st.enabled),
        "lambda_hw_base": float(st.lambda_hw_base),
        "lambda_hw_effective": float(st.lambda_hw_effective),
        "allow_discrete_updates": bool(st.allow_discrete_updates),
        "acc_ref": None if st.acc_ref is None else float(st.acc_ref),
        "acc_ref_source": str(st.acc_ref_source),
        "acc_used_last": None if st.acc_used_last is None else float(st.acc_used_last),
        "in_recovery": bool(st.in_recovery),
        "recovery_start_epoch": int(st.recovery_start_epoch),
        "recovery_good_epochs": int(st.recovery_good_epochs),
        "violation_epoch": int(st.violation_epoch),
        "accuracy_guard_metric_mismatch": bool(st.metric_mismatch),
        "acc_violation": bool(st.acc_violation),
        "guard_triggered": bool(st.guard_triggered),
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


def stable_hw_schedule(cfg: Any, epoch: int, st: StableHWState) -> StableHWState:
    """
    v5.4 NoDoubleScale:
      - schedule produces lambda_hw_base only
      - guard chooses lambda_hw_effective = 0 or base (no extra scaling)
    """
    stable_hw_cfg = getattr(cfg, "stable_hw", None)
    enabled = bool(_get_nested(stable_hw_cfg, "enabled", False))
    st.enabled = enabled
    if not enabled:
        st.lambda_hw_base = float(_get_nested(cfg, "hw.lambda_hw", 0.0) or 0.0)
        st.lambda_hw_effective = st.lambda_hw_base
        st.allow_discrete_updates = True
        return st

    sched = _get_nested(stable_hw_cfg, "lambda_hw_schedule", {}) or {}
    sched_enabled = bool(_cfg_get(sched, "enabled", True))
    if not sched_enabled:
        st.lambda_hw_base = float(_get_nested(cfg, "hw.lambda_hw", 0.0) or 0.0)
        return st

    warmup = int(_cfg_get(sched, "warmup_epochs", 5))
    ramp = int(_cfg_get(sched, "ramp_epochs", 10))
    lam_max = float(_cfg_get(sched, "lambda_hw_max", 0.0) or 0.0)

    # freeze schedule if in recovery and controller requests it
    ctrl = _get_nested(stable_hw_cfg, "accuracy_guard.controller", {}) or _get_nested(stable_hw_cfg, "controller", {}) or {}
    freeze_in_rec = bool(_cfg_get(ctrl, "freeze_schedule_in_recovery", True))
    if freeze_in_rec and st.in_recovery:
        return st

    if epoch < warmup:
        lam = 0.0
    else:
        t = epoch - warmup
        if ramp <= 0:
            lam = lam_max
        else:
            lam = lam_max * min(1.0, max(0.0, float(t) / float(ramp)))
    st.lambda_hw_base = float(max(0.0, lam))
    return st


def apply_accuracy_guard(cfg: Any, epoch: int, metrics: Dict[str, Any], st: StableHWState) -> StableHWState:
    """
    v5.4 Acc-First Hard Gating:
      - if acc drops beyond epsilon_drop vs locked acc_ref:
          lambda_hw_effective = 0
          allow_discrete_updates = False (freeze rho/segments)
      - NoDrift: acc_ref is locked once initialized (baseline_file preferred)
      - NoDoubleScale: do NOT rescale lambda_hw; only choose 0 or base
    """
    stable_hw_cfg = getattr(cfg, "stable_hw", None)
    enabled = bool(_get_nested(stable_hw_cfg, "enabled", False))
    st.enabled = enabled

    # defaults if disabled
    if not enabled:
        st.lambda_hw_effective = float(_get_nested(cfg, "hw.lambda_hw", 0.0) or 0.0)
        st.allow_discrete_updates = True
        st.in_recovery = False
        st.acc_violation = False
        st.guard_triggered = False
        return st

    guard = _get_nested(stable_hw_cfg, "accuracy_guard", {}) or {}
    guard_enabled = bool(_cfg_get(guard, "enabled", True))
    ctrl = _get_nested(guard, "controller", {}) or _get_nested(stable_hw_cfg, "controller", {}) or {}

    mode = str(_cfg_get(ctrl, "mode", "hard"))
    metric_ctrl = str(_cfg_get(ctrl, "metric", "val_acc1"))
    metric_top = guard.get("metric", None)
    if metric_top is not None and str(metric_top) != metric_ctrl:
        st.metric_mismatch = True

    # pick metric value
    acc_used = None
    if metric_ctrl in metrics:
        acc_used = metrics.get(metric_ctrl)
    elif "val_acc1" in metrics:
        acc_used = metrics.get("val_acc1")
    elif "acc1" in metrics:
        acc_used = metrics.get("acc1")
    try:
        acc_used = float(acc_used) if acc_used is not None else None
    except Exception:
        acc_used = None
    st.acc_used_last = acc_used

    # initialize locked acc_ref
    locked = _get_nested(stable_hw_cfg, "locked_acc_ref", {}) or {}
    baseline_stats_path = str(_cfg_get(locked, "baseline_stats_path", "") or "")
    freeze_epoch = int(_cfg_get(locked, "freeze_epoch", 0) or 0)

    if st.acc_ref is None and acc_used is not None and epoch >= freeze_epoch:
        base_acc = _load_baseline_acc(baseline_stats_path)
        if base_acc is not None:
            st.acc_ref = float(base_acc)
            st.acc_ref_source = "baseline_file"
        else:
            # v5.4 fallback: lock to first observed val metric (NO DRIFT)
            st.acc_ref = float(acc_used)
            st.acc_ref_source = "first_val"

    # if guard disabled: just pass through base
    if not guard_enabled or st.acc_ref is None or acc_used is None:
        st.lambda_hw_effective = float(st.lambda_hw_base)
        st.allow_discrete_updates = True
        st.acc_violation = False
        st.guard_triggered = False
        return st

    eps_drop = float(_cfg_get(ctrl, "epsilon_drop", 0.01))
    eps_type = str(_cfg_get(ctrl, "epsilon_drop_type", "abs"))
    if eps_type != "abs":
        # only abs is supported in v5.4; fallback to abs
        eps_type = "abs"

    # violation check (abs drop)
    drop = float(st.acc_ref - acc_used)
    violated = drop > eps_drop
    st.acc_violation = bool(violated)

    cut_hw = bool(_cfg_get(ctrl, "cut_hw_loss_on_violate", True))
    freeze_rho = bool(_cfg_get(ctrl, "freeze_rho_on_violate", True))

    # recovery policy
    rec_min = int(_cfg_get(ctrl, "recovery_min_epochs", 3))
    k_exit = int(_cfg_get(ctrl, "k_exit", 2))
    margin_exit = float(_cfg_get(ctrl, "margin_exit", 0.002))

    # enter recovery
    if violated and (mode == "hard"):
        st.in_recovery = True
        st.violation_epoch = int(epoch)
        if st.recovery_start_epoch < 0:
            st.recovery_start_epoch = int(epoch)
        st.recovery_good_epochs = 0

        st.lambda_hw_effective = 0.0 if cut_hw else float(st.lambda_hw_base)
        st.allow_discrete_updates = False if freeze_rho else True
        st.guard_triggered = True
        return st

    # if already in recovery, test exit conditions
    if st.in_recovery:
        # good if we are within (eps_drop - margin_exit)
        good = (st.acc_ref - acc_used) <= max(0.0, eps_drop - margin_exit)
        if good:
            st.recovery_good_epochs += 1
        else:
            st.recovery_good_epochs = 0

        in_rec_epochs = epoch - st.recovery_start_epoch + 1 if st.recovery_start_epoch >= 0 else 0
        if in_rec_epochs >= rec_min and st.recovery_good_epochs >= k_exit:
            # exit
            st.in_recovery = False
            st.recovery_start_epoch = -1
            st.recovery_good_epochs = 0
            st.guard_triggered = False
            st.lambda_hw_effective = float(st.lambda_hw_base)
            st.allow_discrete_updates = True
            return st

        # still recovering
        st.lambda_hw_effective = 0.0 if cut_hw else float(st.lambda_hw_base)
        st.allow_discrete_updates = False if freeze_rho else True
        st.guard_triggered = True
        return st

    # normal (not violated, not in recovery)
    st.lambda_hw_effective = float(st.lambda_hw_base)
    st.allow_discrete_updates = True
    st.guard_triggered = False
    return st
