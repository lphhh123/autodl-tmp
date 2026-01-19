from __future__ import annotations

from typing import Any, Dict, Optional
import json
from pathlib import Path


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Get nested key from dict/AttrDict-like objects."""
    if obj is None:
        return default
    # AttrDict behaves like dict for get()
    if hasattr(obj, "get"):
        try:
            return obj.get(key, default)
        except Exception:
            pass
    # fallback: attribute
    if hasattr(obj, key):
        return getattr(obj, key)
    return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def stable_hw_log_fields(st: Dict[str, Any]) -> Dict[str, Any]:
    """Small flat dict for tensorboard/csv logging."""
    return {
        "stable_hw/lambda_hw_base": _safe_float(st.get("lambda_hw_base"), 0.0),
        "stable_hw/lambda_hw_effective": _safe_float(st.get("lambda_hw_effective"), 0.0),
        "stable_hw/allow_discrete_updates": bool(st.get("allow_discrete_updates", True)),
        "stable_hw/guard_mode": str(st.get("guard_mode", "HW_OPT")),
        "stable_hw/in_recovery": bool(st.get("in_recovery", False)),
        "stable_hw/acc_ref": _safe_float(st.get("acc_ref"), 0.0) if st.get("acc_ref") is not None else None,
        "stable_hw/acc_used_last": _safe_float(st.get("acc_used_last"), 0.0) if st.get("acc_used_last") is not None else None,
        "stable_hw/acc_violation": bool(st.get("acc_violation", False)),
        "stable_hw/violation_epoch": int(st.get("violation_epoch", -1)) if st.get("violation_epoch") is not None else -1,
        "stable_hw/recovery_good_epochs": int(st.get("recovery_good_epochs", 0)),
    }


def stable_hw_schedule(
    epoch: int,
    stable_hw_cfg: Any,
    st: Dict[str, Any],
    hw_lambda_default: Optional[float] = None,
) -> None:
    """
    Set lambda_hw_base for this epoch (warmup->ramp->stabilize).
    Dict-state implementation to match trainer_* usage.
    """
    if st is None:
        return
    st.setdefault("guard_mode", "HW_OPT")
    st.setdefault("in_recovery", False)
    st.setdefault("allow_discrete_updates", True)

    enabled = bool(_cfg_get(stable_hw_cfg, "enabled", False))
    sched = _cfg_get(stable_hw_cfg, "lambda_hw_schedule", {}) or {}
    sched_enabled = bool(_cfg_get(sched, "enabled", enabled))

    if not enabled or not sched_enabled:
        # legacy behavior
        lam = _safe_float(hw_lambda_default, 0.0)
        st["lambda_hw_base"] = lam
        # default effective = base; guard may override
        st.setdefault("lambda_hw_effective", lam)
        st["lambda_hw"] = st["lambda_hw_base"]
        st["lambda_hw_after_guard"] = st.get("lambda_hw_effective", st["lambda_hw_base"])
        return

    warmup_epochs = int(_cfg_get(sched, "warmup_epochs", 5))
    ramp_epochs = int(_cfg_get(sched, "ramp_epochs", 10))
    lambda_hw_max = _safe_float(_cfg_get(sched, "lambda_hw_max", 0.0), 0.0)
    clamp_min = _safe_float(_cfg_get(sched, "clamp_min", 0.0), 0.0)
    clamp_max = _safe_float(_cfg_get(sched, "clamp_max", 1.0), 1.0)

    # freeze schedule during recovery if asked (spec)
    ctrl = _cfg_get(_cfg_get(stable_hw_cfg, "accuracy_guard", {}), "controller", {}) or {}
    freeze_schedule = bool(_cfg_get(ctrl, "freeze_schedule_in_recovery", True))
    if bool(st.get("in_recovery", False)) and freeze_schedule and st.get("lambda_hw_base") is not None:
        lam = _safe_float(st["lambda_hw_base"], 0.0)
    else:
        if epoch < warmup_epochs:
            lam = 0.0
        elif ramp_epochs <= 0:
            lam = lambda_hw_max
        else:
            t = min(1.0, max(0.0, float(epoch - warmup_epochs) / float(ramp_epochs)))
            lam = t * lambda_hw_max

    lam = max(clamp_min, min(clamp_max, lam))
    st["lambda_hw_base"] = lam

    # default: effective==base; accuracy guard may set to 0 in recovery
    if st.get("lambda_hw_effective") is None:
        st["lambda_hw_effective"] = lam
    st["lambda_hw"] = st["lambda_hw_base"]
    st["lambda_hw_after_guard"] = st.get("lambda_hw_effective", st["lambda_hw_base"])


def _load_locked_acc_ref(stable_hw_cfg: Any, st: Dict[str, Any]) -> None:
    """Load baseline acc reference once (LockedAccRef)."""
    locked = _cfg_get(stable_hw_cfg, "locked_acc_ref", {}) or {}
    enabled = bool(_cfg_get(locked, "enabled", True))
    if not enabled:
        return
    if st.get("acc_ref") is not None:
        return

    baseline_stats_path = _cfg_get(locked, "baseline_stats_path", None)
    if baseline_stats_path:
        p = Path(str(baseline_stats_path))
        if p.exists() and p.is_file():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
                # accept a few common keys
                cand = None
                for k in ("val_acc1", "acc1", "best_val_acc1", "best_acc1"):
                    if isinstance(obj, dict) and k in obj:
                        cand = obj[k]
                        break
                if cand is not None:
                    st["acc_ref"] = _safe_float(cand, None)  # type: ignore[arg-type]
                    st["acc_ref_source"] = f"baseline_stats:{p}"
                    return
            except Exception:
                # fall back to warmup best
                st["acc_ref_source"] = f"baseline_stats_parse_error:{p}"

    # else: will be set by warmup_best when val metrics arrive
    st.setdefault("acc_ref_source", "warmup_best_pending")


def _pick_acc_used(
    epoch: int,
    stable_hw_cfg: Any,
    st: Dict[str, Any],
    val_metric_or_none: Optional[float],
    has_val_this_epoch: bool,
    train_acc1_ema: Optional[float],
) -> Optional[float]:
    """Return accuracy metric used for guard decision, per controller.metric."""
    guard = _cfg_get(stable_hw_cfg, "accuracy_guard", {}) or {}
    ctrl = _cfg_get(guard, "controller", {}) or {}
    metric = str(_cfg_get(ctrl, "metric", _cfg_get(guard, "metric", "val_acc1")))

    # prefer actual val metric if present
    if has_val_this_epoch and val_metric_or_none is not None:
        st["val_acc1_last"] = _safe_float(val_metric_or_none, None)  # type: ignore[arg-type]
        if metric == "val_acc1":
            return _safe_float(val_metric_or_none, None)  # type: ignore[arg-type]

    if metric in ("train_acc1_ema", "acc1_ema"):
        if train_acc1_ema is not None:
            return _safe_float(train_acc1_ema, None)  # type: ignore[arg-type]
        # fallback to state
        if st.get("train_acc1_ema") is not None:
            return _safe_float(st.get("train_acc1_ema"), None)  # type: ignore[arg-type]

    # allow fallback to val_acc1_last if has_val missed but we have last
    if metric == "val_acc1" and st.get("val_acc1_last") is not None:
        return _safe_float(st.get("val_acc1_last"), None)  # type: ignore[arg-type]

    # final fallback: if asked, use ema state
    allow_fallback = bool(_cfg_get(stable_hw_cfg, "allow_train_ema_fallback", False))
    if allow_fallback and st.get("train_acc1_ema") is not None:
        return _safe_float(st.get("train_acc1_ema"), None)  # type: ignore[arg-type]

    return None


def _update_train_ema(stable_hw_cfg: Any, st: Dict[str, Any], acc: float) -> None:
    """Maintain train_acc1_ema in state (legacy support; also useful for fallback)."""
    guard = _cfg_get(stable_hw_cfg, "accuracy_guard", {}) or {}
    use_ema = bool(_cfg_get(guard, "use_ema", False))
    beta = _safe_float(_cfg_get(guard, "ema_beta", 0.9), 0.9)
    if not use_ema:
        return
    prev = st.get("train_acc1_ema")
    if prev is None:
        st["train_acc1_ema"] = float(acc)
    else:
        st["train_acc1_ema"] = float(beta) * float(prev) + (1.0 - float(beta)) * float(acc)


def apply_accuracy_guard(
    epoch: int,
    stable_hw_cfg: Any,
    st: Dict[str, Any],
    val_metric_or_none: Optional[float],
    has_val_this_epoch: bool,
    train_acc1_ema: Optional[float] = None,
) -> None:
    """
    v5.4 Acc-First Hard Gating:
      - if acc drops below (acc_ref - epsilon_drop) => enter RECOVERY:
          lambda_hw_effective = 0
          allow_discrete_updates = False (freeze mapping/layout/track_live/refine)
          (optionally) freeze schedule advancement
      - exit recovery after >=recovery_min_epochs and K-of-N condition satisfied with margin_exit.
    """
    if st is None:
        return
    enabled = bool(_cfg_get(stable_hw_cfg, "enabled", False))
    guard = _cfg_get(stable_hw_cfg, "accuracy_guard", {}) or {}
    guard_enabled = bool(_cfg_get(guard, "enabled", enabled))
    if not enabled or not guard_enabled:
        # passthrough
        if st.get("lambda_hw_effective") is None and st.get("lambda_hw_base") is not None:
            st["lambda_hw_effective"] = st["lambda_hw_base"]
        st["lambda_hw_after_guard"] = st.get("lambda_hw_effective", st.get("lambda_hw_base", 0.0))
        return

    ctrl = _cfg_get(guard, "controller", {}) or {}
    epsilon_drop = _safe_float(_cfg_get(ctrl, "epsilon_drop", _cfg_get(guard, "epsilon_drop", 0.01)), 0.01)
    epsilon_type = str(_cfg_get(ctrl, "epsilon_drop_type", "abs"))
    recovery_min_epochs = int(_cfg_get(ctrl, "recovery_min_epochs", 1))
    cut_hw_loss = bool(_cfg_get(ctrl, "cut_hw_loss_on_violate", True))
    # spec key: freeze_discrete_updates; keep legacy key freeze_discrete_on_violate
    freeze_discrete = bool(
        _cfg_get(
            ctrl,
            "freeze_discrete_updates",
            _cfg_get(ctrl, "freeze_discrete_on_violate", True),
        )
    )

    k_exit = int(_cfg_get(ctrl, "k_exit", 2))
    margin_exit = _safe_float(_cfg_get(ctrl, "margin_exit", 0.0), 0.0)

    # ensure acc_ref is loaded/locked
    _load_locked_acc_ref(stable_hw_cfg, st)

    # if baseline file didn't load, use warmup_best once val metric available
    freeze_epoch = int(_cfg_get(_cfg_get(stable_hw_cfg, "locked_acc_ref", {}) or {}, "freeze_epoch", 0))
    acc_used = _pick_acc_used(epoch, stable_hw_cfg, st, val_metric_or_none, has_val_this_epoch, train_acc1_ema)
    if acc_used is not None:
        st["acc_used_last"] = float(acc_used)
        _update_train_ema(stable_hw_cfg, st, float(acc_used))

    if st.get("acc_ref") is None:
        # set warmup best once we have val metric (or chosen metric)
        if acc_used is not None and epoch >= freeze_epoch:
            st["acc_ref"] = float(acc_used)
            st["acc_ref_source"] = "warmup_best"
        st["acc_violation"] = False
        # no ref -> don't gate; keep effective = base
        st["lambda_hw_effective"] = _safe_float(st.get("lambda_hw_base", 0.0), 0.0)
        st["allow_discrete_updates"] = True
        st["guard_mode"] = "HW_OPT"
        st["in_recovery"] = False
        st["lambda_hw_after_guard"] = st["lambda_hw_effective"]
        return

    acc_ref = float(st["acc_ref"])
    # threshold = acc_ref - eps (abs) or acc_ref*(1-eps) (rel)
    if epsilon_type.lower() == "rel":
        thr = acc_ref * (1.0 - float(epsilon_drop))
    else:
        thr = acc_ref - float(epsilon_drop)

    violate = False
    if acc_used is not None:
        violate = float(acc_used) < float(thr)

    st["acc_violation"] = bool(violate)
    st.setdefault("recovery_good_epochs", 0)
    st.setdefault("recovery_start_epoch", None)

    # --- transitions ---
    if violate:
        st["in_recovery"] = True
        st["guard_mode"] = "RECOVERY"
        st["violation_epoch"] = int(epoch)
        if st.get("recovery_start_epoch") is None:
            st["recovery_start_epoch"] = int(epoch)
        st["recovery_good_epochs"] = 0

        # hard gating
        if cut_hw_loss:
            st["lambda_hw_effective"] = 0.0
        else:
            st["lambda_hw_effective"] = _safe_float(st.get("lambda_hw_base", 0.0), 0.0)

        if freeze_discrete:
            st["allow_discrete_updates"] = False
        else:
            st["allow_discrete_updates"] = True

        st["lambda_hw_after_guard"] = st["lambda_hw_effective"]
        return

    # not violating
    if not bool(st.get("in_recovery", False)):
        st["guard_mode"] = "HW_OPT"
        st["allow_discrete_updates"] = True
        st["lambda_hw_effective"] = _safe_float(st.get("lambda_hw_base", 0.0), 0.0)
        st["lambda_hw_after_guard"] = st["lambda_hw_effective"]
        return

    # currently in recovery: check exit criteria
    # must stay for min epochs
    start = st.get("recovery_start_epoch")
    start = int(start) if start is not None else int(epoch)
    enough_time = (int(epoch) - int(start) + 1) >= int(recovery_min_epochs)

    # K-of-N with margin: acc_used >= thr + margin_exit counts as good epoch
    good = False
    if acc_used is not None:
        good = float(acc_used) >= float(thr + float(margin_exit))
    if good:
        st["recovery_good_epochs"] = int(st.get("recovery_good_epochs", 0)) + 1

    if enough_time and int(st.get("recovery_good_epochs", 0)) >= int(k_exit):
        # exit recovery
        st["in_recovery"] = False
        st["guard_mode"] = "HW_OPT"
        st["allow_discrete_updates"] = True
        st["lambda_hw_effective"] = _safe_float(st.get("lambda_hw_base", 0.0), 0.0)
        st["lambda_hw_after_guard"] = st["lambda_hw_effective"]
        # reset counters
        st["recovery_start_epoch"] = None
        st["recovery_good_epochs"] = 0
        return

    # stay in recovery
    st["guard_mode"] = "RECOVERY"
    st["allow_discrete_updates"] = False if freeze_discrete else True
    st["lambda_hw_effective"] = 0.0 if cut_hw_loss else _safe_float(st.get("lambda_hw_base", 0.0), 0.0)
    st["lambda_hw_after_guard"] = st["lambda_hw_effective"]
