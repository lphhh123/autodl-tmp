from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
from pathlib import Path


@dataclass
class StableHWDecision:
    guard_mode: str
    lambda_hw_base: float
    lambda_hw_effective: float
    allow_discrete_updates: bool
    reason: Dict[str, Any]


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
        phase = str(st.get("schedule_phase", "stabilize"))
    else:
        if epoch < warmup_epochs:
            phase = "warmup"
            lam = 0.0
        elif epoch < (warmup_epochs + ramp_epochs):
            phase = "ramp"
            if ramp_epochs <= 0:
                lam = lambda_hw_max
            else:
                t = min(1.0, max(0.0, float(epoch - warmup_epochs) / float(ramp_epochs)))
                lam = t * lambda_hw_max
        else:
            phase = "stabilize"
            lam = lambda_hw_max

    # ---- v5.4 clamp + schedule_phase (SPEC_C §12B.3) ----
    clamp_min = float(_cfg_get(sched, "clamp_min", 0.0) or 0.0)
    clamp_max = float(_cfg_get(sched, "clamp_max", lambda_hw_max) or lambda_hw_max)
    lam = float(max(clamp_min, min(clamp_max, float(lam))))
    st["lambda_hw_base"] = float(lam)
    st["schedule_phase"] = str(phase)

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
    stable_hw_state: Dict[str, Any],
    val_acc1: Optional[float],
    has_val_this_epoch: bool = True,
    train_acc1_ema: Optional[float] = None,
) -> None:
    """
    v5.4 semantics (SPEC_C §12B.4):
      - acc_used: prefer val_acc1 (explicit downgrade only)
      - acc_ref: ONLY from dense_baseline OR warmup_best(val_acc1), then frozen forever
      - hard gating: if acc_used < acc_ref - eps_drop => lambda_hw_effective=0 + freeze discrete
      - anti-starvation: restart window if stuck in RECOVERY
    """
    st = stable_hw_state
    val_metric_or_none = val_acc1
    if st is None:
        return
    guard_cfg = _cfg_get(stable_hw_cfg, "accuracy_guard", {}) or {}
    lock_cfg = _cfg_get(stable_hw_cfg, "locked_acc_ref", {}) or {}
    ctrl_cfg = _cfg_get(stable_hw_cfg, "controller", {}) or {}

    eps_drop = float(_cfg_get(guard_cfg, "epsilon_drop", 0.01) or 0.01)
    prefer_val = bool(_cfg_get(guard_cfg, "prefer_val_acc1", True))
    allow_train_fallback = bool(_cfg_get(guard_cfg, "allow_train_ema_fallback", False))
    delta_below = float(_cfg_get(guard_cfg, "delta_below_thr", 0.005) or 0.005)
    st["epsilon_drop"] = eps_drop

    warmup_epochs = int(
        _cfg_get(_cfg_get(stable_hw_cfg, "lambda_hw_schedule", {}) or {}, "warmup_epochs", 0) or 0
    )

    st.setdefault("acc_ref_locked", False)
    st.setdefault("acc_ref_source", None)
    st.setdefault("acc_ref", None)
    st.setdefault("acc_ref_warmup_best_val", None)
    st.setdefault("val_acc1_last", None)

    if (not st["acc_ref_locked"]) and bool(_cfg_get(lock_cfg, "prefer_dense_baseline", True)):
        baseline_path = _cfg_get(lock_cfg, "baseline_stats_path", None) or _cfg_get(
            guard_cfg, "baseline_stats_path", None
        )
        st["baseline_stats_path"] = baseline_path
        if baseline_path:
            try:
                with open(str(baseline_path), "r", encoding="utf-8") as f:
                    js = json.load(f)
                v = js.get("val_acc1_best", None)
                if v is None and isinstance(js.get("metrics", None), dict):
                    v = js["metrics"].get("val_acc1_best", None)
                if v is None:
                    v = js.get("best_acc1", None)
                if v is not None:
                    st["acc_ref"] = float(v)
                    st["acc_ref_locked"] = True
                    st["acc_ref_source"] = "dense_baseline"
            except Exception:
                pass

    if has_val_this_epoch and (val_metric_or_none is not None):
        st["val_acc1_last"] = float(val_metric_or_none)

    acc_used = None
    acc_used_source = None
    if prefer_val and has_val_this_epoch and (val_metric_or_none is not None):
        acc_used = float(val_metric_or_none)
        acc_used_source = "val_acc1"
    elif prefer_val and (st.get("val_acc1_last") is not None):
        acc_used = float(st["val_acc1_last"])
        acc_used_source = "val_acc1_last"
    elif allow_train_fallback and (train_acc1_ema is not None):
        acc_used = float(train_acc1_ema)
        acc_used_source = "train_acc1_ema"
    else:
        acc_used = None
        acc_used_source = None

    st["acc_used"] = acc_used
    st["acc_used_source"] = acc_used_source

    if (not st["acc_ref_locked"]) and (epoch < warmup_epochs):
        if acc_used_source in ("val_acc1", "val_acc1_last") and (acc_used is not None):
            prev = st.get("acc_ref_warmup_best_val", None)
            st["acc_ref_warmup_best_val"] = float(acc_used) if prev is None else max(float(prev), float(acc_used))

    if (not st["acc_ref_locked"]) and (epoch >= warmup_epochs):
        wb = st.get("acc_ref_warmup_best_val", None)
        if wb is not None:
            st["acc_ref"] = float(wb)
            st["acc_ref_locked"] = True
            st["acc_ref_source"] = "warmup_best"

    acc_ref = st.get("acc_ref", None)
    acc_ref_f = float(acc_ref) if acc_ref is not None else None

    st.setdefault("recovery_no_improve_epochs", 0)
    st.setdefault("restart_until_epoch", -1)
    st.setdefault("last_restart_epoch", -999999)
    st.setdefault("request_lr_restart", False)

    best_seen = st.get("val_acc1_best_seen", None)
    if best_seen is None and st.get("val_acc1_last") is not None:
        best_seen = float(st["val_acc1_last"])
    st.setdefault("val_acc1_best_seen", best_seen)

    in_restart = epoch <= int(st.get("restart_until_epoch", -1))

    violated = False
    if (acc_ref_f is not None) and (acc_used is not None):
        violated = (acc_ref_f - float(acc_used)) > eps_drop

    far_below = False
    if (acc_ref_f is not None) and (acc_used is not None):
        far_below = float(acc_used) < (acc_ref_f - eps_drop - delta_below)

    if (acc_ref_f is not None) and violated and (has_val_this_epoch and val_metric_or_none is not None):
        prev_best = st.get("_best_seen_for_patience", None)
        cur_best = st.get("val_acc1_best_seen", None)
        if prev_best is None:
            st["_best_seen_for_patience"] = cur_best
        else:
            if (cur_best is not None) and (float(cur_best) > float(prev_best) + 1e-12):
                st["_best_seen_for_patience"] = cur_best
                st["recovery_no_improve_epochs"] = 0
            else:
                st["recovery_no_improve_epochs"] = int(st.get("recovery_no_improve_epochs", 0)) + 1

        patience = int(_cfg_get(ctrl_cfg, "recovery_patience_epochs", 3) or 3)
        restart_len = int(_cfg_get(ctrl_cfg, "restart_window_epochs", 1) or 1)
        min_gap = int(_cfg_get(ctrl_cfg, "min_epochs_between_restarts", 1) or 1)

        if ((st["recovery_no_improve_epochs"] >= patience) or far_below) and (
            epoch - int(st.get("last_restart_epoch", -999999)) >= min_gap
        ):
            st["restart_until_epoch"] = int(epoch + restart_len - 1)
            st["last_restart_epoch"] = int(epoch)
            st["request_lr_restart"] = True
            in_restart = True

    lambda_hw_base = float(st.get("lambda_hw_base", st.get("lambda_hw", 0.0)) or 0.0)

    if in_restart or violated:
        guard_mode = "RESTART_WINDOW" if in_restart else "RECOVERY"
        lambda_hw_eff = 0.0
        allow_discrete = False
        freeze_discrete = True
    else:
        if (acc_ref_f is None) or (acc_used is None):
            guard_mode = "NO_ACCREF"
            lambda_hw_eff = lambda_hw_base
            allow_discrete = True
            freeze_discrete = False
        else:
            guard_mode = "HW_OPT"
            lambda_hw_eff = lambda_hw_base
            allow_discrete = True
            freeze_discrete = False

    st["guard_mode"] = str(guard_mode)
    st["lambda_hw_effective"] = float(lambda_hw_eff)
    st["allow_discrete_updates"] = bool(allow_discrete)
    st["freeze_discrete_updates"] = bool(freeze_discrete)

    return StableHWDecision(
        guard_mode=str(guard_mode),
        lambda_hw_base=float(lambda_hw_base),
        lambda_hw_effective=float(lambda_hw_eff),
        allow_discrete_updates=bool(allow_discrete),
        reason={
            "acc_ref": acc_ref_f,
            "acc_used": float(acc_used) if acc_used is not None else None,
            "acc_used_source": acc_used_source,
            "epsilon_drop": eps_drop,
            "violated": bool(violated),
            "in_restart_window": bool(in_restart),
            "restart_until_epoch": int(st.get("restart_until_epoch", -1)),
        },
    )
