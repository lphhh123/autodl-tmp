from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import json
import os
from pathlib import Path


@dataclass
class StableHWDecision:
    guard_mode: str
    lambda_hw_base: float
    lambda_hw_effective: float
    allow_discrete_updates: bool
    stop_training: bool
    reason: Dict[str, Any]
    state: Dict[str, Any]

"""
StableHW v5.4 field ownership:
  - schedule writes: lambda_hw_base
  - guard writes: lambda_hw_effective (authoritative value for training)
  - lambda_hw_after_guard is legacy/compat alias only
"""


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


def _get_root_and_stable(cfg_or_stable):
    """
    Accept either:
      - full cfg (has .stable_hw)
      - stable_hw_cfg itself
    Return: (root_cfg_or_None, stable_hw_cfg)
    """
    if hasattr(cfg_or_stable, "stable_hw"):
        return cfg_or_stable, cfg_or_stable.stable_hw
    return None, cfg_or_stable


def _get_locked_cfg(cfg_or_stable):
    root, stable = _get_root_and_stable(cfg_or_stable)
    lock = getattr(root, "locked_acc_ref", None) if root is not None else None
    if lock is None:
        lock = getattr(stable, "locked_acc_ref", None)
    return lock


def _get_no_drift_cfg(cfg_or_stable):
    root, stable = _get_root_and_stable(cfg_or_stable)
    nd = getattr(root, "no_drift", None) if root is not None else None
    if nd is None:
        nd = getattr(stable, "no_drift", None)
    return nd


def _get_accuracy_guard_cfg(cfg_or_stable: Any) -> dict:
    _, stable_hw_cfg = _get_root_and_stable(cfg_or_stable)
    guard = _cfg_get(stable_hw_cfg, "accuracy_guard", {}) or {}
    if isinstance(guard, dict):
        guard_cfg = dict(guard)
    else:
        try:
            guard_cfg = {k: guard[k] for k in guard}
        except Exception:
            guard_cfg = {}

    ctrl = _cfg_get(guard, "controller", None)

    # --- legacy fallback: if controller missing, treat guard itself as controller ---
    if not isinstance(ctrl, dict) or len(ctrl) == 0:
        ctrl = {}

        # copy legacy top-level keys into controller
        for k in [
            "enabled",
            "metric",
            "epsilon_drop",
            "guard_mode",
            "freeze_discrete_updates",
            "freeze_schedule_in_recovery",
            "recovery_min_epochs",
            "cut_hw_loss_on_violate",
            "k_exit",
            # legacy aliases (map later if present)
            "freeze_hw_on_drop",
            "freeze_hw_epochs",
            "cut_hw_loss_on_drop",
        ]:
            v = _cfg_get(guard, k, None)
            if v is not None:
                ctrl[k] = v

        # map legacy aliases -> v5.4 names
        if "freeze_hw_on_drop" in ctrl and "freeze_schedule_in_recovery" not in ctrl:
            ctrl["freeze_schedule_in_recovery"] = bool(ctrl.pop("freeze_hw_on_drop"))
        if "freeze_hw_epochs" in ctrl and "recovery_min_epochs" not in ctrl:
            ctrl["recovery_min_epochs"] = int(ctrl.pop("freeze_hw_epochs"))
        if "cut_hw_loss_on_drop" in ctrl and "cut_hw_loss_on_violate" not in ctrl:
            ctrl["cut_hw_loss_on_violate"] = bool(ctrl.pop("cut_hw_loss_on_drop"))

    eps = float(_cfg_get(ctrl, "epsilon_drop", 0.0))
    ctrl["epsilon_drop"] = max(eps, 0.0)

    guard_cfg["controller"] = ctrl
    return guard_cfg


def get_accuracy_metric_key(stable_hw_cfg) -> str:
    guard = _get_accuracy_guard_cfg(stable_hw_cfg)
    ctrl = _cfg_get(guard, "controller", {}) or {}
    mk = _cfg_get(ctrl, "metric", None)
    if mk is None:
        mk = _cfg_get(guard, "metric", "val_acc1")
    return str(mk)


def _safe_float(x: Any, default: float = 0.0) -> float:
    """
    Safe float cast.
    - If x is None: return default (can be None)
    - If cast fails: return default (can be None)
    """
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


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
        "stable_hw/freeze_schedule": bool(st.get("freeze_schedule", False)),
        "stable_hw/acc_margin_last": st.get("acc_margin_last", None),
    }


def stable_hw_schedule(
    epoch: int,
    stable_hw_cfg: Any,
    st: Dict[str, Any],
    hw_lambda_default: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Set lambda_hw_base for this epoch (warmup->ramp->stabilize).
    Dict-state implementation to match trainer_* usage.
    """
    if st is None:
        return st
    st.setdefault("guard_mode", "HW_OPT")
    st.setdefault("in_recovery", False)
    st.setdefault("allow_discrete_updates", True)

    enabled = bool(_cfg_get(stable_hw_cfg, "enabled", False))
    sched = _cfg_get(stable_hw_cfg, "lambda_hw_schedule", {}) or {}
    sched_enabled = bool(_cfg_get(sched, "enabled", enabled))

    if not enabled or not sched_enabled:
        # legacy behavior
        lam = _safe_float(hw_lambda_default, 0.0)
        st["lambda_hw_base"] = float(lam)
        st["schedule_phase"] = str(st.get("schedule_phase", "legacy"))
        st["lambda_hw_effective"] = float(lam)
        st["lambda_hw"] = float(lam)
        st["lambda_hw_after_guard"] = float(lam)
        return st

    warmup_epochs = int(_cfg_get(sched, "warmup_epochs", 5))
    ramp_epochs = int(_cfg_get(sched, "ramp_epochs", 10))
    lambda_hw_max = _safe_float(_cfg_get(sched, "lambda_hw_max", _cfg_get(sched, "max_lambda", 0.0)), 0.0)
    clamp_min = _safe_float(
        _cfg_get(
            sched,
            "clamp_min",
            _cfg_get(
                sched,
                "lambda_hw_min",
                _cfg_get(sched, "min_lambda", _cfg_get(sched, "min_hw_lambda", 0.0)),
            ),
        ),
        0.0,
    )
    clamp_max = _safe_float(_cfg_get(sched, "clamp_max", 1.0), 1.0)

    # freeze schedule during recovery if asked (spec)
    guard = _get_accuracy_guard_cfg(stable_hw_cfg)
    ctrl = _cfg_get(guard, "controller", {}) or {}
    freeze_in_recovery = bool(_cfg_get(ctrl, "freeze_schedule_in_recovery", True))
    if freeze_in_recovery and bool(st.get("in_recovery", False)):
        lam = float(st.get("lambda_hw_base", 0.0))
        st["freeze_schedule"] = True
        st["lambda_hw_base"] = float(lam)
        st["schedule_epoch"] = int(epoch)
        st["lambda_hw_effective"] = float(lam)  # pre-guard
        return st
    st["freeze_schedule"] = False
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
    clamp_min = float(
        _cfg_get(
            sched,
            "clamp_min",
            _cfg_get(
                sched,
                "lambda_hw_min",
                _cfg_get(sched, "min_lambda", _cfg_get(sched, "min_hw_lambda", 0.0)),
            ),
        )
        or 0.0
    )
    clamp_max = float(_cfg_get(sched, "clamp_max", lambda_hw_max) or lambda_hw_max)
    lam = float(max(clamp_min, min(clamp_max, float(lam))))
    st["schedule_phase"] = str(phase)

    # ---- v5.4 canonical writeback (single source of truth) ----
    # lambda_hw_base: schedule output (may be frozen in recovery)
    # lambda_hw_effective: pre-guard value == base; apply_accuracy_guard will overwrite to after-guard
    st["lambda_hw_base"] = float(lam)
    st["schedule_epoch"] = int(epoch)
    st["lambda_hw_effective"] = float(lam)

    # ===== v5.4 LockedAccRef guard: 未锁定 acc_ref 前，硬件项必须强制关闭 =====
    # 目的：acc_ref 必须是“纯精度参考”，不能被硬件项污染
    if bool(st.get("locked_acc_ref", True)) and (st.get("acc_ref", None) is None):
        st["lambda_hw_effective"] = 0.0
    return st


def _load_locked_acc_ref(stable_hw_cfg: Any, st: Dict[str, Any]) -> None:
    """Load baseline acc reference once (LockedAccRef)."""
    locked = _get_locked_cfg(stable_hw_cfg) or {}
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


def _load_baseline_stats(stats_path: str | os.PathLike[str]) -> Dict[str, Any] | None:
    p = Path(str(stats_path))
    if not p.exists() or not p.is_file():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    stats = dict(obj)
    comm_ref = stats.get("comm_ref_ms", stats.get("comm_ref_mb", stats.get("comm_ref", None)))
    if comm_ref is not None:
        stats["comm_ref_ms"] = comm_ref
    return stats


def _pick_acc_used(
    epoch: int,
    stable_hw_cfg: Any,
    st: Dict[str, Any],
    val_metric_or_none: Optional[float],
    has_val_this_epoch: bool,
    train_acc1_ema: Optional[float],
) -> Optional[float]:
    """Return accuracy metric used for guard decision, per controller.metric."""
    guard = _get_accuracy_guard_cfg(stable_hw_cfg)
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
    _, stable_cfg = _get_root_and_stable(stable_hw_cfg)
    allow_fallback = bool(_cfg_get(stable_cfg, "allow_train_ema_fallback", False))
    if allow_fallback and st.get("train_acc1_ema") is not None:
        return _safe_float(st.get("train_acc1_ema"), None)  # type: ignore[arg-type]

    return None


def _update_train_ema(stable_hw_cfg: Any, st: Dict[str, Any], acc: float) -> None:
    """Maintain train_acc1_ema in state (legacy support; also useful for fallback)."""
    guard = _get_accuracy_guard_cfg(stable_hw_cfg)
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
    *,
    epoch: int,
    stable_hw_cfg: Any,
    stable_hw_state: Dict[str, Any],
    val_metric_or_none: Optional[float] = None,
    has_val_this_epoch: bool = True,
    train_ema_or_none: Optional[float] = None,
    # ---- back-compat (trainer older call sites) ----
    val_acc1: Optional[float] = None,
    train_acc1_ema: Optional[float] = None,
) -> Tuple[StableHWDecision, bool]:
    """
    v5.4 Acc-First Hard Gating + LockedAccRef.
    Compatible with BOTH call styles:
      - new: val_metric_or_none / has_val_this_epoch / train_ema_or_none
      - old: val_acc1 / train_acc1_ema
    Returns: (StableHWDecision, allow_discrete_updates)
    """

    st = stable_hw_state
    _, stable_cfg = _get_root_and_stable(stable_hw_cfg)
    gcfg = _get_accuracy_guard_cfg(stable_hw_cfg)
    ctrl = _cfg_get(gcfg, "controller", {}) or {}

    # resolve inputs (prefer explicit new args; fall back to legacy names)
    if val_metric_or_none is None and val_acc1 is not None:
        val_metric_or_none = float(val_acc1)
    if train_ema_or_none is None and train_acc1_ema is not None:
        train_ema_or_none = float(train_acc1_ema)
    if val_metric_or_none is None:
        has_val_this_epoch = False

    eps_drop = float(_cfg_get(ctrl, "epsilon_drop", _cfg_get(gcfg, "epsilon_drop", 0.01)))
    cut_hw = bool(_cfg_get(ctrl, "cut_hw_loss_on_violate", True))
    freeze_discrete = bool(_cfg_get(ctrl, "freeze_discrete_updates", True))
    freeze_sched = bool(_cfg_get(ctrl, "freeze_schedule_in_recovery", True))
    recovery_min_epochs = int(_cfg_get(ctrl, "recovery_min_epochs", 1))
    k_exit = int(_cfg_get(ctrl, "k_exit", 2))
    margin_exit = float(_cfg_get(ctrl, "margin_exit", 0.0))

    acc_used = _pick_acc_used(
        epoch,
        stable_hw_cfg,
        st,
        val_metric_or_none,
        has_val_this_epoch,
        train_ema_or_none,
    )
    if acc_used is not None:
        st["acc_used_last"] = float(acc_used)

    # ---- v5.4 LockedAccRef (must be stable across epochs) ----
    # Priority:
    #   1) locked_acc_ref.baseline_stats_path (dense baseline)
    #   2) warmup_best: during warmup, track best val_acc1; freeze at freeze_epoch
    _load_locked_acc_ref(stable_hw_cfg, st)
    locked = _get_locked_cfg(stable_hw_cfg) or {}
    freeze_epoch = int(
        _cfg_get(
            locked,
            "freeze_epoch",
            _cfg_get(_cfg_get(stable_cfg, "lambda_hw_schedule", {}), "warmup_epochs", 0),
        )
    )

    # update warmup best (ONLY when we have a val metric)
    if has_val_this_epoch and val_metric_or_none is not None:
        cur = float(val_metric_or_none)
        best = st.get("warmup_acc_best")
        st["warmup_acc_best"] = cur if best is None else max(float(best), cur)

    # freeze acc_ref from warmup best when reaching freeze_epoch (end of warmup)
    if st.get("acc_ref") is None and freeze_epoch > 0 and (epoch + 1) >= freeze_epoch:
        if st.get("warmup_acc_best") is not None:
            st["acc_ref"] = float(st["warmup_acc_best"])
            st["acc_ref_source"] = "warmup_best"

    acc_ref = st.get("acc_ref", None)
    if acc_ref is None:
        # No reliable reference yet: hard-disable HW influence (Acc-First)
        st["guard_mode"] = "WARMUP"
        st["in_recovery"] = False
        st["lambda_hw_effective"] = 0.0
        st["lambda_hw_after_guard"] = 0.0
        st["allow_discrete_updates"] = True
        st["stop_on_violation"] = False
        decision = StableHWDecision(
            guard_mode=str(st["guard_mode"]),
            lambda_hw_base=float(st.get("lambda_hw_base", 0.0)),
            lambda_hw_effective=float(st["lambda_hw_effective"]),
            allow_discrete_updates=bool(st["allow_discrete_updates"]),
            stop_training=False,
            reason={
                "acc_ref": None,
                "metric": None,
                "eps_drop": float(eps_drop),
                "margin": 0.0,
                "violate": False,
                "stop_on_violation": False,
            },
            state=st,
        )
        return decision, True

    # choose metric
    metric_key = str(_cfg_get(ctrl, "metric", "val_acc1"))
    metric = None
    if metric_key in ("val_acc1", "acc1", "val"):
        metric = float(val_metric_or_none) if (has_val_this_epoch and val_metric_or_none is not None) else None
    elif metric_key in ("train_acc1_ema", "train_ema"):
        metric = float(train_ema_or_none) if train_ema_or_none is not None else None
    else:
        metric = float(val_metric_or_none) if (has_val_this_epoch and val_metric_or_none is not None) else None

    # persist last seen
    if has_val_this_epoch and val_metric_or_none is not None:
        st["val_acc1_last"] = float(val_metric_or_none)
    if train_ema_or_none is not None:
        st["train_acc1_ema"] = float(train_ema_or_none)

    # threshold check
    violate = False
    if metric is not None:
        violate = (float(acc_ref) - float(metric)) > float(eps_drop)
    margin = float(metric) - (float(acc_ref) - float(eps_drop)) if metric is not None else 0.0
    stop_on_violation = bool(_cfg_get(ctrl, "stop_on_violation", _cfg_get(gcfg, "stop_on_violation", False)))
    stop_training = bool(stop_on_violation and violate)

    # recovery bookkeeping
    if st.get("guard_mode") not in ("RECOVERY", "VIOLATE", "OK", "WARMUP"):
        st["guard_mode"] = "WARMUP"

    if violate:
        st["guard_mode"] = "VIOLATE"
        st["recovery_enter_epoch"] = int(epoch)
        st["recovery_good_streak"] = 0
    else:
        # if previously in recovery, require consecutive good epochs
        if st.get("guard_mode") in ("RECOVERY", "VIOLATE"):
            enter = int(st.get("recovery_enter_epoch", epoch))
            min_ok = (epoch - enter + 1) >= recovery_min_epochs
            ok_margin = (metric is not None) and (float(metric) + margin_exit >= float(acc_ref) - float(eps_drop))
            if min_ok and ok_margin:
                st["recovery_good_streak"] = int(st.get("recovery_good_streak", 0)) + 1
            else:
                st["recovery_good_streak"] = 0

            if int(st.get("recovery_good_streak", 0)) >= k_exit:
                st["guard_mode"] = "OK"
            else:
                st["guard_mode"] = "RECOVERY"
        else:
            st["guard_mode"] = "OK"

    # apply gating
    base = float(st.get("lambda_hw_base", st.get("lambda_hw_effective", st.get("lambda_hw", 0.0))))
    allow_discrete_updates = True
    after = base

    if st["guard_mode"] in ("VIOLATE", "RECOVERY"):
        if cut_hw:
            after = 0.0
        if freeze_discrete:
            allow_discrete_updates = False
        if freeze_sched:
            st["freeze_schedule"] = True
    else:
        st["freeze_schedule"] = False

    # ---- v5.4 canonical state writeback ----
    st["lambda_hw_after_guard"] = float(after)
    st["lambda_hw_effective"] = float(after)  # trainer must ONLY use this
    st["guard_mode"] = str(st["guard_mode"])
    st["in_recovery"] = bool(st["guard_mode"] in ("VIOLATE", "RECOVERY"))
    st["acc_violation"] = bool(violate)
    st["acc_used_last"] = float(acc_used) if acc_used is not None else None
    st["acc_margin_last"] = float(margin)
    st["stop_on_violation"] = bool(stop_training)
    st["epsilon_drop"] = float(eps_drop)
    st["last_guard_decision"] = {
        "guard_mode": str(st["guard_mode"]),
        "violate": bool(violate),
        "metric": float(metric) if metric is not None else None,
        "acc_ref": float(acc_ref) if acc_ref is not None else None,
        "eps_drop": float(eps_drop),
        "margin": float(margin),
        "recovery_good_streak": int(st.get("recovery_good_streak", 0)),
        "k_exit": int(k_exit),
        "stop_on_violation": bool(stop_training),
    }

    # schedule freeze flag mirrors controller intent
    st["freeze_schedule"] = bool(freeze_sched and st["in_recovery"])
    st["allow_discrete_updates"] = bool(allow_discrete_updates)
    decision = StableHWDecision(
        guard_mode=str(st["guard_mode"]),
        lambda_hw_base=float(base),
        lambda_hw_effective=float(after),
        allow_discrete_updates=bool(allow_discrete_updates),
        stop_training=bool(stop_training),
        reason={
            "acc_ref": float(acc_ref) if acc_ref is not None else None,
            "metric": float(metric) if metric is not None else None,
            "eps_drop": float(eps_drop),
            "margin": float(margin),
            "violate": bool(violate),
            "stop_on_violation": bool(stop_training),
        },
        state=st,
    )
    return decision, bool(allow_discrete_updates)


def init_locked_acc_ref(cfg_or_stable, state: dict):
    """
    v5.4 contract:
      locked_acc_ref can be at root-level (preferred) OR under stable_hw (legacy).

    Canonical:
      - manual: set state["acc_ref"] immediately and lock
      - warmup_best / baseline_stats: start UNLOCKED with acc_ref=None,
        then let apply_accuracy_guard() populate/lock it.
    """
    lock = _get_locked_cfg(cfg_or_stable)
    if lock is None or not bool(getattr(lock, "enabled", False)):
        return

    src = str(getattr(lock, "source", "manual")).lower().strip()

    # keep these for logging / completeness
    acc_ref1 = getattr(lock, "acc_ref1", None)
    acc_ref5 = getattr(lock, "acc_ref5", None)
    if acc_ref1 is not None:
        state["acc_ref1"] = float(acc_ref1)
    if acc_ref5 is not None:
        state["acc_ref5"] = float(acc_ref5)

    if src == "manual":
        if acc_ref1 is None:
            raise ValueError("[StableHW] locked_acc_ref.source=manual but acc_ref1 is missing.")
        state["acc_ref"] = float(acc_ref1)
        state["acc_ref_source"] = "manual"
        state["acc_ref_locked"] = True
        return

    # baseline_stats / warmup_best:
    # do NOT lock here; let apply_accuracy_guard() load/freeze
    state.setdefault("acc_ref_source", src)
    state.setdefault("acc_ref_locked", False)
    if "acc_ref" not in state:
        state["acc_ref"] = None


def update_train_acc1_ema(stable_hw_cfg: Any, st: Dict[str, Any], acc1: float) -> None:
    _update_train_ema(stable_hw_cfg, st, float(acc1))


# ===== v5.4: Locked HW refs (NoDrift for proxy refs) =====
def init_hw_refs_from_baseline_stats(cfg, stable_hw_state: dict, stable_hw_cfg=None):
    """
    v5.4 canonical:
      - refs frozen when NoDrift enabled
      - stats_path can come from:
          1) root no_drift.stats_path / baseline_stats_path
          2) stable_hw.no_drift.stats_path / baseline_stats_path (legacy)
          3) locked_acc_ref.baseline_stats_path (compat)
    """
    if stable_hw_cfg is None:
        stable_hw_cfg = getattr(cfg, "stable_hw", None)

    # start from config defaults (already validated)
    stable_hw_state["ref_T"] = float(getattr(getattr(cfg, "hw", None), "latency_ref_ms", 1.0))
    stable_hw_state["ref_E"] = float(getattr(getattr(cfg, "hw", None), "energy_ref_mj", 1.0))
    stable_hw_state["ref_M"] = float(getattr(getattr(cfg, "hw", None), "memory_ref_mb", 1.0))
    base_stats = {}

    # --- Comm reference (ms) ---
    # v5.4 expects comm_ref_ms (milliseconds). Keep backward-compat with older keys.
    comm_ref_ms = float(
        getattr(
            getattr(cfg, "hw", None),
            "comm_ref_ms",
            base_stats.get("comm_ref_ms", base_stats.get("comm_ms", base_stats.get("comm_ref_mb", 1.0))),
        )
    )

    stable_hw_state["comm_ref_ms"] = comm_ref_ms
    # ref_C is the comm reference used by hw_loss normalization (C for comm)
    stable_hw_state["ref_C"] = comm_ref_ms

    nd_cfg = _get_no_drift_cfg(cfg, stable_hw_cfg)
    lock_cfg = _get_locked_cfg(cfg)

    def _enabled(x):
        if isinstance(x, bool):
            return x
        return bool(getattr(x, "enabled", False))

    no_drift_enabled = _enabled(nd_cfg)

    stats_path = None
    if nd_cfg is not None:
        stats_path = getattr(nd_cfg, "stats_path", None) or getattr(nd_cfg, "baseline_stats_path", None)
    if (not stats_path) and (lock_cfg is not None):
        stats_path = getattr(lock_cfg, "baseline_stats_path", None)

    if stats_path:
        loaded = _load_baseline_stats(stats_path)
        if loaded:
            # allow multiple key aliases
            stable_hw_state["ref_T"] = float(
                loaded.get("ref_T", loaded.get("latency_ref_ms", loaded.get("lat_ref_ms", stable_hw_state["ref_T"])))
            )
            stable_hw_state["ref_E"] = float(
                loaded.get("ref_E", loaded.get("energy_ref_mj", loaded.get("en_ref_mj", stable_hw_state["ref_E"])))
            )
            stable_hw_state["ref_M"] = float(
                loaded.get("ref_M", loaded.get("memory_ref_mb", loaded.get("mem_ref_mb", stable_hw_state["ref_M"])))
            )
            stable_hw_state["ref_C"] = float(
                loaded.get(
                    "ref_C",
                    loaded.get(
                        "comm_ref_ms",
                        loaded.get("comm_ref_mb", loaded.get("comm_ref", stable_hw_state["ref_C"])),
                    ),
                )
            )
            stable_hw_state["_refs_loaded_from"] = str(stats_path)

    stable_hw_state["no_drift_enabled"] = bool(no_drift_enabled)
    return stable_hw_state


def _update_hw_refs_when_allowed(stable_hw_state: dict, stats: dict, cfg: dict) -> dict:
    # ====== Original update_hw_refs_from_stats logic moved here ======
    # Only update hardware refs; do NOT update locked_acc_ref / acc_ref here.
    if not stats:
        return stats

    proxy_used = stats.get("proxy_used", {}) if isinstance(stats, dict) else {}
    lat = proxy_used.get("latency_ms", stats.get("latency_ms", stats.get("lat_ms", None)))
    mem = proxy_used.get("memory_mb", stats.get("memory_mb", stats.get("mem_mb", None)))
    energy = proxy_used.get("energy_mj", stats.get("energy_mj", None))
    comm = proxy_used.get("comm_ms", stats.get("comm_ms", None))

    if lat is not None:
        stable_hw_state["ref_T"] = float(lat)
        stable_hw_state["latency_ref_ms"] = float(lat)
    if mem is not None:
        stable_hw_state["ref_M"] = float(mem)
        stable_hw_state["memory_ref_mb"] = float(mem)
    if energy is not None:
        stable_hw_state["ref_E"] = float(energy)
    if comm is not None:
        stable_hw_state["ref_C"] = float(comm)
    stable_hw_state["hw_ref_source"] = "online_stats"
    return stats


def update_hw_refs_from_stats(cfg, stable_hw_state: dict, latest_stats: dict, stable_hw_cfg=None):
    """
    v5.4:
      - only update refs if no_drift.enabled == False
      - update canonical ref_* keys and aliases together
      - supported mode: ema
    """
    if stable_hw_cfg is None:
        stable_hw_cfg = getattr(cfg, "stable_hw", None)

    nd_cfg = _get_no_drift_cfg(cfg, stable_hw_cfg)
    if isinstance(nd_cfg, bool):
        no_drift_enabled = nd_cfg
    else:
        no_drift_enabled = bool(getattr(nd_cfg, "enabled", False))
    if no_drift_enabled:
        # frozen under NoDrift
        stable_hw_state["_contract_ref_frozen"] = True
        return stable_hw_state

    mode = str(getattr(nd_cfg, "ref_update", "frozen")).lower()
    alpha = float(getattr(nd_cfg, "ref_update_alpha", 0.1))
    alpha = max(0.0, min(1.0, alpha))

    def _pos(x, fallback):
        try:
            v = float(x)
            if v > 0:
                return v
        except Exception:
            pass
        return float(fallback if fallback > 0 else 1.0)

    # read latest stats (tolerant)
    T_new = _pos(
        latest_stats.get("latency_ms", latest_stats.get("T_ms", None)),
        stable_hw_state.get("ref_T", 1.0),
    )
    E_new = _pos(
        latest_stats.get("energy_mj", latest_stats.get("E_mj", None)),
        stable_hw_state.get("ref_E", 1.0),
    )
    M_new = _pos(
        latest_stats.get("memory_mb", latest_stats.get("M_mb", None)),
        stable_hw_state.get("ref_M", 1.0),
    )
    C_new = _pos(
        latest_stats.get("comm_ms", latest_stats.get("C_ms", None)),
        stable_hw_state.get("ref_C", 1.0),
    )

    if mode != "ema":
        # v5.4 scope: only ema is supported; fall back safely
        mode = "ema"

    # EMA update
    def _ema(old, new):
        old = _pos(old, new)
        return (1.0 - alpha) * float(old) + alpha * float(new)

    stable_hw_state["ref_T"] = _ema(stable_hw_state.get("ref_T", T_new), T_new)
    stable_hw_state["ref_E"] = _ema(stable_hw_state.get("ref_E", E_new), E_new)
    stable_hw_state["ref_M"] = _ema(stable_hw_state.get("ref_M", M_new), M_new)
    stable_hw_state["ref_C"] = _ema(stable_hw_state.get("ref_C", C_new), C_new)

    # keep aliases in sync
    stable_hw_state["latency_ref_ms"] = stable_hw_state["ref_T"]
    stable_hw_state["energy_ref_mj"] = stable_hw_state["ref_E"]
    stable_hw_state["memory_ref_mb"] = stable_hw_state["ref_M"]
    stable_hw_state["comm_ref_ms"] = stable_hw_state["ref_C"]

    stable_hw_state["_ref_update_mode"] = mode
    stable_hw_state["_ref_update_alpha"] = alpha
    return stable_hw_state
