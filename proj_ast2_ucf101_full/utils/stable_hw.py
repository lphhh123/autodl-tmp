from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
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
) -> Tuple[str, float, bool]:
    """
    v5.4 Acc-First Hard Gating + LockedAccRef.
    Compatible with BOTH call styles:
      - new: val_metric_or_none / has_val_this_epoch / train_ema_or_none
      - old: val_acc1 / train_acc1_ema
    Returns: (guard_mode, lambda_hw_after_guard, allow_discrete_updates)
    """

    st = stable_hw_state
    gcfg = _cfg_get(stable_hw_cfg, "accuracy_guard", {}) or {}
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

    locked = _cfg_get(stable_hw_cfg, "locked_acc_ref", {}) or {}
    freeze_epoch = int(_cfg_get(locked, "freeze_epoch", 0))

    # ===== v5.4 LockedAccRef: warmup-best then freeze =====
    st.setdefault("warmup_best", None)

    # 1) warmup 阶段：只更新 warmup_best（取最大）
    if acc_used is not None and epoch < freeze_epoch:
        prev = st.get("warmup_best", None)
        st["warmup_best"] = float(acc_used) if prev is None else float(max(float(prev), float(acc_used)))
        st["acc_ref_source"] = "warmup_best_tracking"

    # 2) freeze 时刻（epoch >= freeze_epoch）：如果 acc_ref 还没锁定，则锁定为 warmup_best
    if st.get("acc_ref") is None and epoch >= freeze_epoch:
        wb = st.get("warmup_best", None)
        if wb is not None:
            st["acc_ref"] = float(wb)
            st["acc_ref_source"] = "warmup_best_frozen"
        else:
            # 没有 warmup_best（例如 val 一直没跑出来），则保持 pending，等待后续首次可用 acc_used 再锁
            st.setdefault("acc_ref_source", "warmup_best_pending")

    # 3) 兜底：若 freeze 后依旧没有 acc_ref，但此时 acc_used 可用，则锁一次（只锁一次，不漂移）
    if st.get("acc_ref") is None and epoch >= freeze_epoch and acc_used is not None:
        st["acc_ref"] = float(acc_used)
        st["acc_ref_source"] = "post_freeze_first_seen"

    # LockedAccRef must exist (init_locked_acc_ref should have set it)
    acc_ref = st.get("acc_ref", None)
    if acc_ref is None:
        # fallback: if no acc_ref, behave like warmup (do not amplify hw)
        st["guard_mode"] = "WARMUP"
        st["lambda_hw_after_guard"] = float(st.get("lambda_hw_base", st.get("lambda_hw", 0.0)))
        st["allow_discrete_updates"] = True
        return st["guard_mode"], float(st["lambda_hw_after_guard"]), True

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
    base = float(st.get("lambda_hw_effective", st.get("lambda_hw_base", st.get("lambda_hw", 0.0))))
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

    st["lambda_hw_after_guard"] = float(after)
    st["allow_discrete_updates"] = bool(allow_discrete_updates)
    return st["guard_mode"], float(after), bool(allow_discrete_updates)


def init_locked_acc_ref(stable_hw_cfg: Any, st: Dict[str, Any]) -> None:
    """Initialize LockedAccRef once (alias for internal loader)."""
    _load_locked_acc_ref(stable_hw_cfg, st)


# ===== v5.4: Locked HW refs (NoDrift for proxy refs) =====
def init_hw_refs_from_baseline_stats(stable_hw_cfg: Any, st: Dict[str, Any]) -> None:
    """
    Initialize ref_T/ref_E/ref_M/ref_C into stable_hw_state.
    Priority:
      1) If locked_acc_ref.baseline_stats_path exists AND prefer_dense_baseline=True -> lock refs from that file.
      2) Else: leave refs unset (compute_hw_loss will fallback to cfg.hw.*), and allow EMA update later.
    This function is idempotent.
    """
    if st.get("_hw_refs_inited", False):
        return

    locked = _cfg_get(stable_hw_cfg, "locked_acc_ref", {}) or {}
    baseline_stats_path = (
        _cfg_get(locked, "baseline_stats_path", None)
        or _cfg_get(stable_hw_cfg, "baseline_stats_path", None)  # back-compat alias
    )
    prefer_dense = bool(_cfg_get(locked, "prefer_dense_baseline", True))

    norm = _cfg_get(stable_hw_cfg, "normalize", {}) or {}
    eps = float(_cfg_get(norm, "eps", 1e-6))

    def _pos(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            return eps
        if not (v > 0.0):
            return eps
        return max(eps, v)

    if baseline_stats_path and prefer_dense:
        p = Path(str(baseline_stats_path))
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                hw = d.get("last_hw_stats") or d.get("hw_stats") or d.get("hw") or {}
                if not hw and any(k in d for k in ["latency_ms", "energy_mj", "mem_mb", "comm_ms"]):
                    hw = d

                # tolerate multiple key styles
                T = hw.get("latency_ms", hw.get("lat_ms", hw.get("T_ms", hw.get("latency", None))))
                E = hw.get("energy_mj", hw.get("E_mj", hw.get("energy", None)))
                M = hw.get("mem_mb", hw.get("M_mb", hw.get("mem", None)))
                C = hw.get("comm_ms", hw.get("C_ms", hw.get("comm", None)))

                if T is not None:
                    st["ref_T"] = _pos(T)
                if E is not None:
                    st["ref_E"] = _pos(E)
                if M is not None:
                    st["ref_M"] = _pos(M)
                if C is not None:
                    st["ref_C"] = _pos(C)

                st["hw_ref_source"] = f"dense_baseline:{p}"
            except Exception:
                st["hw_ref_source"] = f"dense_baseline_parse_error:{p}"
        else:
            st["hw_ref_source"] = f"dense_baseline_missing:{p}"
    else:
        # no dense baseline locking
        st.setdefault("hw_ref_source", "ema_or_cfg_fallback")

    st["_hw_refs_inited"] = True


def update_hw_refs_from_stats(stable_hw_cfg: Any, st: Dict[str, Any], hw_stats: Dict[str, Any]) -> None:
    """
    EMA update refs from observed hw_stats (ONLY when not locked to dense baseline).
    Guards against negative/zero values to avoid 'negative latency reward' drifting refs.
    """
    src = str(st.get("hw_ref_source", ""))
    if src.startswith("dense_baseline:"):
        return

    norm = _cfg_get(stable_hw_cfg, "normalize", {}) or {}
    eps = float(_cfg_get(norm, "eps", 1e-6))
    beta = float(_cfg_get(norm, "ref_ema_beta", 0.9))  # optional; default 0.9

    def _read_pos(keys):
        for k in keys:
            if k in hw_stats and hw_stats[k] is not None:
                try:
                    v = float(hw_stats[k])
                except Exception:
                    continue
                if v > 0.0:
                    return max(eps, v)
        return None

    T = _read_pos(["latency_ms", "lat_ms", "T_ms", "latency"])
    E = _read_pos(["energy_mj", "E_mj", "energy"])
    M = _read_pos(["mem_mb", "M_mb", "mem"])
    C = _read_pos(["comm_ms", "C_ms", "comm"])

    def _ema(key: str, v: float):
        old = st.get(key, None)
        if old is None:
            st[key] = v
            return
        try:
            oldf = float(old)
        except Exception:
            st[key] = v
            return
        st[key] = beta * max(eps, oldf) + (1.0 - beta) * v

    if T is not None:
        _ema("ref_T", T)
    if E is not None:
        _ema("ref_E", E)
    if M is not None:
        _ema("ref_M", M)
    if C is not None:
        _ema("ref_C", C)

    st["hw_ref_source"] = "ema"
