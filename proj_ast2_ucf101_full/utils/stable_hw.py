from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import json
import math
import os
from pathlib import Path


@dataclass
class StableHWDecision:
    guard_mode: str
    lambda_hw_base: float
    lambda_hw_effective: float
    allow_discrete_updates: bool
    stop_training: bool
    request_lr_restart: bool
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


def _as_enabled(val: Any, default: bool = True) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, dict):
        return bool(val.get("enabled", default))
    return bool(getattr(val, "enabled", default))


def _expand_out_dir_template(path_val: Any, out_dir: Any) -> Any:
    if path_val is None:
        return None
    s = str(path_val)
    if out_dir is None:
        return s
    od = str(out_dir)
    s = s.replace("${out_dir}", od).replace("{out_dir}", od)
    return s


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

    # v5.4 strict: forbid legacy root-level fields (Anti-Loop evidence must match signature)
    contract = getattr(root, "contract", None) if root is not None else None
    is_v54 = bool(contract) and str(getattr(contract, "version", "")).strip() == "v5.4"
    strict = bool(contract) and bool(getattr(contract, "strict", False))
    if is_v54 and strict:
        if getattr(root, "locked_acc_ref", None) not in (None, {}, ""):
            raise ValueError(
                "P0(v5.4): legacy root-level 'locked_acc_ref' is forbidden. "
                "Move it under cfg.stable_hw.locked_acc_ref to avoid silent semantic drift."
            )
        return getattr(stable, "locked_acc_ref", {}) or {}

    # non-strict / legacy compatibility (kept only for old experiments)
    root_locked = getattr(root, "locked_acc_ref", {}) if root is not None else {}
    stable_locked = getattr(stable, "locked_acc_ref", {}) if stable is not None else {}
    return root_locked or stable_locked


def _get_no_drift_cfg(cfg_or_stable, stable_hw_cfg=None):
    """
    v5.4: support both legacy root-level cfg.no_drift and v5 nested cfg.stable_hw.no_drift.
    Also keep backward-compat with old call sites that pass (cfg, stable_hw_cfg).
    """
    root, stable = _get_root_and_stable(cfg_or_stable)

    contract = getattr(root, "contract", None) if root is not None else None
    is_v54 = bool(contract) and str(getattr(contract, "version", "")).strip() == "v5.4"
    strict = bool(contract) and bool(getattr(contract, "strict", False))
    if is_v54 and strict:
        if getattr(root, "no_drift", None) not in (None, {}, ""):
            raise ValueError(
                "P0(v5.4): legacy root-level 'no_drift' is forbidden. "
                "Move it under cfg.stable_hw.no_drift to avoid silent semantic drift."
            )
        return getattr(stable, "no_drift", {}) or {}

    root_nd = getattr(root, "no_drift", {}) if root is not None else {}
    cand = stable_hw_cfg if stable_hw_cfg is not None else stable
    stable_nd = getattr(cand, "no_drift", {}) if cand is not None else {}
    return root_nd or stable_nd


def _get_accuracy_guard_cfg(cfg_or_stable: Any) -> dict:
    _, stable_hw_cfg = _get_root_and_stable(cfg_or_stable)
    # ===== v5.4 SPEC alias: support stable_hw.guard as primary (SPEC_C/D) =====
    guard = _cfg_get(stable_hw_cfg, "accuracy_guard", None)
    if not guard:
        guard = _cfg_get(stable_hw_cfg, "guard", None)
    guard = guard or {}
    if isinstance(guard, dict):
        guard_cfg = dict(guard)
    else:
        try:
            guard_cfg = {k: guard[k] for k in guard}
        except Exception:
            guard_cfg = {}

    ctrl = _cfg_get(guard, "controller", None)

    # ===== v5.4 SPEC field alias mapping (metric_key/threshold/hysteresis/consecutive_trigger) =====
    # NOTE: keep everything in `ctrl` so downstream uses controller.*
    if isinstance(ctrl, dict):
        # metric_key -> metric
        if ("metric" not in ctrl) and ("metric_key" in ctrl):
            ctrl["metric"] = ctrl["metric_key"]
        # threshold -> epsilon_drop (our code uses epsilon_drop)
        if ("epsilon_drop" not in ctrl):
            if "threshold" in ctrl:
                ctrl["epsilon_drop"] = ctrl["threshold"]
            elif "tolerance" in ctrl:
                ctrl["epsilon_drop"] = ctrl["tolerance"]
        # consecutive_trigger -> k_exit (closest existing knob in our guard FSM)
        # 解释：本实现用 k_exit 表示“连续好 epoch 才退出 RECOVERY”，与 spec 的连续触发/恢复最接近
        if ("k_exit" not in ctrl) and ("consecutive_trigger" in ctrl):
            ctrl["k_exit"] = int(ctrl["consecutive_trigger"])
        # keep hysteresis (will be used by patch P0-1b below)
        if ("hysteresis" not in ctrl) and ("hysteresis" in guard_cfg):
            ctrl["hysteresis"] = guard_cfg.get("hysteresis")

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


def stable_hw_log_fields(st: Dict[str, Any], cfg: Any = None) -> Dict[str, Any]:
    """Flat dict for metrics.json / tb; include ref source for SPEC_C audit."""
    return {
        "stable_hw/lambda_hw_base": _safe_float(st.get("lambda_hw_base"), 0.0),
        "stable_hw/lambda_hw_effective": _safe_float(st.get("lambda_hw_effective"), 0.0),
        "stable_hw/allow_discrete_updates": bool(st.get("allow_discrete_updates", True)),
        "stable_hw/guard_mode": str(st.get("guard_mode", "HW_OPT")),
        "stable_hw/in_recovery": bool(st.get("in_recovery", False)),
        "stable_hw/acc_ref": _safe_float(st.get("acc_ref"), 0.0) if st.get("acc_ref") is not None else None,
        "stable_hw/acc_used_last": _safe_float(st.get("acc_used_last"), 0.0)
        if st.get("acc_used_last") is not None
        else None,
        "stable_hw/acc_violation": bool(st.get("acc_violation", False)),
        # NEW: refs audit (SPEC_C 12B.2)
        "stable_hw/hw_ref_source": str(st.get("hw_ref_source", "")),
        "stable_hw/refs_loaded_from": st.get("_refs_loaded_from"),
        "stable_hw/no_drift_effective": bool(st.get("no_drift_effective", st.get("no_drift_enabled", False))),
        "stable_hw/ref_update_mode": str(st.get("ref_update_mode", st.get("_force_ref_update_mode", "frozen"))),
        "stable_hw/ref_T": _safe_float(st.get("ref_T"), 0.0) if st.get("ref_T") is not None else None,
        "stable_hw/ref_E": _safe_float(st.get("ref_E"), 0.0) if st.get("ref_E") is not None else None,
        "stable_hw/ref_M": _safe_float(st.get("ref_M"), 0.0) if st.get("ref_M") is not None else None,
        "stable_hw/ref_C": _safe_float(st.get("ref_C"), 0.0) if st.get("ref_C") is not None else None,
        "schedule_phase": st.get("schedule_phase", ""),
        "lambda_hw_base": _safe_float(st.get("lambda_hw_base", 0.0), 0.0),
        "lambda_hw_eff": _safe_float(st.get("lambda_hw_effective", 0.0), 0.0),
        "lambda_hw_effective": _safe_float(st.get("lambda_hw_effective", 0.0), 0.0),
        "acc_ref": _safe_float(st.get("acc_ref", 0.0), 0.0),
        "acc_ref_source": st.get("acc_ref_source", ""),
        "acc_last": _safe_float(st.get("acc_last", 0.0), 0.0),
        "acc_last_source": st.get("acc_last_source", ""),
        "epsilon_drop": _safe_float(st.get("epsilon_drop", 0.0), 0.0),
        "guard_triggered": bool(st.get("guard_triggered", False)),
        "violate_streak": int(st.get("violate_streak", 0) or 0),
        "recovery_until_epoch": int(st.get("recovery_until_epoch", -1) or -1),
        "freeze_discrete_updates": bool(st.get("freeze_discrete_updates", False)),
        "freeze_schedule_in_recovery": bool(st.get("freeze_schedule_in_recovery", False)),
        "cut_hw_loss_on_violate": bool(st.get("cut_hw_loss_on_violate", False)),
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

    # v5.4: treat missing stable_hw.enabled as enabled (Version-C default);
    # only disable schedule when stable_hw.enabled is explicitly False.
    enabled = bool(_cfg_get(stable_hw_cfg, "enabled", True))
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
    lock_cfg = _get_locked_cfg(stable_hw_cfg)
    lock_enabled = _as_enabled(lock_cfg, default=True)
    if lock_enabled and (st.get("acc_ref", None) is None):
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

    source = str(_cfg_get(locked, "source", _cfg_get(locked, "ref_source", "warmup_best")))
    strict = bool(_cfg_get(locked, "strict", True))
    baseline_stats_path = _cfg_get(locked, "baseline_stats_path", None)
    baseline_stats_path = _expand_out_dir_template(baseline_stats_path, st.get("out_dir"))
    if source == "baseline_stats":
        if not baseline_stats_path:
            if strict:
                raise RuntimeError("[v5.4 LockedAccRef] baseline_stats_path is required (strict=true).")
            st["acc_ref_source"] = "baseline_stats_missing_path"
            return
        p = Path(str(baseline_stats_path))
        if not (p.exists() and p.is_file()):
            if strict:
                raise RuntimeError(f"[v5.4 LockedAccRef] baseline_stats_path invalid: {baseline_stats_path}")
            st["acc_ref_source"] = f"baseline_stats_missing:{p}"
            return
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            allow_placeholder = bool(_cfg_get(locked, "allow_placeholder", False))
            if bool(obj.get("is_placeholder", False)) and not allow_placeholder:
                raise RuntimeError(
                    "[P0][v5.4] Refusing placeholder baseline_stats.json for LockedAccRef.\n"
                    f"baseline_stats_path={baseline_stats_path}\n"
                    "If you REALLY want to use placeholder for smoke only, set:\n"
                    "  stable_hw.locked_acc_ref.allow_placeholder: true\n"
                    "and accept it will invalidate paper-grade results."
                )
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
            if strict:
                raise RuntimeError(f"[v5.4 LockedAccRef] baseline_stats_path invalid: {baseline_stats_path}")
            st["acc_ref_source"] = f"baseline_stats_parse_error:{p}"
            return

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
    hw_ref = stats.get("last_hw_stats") or stats.get("hw_ref") or stats.get("baseline_hw_stats")
    if isinstance(hw_ref, dict):
        latency = hw_ref.get("latency_ms", hw_ref.get("total_latency_ms", None))
        mem_peak = hw_ref.get("mem_peak_mb", hw_ref.get("peak_mem_mb", None))
        energy = hw_ref.get("energy_mj", None)
        if latency is not None:
            stats.setdefault("ref_T", latency)
            stats.setdefault("latency_ref_ms", latency)
        if mem_peak is not None:
            stats.setdefault("ref_M", mem_peak)
            stats.setdefault("memory_ref_mb", mem_peak)
        if energy is not None:
            stats.setdefault("ref_E", energy)
    else:
        latency = stats.get("latency_ms", stats.get("total_latency_ms", None))
        mem_peak = stats.get("mem_peak_mb", stats.get("peak_mem_mb", None))
        energy = stats.get("energy_mj", None)
        if latency is not None:
            stats.setdefault("ref_T", latency)
            stats.setdefault("latency_ref_ms", latency)
        if mem_peak is not None:
            stats.setdefault("ref_M", mem_peak)
            stats.setdefault("memory_ref_mb", mem_peak)
        if energy is not None:
            stats.setdefault("ref_E", energy)
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
            st["acc_used_source"] = metric
            st["acc_fallback_reason"] = "none"
            return _safe_float(val_metric_or_none, None)  # type: ignore[arg-type]

    if metric in ("train_acc1_ema", "acc1_ema"):
        if train_acc1_ema is not None:
            st["acc_used_source"] = "train_acc1_ema"
            st["acc_fallback_reason"] = "none"
            return _safe_float(train_acc1_ema, None)  # type: ignore[arg-type]
        # fallback to state
        if st.get("train_acc1_ema") is not None:
            st["acc_used_source"] = "train_acc1_ema"
            st["acc_fallback_reason"] = "use_state_train_ema"
            return _safe_float(st.get("train_acc1_ema"), None)  # type: ignore[arg-type]

    # allow fallback to val_acc1_last if has_val missed but we have last
    if metric == "val_acc1" and st.get("val_acc1_last") is not None:
        st["acc_used_source"] = f"{metric}_last"
        st["acc_fallback_reason"] = "use_last_val"
        return _safe_float(st.get("val_acc1_last"), None)  # type: ignore[arg-type]

    # final fallback: if asked, use ema state
    _, stable_cfg = _get_root_and_stable(stable_hw_cfg)
    allow_fallback = bool(_cfg_get(stable_cfg, "allow_train_ema_fallback", False))
    if allow_fallback and st.get("train_acc1_ema") is not None:
        st["acc_used_source"] = "train_acc1_ema"
        st["acc_fallback_reason"] = "fallback_train_ema"
        return _safe_float(st.get("train_acc1_ema"), None)  # type: ignore[arg-type]

    st["acc_used_source"] = ""
    st["acc_fallback_reason"] = "no_signal"
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
    prev_guard_mode = str(st.get("guard_mode", "HW_OPT"))
    st.setdefault("request_lr_restart", False)
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
    st["acc_drop_max"] = float(eps_drop)
    st["epsilon_drop"] = float(eps_drop)

    # ===== v5.4 hysteresis: different enter/exit thresholds to avoid oscillation =====
    hyst = float(_cfg_get(ctrl, "hysteresis", 0.0) or 0.0)
    eps_enter = max(eps_drop, 0.0)
    eps_exit = max(eps_enter - hyst, 0.0)
    prev_mode = str(st.get("guard_mode", "HW_OPT")).upper()
    eps_used = eps_enter if prev_mode in ("OK", "HW_OPT", "WARMUP") else eps_exit
    st["guard_eps_enter"] = eps_enter
    st["guard_eps_exit"] = eps_exit
    st["guard_eps_used"] = eps_used
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
        # --- [v5.4 CONTRACT] Fill auditable gating fields for SPEC_E ---
        # metric is current measured accuracy (or whatever metric configured)
        # acc_ref is the reference accuracy (may be None during warmup)
        metric_now = acc_used
        if metric_now is None and has_val_this_epoch and val_metric_or_none is not None:
            metric_now = float(val_metric_or_none)
        st["acc_ref"] = float(acc_ref) if acc_ref is not None else None
        st["acc_now"] = float(metric_now) if metric_now is not None else 0.0

        if acc_ref is not None and metric_now is not None:
            st["acc_drop"] = float(max(0.0, float(acc_ref) - float(metric_now)))
        else:
            st["acc_drop"] = 0.0

        # eps_used is the effective drop threshold (after any hysteresis)
        st["acc_drop_max"] = float(eps_used)
        decision = StableHWDecision(
            guard_mode=str(st["guard_mode"]),
            lambda_hw_base=float(st.get("lambda_hw_base", 0.0)),
            lambda_hw_effective=float(st["lambda_hw_effective"]),
            allow_discrete_updates=bool(st["allow_discrete_updates"]),
            stop_training=False,
            request_lr_restart=False,
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
        st["request_lr_restart"] = False
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
    st["acc_last"] = float(metric) if metric is not None else None
    st["acc_last_source"] = str(metric_key)

    # threshold check
    violate = False
    if metric is not None:
        violate = (float(acc_ref) - float(metric)) > float(eps_used)
    margin = float(metric) - (float(acc_ref) - float(eps_used)) if metric is not None else 0.0
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
            ok_margin = (metric is not None) and (float(metric) + margin_exit >= float(acc_ref) - float(eps_used))
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

    request_lr_restart = bool(st["guard_mode"] == "RECOVERY" and prev_guard_mode != "RECOVERY")

    # ---- v5.4 canonical state writeback ----
    st["lambda_hw_after_guard"] = float(after)
    st["lambda_hw_effective"] = float(after)  # trainer must ONLY use this
    st["guard_mode"] = str(st["guard_mode"])
    st["in_recovery"] = bool(st["guard_mode"] in ("VIOLATE", "RECOVERY"))
    st["acc_violation"] = bool(violate)
    st["guard_triggered"] = bool(violate)
    st["violate_streak"] = int(st.get("violate_streak", 0) + 1) if violate else 0
    st["acc_used_last"] = float(acc_used) if acc_used is not None else None
    st["acc_margin_last"] = float(margin)
    st["stop_on_violation"] = bool(stop_training)
    st["epsilon_drop"] = float(eps_drop)
    st["freeze_discrete_updates"] = bool(freeze_discrete)
    st["freeze_schedule_in_recovery"] = bool(freeze_sched)
    st["cut_hw_loss_on_violate"] = bool(cut_hw)
    if st["guard_mode"] in ("VIOLATE", "RECOVERY"):
        st["recovery_until_epoch"] = int(epoch + recovery_min_epochs)
    else:
        st["recovery_until_epoch"] = int(-1)
    # --- [v5.4 CONTRACT] Fill auditable gating fields for SPEC_E ---
    # metric is current measured accuracy (or whatever metric configured)
    # acc_ref is the reference accuracy (may be None during warmup)
    st["acc_ref"] = float(acc_ref) if acc_ref is not None else None
    st["acc_now"] = float(metric) if metric is not None else 0.0

    if acc_ref is not None and metric is not None:
        st["acc_drop"] = float(max(0.0, float(acc_ref) - float(metric)))
    else:
        st["acc_drop"] = 0.0

    # eps_used is the effective drop threshold (after any hysteresis)
    st["acc_drop_max"] = float(eps_used)
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
    st["request_lr_restart"] = bool(request_lr_restart)
    decision = StableHWDecision(
        guard_mode=str(st["guard_mode"]),
        lambda_hw_base=float(base),
        lambda_hw_effective=float(after),
        allow_discrete_updates=bool(allow_discrete_updates),
        stop_training=bool(stop_training),
        request_lr_restart=bool(request_lr_restart),
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
    if lock is None or not _as_enabled(lock, default=False):
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
def extract_hw_refs_from_baseline(baseline_stats: dict) -> dict:
    """
    Parse baseline_stats.json into hw ref values.
    Required outputs:
      latency_ref_ms, energy_ref_mj, mem_ref_mb, comm_ref
    Accept multiple alias schemas to be robust, but in strict mode caller should reject placeholders.
    """
    if not isinstance(baseline_stats, dict):
        raise ValueError("[v5.4 P0] baseline_stats must be a dict")

    # preferred containers (order matters)
    hw = (
        baseline_stats.get("hw_ref")
        or baseline_stats.get("baseline_hw_refs")
        or baseline_stats.get("baseline_hw_stats")
        or baseline_stats.get("last_hw_stats")
        or {}
    )
    if not isinstance(hw, dict):
        raise ValueError("[v5.4 P0] baseline_stats missing hw_ref/baseline_hw_stats/last_hw_stats dict")

    def _pick(*keys):
        for k in keys:
            v = hw.get(k, None)
            if v is not None:
                return v
        return None

    lat = _pick("latency_ref_ms", "latency_ms", "lat_ms", "ref_latency_ms")
    ene = _pick("energy_ref_mj", "energy_mj", "ene_mj", "ref_energy_mj")
    mem = _pick("mem_ref_mb", "mem_peak_mb", "mem_mb", "ref_mem_mb")
    com = _pick("comm_ref", "comm_ref_ms", "comm_ms", "ref_comm_ms")

    missing = [
        name
        for name, v in {
            "latency_ref_ms": lat,
            "energy_ref_mj": ene,
            "mem_ref_mb": mem,
            "comm_ref": com,
        }.items()
        if v is None
    ]
    if missing:
        raise ValueError(
            f"[v5.4 P0] baseline_stats missing keys: {missing}. "
            "Need latency/energy/mem/comm refs for StableHW NoDrift."
        )

    return {
        "latency_ref_ms": float(lat),
        "energy_ref_mj": float(ene),
        "mem_ref_mb": float(mem),
        "comm_ref": float(com),
    }


def init_hw_refs_from_baseline_stats(cfg: Any, stable_hw_state: Dict[str, Any], stable_hw_cfg: Any = None) -> None:
    """
    v5.4 contract:
      - NoDrift requested => refs must not drift (no ref_update).
      - baseline_stats missing is allowed ONLY if behavior is explicit + auditable and still drift-free.
    Policy (write-stable):
      - If baseline_stats present: init from baseline, freeze if NoDrift requested.
      - If baseline_stats missing + NoDrift requested: mark frozen and let first valid observation initialize refs (still no drift after init).
      - Never silently downgrade to EMA when NoDrift requested.
    """
    stable_hw_cfg = stable_hw_cfg if stable_hw_cfg is not None else getattr(cfg, "stable_hw", None)
    nd_cfg = _get_no_drift_cfg(cfg, stable_hw_cfg=stable_hw_cfg)
    requested_no_drift = _as_enabled(nd_cfg, default=False)

    # v5.4 canonical key is baseline_stats_path; keep baseline_stats as legacy alias.
    baseline_path = (
        getattr(stable_hw_cfg, "baseline_stats_path", "")
        or getattr(stable_hw_cfg, "baseline_stats", "")
        or getattr(cfg, "baseline_stats_path", "")
        or ""
    ).strip()

    baseline_stats = None
    hw_ref_source = ""
    if stable_hw_cfg is not None:
        hw_ref_source = str(getattr(stable_hw_cfg, "hw_ref_source", "") or "").strip()

    if hw_ref_source == "baseline_stats":
        if not baseline_path:
            if getattr(stable_hw_cfg, "strict", True):
                raise RuntimeError(
                    "[v5.4 P0] hw_ref_source=baseline_stats but baseline_stats_path is empty. "
                    "This would silently fallback and violate NoDrift/LockedAccRef auditability."
                )
            baseline_stats = None
        else:
            try:
                baseline_stats = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
            except Exception:
                baseline_stats = None
    elif baseline_path:
        try:
            baseline_stats = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
        except Exception:
            baseline_stats = None

    # Always record requested flag
    stable_hw_state["no_drift_enabled"] = bool(requested_no_drift)

    if baseline_stats is None:
        # baseline missing: do NOT EMA when NoDrift requested
        if requested_no_drift:
            stable_hw_state["_force_ref_update_mode"] = "frozen"
            stable_hw_state["no_drift_effective"] = True
            stable_hw_state["hw_ref_source"] = "missing_baseline_frozen_init_on_first_observation"
            # leave refs as None => first valid observation initializes once
            stable_hw_state.setdefault("hw_refs", {})
            stable_hw_state["hw_refs"].setdefault("latency_ref_ms", None)
            stable_hw_state["hw_refs"].setdefault("energy_ref_mj", None)
            stable_hw_state["hw_refs"].setdefault("mem_ref_mb", None)
            stable_hw_state["hw_refs"].setdefault("comm_ref", None)
            return

        # NoDrift not requested: allow legacy behavior (EMA)
        stable_hw_state["_force_ref_update_mode"] = "ema"
        stable_hw_state["no_drift_effective"] = False
        stable_hw_state["hw_ref_source"] = "missing_baseline_ema"
        stable_hw_state.setdefault("hw_refs", {})
        stable_hw_state["hw_refs"].setdefault(
            "latency_ref_ms", float(getattr(getattr(cfg, "hw", None), "latency_ref_ms", 1.0) or 1.0)
        )
        stable_hw_state["hw_refs"].setdefault(
            "energy_ref_mj", float(getattr(getattr(cfg, "hw", None), "energy_ref_mj", 1.0) or 1.0)
        )
        stable_hw_state["hw_refs"].setdefault(
            "mem_ref_mb", float(getattr(getattr(cfg, "hw", None), "mem_ref_mb", 1.0) or 1.0)
        )
        stable_hw_state["hw_refs"].setdefault(
            "comm_ref", float(getattr(getattr(cfg, "hw", None), "comm_ref", 1.0) or 1.0)
        )
        stable_hw_state["ref_T"] = stable_hw_state["hw_refs"].get("latency_ref_ms")
        stable_hw_state["ref_E"] = stable_hw_state["hw_refs"].get("energy_ref_mj")
        stable_hw_state["ref_M"] = stable_hw_state["hw_refs"].get("mem_ref_mb")
        stable_hw_state["ref_C"] = stable_hw_state["hw_refs"].get("comm_ref")
        return

    # baseline present => init from baseline
    refs = extract_hw_refs_from_baseline(baseline_stats)
    stable_hw_state.setdefault("hw_refs", {})
    stable_hw_state["hw_refs"].update(refs)
    stable_hw_state["hw_ref_source"] = "baseline_stats"
    if requested_no_drift:
        stable_hw_state["_force_ref_update_mode"] = "frozen"
        stable_hw_state["no_drift_effective"] = True
    else:
        stable_hw_state["_force_ref_update_mode"] = "ema"
        stable_hw_state["no_drift_effective"] = False
    stable_hw_state["ref_T"] = stable_hw_state["hw_refs"].get("latency_ref_ms")
    stable_hw_state["ref_E"] = stable_hw_state["hw_refs"].get("energy_ref_mj")
    stable_hw_state["ref_M"] = stable_hw_state["hw_refs"].get("mem_ref_mb")
    stable_hw_state["ref_C"] = stable_hw_state["hw_refs"].get("comm_ref")


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


def update_hw_refs_from_stats(
    cfg: Any,
    stable_hw_state: Dict[str, Any],
    latest_hw_stats: Dict[str, Any],
    stable_hw_cfg: Any = None,
) -> None:
    """
    v5.4 contract:
      - If NoDrift requested => refs must not update (no ref_update).
      - Any downgrade to EMA while NoDrift requested is a hard contract violation.
    """
    stable_hw_cfg = stable_hw_cfg if stable_hw_cfg is not None else getattr(cfg, "stable_hw", None)
    nd_cfg = _get_no_drift_cfg(cfg, stable_hw_cfg=stable_hw_cfg)
    no_drift_enabled = _as_enabled(nd_cfg, default=False)
    ref_update_mode = str(stable_hw_state.get("_force_ref_update_mode", "ema")).lower().strip()

    # CONTRACT SEAL: NoDrift requested => MUST NOT update refs (no hidden fallback)
    if no_drift_enabled and ref_update_mode != "frozen":
        raise RuntimeError(
            f"[SPEC v5.4] no_drift.enabled=True requires ref_update_mode='frozen'. "
            f"Got ref_update_mode='{ref_update_mode}'. Ref updates would violate NoDrift."
        )

    effective_no_drift = bool(no_drift_enabled)  # once requested, it's effectively enforced (or we crash above)

    if effective_no_drift:
        # guarantee: no ref update happens
        stable_hw_state["no_drift_enabled"] = True
        stable_hw_state["no_drift_effective"] = True
        return

    # Otherwise, EMA update allowed (legacy)
    stable_hw_state["no_drift_effective"] = False
    refs = stable_hw_state.get("hw_refs", {}) or {}
    if not refs:
        stable_hw_state["hw_refs"] = refs

    # Initialize-once if missing
    def _init_if_none(k: str, v: Optional[float]) -> None:
        if refs.get(k, None) is None and v is not None and math.isfinite(float(v)) and float(v) > 0.0:
            refs[k] = float(v)

    _init_if_none("latency_ref_ms", latest_hw_stats.get("latency_ref_ms", latest_hw_stats.get("latency_ms", None)))
    _init_if_none("energy_ref_mj", latest_hw_stats.get("energy_ref_mj", latest_hw_stats.get("energy_mj", None)))
    _init_if_none("mem_ref_mb", latest_hw_stats.get("mem_ref_mb", latest_hw_stats.get("mem_mb", None)))
    _init_if_none("comm_ref", latest_hw_stats.get("comm_ref", latest_hw_stats.get("comm_norm", None)))

    # EMA update
    alpha = float(getattr(getattr(stable_hw_cfg, "no_drift", None), "ema_alpha", 0.05) or 0.05) if stable_hw_cfg is not None else 0.05
    for rk, obs_key in [
        ("latency_ref_ms", "latency_ms"),
        ("energy_ref_mj", "energy_mj"),
        ("mem_ref_mb", "mem_mb"),
        ("comm_ref", "comm_norm"),
    ]:
        obs = latest_hw_stats.get(obs_key, None)
        if obs is None:
            continue
        try:
            obs_f = float(obs)
        except Exception:
            continue
        if not math.isfinite(obs_f) or obs_f <= 0.0:
            continue
        cur = refs.get(rk, None)
        if cur is None or not math.isfinite(float(cur)) or float(cur) <= 0.0:
            refs[rk] = obs_f
        else:
            refs[rk] = (1.0 - alpha) * float(cur) + alpha * obs_f

    stable_hw_state["ref_T"] = refs.get("latency_ref_ms")
    stable_hw_state["ref_E"] = refs.get("energy_ref_mj")
    stable_hw_state["ref_M"] = refs.get("mem_ref_mb")
    stable_hw_state["ref_C"] = refs.get("comm_ref")
    stable_hw_state["latency_ref_ms"] = refs.get("latency_ref_ms")
    stable_hw_state["energy_ref_mj"] = refs.get("energy_ref_mj")
    stable_hw_state["mem_ref_mb"] = refs.get("mem_ref_mb")
    stable_hw_state["comm_ref"] = refs.get("comm_ref")
