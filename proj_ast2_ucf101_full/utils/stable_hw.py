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

    state["lambda_hw_base"] = float(lam)
    state["schedule_phase"] = str(phase)
    return float(lam)


def _locked_acc_ref_init(stable_hw_cfg: Any, state: Dict[str, Any]) -> None:
    """
    v5.1 LockedAccRef:
      - Prefer baseline stats file if provided.
      - Else lock from warmup-best (val) once after warmup window.
    """
    lock = _get(stable_hw_cfg, "locked_acc_ref", None) or {}
    # canonical keys
    baseline_path = _get(lock, "baseline_stats_path", None) or _get(stable_hw_cfg, "baseline_stats_path", None)
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
    stable_hw_cfg: Any,
    state: Dict[str, Any],
    val_acc1: Optional[float],
    *,
    has_val_this_epoch: bool = True,
    train_acc1_ema: Optional[float] = None,
) -> GuardDecision:
    """
    v5.1 Acc-First Hard Gating:
      - acc_used: prefer val; fallback to val_last; else train_ema
      - acc_ref: LockedAccRef (from baseline file or warmup-best and then frozen)
      - if acc_used < acc_ref - eps_drop: RECOVERY => lambda_hw_effective=0 and allow_discrete_updates=False
    """
    _locked_acc_ref_init(stable_hw_cfg, state)

    guard = _get(stable_hw_cfg, "accuracy_guard", None) or {}
    eps_drop = _as_float(_get(guard, "epsilon_drop", 0.01), 0.01)

    # decide acc_used
    acc_used = None
    acc_src = "unset"
    if has_val_this_epoch and val_acc1 is not None:
        acc_used = float(val_acc1)
        acc_src = "val"
        state["val_acc1_last"] = float(val_acc1)
    else:
        if state.get("val_acc1_last") is not None:
            acc_used = float(state["val_acc1_last"])
            acc_src = "val_last"
        elif train_acc1_ema is not None:
            acc_used = float(train_acc1_ema)
            acc_src = "train_ema"

    # update warmup best + maybe lock
    _update_warmup_best(epoch, stable_hw_cfg, state, float(val_acc1) if val_acc1 is not None else None)

    acc_ref = state.get("acc_ref", None)
    if acc_ref is None:
        # if we still can't lock, treat current as ref but lock immediately (prevents drift later)
        if acc_used is not None:
            state["acc_ref"] = float(acc_used)
            state["acc_ref_source"] = "fallback_first_acc"
            state["acc_ref_locked"] = True
            acc_ref = state["acc_ref"]
        else:
            acc_ref = 0.0

    acc_used_f = float(acc_used) if acc_used is not None else 0.0
    acc_ref_f = float(acc_ref)
    acc_drop = acc_ref_f - acc_used_f
    violate = (acc_drop > float(eps_drop))

    # schedule base
    lambda_hw_base = float(state.get("lambda_hw_base", state.get("lambda_hw", 0.0)))
    # v5.1: gating produces effective
    if violate:
        guard_mode = "RECOVERY"
        lambda_hw_eff = 0.0
        allow_discrete = False
        state["violate_streak"] = int(state.get("violate_streak", 0)) + 1
    else:
        guard_mode = "HW_OPT"
        lambda_hw_eff = lambda_hw_base
        allow_discrete = True
        state["violate_streak"] = 0

    state["guard_mode"] = str(guard_mode)
    state["epsilon_drop"] = float(eps_drop)
    state["acc_used"] = float(acc_used_f)
    state["acc_used_source"] = str(acc_src)
    state["acc_ref"] = float(acc_ref_f)
    state["acc_drop"] = float(acc_drop)
    state["guard_triggered"] = bool(violate)
    state["lambda_hw_effective"] = float(lambda_hw_eff)
    state["allow_discrete_updates"] = bool(allow_discrete)

    # keep legacy keys for compatibility but DO NOT use them in loss
    state["lambda_hw_after_guard"] = float(lambda_hw_eff)

    return GuardDecision(
        guard_mode=str(guard_mode),
        lambda_hw_base=float(lambda_hw_base),
        lambda_hw_effective=float(lambda_hw_eff),
        allow_discrete_updates=bool(allow_discrete),
        acc_used=float(acc_used_f),
        acc_used_source=str(acc_src),
        acc_ref=float(acc_ref_f),
        acc_drop=float(acc_drop),
        triggered=bool(violate),
    )


# ---- legacy helpers used elsewhere (keep minimal, no drift) ----

def init_hw_refs_from_baseline(state: Dict[str, Any], stable_hw_cfg: Any, baseline_path: str) -> Dict[str, Any]:
    """
    Keep existing behavior, but never overwrite acc_ref once locked.
    """
    state.setdefault("ref_path", baseline_path)
    state.setdefault("refs_inited", True)
    state.setdefault("ref_source", "baseline_path")
    return state


def update_hw_refs_from_stats(stable_hw_cfg: Any, state: Dict[str, Any], epoch_hw_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    v5.1: allowed to update hardware refs (T/E/mem) from stats, but NEVER acc_ref.
    """
    # Only update numeric hw refs if enabled
    ref_up = _get(stable_hw_cfg, "hw_refs_update", None)
    if not ref_up or not _as_bool(_get(ref_up, "enabled", False), False):
        return state

    # Accept common keys
    for k_src, k_dst in [
        ("latency_ms", "ref_latency_ms"),
        ("energy_j", "ref_energy_j"),
        ("mem_mb", "ref_mem_mb"),
        ("comm_mb", "ref_comm_mb"),
    ]:
        if k_src in epoch_hw_stats:
            try:
                state[k_dst] = float(epoch_hw_stats[k_src])
            except Exception:
                pass
    return state


def stable_hw_log_fields(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "guard_mode": state.get("guard_mode"),
        "lambda_hw_base": state.get("lambda_hw_base"),
        "lambda_hw_effective": state.get("lambda_hw_effective"),
        "allow_discrete_updates": state.get("allow_discrete_updates"),
        "acc_ref": state.get("acc_ref"),
        "acc_ref_locked": state.get("acc_ref_locked"),
        "acc_used": state.get("acc_used"),
        "acc_used_source": state.get("acc_used_source"),
        "acc_drop": state.get("acc_drop"),
        "schedule_phase": state.get("schedule_phase"),
        "violate_streak": state.get("violate_streak"),
    }
