from __future__ import annotations

import math
from typing import Any, Dict


def _load_baseline_stats(path: str):
    import json
    import os

    if not path:
        return None
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_baseline_acc_from_stats(stats: dict):
    for k in ("val_acc1", "val_acc", "acc1", "acc"):
        if k in stats:
            try:
                return float(stats[k])
            except Exception:
                pass
    return None


def _get_refs_from_stats(stats: dict):
    def _pick(*keys, default=None):
        for k in keys:
            if k in stats:
                try:
                    return float(stats[k])
                except Exception:
                    pass
        return default

    return {
        "ref_T": _pick("latency_ms", "T_ms", "ref_T"),
        "ref_E": _pick("energy_mj", "E_mj", "ref_E"),
        "ref_M": _pick("mem_mb", "M_mb", "ref_M"),
        "ref_C": _pick("comm_ms", "C_ms", "ref_C"),
    }


def _stable_hw_enabled(stable_hw_cfg) -> bool:
    try:
        return bool(getattr(stable_hw_cfg, "enabled", False))
    except Exception:
        try:
            return bool(stable_hw_cfg.get("enabled", False))
        except Exception:
            return False


def _extract_baseline_refs(baseline_dict: dict) -> tuple[float, float, float]:
    """
    Return (ref_latency_ms, ref_mem_mb, ref_comm_ms) from a baseline_stats.json-like dict.
    Accept multiple key aliases to be robust.
    """

    def pick(keys):
        for k in keys:
            if k in baseline_dict and baseline_dict[k] is not None:
                return float(baseline_dict[k])
        return None

    ref_latency = pick(["ref_latency_ms", "latency_ms", "avg_latency_ms", "mean_latency_ms", "total_latency_ms"])
    ref_mem = pick(["ref_mem_mb", "mem_mb", "avg_mem_mb", "mean_mem_mb", "peak_mem_mb"])
    ref_comm = pick(["ref_comm_ms", "comm_ms", "avg_comm_ms", "mean_comm_ms"])

    if ref_latency is None or ref_mem is None or ref_comm is None:
        raise ValueError(
            f"baseline_stats missing required keys. got keys={sorted(list(baseline_dict.keys()))}. "
            f"need latency({['latency_ms','avg_latency_ms',...]}), mem({['mem_mb','avg_mem_mb',...]}), "
            f"comm({['comm_ms','avg_comm_ms',...]})"
        )
    return ref_latency, ref_mem, ref_comm


def init_hw_refs_from_baseline(
    stable_hw_state: dict,
    stable_hw_cfg,
    baseline_stats_path: str,
) -> dict:
    """
    Initialize stable_hw_state refs from baseline_stats.json.
    No-op if already refs_inited=True.
    """
    if stable_hw_state.get("refs_inited", False):
        return stable_hw_state
    if not _stable_hw_enabled(stable_hw_cfg):
        return stable_hw_state

    import json
    from pathlib import Path

    p = Path(baseline_stats_path)
    if not p.exists():
        raise FileNotFoundError(f"[StableHW] baseline_stats_path not found: {p}")

    d = json.loads(p.read_text(encoding="utf-8"))
    ref_latency, ref_mem, ref_comm = _extract_baseline_refs(d)

    stable_hw_state["ref_latency_ms"] = float(ref_latency)
    stable_hw_state["ref_mem_mb"] = float(ref_mem)
    stable_hw_state["ref_comm_ms"] = float(ref_comm)
    stable_hw_state.setdefault("ref_T", float(ref_latency))
    stable_hw_state.setdefault("ref_M", float(ref_mem))
    stable_hw_state.setdefault("ref_C", float(ref_comm))
    stable_hw_state["refs_inited"] = True
    stable_hw_state["ref_source"] = "baseline_stats"
    stable_hw_state["ref_path"] = str(p)
    return stable_hw_state


def _get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_path(obj: Any, path: str, default=None):
    cur = obj
    for k in path.split("."):
        cur = _get(cur, k, None)
        if cur is None:
            return default
    return cur


def stable_hw_schedule(epoch: int, stable_hw_cfg, stable_hw_state: Dict[str, Any], total_epochs: int | None = None):
    """
    Schema-aligned schedule:
      stable_hw.lambda_hw (base)
      stable_hw.lambda_hw_schedule.{enabled, lambda_hw_max, lambda_hw_min,
                                   warmup_ratio, ramp_ratio, hold_ratio,
                                   cosine_decay, clamp_min, clamp_max}
    """
    if stable_hw_state is None:
        return

    base_lam = float(_get(stable_hw_cfg, "lambda_hw", 0.0) or 0.0)
    sched = _get(stable_hw_cfg, "lambda_hw_schedule", None)

    enabled = bool(_get(sched, "enabled", False)) if sched is not None else False
    if not enabled:
        stable_hw_state["lambda_hw"] = base_lam
        stable_hw_state["schedule_phase"] = "disabled"
        return

    if total_epochs is None:
        total_epochs = int(stable_hw_state.get("total_epochs", 0) or 0)
    if total_epochs <= 0:
        total_epochs = int(_get(sched, "total_epochs", 0) or 0)
    if total_epochs <= 0:
        total_epochs = 1

    lam_min = float(_get(sched, "lambda_hw_min", 0.0) or 0.0)
    lam_max = float(_get(sched, "lambda_hw_max", base_lam) or base_lam)
    lam_max = max(lam_max, lam_min)

    warmup_ratio = float(_get(sched, "warmup_ratio", 0.0) or 0.0)
    ramp_ratio = float(_get(sched, "ramp_ratio", 0.0) or 0.0)
    hold_ratio = float(_get(sched, "hold_ratio", 0.0) or 0.0)
    cosine_decay = bool(_get(sched, "cosine_decay", False))

    clamp_min = _get(sched, "clamp_min", None)
    clamp_max = _get(sched, "clamp_max", None)
    if clamp_min is not None:
        lam_min = max(lam_min, float(clamp_min))
    if clamp_max is not None:
        lam_max = min(lam_max, float(clamp_max))
        lam_max = max(lam_max, lam_min)

    e = int(epoch)
    e = max(0, min(e, total_epochs - 1))

    w = int(round(total_epochs * warmup_ratio))
    r = int(round(total_epochs * ramp_ratio))
    h = int(round(total_epochs * hold_ratio))
    w = max(0, w)
    r = max(0, r)
    h = max(0, h)

    if w + r + h > total_epochs:
        overflow = (w + r + h) - total_epochs
        cut_h = min(h, overflow)
        h -= cut_h
        overflow -= cut_h
        cut_r = min(r, overflow)
        r -= cut_r
        overflow -= cut_r
        cut_w = min(w, overflow)
        w -= cut_w
        overflow -= cut_w

    warm_end = w
    ramp_end = w + r
    hold_end = w + r + h

    if e < warm_end:
        t = (e + 1) / max(1, warm_end)
        lam = lam_min * t
        phase = "warmup"
    elif e < ramp_end:
        t = (e - warm_end + 1) / max(1, (ramp_end - warm_end))
        lam = lam_min + (lam_max - lam_min) * t
        phase = "ramp"
    elif e < hold_end:
        lam = lam_max
        phase = "hold"
    else:
        if cosine_decay and (total_epochs - hold_end) > 0:
            t = (e - hold_end) / max(1, (total_epochs - hold_end - 1))
            lam = lam_min + 0.5 * (lam_max - lam_min) * (1.0 + math.cos(math.pi * t))
            phase = "cosine_decay"
        else:
            lam = lam_max
            phase = "post_hold"

    stable_hw_state["lambda_hw"] = float(max(lam_min, min(lam, lam_max)))
    stable_hw_state["schedule_phase"] = phase


def apply_accuracy_guard(
    stable_hw_cfg,
    stable_hw_state: Dict[str, Any],
    acc1: float,
    epoch: int,
):
    """
    Schema-aligned:
      stable_hw.accuracy_guard.enabled
      stable_hw.accuracy_guard.target_acc1
      stable_hw.accuracy_guard.margin
      stable_hw.accuracy_guard.use_ema
      stable_hw.accuracy_guard.ema_beta
      stable_hw.accuracy_guard.on_violate.max_consecutive
      stable_hw.accuracy_guard.on_violate.scale_lambda_hw
      stable_hw.accuracy_guard.on_violate.freeze_rho_epochs
    Effects:
      - updates stable_hw_state["acc_ema"]
      - tracks stable_hw_state["consecutive_violations"]
      - scales stable_hw_state["lambda_hw"]
      - sets stable_hw_state["rho_frozen_until_epoch"]
    """
    guard = _get(stable_hw_cfg, "accuracy_guard", None)
    if not guard or not bool(_get(guard, "enabled", False)):
        stable_hw_state["acc_drop"] = 0.0
        return stable_hw_state

    target = float(_get(guard, "target_acc1", 1.0) or 1.0)
    margin = float(_get(guard, "margin", 0.0) or 0.0)
    use_ema = bool(_get(guard, "use_ema", True))
    beta = float(_get(guard, "ema_beta", 0.9) or 0.9)

    onv = _get(guard, "on_violate", None)
    max_consec = int(_get(onv, "max_consecutive", 2) or 2)
    scale_lam = float(_get(onv, "scale_lambda_hw", 0.5) or 0.5)
    freeze_rho = int(_get(onv, "freeze_rho_epochs", 0) or 0)

    acc = float(acc1)
    if use_ema:
        prev = float(stable_hw_state.get("acc_ema", acc))
        ema = beta * prev + (1.0 - beta) * acc
        stable_hw_state["acc_ema"] = float(ema)
        acc_eff = float(ema)
    else:
        stable_hw_state["acc_ema"] = float(acc)
        acc_eff = float(acc)

    violate = acc_eff < (target - margin)
    stable_hw_state["acc_drop"] = float((target - margin) - acc_eff) if violate else 0.0

    consec = int(stable_hw_state.get("consecutive_violations", 0) or 0)
    consec = consec + 1 if violate else 0
    stable_hw_state["consecutive_violations"] = int(consec)

    if violate and consec >= max_consec:
        lam = float(stable_hw_state.get("lambda_hw", float(_get(stable_hw_cfg, "lambda_hw", 0.0) or 0.0)))
        lam = lam * max(0.0, min(1.0, scale_lam))
        sched = _get(stable_hw_cfg, "lambda_hw_schedule", None)
        lam_min = float(_get(sched, "lambda_hw_min", 0.0) or 0.0) if sched else 0.0
        lam_max = float(_get(sched, "lambda_hw_max", lam) or lam) if sched else lam
        lam = max(lam_min, min(lam, max(lam_min, lam_max)))
        stable_hw_state["lambda_hw"] = float(lam)

        if freeze_rho > 0:
            until = int(epoch) + int(freeze_rho)
            prev_until = int(stable_hw_state.get("rho_frozen_until_epoch", -1) or -1)
            stable_hw_state["rho_frozen_until_epoch"] = int(max(prev_until, until))

    return stable_hw_state


def update_hw_refs_from_stats(stable_hw_cfg, stable_hw_state: Dict[str, Any], epoch_hw_stats: Dict[str, float]):
    """
    Two-layer ref management:
      - stable_hw.normalize.{enabled, ref_source, refs{latency_ms,energy_mj,peak_temp_c}}
      - stable_hw.hw_refs_update.{enabled, update_every_epochs, ema_beta, clamp_factor_min, clamp_factor_max}
    """
    if stable_hw_state is None:
        return stable_hw_state

    normalize = _get(stable_hw_cfg, "normalize", None)
    if not normalize or not bool(_get(normalize, "enabled", True)):
        return stable_hw_state

    epoch = int(epoch_hw_stats.get("epoch", stable_hw_state.get("epoch", 0) or 0))

    refs = stable_hw_state.get("refs", None)
    if not isinstance(refs, dict):
        refs = {}
        stable_hw_state["refs"] = refs

    def _set_refs(lat, en, temp):
        refs["latency_ms"] = float(lat)
        refs["energy_mj"] = float(en)
        refs["peak_temp_c"] = float(temp)

    ref_source = str(_get(normalize, "ref_source", "first_epoch") or "first_epoch")

    if (not refs) or any(k not in refs for k in ("latency_ms", "energy_mj", "peak_temp_c")):
        if ref_source == "fixed":
            fixed = _get(normalize, "refs", None)
            _set_refs(
                _get(fixed, "latency_ms", epoch_hw_stats.get("latency_ms", 1.0) or 1.0),
                _get(fixed, "energy_mj", epoch_hw_stats.get("energy_mj", 1.0) or 1.0),
                _get(fixed, "peak_temp_c", epoch_hw_stats.get("peak_temp_c", 1.0) or 1.0),
            )
            stable_hw_state["refs_source"] = "fixed"
        elif ref_source == "dense_baseline":
            dense = epoch_hw_stats.get("dense_baseline", None)
            if isinstance(dense, dict):
                _set_refs(
                    dense.get("latency_ms", epoch_hw_stats.get("latency_ms", 1.0) or 1.0),
                    dense.get("energy_mj", epoch_hw_stats.get("energy_mj", 1.0) or 1.0),
                    dense.get("peak_temp_c", epoch_hw_stats.get("peak_temp_c", 1.0) or 1.0),
                )
            else:
                _set_refs(
                    epoch_hw_stats.get("latency_ms", 1.0) or 1.0,
                    epoch_hw_stats.get("energy_mj", 1.0) or 1.0,
                    epoch_hw_stats.get("peak_temp_c", 1.0) or 1.0,
                )
            stable_hw_state["refs_source"] = "dense_baseline"
        else:
            _set_refs(
                epoch_hw_stats.get("latency_ms", 1.0) or 1.0,
                epoch_hw_stats.get("energy_mj", 1.0) or 1.0,
                epoch_hw_stats.get("peak_temp_c", 1.0) or 1.0,
            )
            stable_hw_state["refs_source"] = "first_epoch"

    upd = _get(stable_hw_cfg, "hw_refs_update", None)
    if not upd or not bool(_get(upd, "enabled", False)):
        return stable_hw_state

    every = int(_get(upd, "update_every_epochs", 1) or 1)
    if every <= 0:
        every = 1
    if (epoch % every) != 0:
        return stable_hw_state

    beta = float(_get(upd, "ema_beta", 0.9) or 0.9)
    cf_min = float(_get(upd, "clamp_factor_min", 0.5) or 0.5)
    cf_max = float(_get(upd, "clamp_factor_max", 2.0) or 2.0)
    cf_min = max(0.0, min(cf_min, cf_max))

    cur_lat = float(epoch_hw_stats.get("latency_ms", refs.get("latency_ms", 1.0) or 1.0))
    cur_en = float(epoch_hw_stats.get("energy_mj", refs.get("energy_mj", 1.0) or 1.0))
    cur_tp = float(epoch_hw_stats.get("peak_temp_c", refs.get("peak_temp_c", 1.0) or 1.0))

    def _ema(prev, cur):
        return beta * prev + (1.0 - beta) * cur

    new_lat = _ema(float(refs["latency_ms"]), cur_lat)
    new_en = _ema(float(refs["energy_mj"]), cur_en)
    new_tp = _ema(float(refs["peak_temp_c"]), cur_tp)

    def _clamp(prev, val):
        lo = prev * cf_min
        hi = prev * cf_max
        return max(lo, min(val, hi))

    refs["latency_ms"] = float(_clamp(float(refs["latency_ms"]), new_lat))
    refs["energy_mj"] = float(_clamp(float(refs["energy_mj"]), new_en))
    refs["peak_temp_c"] = float(_clamp(float(refs["peak_temp_c"]), new_tp))
    return stable_hw_state
