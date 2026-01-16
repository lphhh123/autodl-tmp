from __future__ import annotations

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
    base_acc = _get_baseline_acc_from_stats(d)
    if base_acc is not None:
        stable_hw_state["baseline_acc"] = float(base_acc)
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


def stable_hw_schedule(epoch: int, cfg: dict, state: dict) -> float:
    """
    Epoch-based schedule (SPEC):
      warmup_epochs: 0 -> lambda_hw_min
      ramp_epochs  : linear lambda_hw_min -> lambda_hw_max
      stabilize_epochs: hold at lambda_hw_max
      decay_epochs : linear lambda_hw_max -> lambda_hw_min
    """
    sch = (cfg or {}).get("lambda_hw_schedule", {}) or {}
    lam_min = float(sch.get("lambda_hw_min", 0.0) or 0.0)
    lam_max = float(sch.get("lambda_hw_max", 0.0) or 0.0)

    warm = int(sch.get("warmup_epochs", 0) or 0)
    ramp = int(sch.get("ramp_epochs", 0) or 0)
    stab = int(sch.get("stabilize_epochs", 0) or 0)
    deca = int(sch.get("decay_epochs", 0) or 0)

    e = int(epoch)
    if warm > 0 and e < warm:
        lam = lam_min
    else:
        e2 = e - warm
        if ramp > 0 and e2 < ramp:
            t = float(e2) / float(max(1, ramp))
            lam = lam_min + t * (lam_max - lam_min)
        else:
            e3 = e2 - ramp
            if stab > 0 and e3 < stab:
                lam = lam_max
            else:
                e4 = e3 - stab
                if deca > 0 and e4 < deca:
                    t = float(e4) / float(max(1, deca))
                    lam = lam_max + t * (lam_min - lam_max)
                else:
                    lam = lam_min

    # apply guard scaling / disable
    scale = float(state.get("lambda_hw_scale", 1.0) or 1.0)
    if bool(state.get("hw_disabled", False)):
        lam = 0.0
    else:
        lam = lam * max(0.0, scale)

    state["lambda_hw_base"] = float(lam_min if lam_max == 0 else lam)
    state["lambda_hw_after_schedule"] = float(lam)
    return float(lam)


def apply_accuracy_guard(epoch: int, stable_hw_cfg: dict, state: dict, current_metric: float) -> dict:
    """
    SPEC accuracy_guard:
      - metric: (we pass current_metric in)
      - epsilon_drop: violation if (ref_metric < baseline_metric - epsilon_drop)
      - optional EMA
      - on_violate: scale_lambda_hw, freeze_rho_epochs, max_violations, disable_hw_after_max_violations
    Must populate state keys for trainer logging:
      baseline_acc, current_acc, current_acc_ema, epsilon_drop,
      violate_streak, guard_triggered, lambda_hw_scale, hw_disabled, rho_frozen_until_epoch
    """
    guard = (stable_hw_cfg or {}).get("accuracy_guard", {}) or {}
    eps = float(guard.get("epsilon_drop", 0.0) or 0.0)
    use_ema = bool(guard.get("use_ema", True))
    ema_beta = float(guard.get("ema_beta", 0.9) or 0.9)

    onv = (guard.get("on_violate", {}) or {})
    scale_mul = float(onv.get("scale_lambda_hw", 1.5) or 1.5)
    freeze_epochs = int(onv.get("freeze_rho_epochs", 1) or 1)
    max_viol = int(onv.get("max_violations", 2) or 2)
    disable_after = bool(onv.get("disable_hw_after_max_violations", False))

    cur = float(current_metric)
    state["current_acc"] = cur
    state["epsilon_drop"] = eps

    base = state.get("baseline_acc", None)
    if base is None:
        state["baseline_acc"] = cur
        base = cur

    if use_ema:
        prev = state.get("current_acc_ema", None)
        ema = cur if prev is None else (ema_beta * float(prev) + (1.0 - ema_beta) * cur)
        state["current_acc_ema"] = float(ema)
        ref = float(ema)
    else:
        state["current_acc_ema"] = None
        ref = float(cur)

    violate = (ref < float(base) - eps)
    state["acc_drop"] = float((float(base) - eps) - ref) if violate else 0.0
    if violate:
        streak = int(state.get("violate_streak", 0) or 0) + 1
        state["violate_streak"] = streak
        state["guard_triggered"] = True

        prev_scale = float(state.get("lambda_hw_scale", 1.0) or 1.0)
        state["lambda_hw_scale"] = float(prev_scale * max(1.0, scale_mul))

        state["rho_frozen_until_epoch"] = int(epoch + max(1, freeze_epochs))

        if disable_after and streak >= max_viol:
            state["hw_disabled"] = True
    else:
        state["violate_streak"] = 0
        state["guard_triggered"] = False

    state["hw_disabled"] = bool(state.get("hw_disabled", False))
    state["lambda_hw_scale"] = float(state.get("lambda_hw_scale", 1.0) or 1.0)

    return state


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
