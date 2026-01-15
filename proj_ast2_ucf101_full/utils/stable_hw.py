from __future__ import annotations

from typing import Any, Dict

from utils.config_utils import get_nested


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


def stable_hw_schedule(
    stable_hw_cfg: Any,
    state: Dict,
    epoch: int,
    step_in_epoch: int = 0,
    steps_per_epoch: int = 1,
) -> Dict:
    """
    Apply warmup->ramp->stabilize schedule and clamp.
    Reads:
      stable_hw.lambda_hw_schedule:
        enabled, warmup_epochs, ramp_epochs, stabilize_epochs,
        lambda_hw_max, lambda_hw_min,
        clamp_min, clamp_max
    """
    if stable_hw_cfg is None:
        return state

    sched = getattr(stable_hw_cfg, "lambda_hw_schedule", None)
    if sched is None or not bool(getattr(sched, "enabled", True)):
        return state

    warm = int(getattr(sched, "warmup_epochs", 0) or 0)
    ramp = int(getattr(sched, "ramp_epochs", 0) or 0)
    stab = int(getattr(sched, "stabilize_epochs", 0) or 0)
    _ = (step_in_epoch, steps_per_epoch, stab)

    lam_max = float(getattr(sched, "lambda_hw_max", 0.0) or 0.0)
    lam_min = float(getattr(sched, "lambda_hw_min", 0.0) or 0.0)

    clamp_min = float(getattr(sched, "clamp_min", lam_min) or lam_min)
    clamp_max = float(getattr(sched, "clamp_max", lam_max) or lam_max)

    e = int(epoch)

    if e < warm:
        phase = "warmup"
        lam = 0.0
    elif e < warm + ramp:
        phase = "ramp"
        t = (e - warm) / max(1, ramp)
        lam = lam_min + (lam_max - lam_min) * float(t)
    else:
        phase = "stabilize"
        lam = lam_max

    lam = max(float(clamp_min), min(float(clamp_max), float(lam)))

    state["lambda_hw"] = float(lam)
    state["schedule_phase"] = phase
    state["schedule_epoch"] = e
    return state


def apply_accuracy_guard(stable_hw_cfg, state: dict, acc_now: float, logger=None) -> dict:
    """
    SPEC/Config-aligned accuracy guard.
    Reads keys exactly from:
      stable_hw.accuracy_guard:
        metric: "acc1"
        use_ema: true/false
        ema_beta: 0.98
        epsilon_drop: 0.02
        max_consecutive: 3
        on_violate:
          scale_lambda_hw: 0.5
          freeze_rho_epochs: 1
          disable_hw_after_max_violations: false
          recover:
            enable: true
            patience_epochs: 1
            restore_lambda_hw: true
    """
    if stable_hw_cfg is None:
        return state

    guard = getattr(stable_hw_cfg, "accuracy_guard", None)
    if guard is None:
        return state

    use_ema = bool(getattr(guard, "use_ema", True))
    ema_beta = float(getattr(guard, "ema_beta", 0.98) or 0.98)
    epsilon_drop = float(getattr(guard, "epsilon_drop", 0.02) or 0.02)
    max_consecutive = int(getattr(guard, "max_consecutive", 3) or 3)

    onv = getattr(guard, "on_violate", None)
    scale_lambda_hw = float(getattr(onv, "scale_lambda_hw", 0.5) or 0.5) if onv else 0.5
    freeze_rho_epochs = int(getattr(onv, "freeze_rho_epochs", 1) or 1) if onv else 1
    disable_hw_after = bool(getattr(onv, "disable_hw_after_max_violations", False)) if onv else False

    recover = getattr(onv, "recover", None) if onv else None
    recover_enable = bool(getattr(recover, "enable", True)) if recover else True
    recover_patience = int(getattr(recover, "patience_epochs", 1) or 1) if recover else 1
    recover_restore_lambda = bool(getattr(recover, "restore_lambda_hw", True)) if recover else True

    acc_best = state.get("acc_best", None)
    if acc_best is None:
        state["acc_best"] = float(acc_now)
        state["acc_ema"] = float(acc_now)
        state["acc_drop"] = 0.0
        state["acc_violations"] = 0
        state["rho_freeze_epochs_left"] = 0
        state["hw_disabled"] = False
        state["recover_wait"] = 0
        state["lambda_hw_pre_guard"] = float(state.get("lambda_hw", 0.0))
        return state

    if use_ema:
        state["acc_ema"] = float(ema_beta) * float(state.get("acc_ema", acc_now)) + (1.0 - float(ema_beta)) * float(acc_now)
        acc_ref = float(state["acc_ema"])
    else:
        acc_ref = float(acc_now)

    state["acc_best"] = max(float(state["acc_best"]), float(acc_now))
    best = float(state["acc_best"])

    drop = max(0.0, best - acc_ref)
    state["acc_drop"] = float(drop)

    violated = drop > float(epsilon_drop)

    if not violated:
        state["acc_violations"] = 0
        if recover_enable:
            state["recover_wait"] = int(state.get("recover_wait", 0)) + 1
            if int(state["recover_wait"]) >= int(recover_patience):
                if recover_restore_lambda and ("lambda_hw_pre_guard" in state):
                    state["lambda_hw"] = float(state["lambda_hw_pre_guard"])
                state["recover_wait"] = 0
        return state

    state["recover_wait"] = 0
    state["acc_violations"] = int(state.get("acc_violations", 0)) + 1

    if "lambda_hw_pre_guard" not in state:
        state["lambda_hw_pre_guard"] = float(state.get("lambda_hw", 0.0))

    cur = float(state.get("lambda_hw", 0.0))
    newv = max(0.0, cur * float(scale_lambda_hw))
    state["lambda_hw"] = float(newv)

    state["rho_freeze_epochs_left"] = max(int(state.get("rho_freeze_epochs_left", 0)), int(freeze_rho_epochs))

    if disable_hw_after and int(state["acc_violations"]) >= int(max_consecutive):
        state["hw_disabled"] = True
        state["lambda_hw"] = 0.0

    if logger is not None:
        logger.info(
            f"[ACC_GUARD] violated drop={drop:.4f} > eps={epsilon_drop:.4f} | "
            f"lambda_hw {cur:.4g}->{state['lambda_hw']:.4g} | "
            f"viol={state['acc_violations']}/{max_consecutive} | "
            f"rho_freeze_left={state['rho_freeze_epochs_left']} | hw_disabled={state.get('hw_disabled', False)}"
        )
    return state


def update_hw_refs_from_stats(stable_hw_cfg: Any, stable_hw_state: Dict, stats: Dict) -> Dict:
    """
    Epoch-end ref update.
    - If ref_source == baseline_stats: never EMA-update.
    - If ref_source == ema: EMA-update with beta=stable_hw.hw_refs_update.ema_beta.
    - Bootstrap if not refs_inited.
    """
    if stable_hw_state is None:
        return stable_hw_state
    if not _stable_hw_enabled(stable_hw_cfg):
        return stable_hw_state

    if not bool(stable_hw_state.get("refs_inited", False)):
        ref_mode = str(getattr(stable_hw_cfg, "ref_mode", "ema")).lower()
        if ref_mode == "dense_baseline":
            base_cfg = getattr(stable_hw_cfg, "dense_baseline", None)
            base_path = str(getattr(base_cfg, "baseline_stats_path", "")) if base_cfg else ""
            bs = _load_baseline_stats(base_path)
            if isinstance(bs, dict):
                refs = _get_refs_from_stats(bs)
                for k, v in refs.items():
                    if v is not None:
                        stable_hw_state[k] = float(v)
                stable_hw_state["refs_inited"] = True
                stable_hw_state["ref_source"] = f"dense_baseline:{base_path}"

    ref_up = get_nested(stable_hw_cfg, "hw_refs_update", None)
    ref_source = str(get_nested(ref_up, "ref_source", "ema")) if ref_up else "ema"
    beta = float(get_nested(ref_up, "ema_beta", 0.95)) if ref_up else 0.95

    # baseline_stats source => do not update here
    if ref_source == "baseline_stats" and bool(stable_hw_state.get("refs_inited", False)):
        return stable_hw_state

    def _f(key: str, default: float = 0.0) -> float:
        v = stats.get(key, default)
        try:
            import torch
            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item())
        except Exception:
            pass
        return float(v)

    eps = float(get_nested(getattr(stable_hw_cfg, "normalize", None), "eps", 1e-6) or 1e-6)
    cur_T = max(eps, _f("latency_ms", 0.0))
    cur_E = max(eps, _f("energy_mj", 0.0))
    cur_M = max(eps, _f("mem_mb", 0.0))
    cur_C = max(0.0, _f("comm_ms", 0.0))

    if not bool(stable_hw_state.get("refs_inited", False)):
        stable_hw_state["ref_T"] = cur_T
        stable_hw_state["ref_E"] = cur_E
        stable_hw_state["ref_M"] = cur_M
        stable_hw_state["ref_C"] = cur_C
        stable_hw_state["refs_inited"] = True
        stable_hw_state["ref_source"] = stable_hw_state.get("ref_source", "bootstrap_first_obs")
        return stable_hw_state

    stable_hw_state["ref_T"] = beta * float(stable_hw_state.get("ref_T", cur_T)) + (1 - beta) * cur_T
    stable_hw_state["ref_E"] = beta * float(stable_hw_state.get("ref_E", cur_E)) + (1 - beta) * cur_E
    stable_hw_state["ref_M"] = beta * float(stable_hw_state.get("ref_M", cur_M)) + (1 - beta) * cur_M
    stable_hw_state["ref_C"] = beta * float(stable_hw_state.get("ref_C", cur_C)) + (1 - beta) * cur_C
    return stable_hw_state
