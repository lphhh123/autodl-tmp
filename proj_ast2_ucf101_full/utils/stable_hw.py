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


def stable_hw_schedule(epoch: int, stable_hw_cfg: Any, state: Dict) -> None:
    if not bool(getattr(stable_hw_cfg, "enabled", False)):
        state["lambda_hw_base"] = 0.0
        state["lambda_hw"] = 0.0
        return

    sched = getattr(stable_hw_cfg, "lambda_hw_schedule", None)
    lam_max = (
        float(getattr(sched, "lambda_hw_max", 0.0))
        if sched
        else float(state.get("lambda_hw", 0.0))
    )
    warmup_epochs = int(getattr(sched, "warmup_epochs", 0)) if sched else 0
    ramp_epochs = int(getattr(sched, "ramp_epochs", 0)) if sched else 0

    if warmup_epochs > 0 and epoch < warmup_epochs:
        lam_base = 0.0
        phase = "warmup"
    else:
        t = max(0, epoch - warmup_epochs)
        if ramp_epochs <= 0:
            lam_base = lam_max
            phase = "steady"
        else:
            frac = min(1.0, float(t) / float(ramp_epochs))
            lam_base = lam_max * frac
            phase = "ramp" if frac < 1.0 else "steady"

    state["lambda_hw_base"] = float(lam_base)
    state["schedule_phase"] = str(phase)

    scale = float(state.get("lambda_hw_scale", 1.0))
    if bool(state.get("hw_disabled", False)):
        state["lambda_hw"] = 0.0
    else:
        state["lambda_hw"] = float(lam_base) * scale


def apply_accuracy_guard(acc1: float, stable_hw_cfg: Any, state: Dict) -> None:
    guard = getattr(stable_hw_cfg, "accuracy_guard", None)
    if not guard or not bool(getattr(guard, "enabled", True)):
        return

    eps_drop = float(getattr(guard, "epsilon_drop", 0.0))
    max_consecutive = int(getattr(guard, "max_consecutive_violations", 2))
    scale_down = float(getattr(guard, "lambda_scale_down", 0.5))
    freeze_epochs = int(getattr(guard, "freeze_rho_epochs", 0))

    baseline = state.get("baseline_acc", None)
    if baseline is None:
        base_cfg = getattr(stable_hw_cfg, "dense_baseline", None)
        base_path = str(getattr(base_cfg, "baseline_stats_path", "")) if base_cfg else ""
        bs = _load_baseline_stats(base_path)
        if isinstance(bs, dict):
            bacc = _get_baseline_acc_from_stats(bs)
            if bacc is not None:
                baseline = float(bacc)

    if baseline is None:
        baseline = float(acc1)

    beta = float(getattr(guard, "acc_ema_beta", 0.9))
    acc_ema = float(state.get("current_acc_ema", baseline))
    acc_ema = beta * acc_ema + (1.0 - beta) * float(acc1)

    acc_drop = max(0.0, float(baseline) - float(acc_ema))

    state["baseline_acc"] = float(baseline)
    state["current_acc"] = float(acc1)
    state["current_acc_ema"] = float(acc_ema)
    state["acc_drop"] = float(acc_drop)
    state["epsilon_drop"] = float(eps_drop)

    violated = acc_drop > eps_drop
    streak = int(state.get("violate_streak", 0))

    if violated:
        streak += 1
        scale = float(state.get("lambda_hw_scale", 1.0))
        scale = max(0.0, scale * float(scale_down))
        state["lambda_hw_scale"] = float(scale)
        state["guard_triggered"] = True

        if freeze_epochs > 0:
            cur_epoch = int(state.get("epoch", 0))
            state["rho_frozen_until_epoch"] = max(
                int(state.get("rho_frozen_until_epoch", 0)), cur_epoch + freeze_epochs
            )

        if streak >= max_consecutive:
            state["hw_disabled"] = True
            state["lambda_hw_scale"] = 0.0

    else:
        streak = 0
        if bool(state.get("hw_disabled", False)):
            recover = bool(getattr(guard, "recover_enable_hw", True))
            if recover:
                state["hw_disabled"] = False
                state["lambda_hw_scale"] = float(getattr(guard, "recover_lambda_scale", 0.25))

    state["violate_streak"] = int(streak)

    lam_base = float(state.get("lambda_hw_base", state.get("lambda_hw", 0.0)))
    if bool(state.get("hw_disabled", False)):
        state["lambda_hw"] = 0.0
    else:
        state["lambda_hw"] = lam_base * float(state.get("lambda_hw_scale", 1.0))

    state["lambda_hw_after_guard"] = float(state.get("lambda_hw", 0.0))


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
