from __future__ import annotations
import os
from typing import Any, Dict


REQUIRED_SIGNATURE_FIELDS = [
    "moves_enabled",
    "lookahead_k",
    "bandit_type",
    "policy_switch",
    "policy_switch_mode",
    "cache_enabled",
    "cache_key_schema_version",
    "acc_first_hard_gating_enabled",
    "locked_acc_ref_enabled",
    "acc_ref_source",
    "no_drift_enabled",
    "no_double_scale_enabled",
    "seed_global",
    "seed_problem",
    "config_fingerprint",
    "git_commit_or_version",
]


def _cfg_get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _enabled(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return bool(x)
    if x is None:
        return bool(default)
    v = _cfg_get(x, "enabled", None)
    return bool(default if v is None else v)


def get_git_commit_or_version() -> str:
    return (
        os.environ.get("GIT_COMMIT")
        or os.environ.get("GITHUB_SHA")
        or os.environ.get("VERSION")
        or "unknown"
    )


def build_run_signature(
    cfg: Any,
    *,
    cfg_hash: str,
    seed_global: int,
    seed_problem: int,
    mode: str,
) -> Dict[str, Any]:
    detailed_cfg = _cfg_get(cfg, "detailed_place", None)
    lookahead_cfg = _cfg_get(detailed_cfg, "lookahead", _cfg_get(cfg, "lookahead", {})) or {}
    policy_switch_cfg = _cfg_get(detailed_cfg, "policy_switch", _cfg_get(cfg, "policy_switch", {})) or {}

    action_families = _cfg_get(policy_switch_cfg, "action_families", None)
    moves_enabled = bool(action_families) if action_families is not None else bool(_cfg_get(cfg, "moves_enabled", False))
    lookahead_k = int(_cfg_get(lookahead_cfg, "topk", _cfg_get(lookahead_cfg, "k", 0) or 0))
    bandit_type = str(_cfg_get(policy_switch_cfg, "bandit_type", "eps_greedy"))
    policy_switch_enabled = _enabled(policy_switch_cfg, default=False)
    policy_switch_mode = str(_cfg_get(policy_switch_cfg, "mode", "bandit" if policy_switch_enabled else "none"))
    cache_size = int(_cfg_get(policy_switch_cfg, "cache_size", 0) or 0)
    cache_key_schema_version = str(_cfg_get(policy_switch_cfg, "cache_key_schema_version", "v5.4"))
    cache_enabled = bool(policy_switch_enabled and cache_size > 0)

    stable_hw_cfg = _cfg_get(cfg, "stable_hw", None)
    locked_cfg = _cfg_get(cfg, "locked_acc_ref", None)
    no_drift_cfg = _cfg_get(cfg, "no_drift", _cfg_get(stable_hw_cfg, "no_drift", None))
    no_double_cfg = _cfg_get(cfg, "no_double_scale", _cfg_get(stable_hw_cfg, "no_double_scale", None))

    locked_acc_ref_enabled = _enabled(locked_cfg, default=False)
    no_drift_enabled = _enabled(no_drift_cfg, default=True)
    no_double_scale_enabled = _enabled(no_double_cfg, default=True)

    acc_guard = _cfg_get(stable_hw_cfg, "accuracy_guard", None)
    controller = _cfg_get(acc_guard, "controller", None)
    acc_first_hard_gating_enabled = bool(_enabled(controller, default=False) or _enabled(stable_hw_cfg, default=False))

    acc_ref_source = "locked" if locked_acc_ref_enabled else str(_cfg_get(controller, "acc_ref_source", "online"))

    sig = {
        "moves_enabled": moves_enabled,
        "lookahead_k": lookahead_k,
        "bandit_type": bandit_type,
        "policy_switch": policy_switch_enabled,
        "policy_switch_mode": policy_switch_mode,
        "cache_enabled": cache_enabled,
        "cache_key_schema_version": cache_key_schema_version,
        "acc_first_hard_gating_enabled": acc_first_hard_gating_enabled,
        "locked_acc_ref_enabled": locked_acc_ref_enabled,
        "acc_ref_source": acc_ref_source,
        "no_drift_enabled": no_drift_enabled,
        "no_double_scale_enabled": no_double_scale_enabled,
        "seed_global": int(seed_global),
        "seed_problem": int(seed_problem),
        "config_fingerprint": str(cfg_hash),
        "git_commit_or_version": get_git_commit_or_version(),
        "mode": str(mode),
    }

    missing = [k for k in REQUIRED_SIGNATURE_FIELDS if k not in sig]
    if missing:
        raise ValueError(f"[v5.4] signature missing fields: {missing}")

    return sig
