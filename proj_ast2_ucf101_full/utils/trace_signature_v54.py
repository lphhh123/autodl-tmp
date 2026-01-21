from __future__ import annotations
from typing import Any, Dict, Optional, List
import os
from .stable_hash import stable_hash

REQUIRED_SIGNATURE_FIELDS: List[str] = [
    "method_name",
    "config_fingerprint",
    "seed_global",
    "seed_problem",
    "git_commit_or_version",
    "acc_first_hard_gating",
    "locked_acc_ref_enabled",
    "no_drift_enabled",
    "no_double_scale_enabled",
    "action_families",
    "moves_enabled",
    "lookahead_k",
    "bandit_type",
    "policy_switch_mode",
    "cache_enabled",
    "cache_key_schema_version",
]


def _get(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_path(obj: Any, path: str, default=None):
    cur = obj
    for p in path.split("."):
        cur = _get(cur, p, None)
        if cur is None:
            return default
    return cur


def compute_config_fingerprint_v54(cfg) -> str:
    # fingerprint ONLY from stable, spec-relevant knobs
    src = {
        "mode": _get(cfg, "mode", None),
        "train_seed": _get_path(cfg, "train.seed", None),
        "stable_hw": _get(cfg, "stable_hw", None),
        "locked_acc_ref": _get(cfg, "locked_acc_ref", None),  # legacy mirror
        "no_drift": _get(cfg, "no_drift", None),  # legacy mirror
        "detailed_place": _get(cfg, "detailed_place", None),
        "policy_switch": _get(cfg, "policy_switch", None),
        "hw": _get(cfg, "hw", None),
        "loss": _get(cfg, "loss", None),
    }
    return stable_hash(src)


def build_signature_v54(
    cfg,
    *,
    method_name: str,
    seed_problem: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    fp = compute_config_fingerprint_v54(cfg)

    seed_global = int(_get_path(cfg, "train.seed", 0) or 0)
    if seed_problem is None:
        seed_problem = int(_get_path(cfg, "problem.seed", seed_global) or seed_global)

    # StableHW flags (prefer stable_hw migrated config)
    acc_first_hard_gating = bool(_get_path(cfg, "stable_hw.accuracy_guard.enabled", False))
    locked_acc_ref_enabled = bool(
        _get_path(cfg, "stable_hw.locked_acc_ref.enabled", False) or _get_path(cfg, "locked_acc_ref.enabled", False)
    )
    no_drift_enabled = bool(
        _get_path(cfg, "stable_hw.no_drift.enabled", False) or _get_path(cfg, "no_drift.enabled", False)
    )
    no_double_scale_enabled = bool(_get_path(cfg, "stable_hw.no_double_scale.enabled", True))

    # Ours-B2+ knobs (layout)
    action_probs = _get_path(cfg, "detailed_place.action_probs", None)
    if isinstance(action_probs, dict):
        action_families = list(action_probs.keys())
    else:
        action_families = list(_get_path(cfg, "policy_switch.action_families", []) or [])
    moves_enabled = bool(action_families)
    lookahead_k = int(_get_path(cfg, "detailed_place.lookahead_k", _get_path(cfg, "detailed_place.lookahead.k", 0)) or 0)
    bandit_type = str(_get_path(cfg, "policy_switch.bandit_type", "none") or "none")
    ps_enabled = bool(_get_path(cfg, "policy_switch.enabled", False))
    ps_mode = str(_get_path(cfg, "policy_switch.mode", "bandit" if ps_enabled else "none"))
    if not ps_enabled:
        ps_mode = "none"

    cache_enabled = bool(_get_path(cfg, "detailed_place.cache_enabled", _get_path(cfg, "detailed_place.cache.enabled", True)))
    cache_key_schema_version = int(_get_path(cfg, "detailed_place.cache_key_schema_version", 1) or 1)

    git_commit_or_version = (
        os.environ.get("GIT_COMMIT", None) or os.environ.get("PROJECT_VERSION", None) or "v5.4"
    )

    sig = {
        "method_name": method_name,
        "config_fingerprint": fp,
        "seed_global": seed_global,
        "seed_problem": int(seed_problem),
        "git_commit_or_version": git_commit_or_version,
        "acc_first_hard_gating": acc_first_hard_gating,
        "locked_acc_ref_enabled": locked_acc_ref_enabled,
        "no_drift_enabled": no_drift_enabled,
        "no_double_scale_enabled": no_double_scale_enabled,
        "action_families": action_families,
        "moves_enabled": moves_enabled,
        "lookahead_k": lookahead_k,
        "bandit_type": bandit_type,
        "policy_switch_mode": ps_mode,
        "cache_enabled": cache_enabled,
        "cache_key_schema_version": cache_key_schema_version,
    }

    if overrides:
        sig.update(overrides)

    # hard assertion here (fail fast)
    missing = [k for k in REQUIRED_SIGNATURE_FIELDS if k not in sig]
    if missing:
        raise ValueError(f"trace signature missing required fields: {missing}")
    return sig
