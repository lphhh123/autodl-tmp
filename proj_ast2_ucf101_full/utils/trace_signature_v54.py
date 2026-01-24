from __future__ import annotations

from typing import Any, Dict, Optional, List
import json
import os

from .stable_hash import stable_hash


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

# v5.4 trace signature contract (SPEC_E §6)
REQUIRED_SIGNATURE_FIELDS: List[str] = [
    "method_name",
    "config_fingerprint",
    "git_commit_or_version",
    "seed_global",
    "seed_problem",
    # Ours-B2+ knobs
    "moves_enabled",
    "lookahead_k",
    "bandit_type",
    "policy_switch_mode",
    "cache_enabled",
    "cache_key_schema_version",
    # StableHW contracts
    "acc_first_hard_gating_enabled",
    "locked_acc_ref_enabled",
    "acc_ref_source",
    "no_drift_enabled",
    "no_double_scale_enabled",
]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_path(obj: Any, path: str, default: Any = None) -> Any:
    cur = obj
    for p in path.split("."):
        if cur is None:
            return default
        cur = _get(cur, p, None)
        if cur is None:
            return default
    return cur


def compute_config_fingerprint_v54(cfg: Any) -> str:
    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg
    return stable_hash(cfg_dict)


def _boolish(v: Any) -> bool:
    if v is None:
        raise ValueError("required boolean is missing")
    return bool(v)


def _parse_no_double_scale(cfg: Any) -> bool:
    nds = _get_path(cfg, "stable_hw.no_double_scale", None)
    if isinstance(nds, dict):
        return _boolish(nds.get("enabled", None))
    if isinstance(nds, (bool, int)):
        return bool(nds)
    nds2 = _get_path(cfg, "stable_hw.no_double_scale.enabled", None)
    if nds2 is not None:
        return bool(nds2)
    raise ValueError("missing cfg.stable_hw.no_double_scale (bool or {enabled: ...})")


def build_signature_v54(
    cfg: Any,
    method_name: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if _get(cfg, "stable_hw", None) is None:
        raise ValueError("missing cfg.stable_hw; call validate_and_fill_defaults() first")

    strict = bool(_get_path(cfg, "contract.strict", _get_path(cfg, "_contract.strict", False)))
    if strict:
        for legacy_key in ("locked_acc_ref", "no_drift", "no_double_scale", "accuracy_guard", "hard_gating"):
            if _get(cfg, legacy_key, None) is not None:
                raise ValueError(
                    f"P0(v5.4 strict): legacy root-level {legacy_key} forbidden; "
                    "use cfg.stable_hw.* only"
                )

    fp = compute_config_fingerprint_v54(cfg)

    seed_global = int(_get_path(cfg, "seed", _get_path(cfg, "train.seed", 0)) or 0)
    seed_problem = int(_get_path(cfg, "problem.seed", 0) or 0)

    # -------- Ours-B2+ knobs (layout search) --------
    action_probs = _get_path(cfg, "detailed_place.action_probs", None)
    if not isinstance(action_probs, dict):
        action_probs = {}
    moves_enabled = sorted([str(k) for k in action_probs.keys()])

    lookahead_k = int(_get_path(cfg, "detailed_place.lookahead.k", 0) or 0)

    ps_enabled = bool(_get_path(cfg, "detailed_place.policy_switch.enabled", False))
    ps_mode = str(_get_path(cfg, "detailed_place.policy_switch.mode", "off") or "off")
    policy_switch_mode = ps_mode if ps_enabled else "off"

    bandit_type = str(
        _get_path(
            cfg,
            "detailed_place.policy_switch.bandit_type",
            _get_path(cfg, "detailed_place.policy_switch.bandit.type", "none"),
        )
        or "none"
    )

    cache_size = int(_get_path(cfg, "detailed_place.policy_switch.cache_size", 0) or 0)
    cache_enabled = bool(ps_enabled and cache_size > 0)
    cache_key_schema_version = str(
        _get_path(cfg, "detailed_place.policy_switch.cache_key_schema_version", "v5.4") or "v5.4"
    )

    # -------- StableHW contracts --------
    acc_first_hard_gating_enabled = _boolish(_get_path(cfg, "stable_hw.accuracy_guard.enabled", None))

    locked_acc_ref_enabled = _get_path(cfg, "stable_hw.locked_acc_ref.enabled", None)
    locked_acc_ref_enabled = _boolish(locked_acc_ref_enabled)

    acc_ref_source = _get_path(cfg, "stable_hw.locked_acc_ref.source", None)
    if acc_ref_source is None:
        raise ValueError("missing stable_hw.locked_acc_ref.source")

    no_drift_enabled = _get_path(cfg, "stable_hw.no_drift.enabled", None)
    no_drift_enabled = _boolish(no_drift_enabled)

    no_double_scale_enabled = _parse_no_double_scale(cfg)

    git_commit_or_version = (
        os.environ.get("GIT_COMMIT")
        or os.environ.get("GITHUB_SHA")
        or os.environ.get("PROJECT_GIT_SHA")
        or "code_only"
    )

    sig: Dict[str, Any] = {
        "method_name": str(method_name),
        "config_fingerprint": fp,
        "git_commit_or_version": str(git_commit_or_version),
        "seed_global": seed_global,
        "seed_problem": seed_problem,
        "moves_enabled": moves_enabled,
        "lookahead_k": lookahead_k,
        "bandit_type": bandit_type,
        "policy_switch_mode": policy_switch_mode,
        "cache_enabled": cache_enabled,
        "cache_key_schema_version": cache_key_schema_version,
        "acc_first_hard_gating_enabled": acc_first_hard_gating_enabled,
        "locked_acc_ref_enabled": locked_acc_ref_enabled,
        "acc_ref_source": str(acc_ref_source),
        "no_drift_enabled": no_drift_enabled,
        "no_double_scale_enabled": no_double_scale_enabled,
        "action_families": moves_enabled,  # optional
    }

    if overrides:
        sig.update(overrides)

    missing = [k for k in REQUIRED_SIGNATURE_FIELDS if k not in sig]
    if missing:
        raise ValueError(f"missing required signature fields: {missing}")
    none_fields = [k for k in REQUIRED_SIGNATURE_FIELDS if sig.get(k, None) is None]
    if none_fields:
        raise ValueError(f"signature required fields are None: {none_fields}")

    return sig
