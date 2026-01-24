# utils/trace_contract_v54.py
# Contract: SPEC_E_AntiLoop_Contracts_and_Smoke_v5.4.md (Appendix A)

import hashlib
from typing import Any

from omegaconf import OmegaConf

from .trace_signature_v54 import stable_json_dumps
from .trace_signature_v54 import REQUIRED_SIGNATURE_FIELDS

SCHEMA_VERSION_V54 = "5.4"

# Keep a superset to avoid breaking existing layout pipelines,
# but enforce SPEC_E-required payload keys for contract-critical events.
ALLOWED_EVENT_TYPES_V54 = {
    "trace_header",
    "gating",
    "proxy_sanitize",
    "ref_update",
    "finalize",
    "contract_violation",
    "contract_override",

    # non-contract-critical / legacy (allowed but not strictly keyed here)
    "step",
    "layout_step",
    "pareto_add",
    "duplicate",
    "boundary",
}

# ===== SPEC_E Appendix A1.3 (trace_header) =====
REQUIRED_TRACE_HEADER_KEYS = [
    # HardGate-B canonical evidence chain
    "requested_config_snapshot",
    "effective_config_snapshot",
    "contract_overrides",
    "seal_digest",
    "signature",

    # --- SPEC_E: Acc-First / Locked / Stability required fields (auditable) ---
    "acc_first_hard_gating_enabled",
    "locked_acc_ref_enabled",
    "acc_ref_source",
    "no_drift_enabled",
    "no_double_scale_enabled",

    # --- SPEC_E: Repro required fields (auditable) ---
    "seed_global",
    "seed_problem",
    "config_fingerprint",
    "git_commit_or_version",
]

# ===== SPEC_E Appendix A1.2 (gating) =====
REQUIRED_GATING_KEYS = [
    "candidate_id",
    "gate",
    "reason_code",
    "acc_ref",
    "acc_used",
    "acc_drop",
    "acc_drop_max",
    "lambda_hw_effective",
    "hw_loss_raw",
    "hw_loss_used",
    "hw_scale_schema_version",
    "hw_metric_ref",
    "hw_metric_raw",
    "hw_metric_normed",
    "total_loss_acc_part",
    "total_loss_hw_part",
    "total_loss",
]

# ===== Layout agent step payload (layout pipeline only) =====
REQUIRED_LAYOUT_STEP_KEYS = [
    "iter",
    "stage",
    "op",
    "op_args_json",
    "accepted",
    "total_scalar",
    "comm_norm",
    "therm_norm",
    "pareto_added",
    "duplicate_penalty",
    "boundary_penalty",
    "seed_id",
    "time_ms",
    "signature",
]

# ===== SPEC_E Appendix A1.4 (proxy_sanitize) =====
REQUIRED_PROXY_SANITIZE_KEYS = [
    "candidate_id",
    "metric",
    "raw_value",
    "used_value",
    "penalty_added",
]

# ===== SPEC_E (ref_update; used only when NoDrift disabled) =====
REQUIRED_REF_UPDATE_KEYS = [
    "ref_name",
    "old_value",
    "new_value",
    "update_type",
    "allowed_by_no_drift",
    "requested_mode",
    "effective_mode",
    "reason",
]

# ===== SPEC_E finalize (Appendix A; minimal enforceable keys) =====
REQUIRED_FINALIZE_KEYS = [
    "reason",
    "steps_done",
    "best_solution_valid",
]

REQUIRED_EVENT_PAYLOAD_KEYS_V54 = {
    "trace_header": REQUIRED_TRACE_HEADER_KEYS,
    "gating": REQUIRED_GATING_KEYS,
    "proxy_sanitize": REQUIRED_PROXY_SANITIZE_KEYS,
    "ref_update": REQUIRED_REF_UPDATE_KEYS,
    "finalize": REQUIRED_FINALIZE_KEYS,
    "contract_violation": ["reason", "expected_seal_digest", "actual_seal_digest"],
    "contract_override": ["reason", "requested", "effective", "details"],

    # Allow free-form payload for these (legacy / non-contract-critical)
    "step": [],
    "layout_step": REQUIRED_LAYOUT_STEP_KEYS,
    "pareto_add": [],
    "duplicate": [],
    "boundary": [],
}


class TraceContractV54:
    CONTRACT_VERSION = "v5.4"

    @staticmethod
    def encode_config_snapshot(cfg, resolve: bool = True):
        if cfg is None:
            return None
        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=resolve)
        if isinstance(cfg, dict):
            return dict(cfg)
        return {"_unsupported_cfg_snapshot_type": str(type(cfg))}

    @staticmethod
    def validate_event(event_type: str, payload: dict) -> bool:
        required = REQUIRED_EVENT_PAYLOAD_KEYS_V54.get(event_type, [])
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"trace_contract_v54: {event_type} missing required keys: {missing}")
        return True


def assert_trace_header_v54(payload: dict, strict: bool = True) -> None:
    if not isinstance(payload, dict):
        raise TypeError("trace_header payload must be a dict")
    TraceContractV54.validate_event("trace_header", payload)
    signature = payload.get("signature", {})
    if not isinstance(signature, dict):
        raise TypeError("trace_header.signature must be a dict")
    if strict:
        missing = [k for k in REQUIRED_SIGNATURE_FIELDS if k not in signature]
        if missing:
            raise ValueError(f"trace_header.signature missing required fields: {missing}")


def compute_effective_cfg_digest_v54(cfg_or_snapshot: Any) -> str:
    """
    v5.4 LEGAL seal digest:
      sha256( effective_config_snapshot )
    where effective_config_snapshot is a resolved dict with ALL meta stripped:
      - remove top-level keys starting with "_contract"
      - remove top-level key "contract" entirely (meta, not training semantics)
    """

    def _to_plain(obj: Any) -> Any:
        if OmegaConf is not None and OmegaConf.is_config(obj):
            return OmegaConf.to_container(obj, resolve=True)
        return obj

    def _strip_meta(d: Any) -> Any:
        if not isinstance(d, dict):
            return d
        out = {}
        for k, v in d.items():
            if isinstance(k, str) and (k.startswith("_contract") or k == "contract"):
                continue
            out[k] = _strip_meta(v) if isinstance(v, dict) else v
        return out

    plain = _to_plain(cfg_or_snapshot)
    plain = _strip_meta(plain)
    return compute_snapshot_sha256_v54(plain)


def strip_seal_fields_v54(snapshot: dict) -> dict:
    """
    Return a COPY of snapshot where self-referential seal fields are removed.
    This MUST match what is stored as trace_header.effective_config_snapshot.
    """
    if not isinstance(snapshot, dict):
        return snapshot
    snap = dict(snapshot)

    c = snap.get("contract", None)
    if isinstance(c, dict) and "seal_digest" in c:
        c2 = dict(c)
        c2.pop("seal_digest", None)
        snap["contract"] = c2
    return snap


def compute_snapshot_sha256_v54(snapshot_obj: dict) -> str:
    """
    sha256(stable_json_dumps(snapshot_obj)) — canonical seal definition for v5.4.
    """
    s = stable_json_dumps(snapshot_obj).encode("utf-8")
    return hashlib.sha256(s).hexdigest()
