# utils/trace_contract_v54.py
# Contract: SPEC_E_AntiLoop_Contracts_and_Smoke_v5.4.md (Appendix A)

SCHEMA_VERSION_V54 = "5.4"

# Keep a superset to avoid breaking existing layout pipelines,
# but enforce SPEC_E-required payload keys for contract-critical events.
ALLOWED_EVENT_TYPES_V54 = {
    "trace_header",
    "gating",
    "proxy_sanitize",
    "ref_update",
    "finalize",

    # non-contract-critical / legacy (allowed but not strictly keyed here)
    "step",
    "pareto_add",
    "duplicate",
    "boundary",
}

# ===== SPEC_E Appendix A1.3 (trace_header) =====
REQUIRED_TRACE_HEADER_KEYS = [
    "requested_config",
    "effective_config",
    "contract_overrides",
    "requested",
    "effective",
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

    # Allow free-form payload for these (legacy / non-contract-critical)
    "step": [],
    "pareto_add": [],
    "duplicate": [],
    "boundary": [],
}
