# proj_ast2_ucf101_full/utils/trace_contract_v54.py
# v5.4 Anti-Loop Trace Contract (SPEC_E authoritative)

SCHEMA_VERSION_V54 = "5.4"

ALLOWED_EVENT_TYPES_V54 = (
    "trace_header",
    "gating",
    "proxy_sanitize",
    "ref_update",
    "step",
    "finalize",
)

# ---- payload required keys (SPEC_E) ----
REQUIRED_TRACE_HEADER_KEYS = ("signature", "requested_config", "effective_config")
REQUIRED_GATING_KEYS = ("candidate_id", "gate", "acc_drop", "acc_drop_max")
REQUIRED_PROXY_SANITIZE_KEYS = ("metric", "raw_value", "used_value", "penalty_added")
REQUIRED_REF_UPDATE_KEYS = ("key", "old_value", "new_value", "reason")
REQUIRED_FINALIZE_KEYS = ("status", "summary")

# v5.4 layout/HeurAgenix step evidence (anti-loop legal record)
REQUIRED_STEP_KEYS = (
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
)

REQUIRED_EVENT_PAYLOAD_KEYS_V54 = {
    "trace_header": REQUIRED_TRACE_HEADER_KEYS,
    "gating": REQUIRED_GATING_KEYS,
    "proxy_sanitize": REQUIRED_PROXY_SANITIZE_KEYS,
    "ref_update": REQUIRED_REF_UPDATE_KEYS,
    "step": REQUIRED_STEP_KEYS,
    "finalize": REQUIRED_FINALIZE_KEYS,
}
