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
REQUIRED_TRACE_HEADER_KEYS = (
    "signature",
    "requested_config",
    "effective_config",
    "no_drift_enabled",
    "acc_ref_source",
)
REQUIRED_GATING_KEYS = (
    "candidate_id",
    "gate",
    "acc_ref",
    "acc_now",
    "acc_drop",
    "acc_drop_max",
    "acc_used_source",
    "acc_used_value",
    "lambda_hw_base",
    "lambda_hw_effective",
    "hw_loss_raw",
    "hw_loss_used",
    "total_loss_scalar",
    "total_loss_acc_part",
    "total_loss_hw_part",
    "hw_metric_ref",
    "hw_metric_raw",
    "hw_metric_normed",
    "hw_scale_schema_version",
)
REQUIRED_PROXY_SANITIZE_KEYS = (
    "metric",
    "raw_value",
    "used_value",
    "penalty_added",
    "clamp_min",
    "clamp_max",
    "note",
    "source",
)
REQUIRED_REF_UPDATE_KEYS = ("ref_name", "old_value", "new_value", "reason")
REQUIRED_FINALIZE_KEYS = ("reason", "steps_done", "best_solution_valid")

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
    "duplicate_penalty",
    "boundary_penalty",
    "seed_id",
    "time_ms",
)

REQUIRED_EVENT_PAYLOAD_KEYS_V54 = {
    "trace_header": REQUIRED_TRACE_HEADER_KEYS,
    "gating": REQUIRED_GATING_KEYS,
    "proxy_sanitize": REQUIRED_PROXY_SANITIZE_KEYS,
    "ref_update": REQUIRED_REF_UPDATE_KEYS,
    "step": REQUIRED_STEP_KEYS,
    "finalize": REQUIRED_FINALIZE_KEYS,
}
