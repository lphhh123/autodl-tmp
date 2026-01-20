# v5.4: trace schema must match actual written rows (20 columns)
TRACE_FIELDS = [
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
    # per-step deltas / anti-loop diagnostics (must exist even when steps=0 -> init row zeros)
    "delta_total",
    "delta_comm",
    "delta_therm",
    "tabu_hit",
    "inverse_hit",
    "cooldown_hit",
]
