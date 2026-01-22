"""
Unified trace schema for SPEC v5.4.

原则：
- 兼容旧字段（保留原 TRACE_FIELDS 前半段）
- 新增 v5.4 addendum 必需的：身份/单位/预算轴/签名分离/LLM 透明度/累积统计
"""
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
    "delta_total",
    "delta_comm",
    "delta_therm",
    "tabu_hit",
    "inverse_hit",
    "cooldown_hit",
    "policy",
    "move",
    "lookahead_k",
    "cache_hit",
    "cache_key",
    # --- SPEC_B additions (v5.4) ---
    "objective_hash",
    "eval_calls_cum",
    "cache_hit_cum",
    "cache_miss_cum",
    "cache_saved_eval_calls_cum",
    "llm_used",
    "llm_fail_count",
    "fallback_reason",
    "wall_time_ms_cum",
    "accepted_steps_cum",
    "sim_eval_calls_cum",
    "lookahead_enabled",
    "lookahead_r",
    "notes",
]
