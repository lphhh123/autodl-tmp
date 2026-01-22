"""
Unified trace schema for SPEC v5.4.

原则：
- 兼容旧字段（保留原 TRACE_FIELDS 前半段）
- 新增 v5.4 addendum 必需的：身份/单位/预算轴/签名分离/LLM 透明度/累积统计
"""
TRACE_FIELDS = [
    # --- legacy core ---
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
    "signature",

    "wall_time_ms",
    "evaluator_calls",
    "accepted_steps",
    "selected_idx",
    "tabu_hit",
    "inverse_hit",
    "cooldown_hit",

    "cache_hit",
    "cache_miss",
    "cache_size",
    "cache_key",

    "move_family",
    "selector",
    "lookahead",

    # --- v5.4 addendum required ---
    "assign_signature",
    "op_signature",

    "seed_id",
    "objective_hash",
    "eval_version",
    "time_unit",
    "dist_unit",

    "wall_time_ms_cum",
    "eval_calls_cum",
    "accepted_steps_cum",

    "cache_hit_cum",
    "cache_miss_cum",

    "budget_mode",
    "budget_total",
    "budget_remaining",

    "use_llm",
    "llm_model",
    "llm_prompt_tokens",
    "llm_completion_tokens",
    "llm_latency_ms",
]
