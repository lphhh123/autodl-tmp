# -*- coding: utf-8 -*-
# NOTE: This smoke is bound to SPEC_E v5.4; do not change fields without updating SPEC_E + trace_contract_v54.py
"""
Smoke: v5.4 requires trace file exists even when steps=0 / early stop.
This script generates a minimal trace.csv compatible file and a minimal
trace_events.jsonl header+finalize to validate schema contracts quickly.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from utils.trace_guard import init_trace_dir, finalize_trace_dir, append_trace_event_v54
from utils.trace_schema import TRACE_FIELDS
from utils.stable_hash import stable_hash
from utils.trace_signature_v54 import build_signature_v54, REQUIRED_SIGNATURE_FIELDS


def _row_to_list(row: dict) -> list:
    return [row.get(k, "") for k in TRACE_FIELDS]


def _write_trace_csv_v54(out_dir: Path, seed: int) -> Path:
    trace_csv = out_dir / "trace.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    objective_hash = stable_hash({"smoke": "trace_zero_step"})
    assign_signature = "assign:null"
    cache_key = stable_hash(
        {
            "assign": assign_signature,
            "objective_hash": objective_hash,
            "eval_version": "v5.4",
            "cfg_hash_subset": stable_hash({"smoke": "trace_zero_step"}),
        }
    )

    init_row = {
        "iter": -1,
        "stage": "init",
        "op": "init",
        "op_args_json": json.dumps({"op": "init"}, ensure_ascii=False),
        "accepted": 1,
        "total_scalar": 0.0,
        "comm_norm": 0.0,
        "therm_norm": 0.0,
        "duplicate_penalty": 0.0,
        "boundary_penalty": 0.0,
        "signature": assign_signature,
        "assign_signature": assign_signature,
        "op_signature": stable_hash({"op": "init"}),
        "seed_id": int(seed),
        "objective_hash": objective_hash,
        "eval_version": "v5.4",
        "time_unit": "ms",
        "dist_unit": "mm",
        "wall_time_ms": 0,
        "wall_time_ms_cum": 0,
        "evaluator_calls": 0,
        "eval_calls_cum": 0,
        "accepted_steps": 0,
        "accepted_steps_cum": 0,
        "selected_idx": -1,
        "tabu_hit": 0,
        "inverse_hit": 0,
        "cooldown_hit": 0,
        "cache_hit": 0,
        "cache_miss": 0,
        "cache_hit_cum": 0,
        "cache_miss_cum": 0,
        "cache_size": 0,
        "cache_key": cache_key,
        "move_family": "init",
        "selector": "init",
        "lookahead": 0,
        "budget_mode": "steps",
        "budget_total": 0,
        "budget_remaining": 0,
        "use_llm": 0,
        "llm_model": "",
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "llm_latency_ms": 0,
    }

    fin_row = {
        "iter": 0,
        "stage": "finalize",
        "op": "finalize",
        "op_args_json": json.dumps({"op": "finalize"}, ensure_ascii=False),
        "accepted": 1,
        "total_scalar": 0.0,
        "comm_norm": 0.0,
        "therm_norm": 0.0,
        "duplicate_penalty": 0.0,
        "boundary_penalty": 0.0,
        "signature": assign_signature,
        "assign_signature": assign_signature,
        "op_signature": stable_hash({"op": "finalize"}),
        "seed_id": int(seed),
        "objective_hash": objective_hash,
        "eval_version": "v5.4",
        "time_unit": "ms",
        "dist_unit": "mm",
        "wall_time_ms": int((time.time() - start_time) * 1000),
        "wall_time_ms_cum": int((time.time() - start_time) * 1000),
        "evaluator_calls": 0,
        "eval_calls_cum": 0,
        "accepted_steps": 0,
        "accepted_steps_cum": 0,
        "selected_idx": -1,
        "tabu_hit": 0,
        "inverse_hit": 0,
        "cooldown_hit": 0,
        "cache_hit": 0,
        "cache_miss": 0,
        "cache_hit_cum": 0,
        "cache_miss_cum": 0,
        "cache_size": 0,
        "cache_key": cache_key,
        "move_family": "finalize",
        "selector": "finalize",
        "lookahead": 0,
        "budget_mode": "steps",
        "budget_total": 0,
        "budget_remaining": 0,
        "use_llm": 0,
        "llm_model": "",
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "llm_latency_ms": 0,
    }

    with trace_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(TRACE_FIELDS)
        w.writerow(_row_to_list(init_row))
        w.writerow(_row_to_list(fin_row))

    return trace_csv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Minimal config-like dict for signature builder
    cfg_min = {
        "seed": int(args.seed),
        "out_dir": str(out_dir),
        "stable_hw": {
            "enabled": True,
            "accuracy_guard": {"enabled": True, "controller": {"guard_mode": "hard"}},
            "locked_acc_ref": {"enabled": True, "source": "manual"},
            "no_drift": {"enabled": True},
            "no_double_scale": {"enabled": True},
        },
        "locked_acc_ref": {"enabled": True, "source": "manual"},
        "no_drift": {"enabled": True},
        "no_double_scale": {"enabled": True},
        "layout": {"moves_enabled": False},
        "policy_switch": {"enabled": False},
        "cache": {"enabled": False},
        "detailed_place": {
            "action_probs": {},
            "lookahead": {"k": 0},
            "policy_switch": {"enabled": False, "cache_size": 0, "cache_key_schema_version": "v5.4"},
        },
    }
    sig = build_signature_v54(cfg_min, method_name="smoke_trace_zero_step")

    trace_csv = _write_trace_csv_v54(out_dir, int(args.seed))

    trace_dir = out_dir / "trace"
    run_id = stable_hash({"mode": "smoke_steps0", "seed": int(args.seed)})
    init_trace_dir(
        trace_dir,
        signature=sig,
        run_meta={"run_id": run_id, "seed_id": int(args.seed), "mode": "smoke_steps0"},
        required_signature_keys=REQUIRED_SIGNATURE_FIELDS,
        resolved_config=cfg_min,
    )
    trace_events_path = trace_dir / "trace_events.jsonl"
    append_trace_event_v54(
        trace_events_path,
        "trace_header",
        payload={
            "requested_config": {},
            "effective_config": {},
            "contract_overrides": [],
            "requested": {"mode": "smoke_steps0"},
            "effective": {"mode": "smoke_steps0"},
            "signature": sig,
            "no_drift_enabled": True,
            "acc_ref_source": "manual",
        },
        run_id=run_id,
        step=0,
    )
    finalize_trace_dir(
        trace_events_path,
        reason="steps0",
        steps_done=0,
        best_solution_valid=True,
    )

    # also drop a minimal json file to help debugging
    (out_dir / "smoke_signature.json").write_text(
        json.dumps(sig, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[SMOKE] wrote:", str(trace_csv))


if __name__ == "__main__":
    main()
