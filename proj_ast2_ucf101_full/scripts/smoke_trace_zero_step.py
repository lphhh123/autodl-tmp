# -*- coding: utf-8 -*-
"""
Smoke: v5.4 requires trace file exists even when steps=0 / early stop.
This script generates a minimal trace_v54.csv compatible file and a minimal
trace_events.jsonl header+finalize to validate schema contracts quickly.
"""

import argparse
import csv
import json
import time
from pathlib import Path

from utils.trace_guard import ensure_trace_events, finalize_trace_events
from utils.trace_schema import TRACE_FIELDS
from utils.trace_signature_v54 import build_signature_v54


def _write_trace_csv_v54(out_dir: Path, seed: int) -> Path:
    trace_csv = out_dir / "trace_v54.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Minimal init row + finalize row (steps=0)
    # NOTE: trace.csv signature column is assign signature (assign:...), not run signature dict.
    init_row = [
        0,
        "init",
        "init",
        json.dumps({"op": "init"}, ensure_ascii=False),
        1,
        0.0,
        0.0,
        0.0,
        0,
        0.0,
        0.0,
        int(seed),
        0,
        "assign:null",
        0,
        0,
        0,
        0,
        0,
        0,
        "init",
        "init",
        0,
        0,
        "cache:null",
    ]
    fin_row = [
        1,
        "finalize",
        "finalize",
        json.dumps({"op": "finalize"}, ensure_ascii=False),
        1,
        0.0,
        0.0,
        0.0,
        0,
        0.0,
        0.0,
        int(seed),
        int((time.time() - start_time) * 1000),
        "assign:null",
        0,
        0,
        0,
        0,
        0,
        0,
        "finalize",
        "finalize",
        0,
        0,
        "",
    ]

    with trace_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(TRACE_FIELDS)
        w.writerow(init_row)
        w.writerow(fin_row)

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

    trace_events = out_dir / "trace_events.jsonl"
    ensure_trace_events(trace_events, payload={"signature": sig, "note": "steps0"})
    finalize_trace_events(trace_events, payload={"reason": "steps0", "steps_done": 0, "best_solution_valid": False})

    # also drop a minimal json file to help debugging
    (out_dir / "smoke_signature.json").write_text(
        json.dumps(sig, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("[SMOKE] wrote:", str(trace_csv))


if __name__ == "__main__":
    main()
