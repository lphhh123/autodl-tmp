#!/usr/bin/env python3
"""
Smoke: v5.4 trace_events.jsonl must start with trace_header and end with finalize.
This script writes a minimal trace_header + finalize (steps0) and validates the contract.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults
from utils.stable_hash import stable_hash
from utils.trace_contract_v54 import REQUIRED_GATING_KEYS, REQUIRED_PROXY_SANITIZE_KEYS
from utils.trace_guard import append_trace_event_v54, finalize_trace_dir, init_trace_dir_v54
from utils.trace_signature_v54 import REQUIRED_SIGNATURE_FIELDS, build_signature_v54


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _validate_payload(event_type: str, payload: dict) -> list[str]:
    errors = []
    if event_type == "trace_header":
        for key in ("signature", "no_drift_enabled", "acc_ref_source"):
            if key not in payload:
                errors.append(f"trace_header missing {key}")
        for key in ("requested_config", "effective_config", "contract_overrides", "requested", "effective"):
            if key not in payload:
                errors.append(f"trace_header missing {key}")
    elif event_type == "gating":
        for key in REQUIRED_GATING_KEYS:
            if key not in payload:
                errors.append(f"gating missing {key}")
    elif event_type == "proxy_sanitize":
        for key in REQUIRED_PROXY_SANITIZE_KEYS:
            if key not in payload:
                errors.append(f"proxy_sanitize missing {key}")
    elif event_type == "ref_update":
        for key in ("key", "old_value", "new_value", "reason"):
            if key not in payload:
                errors.append(f"ref_update missing {key}")
    elif event_type == "finalize":
        for key in ("status", "summary"):
            if key not in payload:
                errors.append(f"finalize missing {key}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="config path")
    ap.add_argument("--out_dir", default="outputs/smoke_trace_events_contract")
    args = ap.parse_args()

    cfg = validate_and_fill_defaults(load_config(args.cfg), mode="version_c")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = stable_hash({"mode": "smoke_trace_events_contract", "cfg": str(args.cfg)})
    signature = build_signature_v54(cfg, method_name="smoke_trace_events_contract")

    trace_meta = init_trace_dir_v54(
        base_dir=out_dir / "trace",
        run_id=str(run_id),
        cfg=cfg,
        signature=signature,
        signature_v54=signature,
        required_signature_fields=REQUIRED_SIGNATURE_FIELDS,
        run_meta={"mode": "smoke_trace_events_contract", "seed_id": 0, "run_id": str(run_id)},
        extra_manifest={"task": "smoke_trace_events_contract", "out_dir": str(out_dir)},
    )
    trace_events_path = Path(trace_meta["trace_events"])

    append_trace_event_v54(
        trace_events_path,
        "trace_header",
        payload={
            "requested_config": {},
            "effective_config": {},
            "contract_overrides": [],
            "requested": {"mode": "version_c"},
            "effective": {"mode": "version_c"},
            "signature": signature,
            "no_drift_enabled": True,
            "acc_ref_source": "locked",
        },
        run_id=str(run_id),
        step=0,
    )

    finalize_trace_dir(
        trace_events_path,
        reason="steps0",
        steps_done=0,
        best_solution_valid=True,
    )

    rows = _load_jsonl(trace_events_path)
    if not rows:
        print("[SMOKE] trace_events.jsonl is empty")
        return 1
    if rows[0].get("event_type") != "trace_header":
        print(f"[SMOKE] first event_type is not trace_header: {rows[0].get('event_type')}")
        return 1
    if rows[-1].get("event_type") != "finalize":
        print(f"[SMOKE] last event_type is not finalize: {rows[-1].get('event_type')}")
        return 1
    for idx, row in enumerate(rows):
        event_type = row.get("event_type")
        payload = row.get("payload", {})
        errors = _validate_payload(event_type, payload)
        if errors:
            joined = ", ".join(errors)
            print(f"[SMOKE] event {idx} ({event_type}) failed contract: {joined}")
            return 1
    print("[SMOKE] trace_events contract ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
