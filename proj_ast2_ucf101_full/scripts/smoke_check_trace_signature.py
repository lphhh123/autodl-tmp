# -*- coding: utf-8 -*-
"""Smoke check for v5.4 trace signature + schema in trace.csv."""

import argparse
import csv
import json
from pathlib import Path

from utils.trace_guard import validate_required_signature
from utils.trace_schema import TRACE_FIELDS


def _load_trace_rows(trace_path: Path):
    with trace_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise RuntimeError(f"trace file is empty: {trace_path}")
    return rows


def _check_trace_csv(trace_path: Path) -> None:
    rows = _load_trace_rows(trace_path)
    header = rows[0]
    if header != TRACE_FIELDS:
        raise AssertionError(f"trace header mismatch: got={header} expected={TRACE_FIELDS}")

    signature_idx = header.index("signature")
    for idx, row in enumerate(rows[1:], start=1):
        if len(row) != len(TRACE_FIELDS):
            raise AssertionError(
                f"trace row {idx} has {len(row)} cols but TRACE_FIELDS has {len(TRACE_FIELDS)}"
            )
        sig = row[signature_idx]
        if not sig:
            raise AssertionError(f"trace row {idx} has empty signature")


def _check_trace_events(trace_path: Path) -> None:
    events_path = trace_path.parent / "trace_events.jsonl"
    if not events_path.exists():
        return
    with events_path.open("r", encoding="utf-8") as f:
        first = f.readline().strip()
    if not first:
        raise AssertionError("trace_events.jsonl is empty")
    obj = json.loads(first)
    if obj.get("event_type") != "trace_header":
        raise AssertionError("trace_events.jsonl first record must be event_type=trace_header")
    payload = obj.get("payload", {})
    signature = payload.get("signature", {})
    if not isinstance(signature, dict):
        raise AssertionError("trace_header payload.signature must be dict")
    validate_required_signature(signature)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=str, required=True)
    args = ap.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.exists():
        raise FileNotFoundError(f"trace file not found: {trace_path}")

    _check_trace_csv(trace_path)
    _check_trace_events(trace_path)

    print("[SMOKE] trace signature/schema OK")


if __name__ == "__main__":
    main()
