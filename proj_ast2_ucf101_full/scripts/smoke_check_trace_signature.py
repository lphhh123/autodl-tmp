# -*- coding: utf-8 -*-
"""Smoke check for v5.4 trace signature + schema in trace.csv."""

import argparse
import csv
import json
from pathlib import Path

from utils.trace_schema import TRACE_FIELDS
from utils.trace_signature_v54 import REQUIRED_SIGNATURE_FIELDS


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
    trace_dir = trace_path.parent / "trace"
    if trace_path.parent.name == "trace":
        trace_dir = trace_path.parent
    header_path = trace_dir / "trace_header.json"
    if not header_path.exists():
        return
    header = json.loads(header_path.read_text(encoding="utf-8"))
    signature = header.get("signature", {})
    if not isinstance(signature, dict):
        raise AssertionError("trace_header.signature must be dict")
    missing = [k for k in REQUIRED_SIGNATURE_FIELDS if k not in signature]
    if missing:
        raise AssertionError(f"trace_header signature missing required fields: {missing}")

    for fn in ["manifest.json", "trace_summary.json"]:
        p = trace_dir / fn
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")


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
