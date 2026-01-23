# -*- coding: utf-8 -*-
"""Smoke check for v5.4 gating evidence fields in trace_events.jsonl."""

import argparse
import json
from pathlib import Path


def _find_first_gating_event(trace_events_path: Path) -> dict | None:
    with trace_events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("event_type") == "gating":
                return obj
    return None


def _as_float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _check_gating_event(event: dict) -> None:
    payload = event.get("payload", {}) if isinstance(event, dict) else {}
    guard_mode = str(payload.get("guard_mode", "")).upper()
    acc_ref = _as_float(payload.get("acc_ref", 0.0), 0.0)
    acc_now = _as_float(payload.get("acc_now", 0.0), 0.0)
    acc_drop = _as_float(payload.get("acc_drop", 0.0), 0.0)
    acc_drop_max = _as_float(payload.get("acc_drop_max", 0.0), 0.0)

    if acc_drop_max <= 0.0:
        raise AssertionError("gating acc_drop_max must be > 0 for SPEC_E evidence")

    if guard_mode != "WARMUP":
        if acc_ref <= 0.0:
            raise AssertionError("gating acc_ref must be > 0 outside warmup")
        if acc_now <= 0.0:
            raise AssertionError("gating acc_now must be > 0 outside warmup")

    if acc_ref > 0.0 or acc_now > 0.0:
        expected_drop = max(0.0, acc_ref - acc_now)
        if abs(acc_drop - expected_drop) > 1e-6:
            raise AssertionError(
                f"gating acc_drop mismatch: acc_drop={acc_drop} expected={expected_drop}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_dir", type=str, required=True)
    args = ap.parse_args()

    trace_dir = Path(args.trace_dir)
    trace_events_path = trace_dir / "trace_events.jsonl"
    if not trace_events_path.exists():
        print(f"[SMOKE] trace_events.jsonl missing at {trace_events_path}; skipping gating evidence check")
        return

    event = _find_first_gating_event(trace_events_path)
    if event is None:
        raise AssertionError("no gating event found in trace_events.jsonl")

    _check_gating_event(event)
    print("[SMOKE] gating evidence OK")


if __name__ == "__main__":
    main()
