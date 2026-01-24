"""Ensure baseline_stats.json exists for LockedAccRef / StableHW.

This script is intentionally lightweight:
  * It does NOT require torch / dataset availability.
  * It guarantees a valid JSON schema for `utils.stable_hw.init_hw_refs_from_baseline_stats()`.

If you have the full training/eval environment, you can later extend
the placeholder parts to produce true baseline values. For now, we write
safe placeholders so Version-C pipelines do not crash and StableHW can lock a reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _is_valid_baseline_stats(obj: Dict[str, Any]) -> bool:
    # Accepts multiple aliases in stable_hw.init_hw_refs_from_baseline_stats();
    # we enforce the minimal schema we write: acc_ref + last_hw_stats.
    if not isinstance(obj, dict):
        return False
    if "acc_ref" not in obj and "acc_ref_fixed" not in obj and "val_top1" not in obj:
        return False
    hw = obj.get("last_hw_stats") or obj.get("hw_ref") or obj.get("baseline_hw_stats")
    if not isinstance(hw, dict):
        return False
    for k in ("latency_ms", "energy_mj", "mem_peak_mb"):
        if k not in hw:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Config YAML (baseline).")
    ap.add_argument("--out", required=True, help="Output baseline_stats.json path.")
    ap.add_argument("--force", action="store_true", help="Overwrite even if file exists.")
    ap.add_argument(
        "--write_placeholder",
        action="store_true",
        help="Write a PLACEHOLDER baseline_stats.json (SMOKE ONLY). Default: False.",
    )
    ap.add_argument(
        "--placeholder_reason",
        type=str,
        default="smoke_only",
        help="Reason string recorded into placeholder baseline_stats.json",
    )
    ap.add_argument(
        "--placeholder_acc",
        type=float,
        default=0.0,
        help="Placeholder acc value (SMOKE ONLY)",
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if not args.force:
        existing = _read_json(args.out)
        if existing is not None and _is_valid_baseline_stats(existing):
            print(f"[ensure_baseline_stats] OK (exists): {args.out}")
            return

    cfg = load_config(args.cfg)
    cfg = validate_and_fill_defaults(cfg)

    if not args.write_placeholder:
        raise SystemExit(
            "[P0][v5.4] baseline_stats.json not found or invalid.\n"
            "Refusing to auto-generate placeholder baseline_stats (would break LockedAccRef contract).\n"
            "Please generate real baseline_stats via the official baseline/eval pipeline, or\n"
            "explicitly pass --write_placeholder ONLY for smoke/debug and then set\n"
            "stable_hw.locked_acc_ref.allow_placeholder=true (and accept it will be flagged in trace)."
        )

    placeholder = {
        "schema_version": "v5.4_baseline_stats_v1",
        "is_placeholder": True,
        "placeholder_reason": "auto_created_by_ensure_baseline_stats_when_missing",
        "val_acc1": 0.0,
        "last_hw_stats": {"lat_ms": 1.0, "power_mw": 1.0, "mem_mb": 1.0},
        "acc_ref": 0.0,
        "note": "placeholder baseline_stats (NOT real). Replace with a real baseline run.",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(placeholder, f, indent=2, ensure_ascii=False)
    print(f"[WARN][v5.4] Wrote PLACEHOLDER baseline_stats to: {args.out}")


if __name__ == "__main__":
    main()
