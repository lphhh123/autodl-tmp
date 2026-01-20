"""Ensure baseline_stats.json exists for LockedAccRef / StableHW.

This script is intentionally lightweight:
  * It does NOT require torch / dataset availability.
  * It guarantees a valid JSON schema for `utils.stable_hw.init_hw_refs_from_baseline_stats()`.

If you have the full training/eval environment, you can later extend
the placeholder parts to produce true baseline values. For now, we write
safe placeholders so Version-C pipelines do not crash and StableHW can lock a reference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any, Dict, Optional

from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults


def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


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


def _safe_float(x: Any, default: float) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return default
        if v in (float("inf"), float("-inf")):
            return default
        return v
    except Exception:
        return default


def _placeholder_hw_ref(cfg: Any) -> Dict[str, float]:
    # Prefer a user-provided target_latency_ms if present; otherwise 1.0 to keep ratios defined.
    lat = 1.0
    if hasattr(cfg, "hw") and isinstance(getattr(cfg.hw, "target_latency_ms", None), (int, float)):
        lat = max(1e-6, float(cfg.hw.target_latency_ms))
    return {"latency_ms": float(lat), "energy_mj": 1.0, "mem_peak_mb": 1.0}


def _placeholder_acc_ref(cfg: Any) -> float:
    # If config pins an acc_ref_fixed, honor it; else default 0.0 (conservative; never falsely blocks).
    try:
        if hasattr(cfg, "stable_hw") and hasattr(cfg.stable_hw, "locked_acc_ref"):
            fixed = getattr(cfg.stable_hw.locked_acc_ref, "acc_ref_fixed", None)
            if fixed is not None:
                return _safe_float(fixed, 0.0)
    except Exception:
        pass
    return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Config YAML (baseline).")
    ap.add_argument("--out", required=True, help="Output baseline_stats.json path.")
    ap.add_argument("--force", action="store_true", help="Overwrite even if file exists.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if not args.force:
        existing = _read_json(args.out)
        if existing is not None and _is_valid_baseline_stats(existing):
            print(f"[ensure_baseline_stats] OK (exists): {args.out}")
            return

    cfg = load_config(args.cfg)
    cfg = validate_and_fill_defaults(cfg)

    baseline = {
        "acc_ref": _placeholder_acc_ref(cfg),
        "last_hw_stats": _placeholder_hw_ref(cfg),
        "created_at_unix": int(time.time()),
        "cfg_path": os.path.abspath(args.cfg),
        "cfg_fingerprint_sha1": _sha1_text(json.dumps(cfg, sort_keys=True, default=str)),
        "note": (
            "Placeholder baseline stats generated without running full evaluation. "
            "To use true LockedAccRef, regenerate this file from a real baseline run."
        ),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"[ensure_baseline_stats] Wrote: {args.out}")


if __name__ == "__main__":
    main()
