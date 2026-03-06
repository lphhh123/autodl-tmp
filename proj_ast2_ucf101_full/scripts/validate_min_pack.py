#!/usr/bin/env python3
"""Validate a MIN-pack run directory.

This script is intentionally lightweight and dependency-free.
It is used by scripts/pack_B_min.sh to avoid uploading non-diagnosable packs.

Checks:
  - required files exist
  - budget.json indicates a budget-full run (budget_exhausted && actual_eval_calls == primary_limit.limit)
  - trace_meta.json contains mpvs stats when mpvs is enabled
  - report.json contains selected_total_scalar and key efficiency metrics

Outputs a JSON report with a stable schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


REQUIRED = [
    "report.json",
    "budget.json",
    "effective_config_snapshot.yaml",
    "manifest.json",
    "trace_meta.json",
]


def _read_json(p: Path, default: Any) -> Any:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _budget_full(budget: Dict[str, Any]) -> bool:
    try:
        exhausted = bool(budget.get("budget_exhausted", False))
        actual = int(budget.get("actual_eval_calls", -1))
        lim = int((budget.get("primary_limit", {}) or {}).get("limit", -1))
        return exhausted and lim > 0 and actual == lim
    except Exception:
        return False


def validate_run_dir(run_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "ok": True,
        "missing": [],
        "budget_full": False,
        "warnings": [],
    }

    for fn in REQUIRED:
        if not (run_dir / fn).exists():
            out["missing"].append(fn)

    if out["missing"]:
        out["ok"] = False

    budget = _read_json(run_dir / "budget.json", {})
    out["budget_full"] = _budget_full(budget)
    if not out["budget_full"]:
        out["warnings"].append("budget_not_full")

    report = _read_json(run_dir / "report.json", {})
    if "selected_total_scalar" not in report:
        out["ok"] = False
        out["warnings"].append("missing_selected_total_scalar")

    # Efficiency metrics are required for diagnosing eval-call fixed tax.
    for k in [
        "calls_per_iter_overall",
        "calls_per_iter_lastN",
        "steps_per_1k_calls_lastN",
        "improve_steps_per_1k_calls_lastN",
    ]:
        if k not in report:
            out["warnings"].append(f"missing_metric:{k}")

    meta = _read_json(run_dir / "trace_meta.json", {})
    mpvs = meta.get("mpvs") if isinstance(meta, dict) else None
    if isinstance(mpvs, dict):
        out["mpvs_present"] = True
        # minimal sanity: some trigger/call accounting keys should exist
        for k in ["plans_scored", "calls_by_src", "trig_enabled"]:
            if k not in mpvs:
                out["warnings"].append(f"mpvs_missing:{k}")
    else:
        out["mpvs_present"] = False
        # If report says MPVS was enabled but meta lacks it, that's a diagnosability failure.
        if report.get("mpvs") is not None:
            out["ok"] = False
            out["warnings"].append("mpvs_enabled_but_meta_missing")

    # Optional: stdout_tail exists in MIN stage.
    if not any((run_dir / f).exists() for f in ["stdout_tail_400.log", "stdout_tail_200.log", "stdout_tail_800.log"]):
        out["warnings"].append("missing_stdout_tail")

    # Optional: run_summary.json
    if (run_dir / "run_summary.json").exists():
        out["run_summary_present"] = True
    else:
        out["run_summary_present"] = False
        out["warnings"].append("missing_run_summary")

    out["selected_total_scalar"] = report.get("selected_total_scalar")
    out["actual_eval_calls"] = budget.get("actual_eval_calls")
    out["primary_limit"] = (budget.get("primary_limit", {}) or {}).get("limit")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    rep = validate_run_dir(run_dir)

    s = json.dumps(rep, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(s, encoding="utf-8")
    else:
        print(s)

    # Exit code indicates hard validity only (missing required files / schema broken).
    sys.exit(0 if rep.get("ok", False) else 2)


if __name__ == "__main__":
    main()
