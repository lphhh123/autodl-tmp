"""Smoke check that layout_signature is assign-only (SPEC_B v5.4)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _load_lines(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"stats log not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise AssertionError(f"No records found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_path",
        type=str,
        default=None,
        help="Path to logs/version_c_stats.jsonl.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Training output dir containing logs/version_c_stats.jsonl.",
    )
    args = parser.parse_args()

    if not args.log_path and not args.out_dir:
        raise SystemExit("Provide --log_path or --out_dir.")

    if args.log_path:
        log_path = Path(args.log_path)
    else:
        log_path = Path(args.out_dir) / "logs" / "version_c_stats.jsonl"

    rows = _load_lines(log_path)
    for idx, row in enumerate(rows):
        layout_sig = row.get("layout_signature")
        mapping_sig = row.get("mapping_signature")
        if not layout_sig:
            raise AssertionError(f"Missing layout_signature at line {idx + 1}")
        if not str(layout_sig).startswith("assign:"):
            raise AssertionError(f"layout_signature not assign-only at line {idx + 1}: {layout_sig!r}")
        if HEX40.match(str(layout_sig)) or HEX64.match(str(layout_sig)):
            raise AssertionError(f"layout_signature looks like hash at line {idx + 1}: {layout_sig!r}")
        if not mapping_sig:
            raise AssertionError(f"Missing mapping_signature at line {idx + 1}")

    print(f"[OK] {log_path} layout_signature is assign-only and mapping_signature present.")


if __name__ == "__main__":
    main()
