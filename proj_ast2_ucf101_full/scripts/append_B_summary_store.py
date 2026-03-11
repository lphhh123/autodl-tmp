#!/usr/bin/env python3
"""Append a compact sweep summary into a persistent store under outputs/B/_summary_store.

Dedup by run_id, keep store small and stable.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Set


def read_tsv(p: Path) -> List[Dict[str, str]]:
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in r]


def write_tsv(rows: List[Dict[str, str]], out: Path) -> None:
    if not rows:
        out.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in cols})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-tsv", required=True)
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--store-name", default="B_key_runs_store.tsv")
    args = ap.parse_args()

    in_tsv = Path(args.in_tsv)
    store_dir = Path(args.store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)
    store_tsv = store_dir / str(args.store_name)

    new_rows = read_tsv(in_tsv)
    old_rows = read_tsv(store_tsv)

    seen: Set[str] = set()
    for r in old_rows:
        rid = (r.get("run_id") or "").strip()
        if rid:
            seen.add(rid)

    merged = list(old_rows)
    for r in new_rows:
        rid = (r.get("run_id") or "").strip()
        if not rid or rid in seen:
            continue
        merged.append(r)
        seen.add(rid)

    def key(r: Dict[str, str]):
        try:
            b = int(float(r.get("budget_calls", "0") or "0"))
        except Exception:
            b = 0
        try:
            sd = int(float(r.get("seed", "-1") or "-1"))
        except Exception:
            sd = -1
        return (
            r.get("exp_key", ""),
            r.get("run_tag", ""),
            b,
            r.get("instance", ""),
            sd,
            r.get("run_id", ""),
        )

    merged.sort(key=key)
    write_tsv(merged, store_tsv)
    print(f"[STORE] rows={len(merged)} -> {store_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
