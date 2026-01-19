import csv
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse

REQUIRED_COLS = [
    "iter",
    "stage",
    "op",
    "op_args_json",
    "accepted",
    "total_scalar",
    "comm_norm",
    "therm_norm",
    "pareto_added",
    "duplicate_penalty",
    "boundary_penalty",
    "seed_id",
    "time_ms",
    "signature",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=str, required=True, help="path to trace.csv")
    args = ap.parse_args()

    p = Path(args.trace)
    if not p.exists():
        raise FileNotFoundError(str(p))

    with p.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        cols = r.fieldnames or []
        for c in REQUIRED_COLS:
            if c not in cols:
                raise AssertionError(f"trace schema missing col: {c}")

        rows = list(r)
        if len(rows) == 0:
            raise AssertionError("trace.csv has no rows (must contain init row even when steps=0)")

        init = rows[0]
        if str(init.get("stage", "")) != "init":
            raise AssertionError("first row stage must be 'init'")
        if str(init.get("op", "")) != "init":
            raise AssertionError("first row op must be 'init'")
        if str(init.get("accepted", "")) not in ("1", "True", "true"):
            raise AssertionError("first row accepted must be 1/True")
        if not str(init.get("signature", "")).startswith("assign:"):
            raise AssertionError("first row signature must start with 'assign:'")

    print("[SMOKE] trace schema OK:", str(p))


if __name__ == "__main__":
    main()
