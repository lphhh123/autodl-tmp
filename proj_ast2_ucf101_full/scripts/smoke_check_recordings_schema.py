import json
import sys


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python smoke_check_recordings_schema.py <recordings.jsonl>")
    path = sys.argv[1]
    required = {
        "iter",
        "stage",
        "op",
        "accepted",
        "total_scalar",
        "comm_norm",
        "therm_norm",
        "pareto_added",
        "duplicate_penalty",
        "boundary_penalty",
        "seed_id",
        "time_ms",
    }
    with open(path, "r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            miss = sorted(list(required - set(obj.keys())))
            if miss:
                raise SystemExit(f"Missing keys at line {ln}: {miss}")
    print(f"OK schema: {path}")


if __name__ == "__main__":
    main()
