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
        # v5.4 required for trace/signature alignment
        "op_args_json",
        "signature",
    }
    seen_any = False
    with open(path, "r", encoding="utf-8") as f:
        for ln, raw in enumerate(f, 1):
            raw = raw.strip()
            if not raw:
                continue
            seen_any = True
            obj = json.loads(raw)
            miss = sorted(list(required - set(obj.keys())))
            if miss:
                raise SystemExit(f"Missing keys at line {ln}: {miss}")
            signature = obj.get("signature")
            if not isinstance(signature, str):
                raise SystemExit(f"signature must be string (assign:...) at line {ln}")
            if not signature.startswith("assign:"):
                raise SystemExit(f"signature must start with 'assign:' at line {ln}")
    if not seen_any:
        raise SystemExit(f"recordings file has no valid jsonl rows: {path}")
    print(f"OK schema: {path}")


if __name__ == "__main__":
    main()
