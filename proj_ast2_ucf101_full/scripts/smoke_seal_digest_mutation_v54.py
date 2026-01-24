import argparse
import json
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults
from utils.contract_seal import assert_cfg_sealed_or_violate
from utils.trace_guard import init_trace_dir
from utils.trace_signature_v54 import build_signature_v54


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/vc_phase3_full_ucf101.yaml")
    parser.add_argument("--out_dir", type=str, default="outputs/smoke_seal_digest_mutation_v54")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    cfg = validate_and_fill_defaults(cfg, mode="version_c")
    seal_digest = str(getattr(getattr(cfg, "contract", None), "seal_digest", ""))
    if not seal_digest:
        raise RuntimeError("Missing contract.seal_digest after validation.")

    trace_dir = Path(args.out_dir) / "trace"
    meta = init_trace_dir(
        trace_dir=trace_dir,
        signature=build_signature_v54(cfg, method_name="smoke_seal_digest_mutation_v54"),
        run_meta={"mode": "smoke_seal_digest_mutation_v54"},
        resolved_config=cfg.to_dict() if hasattr(cfg, "to_dict") else cfg,
        requested_config=getattr(getattr(cfg, "_contract", None), "requested_config_snapshot", None),
        contract_overrides=getattr(getattr(cfg, "_contract", None), "overrides", []),
    )
    trace_events_path = Path(meta["trace_events"])

    cfg.stable_hw.min_latency_ms = float(getattr(cfg.stable_hw, "min_latency_ms", 1.0)) + 1.0

    try:
        assert_cfg_sealed_or_violate(cfg, seal_digest, trace_events_path, step=0)
    except RuntimeError:
        events = trace_events_path.read_text(encoding="utf-8").splitlines()
        if not events:
            raise AssertionError("Expected contract_violation event, but trace_events.jsonl is empty.")
        parsed = [json.loads(line) for line in events if line.strip()]
        matched = [
            e for e in parsed
            if e.get("event_type") == "contract_violation"
            and e.get("payload", {}).get("reason") == "cfg_mutated_after_seal"
        ]
        if not matched:
            raise AssertionError("Expected contract_violation with reason=cfg_mutated_after_seal.")
        print("[SMOKE] PASS: seal_digest mutation detected and logged")
        return

    raise AssertionError("Expected RuntimeError on cfg mutation, but no error was raised.")


if __name__ == "__main__":
    main()
