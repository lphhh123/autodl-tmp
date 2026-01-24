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
from utils.trace_guard import init_trace_dir
from utils.trace_signature_v54 import build_signature_v54


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/vc_phase3_full_ucf101.yaml")
    parser.add_argument("--out_dir", type=str, default="outputs/smoke_trace_header_audit_fields_v54")
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    cfg = validate_and_fill_defaults(cfg, mode="version_c")

    resolved_cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else cfg
    requested_cfg = getattr(getattr(cfg, "_contract", None), "requested_config_snapshot", None)
    signature = build_signature_v54(cfg, method_name="smoke_trace_header_audit_fields_v54")

    trace_dir = Path(args.out_dir) / "trace"
    meta = init_trace_dir(
        trace_dir=trace_dir,
        signature=signature,
        run_meta={"mode": "smoke_trace_header_audit_fields_v54"},
        resolved_config=resolved_cfg,
        requested_config=requested_cfg,
        contract_overrides=getattr(getattr(cfg, "_contract", None), "overrides", []),
    )

    header_path = Path(meta["trace_header"])
    header = json.loads(header_path.read_text(encoding="utf-8"))
    required = [
        "requested_config_snapshot",
        "effective_config_snapshot",
        "contract_overrides",
        "seal_digest",
        "signature",
    ]
    missing = [k for k in required if k not in header]
    if missing:
        raise AssertionError(f"trace_header.json missing keys: {missing}")

    print("[SMOKE] trace_header audit fields present")


if __name__ == "__main__":
    main()
