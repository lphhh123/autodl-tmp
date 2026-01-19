import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse

from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    ap.add_argument("--mode", type=str, default="version_c")
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    cfg = validate_and_fill_defaults(cfg, mode=args.mode)

    # ---- SPEC_E: NoDoubleScale must be true and legacy lambdas must be 0 when stable_hw enabled
    stable_en = bool(getattr(getattr(cfg, "stable_hw", None), "enabled", False))
    if stable_en:
        if hasattr(cfg, "loss") and float(getattr(cfg.loss, "lambda_hw", 0.0) or 0.0) != 0.0:
            raise AssertionError("NoDoubleScale violated: loss.lambda_hw must be 0 when stable_hw.enabled=True")
        if hasattr(cfg, "hw") and float(getattr(cfg.hw, "lambda_hw", 0.0) or 0.0) != 0.0:
            raise AssertionError("NoDoubleScale violated: hw.lambda_hw must be 0 when stable_hw.enabled=True")
        if not bool(getattr(cfg.stable_hw, "no_double_scale", True)):
            raise AssertionError("stable_hw.no_double_scale must be True in v5.4")

    print("[SMOKE] config no_drift OK. stable_hw.enabled=", stable_en)


if __name__ == "__main__":
    main()
