"""Smoke check for StableHW schema alignment."""
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse

from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults


def _assert_type(name: str, value, expected) -> None:
    if not isinstance(value, expected):
        raise AssertionError(f"{name} expected {expected}, got {type(value)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    cfg = validate_and_fill_defaults(cfg, mode="version_c")

    stable_hw = getattr(cfg, "stable_hw", None)
    if stable_hw is None:
        raise AssertionError("stable_hw config missing")

    guard = getattr(stable_hw, "accuracy_guard", None)
    schedule = getattr(stable_hw, "lambda_hw_schedule", None)

    if guard is None:
        raise AssertionError("stable_hw.accuracy_guard missing")
    if schedule is None:
        raise AssertionError("stable_hw.lambda_hw_schedule missing")

    print("[SMOKE] stable_hw.accuracy_guard", guard)
    print("[SMOKE] stable_hw.lambda_hw_schedule", schedule)

    _assert_type("stable_hw.accuracy_guard.use_ema", guard.use_ema, bool)
    _assert_type("stable_hw.accuracy_guard.ema_beta", guard.ema_beta, (float, int))
    _assert_type("stable_hw.accuracy_guard.epsilon_drop", guard.epsilon_drop, (float, int))
    _assert_type("stable_hw.accuracy_guard.max_consecutive", guard.max_consecutive, int)

    on_violate = getattr(guard, "on_violate", None)
    if on_violate is None:
        raise AssertionError("stable_hw.accuracy_guard.on_violate missing")

    _assert_type("stable_hw.accuracy_guard.on_violate.scale_lambda_hw", on_violate.scale_lambda_hw, (float, int))
    _assert_type("stable_hw.accuracy_guard.on_violate.freeze_rho_epochs", on_violate.freeze_rho_epochs, int)
    _assert_type(
        "stable_hw.accuracy_guard.on_violate.disable_hw_after_max_violations",
        on_violate.disable_hw_after_max_violations,
        bool,
    )

    recover = getattr(on_violate, "recover", None)
    if recover is None:
        raise AssertionError("stable_hw.accuracy_guard.on_violate.recover missing")

    _assert_type("stable_hw.accuracy_guard.on_violate.recover.enable", recover.enable, bool)
    _assert_type("stable_hw.accuracy_guard.on_violate.recover.patience_epochs", recover.patience_epochs, int)
    _assert_type("stable_hw.accuracy_guard.on_violate.recover.restore_lambda_hw", recover.restore_lambda_hw, bool)

    _assert_type("stable_hw.lambda_hw_schedule.enabled", schedule.enabled, bool)
    _assert_type("stable_hw.lambda_hw_schedule.warmup_epochs", schedule.warmup_epochs, int)
    _assert_type("stable_hw.lambda_hw_schedule.ramp_epochs", schedule.ramp_epochs, int)
    _assert_type("stable_hw.lambda_hw_schedule.stabilize_epochs", schedule.stabilize_epochs, int)
    _assert_type("stable_hw.lambda_hw_schedule.lambda_hw_max", schedule.lambda_hw_max, (float, int))
    _assert_type("stable_hw.lambda_hw_schedule.lambda_hw_min", schedule.lambda_hw_min, (float, int))
    _assert_type("stable_hw.lambda_hw_schedule.clamp_min", schedule.clamp_min, (float, int))
    _assert_type("stable_hw.lambda_hw_schedule.clamp_max", schedule.clamp_max, (float, int))

    print("[SMOKE] StableHW schema OK")


if __name__ == "__main__":
    main()
