"""Smoke check for StableHW schema alignment (v5.4 canonical)."""
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

    locked = getattr(stable_hw, "locked_acc_ref", None)
    guard = getattr(stable_hw, "accuracy_guard", None)
    sched = getattr(stable_hw, "lambda_hw_schedule", None)

    if locked is None:
        raise AssertionError("stable_hw.locked_acc_ref missing")
    if guard is None:
        raise AssertionError("stable_hw.accuracy_guard missing")
    if sched is None:
        raise AssertionError("stable_hw.lambda_hw_schedule missing")

    ctrl = getattr(guard, "controller", None)
    if ctrl is None:
        raise AssertionError("stable_hw.accuracy_guard.controller missing (v5.4 canonical)")

    print("[SMOKE] stable_hw.locked_acc_ref", locked)
    print("[SMOKE] stable_hw.accuracy_guard", guard)
    print("[SMOKE] stable_hw.accuracy_guard.controller", ctrl)
    print("[SMOKE] stable_hw.lambda_hw_schedule", sched)

    # locked
    _assert_type("stable_hw.locked_acc_ref.baseline_stats_path", locked.baseline_stats_path, (str, type(None)))
    _assert_type("stable_hw.locked_acc_ref.freeze_epoch", locked.freeze_epoch, int)
    _assert_type("stable_hw.locked_acc_ref.prefer_dense_baseline", locked.prefer_dense_baseline, bool)

    # controller canonical
    _assert_type("stable_hw.accuracy_guard.controller.mode", ctrl.mode, str)
    _assert_type("stable_hw.accuracy_guard.controller.metric", ctrl.metric, str)
    _assert_type("stable_hw.accuracy_guard.controller.epsilon_drop", ctrl.epsilon_drop, (float, int))
    _assert_type("stable_hw.accuracy_guard.controller.freeze_rho_on_violate", ctrl.freeze_rho_on_violate, bool)
    _assert_type("stable_hw.accuracy_guard.controller.cut_hw_loss_on_violate", ctrl.cut_hw_loss_on_violate, bool)
    _assert_type("stable_hw.accuracy_guard.controller.recovery_min_epochs", ctrl.recovery_min_epochs, int)
    _assert_type("stable_hw.accuracy_guard.controller.freeze_schedule_in_recovery", ctrl.freeze_schedule_in_recovery, bool)
    _assert_type("stable_hw.accuracy_guard.controller.k_exit", ctrl.k_exit, int)
    _assert_type("stable_hw.accuracy_guard.controller.margin_exit", ctrl.margin_exit, (float, int))

    # schedule
    _assert_type("stable_hw.lambda_hw_schedule.enabled", sched.enabled, bool)
    _assert_type("stable_hw.lambda_hw_schedule.warmup_epochs", sched.warmup_epochs, int)
    _assert_type("stable_hw.lambda_hw_schedule.ramp_epochs", sched.ramp_epochs, int)
    _assert_type("stable_hw.lambda_hw_schedule.lambda_hw_max", sched.lambda_hw_max, (float, int))

    print("[SMOKE] StableHW schema OK (v5.4)")


if __name__ == "__main__":
    main()
