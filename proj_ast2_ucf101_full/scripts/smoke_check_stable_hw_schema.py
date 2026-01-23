# NOTE: This smoke is bound to SPEC_E v5.4; do not change fields without updating SPEC_E + trace_contract_v54.py
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


def _is_mapping(x) -> bool:
    if x is None:
        return False
    if isinstance(x, dict):
        return True
    try:
        from omegaconf import DictConfig  # type: ignore
        return isinstance(x, DictConfig)
    except Exception:
        return False


def _assert_mapping(path: str, x) -> None:
    if not _is_mapping(x):
        raise AssertionError(f"{path} must be a mapping (dict/DictConfig); got {type(x)}")


def _assert_type(name: str, value, expected) -> None:
    if not isinstance(value, expected):
        raise AssertionError(f"{name} expected {expected}, got {type(value)}")


def _get(x, k):
    if isinstance(x, dict):
        return x.get(k)
    return getattr(x, k, None)


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
    norm = getattr(stable_hw, "normalize", None)

    if locked is None:
        raise AssertionError("stable_hw.locked_acc_ref missing")
    if guard is None:
        raise AssertionError("stable_hw.accuracy_guard missing")
    if sched is None:
        raise AssertionError("stable_hw.lambda_hw_schedule missing")
    if norm is None:
        raise AssertionError("stable_hw.normalize missing")

    ctrl = getattr(guard, "controller", None)
    if ctrl is None:
        raise AssertionError("stable_hw.accuracy_guard.controller missing (v5.4 canonical)")

    print("[SMOKE] stable_hw.locked_acc_ref", locked)
    print("[SMOKE] stable_hw.accuracy_guard", guard)
    print("[SMOKE] stable_hw.accuracy_guard.controller", ctrl)
    print("[SMOKE] stable_hw.lambda_hw_schedule", sched)

    # locked
    _assert_mapping("stable_hw.locked_acc_ref", locked)
    _assert_type("stable_hw.locked_acc_ref.baseline_stats_path", _get(locked, "baseline_stats_path"), (str, type(None)))
    _assert_type("stable_hw.locked_acc_ref.freeze_epoch", _get(locked, "freeze_epoch"), int)
    _assert_type("stable_hw.locked_acc_ref.prefer_dense_baseline", _get(locked, "prefer_dense_baseline"), bool)

    # controller canonical (v5.4)
    _assert_type("stable_hw.accuracy_guard.controller.mode", _get(ctrl, "mode"), str)
    _assert_type("stable_hw.accuracy_guard.controller.metric", _get(ctrl, "metric"), str)
    _assert_type("stable_hw.accuracy_guard.controller.epsilon_drop", _get(ctrl, "epsilon_drop"), (float, int))
    _assert_type("stable_hw.accuracy_guard.controller.recovery_min_epochs", _get(ctrl, "recovery_min_epochs"), int)
    _assert_type("stable_hw.accuracy_guard.controller.cut_hw_loss_on_violate", _get(ctrl, "cut_hw_loss_on_violate"), bool)
    _assert_type("stable_hw.accuracy_guard.controller.freeze_discrete_updates", _get(ctrl, "freeze_discrete_updates"), bool)
    _assert_type("stable_hw.accuracy_guard.controller.freeze_schedule_in_recovery", _get(ctrl, "freeze_schedule_in_recovery"), bool)
    _assert_type("stable_hw.accuracy_guard.controller.k_exit", _get(ctrl, "k_exit"), int)
    _assert_type("stable_hw.accuracy_guard.controller.margin_exit", _get(ctrl, "margin_exit"), (float, int))

    # schedule
    _assert_type("stable_hw.lambda_hw_schedule.enabled", _get(sched, "enabled"), bool)
    _assert_type("stable_hw.lambda_hw_schedule.warmup_epochs", _get(sched, "warmup_epochs"), int)
    _assert_type("stable_hw.lambda_hw_schedule.ramp_epochs", _get(sched, "ramp_epochs"), int)
    _assert_type("stable_hw.lambda_hw_schedule.lambda_hw_max", _get(sched, "lambda_hw_max"), (float, int))

    # normalize
    _assert_type("stable_hw.normalize.mode", _get(norm, "mode"), str)

    print("[SMOKE] StableHW schema OK (v5.4)")


if __name__ == "__main__":
    main()
