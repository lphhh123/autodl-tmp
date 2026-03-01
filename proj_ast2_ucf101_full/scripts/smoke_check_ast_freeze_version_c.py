# NOTE: Bound to SPEC_E v5.4. This is a cheap smoke to avoid 3-day blind runs.
"""Smoke check: Version-C must freeze/pause AST schedule when StableHW is in recovery.

This prevents token_keep from continuing to drop while StableHW (Acc-first hard gating)
tries to recover validation accuracy; otherwise runs get stuck in VIOLATE forever.
"""
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults
from trainer.trainer_version_c import compute_ast_schedule_effective_with_stable_hw_freeze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="configs/vc_phase3_full_ucf101_A_main.yaml")
    args = ap.parse_args()

    cfg = validate_and_fill_defaults(load_config(args.cfg), mode="version_c")

    st = {"ast_sched_virtual_epoch": 0}

    # Epoch0 (not frozen): should advance virtual epoch to 1
    s0, e0 = compute_ast_schedule_effective_with_stable_hw_freeze(cfg, st, outer=0)
    assert isinstance(s0, dict)
    assert int(e0) == 0
    assert int(st["ast_sched_virtual_epoch"]) == 1

    # Freeze: schedule must NOT advance and must be dense
    st["freeze_schedule"] = True
    sF, eF = compute_ast_schedule_effective_with_stable_hw_freeze(cfg, st, outer=1)
    assert int(st["ast_sched_virtual_epoch"]) == 1, "virtual epoch must pause while frozen"
    assert bool(sF.get("force_dense", False)) is True
    assert abs(float(sF.get("rho_token", 0.0)) - 1.0) < 1e-12
    assert abs(float(sF.get("lambda_ast", 1.0)) - 0.0) < 1e-12

    # Unfreeze: should resume from virtual epoch 1, and then advance to 2
    st["freeze_schedule"] = False
    s1, e1 = compute_ast_schedule_effective_with_stable_hw_freeze(cfg, st, outer=2)
    assert int(e1) == 1, "must resume from paused virtual epoch (not jump to outer index)"
    assert int(st["ast_sched_virtual_epoch"]) == 2

    print("[SMOKE-AST-FREEZE] OK")
    print("  e0 =", e0, "s0.phase =", s0.get("phase"))
    print("  freeze.phase =", sF.get("phase"), "rho =", sF.get("rho_token"), "lambda_ast =", sF.get("lambda_ast"))
    print("  e1 =", e1, "s1.phase =", s1.get("phase"))


if __name__ == "__main__":
    main()
