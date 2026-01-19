"""
Runtime smoke check for v5.4 StableHW controller semantics.
Goal: verify schedule->guard->effective is a single closed loop.
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
from utils.stable_hw import stable_hw_schedule, apply_accuracy_guard, init_locked_acc_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, required=True)
    args = ap.parse_args()

    cfg = validate_and_fill_defaults(load_config(args.cfg), mode="version_c")
    shw = getattr(cfg, "stable_hw", None)
    assert shw is not None and bool(getattr(shw, "enabled", False)), "stable_hw must be enabled for this smoke"

    st = {}
    # LockedAccRef must exist (but we don't require baseline_stats file here)
    init_locked_acc_ref(shw, st)
    if st.get("acc_ref") is None:
        # fallback for smoke: set a deterministic ref
        st["acc_ref"] = 0.80
    st["_acc_ref_once"] = float(st["acc_ref"])

    # epoch0 schedule
    stable_hw_schedule(0, shw, st)
    base0 = float(st["lambda_hw_base"])
    assert float(st["lambda_hw_effective"]) == base0, "pre-guard effective must equal base"

    # violate (acc below ref - eps)
    apply_accuracy_guard(
        epoch=0,
        stable_hw_cfg=shw,
        stable_hw_state=st,
        val_metric_or_none=float(st["acc_ref"]) - 1.0,  # force violate
        has_val_this_epoch=True,
        train_ema_or_none=None,
    )
    assert st["guard_mode"] in ("VIOLATE", "RECOVERY"), "guard must enter violate/recovery when forced"
    assert float(st["lambda_hw_effective"]) <= base0 + 1e-12, "effective must not exceed base in violate"
    assert bool(st.get("in_recovery", False)) is True, "in_recovery must be True in violate/recovery"

    # recover (acc above ref + margin)
    apply_accuracy_guard(
        epoch=1,
        stable_hw_cfg=shw,
        stable_hw_state=st,
        val_metric_or_none=float(st["acc_ref"]) + 1.0,  # force good acc
        has_val_this_epoch=True,
        train_ema_or_none=None,
    )
    # may still be in RECOVERY depending on k_exit, but effective must be defined and finite
    assert "lambda_hw_effective" in st and abs(float(st["lambda_hw_effective"])) < 1e9

    print("[SMOKE-RUNTIME] OK")
    print("  acc_ref =", st["acc_ref"])
    print("  lambda_hw_base =", st["lambda_hw_base"])
    print("  lambda_hw_effective =", st["lambda_hw_effective"])
    print("  guard_mode =", st.get("guard_mode"))
    print("  in_recovery =", st.get("in_recovery"))
    print("  freeze_schedule =", st.get("freeze_schedule"))


if __name__ == "__main__":
    main()
