import argparse

from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults
from utils.stable_hw import (
    apply_accuracy_guard,
    init_hw_refs_from_baseline_stats,
    init_locked_acc_ref,
    stable_hw_schedule,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="configs/vc_phase3_full_ucf101.yaml")
    ap.add_argument("--mode", type=str, default="version_c")
    args = ap.parse_args()

    cfg = load_config(args.cfg)
    cfg = validate_and_fill_defaults(cfg, mode=args.mode)

    st = {}
    init_hw_refs_from_baseline_stats(cfg, st, cfg.stable_hw)
    init_locked_acc_ref(cfg, st)

    # Case 1: acc_ref not ready => lambda_hw must be 0
    stable_hw_schedule(0, cfg.stable_hw, st)
    decision, _ = apply_accuracy_guard(cfg, st, val_metric_or_none=None, epoch=0, has_val_this_epoch=False)
    assert st.get("lambda_hw_effective", 0.0) == 0.0, "acc_ref missing => lambda_hw_effective must be 0"

    # Case 2: set a locked acc_ref and violate => lambda_hw must be 0
    st["acc_ref"] = 0.9
    st["acc_ref_locked"] = True
    stable_hw_schedule(1, cfg.stable_hw, st)
    decision2, _ = apply_accuracy_guard(cfg, st, val_metric_or_none=0.0, epoch=1, has_val_this_epoch=True)
    assert st.get("lambda_hw_effective", 0.0) == 0.0, "violate => lambda_hw_effective must be 0"

    print("[OK] smoke_stable_hw_gate: gating behaves as expected.")


if __name__ == "__main__":
    main()
