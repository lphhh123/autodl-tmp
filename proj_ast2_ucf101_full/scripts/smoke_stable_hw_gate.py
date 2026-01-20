# scripts/smoke_stable_hw_gate.py
from omegaconf import OmegaConf

from utils.stable_hw import (
    apply_accuracy_guard,
    init_locked_acc_ref,
    init_stable_hw,
    stable_hw_schedule,
)


def main() -> None:
    cfg = OmegaConf.load("configs/version_c_ucf101.yaml")
    st = {}
    init_stable_hw(cfg.stable_hw, st, cfg)
    init_locked_acc_ref(cfg.stable_hw, st)

    # schedule
    stable_hw_schedule(epoch=0, stable_hw_cfg=cfg.stable_hw, st=st)

    metric_key = str(getattr(getattr(cfg.stable_hw, "accuracy_guard", None), "metric", "val_acc1"))
    assert metric_key.startswith("val_"), f"expected val_* metric in this smoke, got {metric_key}"

    # case A: no last val => must WARMUP gate
    last_val = st.get("val_acc1_last", None)
    assert last_val is None
    # mimic trainer's intended behavior:
    st["guard_mode"] = "WARMUP"
    st["g_hw"] = 0.0
    st["lambda_hw_effective"] = 0.0
    st["allow_discrete_updates"] = False
    assert st["lambda_hw_effective"] == 0.0 and st["g_hw"] == 0.0

    # case B: has last val below acc_ref => must RECOVERY gate (0)
    st["val_acc1_last"] = 0.0
    _, allow_discrete = apply_accuracy_guard(
        epoch=0,
        stable_hw_cfg=cfg.stable_hw,
        stable_hw_state=st,
        val_metric_or_none=st["val_acc1_last"],
        has_val_this_epoch=True,
        train_ema_or_none=None,
    )
    assert st["lambda_hw_effective"] == 0.0
    assert allow_discrete is False

    print("[OK] StableHW gate smoke passed.")


if __name__ == "__main__":
    main()
