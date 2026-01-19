"""Smoke check for StableHW contract invariants (SPEC v5.4)."""
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse

import torch

from hw_proxy.hw_loss import compute_hw_loss
from hw_proxy.layer_hw_proxy import LayerHwProxy
from mapping.mapping_solver import MappingSolver
from mapping.segments import Segment
from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    cfg = validate_and_fill_defaults(cfg, mode="version_c")

    if float(getattr(cfg.hw, "lambda_hw", 0.0) or 0.0) != 0.0:
        raise AssertionError("cfg.hw.lambda_hw must be 0 under v5.4 NoDoubleScale")

    stable_hw_cfg = getattr(cfg, "stable_hw", None)
    if stable_hw_cfg is None:
        raise AssertionError("stable_hw config missing")
    if not bool(getattr(getattr(stable_hw_cfg, "normalize", None), "enabled", True)):
        raise AssertionError("stable_hw.normalize.enabled must be true for contract check")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hw_proxy = LayerHwProxy(cfg.hw.device_name, cfg.hw.gpu_yaml, cfg.hw.proxy_weight_dir)
    mapping_solver = MappingSolver(strategy="greedy_local", mem_limit_factor=0.9)

    segments = [
        Segment(
            id=0,
            layer_ids=[0],
            flops=1e9,
            bytes=1e6,
            seq_len=128,
            embed_dim=256,
            num_heads=4,
            mlp_ratio=4.0,
            precision=1,
            traffic_in_bytes=1e6,
            traffic_out_bytes=1e6,
            kind="attn",
        )
    ]
    eff_specs = {
        "peak_flops": torch.tensor([1e12], device=device, requires_grad=True),
        "peak_bw": torch.tensor([1e11], device=device),
        "mem_gb": torch.tensor([16.0], device=device),
        "tdp_w": torch.tensor([200.0], device=device),
        "area_mm2": torch.tensor([200.0], device=device),
    }

    stable_hw_state = {"lambda_hw_effective_after_guard": 0.0}

    L_hw, hw_stats = compute_hw_loss(
        cfg,
        hw_proxy,
        model_info={},
        stable_hw_cfg=stable_hw_cfg,
        stable_hw_state=stable_hw_state,
        segments=segments,
        mapping=[0],
        mapping_sig=None,
        segments_sig=None,
        eff_specs=eff_specs,
        layout_positions=None,
        mapping_solver=mapping_solver,
        wafer_layout=None,
        alpha=None,
    )

    if "L_hw_norm" not in hw_stats:
        raise AssertionError("L_hw_norm missing from hw_stats")
    L_hw_norm = torch.as_tensor(float(hw_stats["L_hw_norm"]), device=L_hw.device, dtype=L_hw.dtype)
    L_hw_guarded = L_hw - L_hw_norm
    if not torch.isfinite(L_hw_guarded).all():
        raise AssertionError("Guarded HW loss is not finite")

    print("[SMOKE] StableHW contract OK")
    print("  L_hw_total =", float(L_hw.detach().cpu().item()))
    print("  L_hw_norm  =", float(L_hw_norm.detach().cpu().item()))
    print("  L_hw_guarded =", float(L_hw_guarded.detach().cpu().item()))


if __name__ == "__main__":
    main()
