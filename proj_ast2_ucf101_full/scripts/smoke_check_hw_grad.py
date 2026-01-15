"""Smoke check for StableHW differentiability in compute_hw_loss."""
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.hw.lambda_chip = float(getattr(cfg.hw, "lambda_chip", 0.0) or 1.0)

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
        ),
        Segment(
            id=1,
            layer_ids=[1],
            flops=2e9,
            bytes=2e6,
            seq_len=128,
            embed_dim=256,
            num_heads=4,
            mlp_ratio=4.0,
            precision=1,
            traffic_in_bytes=2e6,
            traffic_out_bytes=2e6,
            kind="mlp",
        ),
    ]

    S = 2
    eff_specs = {
        "peak_flops": torch.tensor([1e12, 1.1e12], device=device, requires_grad=True),
        "peak_bw": torch.tensor([1e11, 1.1e11], device=device),
        "mem_gb": torch.tensor([16.0, 16.0], device=device),
        "tdp_w": torch.tensor([200.0, 210.0], device=device),
    }

    alpha = torch.randn(S, 2, device=device, requires_grad=True)
    mapping = [0, 1]

    L_hw, _ = compute_hw_loss(
        cfg,
        hw_proxy,
        model_info={},
        stable_hw_cfg=getattr(cfg, "stable_hw", None),
        stable_hw_state={},
        segments=segments,
        mapping=mapping,
        mapping_sig=None,
        segments_sig=None,
        eff_specs=eff_specs,
        layout_positions=None,
        mapping_solver=mapping_solver,
        wafer_layout=None,
        alpha=alpha,
    )

    L_hw.backward()

    if alpha.grad is None:
        raise AssertionError("alpha.grad is None; HW loss is not differentiable")
    if torch.all(alpha.grad == 0):
        raise AssertionError("alpha.grad is all zeros; HW loss gradient is cut off")

    print("[SMOKE] HW gradient OK; grad norm=", float(alpha.grad.norm().item()))


if __name__ == "__main__":
    main()
