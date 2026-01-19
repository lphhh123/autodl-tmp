"""Smoke check for StableHW differentiability in compute_hw_loss (no proxy_weights required)."""
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import numpy as np
import torch

from hw_proxy.hw_loss import compute_hw_loss
from mapping.mapping_solver import MappingSolver
from mapping.segments import Segment
from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults


class AnalyticProxy:
    """A tiny differentiable proxy used ONLY for smoke tests."""

    def predict_layers_batch(self, layers_cfg):
        lat = []
        mem = []
        power = []
        eps = 1e-12
        for x in layers_cfg:
            dev = x["device_cfg"]
            peak_flops = float(dev.get("peak_flops", 1.0))
            peak_bw = float(dev.get("peak_bw", 1.0))
            tdp = float(dev.get("tdp_w", 200.0))
            flops = float(x.get("flops", 0.0))
            byt = float(x.get("bytes", 0.0))
            lat_ms = (flops / (peak_flops + eps)) * 1e3 + (byt / (peak_bw + eps)) * 1e3
            lat.append(max(lat_ms, 0.0))
            mem.append(max(byt / 1e6, 0.0))
            power.append(max(0.5 * tdp, 0.0))
        return {
            "lat_ms": np.asarray(lat, dtype=np.float32),
            "mem_mb": np.asarray(mem, dtype=np.float32),
            "power_w": np.asarray(power, dtype=np.float32),
        }

    def predict_layers_batch_torch(self, layers_cfg, device):
        lat = []
        mem = []
        power = []
        eps = torch.tensor(1e-12, device=device)
        for x in layers_cfg:
            dev = x["device_cfg"]
            peak_flops = dev.get("peak_flops", torch.tensor(1.0, device=device))
            peak_bw = dev.get("peak_bw", torch.tensor(1.0, device=device))
            tdp = dev.get("tdp_w", torch.tensor(200.0, device=device))
            flops = torch.tensor(float(x.get("flops", 0.0)), device=device)
            byt = torch.tensor(float(x.get("bytes", 0.0)), device=device)
            lat_ms = (flops / (peak_flops + eps)) * 1e3 + (byt / (peak_bw + eps)) * 1e3
            lat.append(torch.clamp(lat_ms, min=0.0))
            mem.append(torch.clamp(byt / 1e6, min=0.0))
            power.append(torch.clamp(0.5 * tdp, min=0.0))
        return {
            "lat_ms": torch.stack(lat, dim=0),
            "mem_mb": torch.stack(mem, dim=0),
            "power_w": torch.stack(power, dim=0),
        }


class NegativeProxy(AnalyticProxy):
    """Return strictly negative proxy outputs to test I4/S3 guard (must NOT reduce loss)."""

    def predict_layers_batch(self, layers_cfg):
        out = super().predict_layers_batch(layers_cfg)
        for k in list(out.keys()):
            out[k] = -out[k]
        return out

    def predict_layers_batch_torch(self, layers_cfg, device):
        out = super().predict_layers_batch_torch(layers_cfg, device)
        for k in list(out.keys()):
            out[k] = -out[k]
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    cfg = validate_and_fill_defaults(cfg, mode="version_c")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg.hw.lambda_area = 1.0
    cfg.hw.area_budget_mm2 = 100.0
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
        "peak_bw": torch.tensor([1e11, 1.1e11], device=device, requires_grad=True),
        "mem_gb": torch.tensor([16.0, 16.0], device=device),
        "tdp_w": torch.tensor([200.0, 210.0], device=device),
        "area_mm2": torch.tensor([200.0, 200.0], device=device, requires_grad=True),
    }
    alpha = torch.randn(S, 2, device=device, requires_grad=True)
    mapping = [0, 1]

    hw_proxy = AnalyticProxy()

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

    assert alpha.grad is not None and float(alpha.grad.norm().item()) > 0.0, "alpha.grad is missing/zero"
    assert (
        eff_specs["area_mm2"].grad is not None
        and float(eff_specs["area_mm2"].grad.norm().item()) > 0.0
    ), "area grad missing/zero"

    # ===== S3: negative proxy guard must NOT reduce loss =====
    stable_state = {}

    # 1) run once to get reasonable refs from the positive proxy
    _, s0 = compute_hw_loss(
        cfg,
        hw_proxy,
        model_info={},
        stable_hw_cfg=getattr(cfg, "stable_hw", None),
        stable_hw_state=stable_state,
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
    # set refs equal to current clamped values so normalized baseline ~ 0
    stable_state["ref_T"] = float(s0.get("clamped_latency_ms", 1.0))
    stable_state["ref_E"] = float(s0.get("clamped_energy_mj", 1.0))
    stable_state["ref_M"] = float(s0.get("clamped_mem_mb", 1.0))
    stable_state["ref_C"] = float(s0.get("clamped_comm_ms", 1.0))

    # 2) baseline loss with positive proxy under these refs
    L_base, _ = compute_hw_loss(
        cfg,
        hw_proxy,
        model_info={},
        stable_hw_cfg=getattr(cfg, "stable_hw", None),
        stable_hw_state=stable_state,
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

    # 3) loss with negative proxy must NOT be smaller than baseline
    neg_proxy = NegativeProxy()
    L_neg, s_neg = compute_hw_loss(
        cfg,
        neg_proxy,
        model_info={},
        stable_hw_cfg=getattr(cfg, "stable_hw", None),
        stable_hw_state=stable_state,
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

    assert float(s_neg.get("clamped_latency_ms", -1.0)) >= 0.0, "clamped latency must be non-negative"
    assert float(s_neg.get("clamped_mem_mb", -1.0)) >= 0.0, "clamped mem must be non-negative"
    assert float(s_neg.get("clamped_energy_mj", -1.0)) >= 0.0, "clamped energy must be non-negative"
    assert float(s_neg.get("clamped_comm_ms", -1.0)) >= 0.0, "clamped comm must be non-negative"

    lb = float(L_base.detach().cpu().item())
    ln = float(L_neg.detach().cpu().item())
    assert ln >= lb - 1e-6, f"negative proxy must NOT reduce loss: L_neg={ln}, L_base={lb}"

    print("[SMOKE] Negative proxy guard OK; L_base=", lb, "L_neg=", ln)

    print("[SMOKE] HW gradient OK; alpha grad norm=", float(alpha.grad.norm().item()))
    print("[SMOKE] Area gradient OK; area grad norm=", float(eff_specs["area_mm2"].grad.norm().item()))


if __name__ == "__main__":
    main()
