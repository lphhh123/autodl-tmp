# NOTE: This smoke is bound to SPEC_E v5.4; do not change fields without updating SPEC_E + trace_contract_v54.py
"""Smoke check for StableHW contract invariants (SPEC v5.4)."""
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
from utils.config import AttrDict, load_config
from utils.config_validate import validate_and_fill_defaults
from utils.stable_hw import update_hw_refs_from_stats


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


class NegativeProxy:
    """Proxy that emits invalid/negative values to validate sanitize behavior."""

    def predict_from_model_info(self, _model_info):
        return {"latency_ms": -5.0, "mem_mb": float("nan"), "energy_mj": float("inf")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    cfg = validate_and_fill_defaults(cfg, mode="version_c")

    if float(getattr(cfg.hw, "lambda_hw", 0.0) or 0.0) != 0.0:
        raise AssertionError("cfg.hw.lambda_hw must be 0 under v5.4 NoDoubleScale")
    if float(getattr(getattr(cfg, "loss", None), "lambda_hw", 0.0) or 0.0) != 0.0:
        raise AssertionError("cfg.loss.lambda_hw must be 0 under v5.4 NoDoubleScale")

    stable_hw_cfg = getattr(cfg, "stable_hw", None)
    if stable_hw_cfg is None:
        raise AssertionError("stable_hw config missing")
    if not bool(getattr(getattr(stable_hw_cfg, "normalize", None), "enabled", True)):
        raise AssertionError("stable_hw.normalize.enabled must be true for contract check")

    guard = getattr(stable_hw_cfg, "accuracy_guard", None)
    if guard is None:
        raise AssertionError("stable_hw.accuracy_guard missing")
    ctrl = getattr(guard, "controller", None)
    if ctrl is None:
        legacy_keys = [
            "enabled",
            "metric",
            "epsilon_drop",
            "guard_mode",
            "freeze_discrete_updates",
            "freeze_schedule_in_recovery",
            "recovery_min_epochs",
            "cut_hw_loss_on_violate",
            "k_exit",
            "freeze_hw_on_drop",
            "freeze_hw_epochs",
            "cut_hw_loss_on_drop",
        ]
        if any(getattr(guard, k, None) is not None for k in legacy_keys):
            print("[WARN] accuracy_guard.controller missing; legacy keys detected.", file=sys.stderr)
        else:
            raise AssertionError("stable_hw.accuracy_guard.controller missing")

    if getattr(cfg, "locked_acc_ref", None) is not None or getattr(cfg, "no_drift", None) is not None:
        raise AssertionError("[v5.4 P0][HardGate-A] root-level locked_acc_ref/no_drift forbidden in strict mode.")

    if getattr(stable_hw_cfg, "locked_acc_ref", None) is None:
        raise AssertionError("stable_hw.locked_acc_ref missing (canonical).")
    if getattr(stable_hw_cfg, "no_drift", None) is None:
        raise AssertionError("stable_hw.no_drift missing (canonical).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hw_proxy = AnalyticProxy()
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

    stable_hw_state = {}

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

    # ---- NoDrift: enabled means refs must not update ----
    no_drift_cfg = AttrDict({"no_drift": AttrDict({"enabled": True})})
    stable_state = {"latency_ref_ms": 1.0, "memory_ref_mb": 2.0}
    update_hw_refs_from_stats(no_drift_cfg, stable_state, {"latency_ms": 9.0, "mem_mb": 9.0})
    if stable_state["latency_ref_ms"] != 1.0 or stable_state["memory_ref_mb"] != 2.0:
        raise AssertionError("NoDrift contract violated: refs updated while enabled")

    # ---- regression: real call-path must not crash (cfg + stable_hw_cfg) ----
    stable_state2 = {"latency_ref_ms": 1.0, "memory_ref_mb": 2.0}
    # should be no-op under NoDrift, but must not raise
    update_hw_refs_from_stats(cfg, stable_state2, {"latency_ms": 9.0, "mem_mb": 9.0}, stable_hw_cfg=stable_hw_cfg)

    # ---- sanitize negative proxy values ----
    neg_proxy = NegativeProxy()
    cfg.stable_hw.min_latency_ms = float(getattr(cfg.stable_hw, "min_latency_ms", 1e-3))
    _, neg_stats = compute_hw_loss(
        cfg,
        neg_proxy,
        model_info={},
        stable_hw_cfg=stable_hw_cfg,
        stable_hw_state={},
    )
    if float(neg_stats.get("latency_ms_sanitized", 0.0)) < float(cfg.stable_hw.min_latency_ms):
        raise AssertionError("sanitize latency failed: latency_ms_sanitized below min_latency_ms")

    print("[SMOKE] StableHW contract OK")
    print("  L_hw_total =", float(L_hw.detach().cpu().item()))
    print("  L_hw_norm  =", float(L_hw_norm.detach().cpu().item()))
    print("  L_hw_guarded =", float(L_hw_guarded.detach().cpu().item()))


if __name__ == "__main__":
    main()
