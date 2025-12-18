"""Phase 4 layout experiments."""
from __future__ import annotations

import torch

from utils.distributed_utils import get_device
from utils.logging_utils import setup_logger
from version_c.wafer_layout import WaferLayout, layout_boundary_penalty, layout_overlap_penalty
from version_c.wafer_legalize import legalize_layout
from hw_proxy.feature_builder import DeviceSpec


def run_phase4(cfg):
    device = get_device(cfg.train.device)
    logger = setup_logger(cfg.train.output_dir)

    devices = [
        DeviceSpec(
            name="grid",
            peak_flops=1e12,
            peak_bw=1e12,
            mem_size_gb=16,
            area_mm2=200,
            tdp_watt=200,
            energy_per_bit_pj=10.0,
        )
        for _ in range(cfg.version_c.num_slots)
    ]

    layout = WaferLayout(cfg.layout.wafer_radius_mm, devices).to(device)
    optim = torch.optim.Adam(layout.parameters(), lr=1e-2)
    for step in range(10):
        optim.zero_grad()
        pos = layout()
        loss = layout_boundary_penalty(pos, cfg.layout.wafer_radius_mm) + layout_overlap_penalty(pos, [d.area_mm2 for d in devices])
        loss.backward()
        optim.step()
        logger.info(f"step {step} layout loss {loss.item():.4f}")

    legalized = legalize_layout(layout().detach(), [d.area_mm2 for d in devices], cfg.layout.wafer_radius_mm)
    logger.info(f"Final legalized positions shape: {legalized.shape}")
