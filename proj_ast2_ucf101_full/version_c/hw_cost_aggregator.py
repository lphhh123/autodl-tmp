"""Aggregates hardware-related costs for Version-C."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from hw_proxy.feature_builder import DeviceSpec
from hw_proxy.layer_hw_proxy import LayerHwProxy
from .segmenter import Segment
from .wafer_layout import layout_boundary_penalty, layout_comm_cost, layout_overlap_penalty, layout_thermal_penalty


class HwCostAggregator:
    def __init__(
        self,
        hw_proxy: LayerHwProxy,
        lambda_T: float,
        lambda_E: float,
        lambda_mem: float,
        lambda_area: float,
        lambda_thermal: float,
        lambda_comm: float,
        lambda_chip: float,
    ) -> None:
        self.hw_proxy = hw_proxy
        self.lambda_T = lambda_T
        self.lambda_E = lambda_E
        self.lambda_mem = lambda_mem
        self.lambda_area = lambda_area
        self.lambda_thermal = lambda_thermal
        self.lambda_comm = lambda_comm
        self.lambda_chip = lambda_chip

    def compute_cost(
        self,
        segments: List[Segment],
        chip_specs: List[DeviceSpec],
        mapping: Dict[int, int],
        layout: torch.Tensor,
        alpha: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        stats: Dict[str, float] = {"T_ms": 0.0, "E_joule": 0.0, "M_peak": 0.0, "A_total": 0.0, "T_max": 0.0, "C_comm": 0.0, "N_chip": 0.0}

        device_time = {i: 0.0 for i in range(len(chip_specs))}
        for seg in segments:
            slot = mapping.get(seg.id, 0)
            spec = chip_specs[slot]
            device_time[slot] += seg.flops / max(1e-6, spec.peak_flops)
        makespan = max(device_time.values()) if device_time else 0.0
        stats["T_ms"] = makespan * 1e3

        positions = layout
        comm_graph = []
        stats["C_comm"] = float(layout_comm_cost(positions, comm_graph, energy_per_bit=1.0).detach().cpu())
        stats["A_total"] = float(sum(spec.area_mm2 for spec in chip_specs))
        stats["N_chip"] = float((1.0 - alpha[:, 0]).sum().item())

        loss = (
            self.lambda_T * stats["T_ms"]
            + self.lambda_E * stats["E_joule"]
            + self.lambda_mem * stats["M_peak"]
            + self.lambda_area * stats["A_total"]
            + self.lambda_comm * stats["C_comm"]
        )
        loss_tensor = torch.tensor(loss, dtype=torch.float32, device=layout.device)
        loss_tensor = loss_tensor + self.lambda_thermal * layout_overlap_penalty(positions, [spec.area_mm2 for spec in chip_specs])
        loss_tensor = loss_tensor + self.lambda_thermal * layout_boundary_penalty(positions, wafer_radius=max(positions.norm(dim=1).max().item(), 1.0))
        return loss_tensor, stats
