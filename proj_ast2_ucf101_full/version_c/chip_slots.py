"""Gumbel-Softmax chip slot manager."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from hw_proxy.feature_builder import DeviceSpec
from .chip_types import ChipType


class ChipSlotManager(nn.Module):
    def __init__(self, chip_types: List[ChipType], num_slots: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.chip_types = chip_types
        self.num_slots = num_slots
        self.num_types = len(chip_types) + 1  # +1 for empty
        self.logits = nn.Parameter(torch.zeros(num_slots, self.num_types))
        self.temperature = temperature

    def forward(self, hard: bool = False) -> torch.Tensor:
        return F.gumbel_softmax(self.logits, tau=self.temperature, hard=hard, dim=-1)

    def get_effective_specs(self, alpha: torch.Tensor) -> List[DeviceSpec]:
        specs: List[DeviceSpec] = []
        for slot in range(self.num_slots):
            coeffs = alpha[slot]
            peak_flops = 0.0
            peak_bw = 0.0
            mem_size = 0.0
            area = 0.0
            tdp = 0.0
            energy = 0.0
            for i, chip in enumerate(self.chip_types, start=1):
                prob = coeffs[i].item()
                peak_flops += prob * chip.peak_flops
                peak_bw += prob * chip.peak_bw
                mem_size += prob * chip.mem_size_gb
                area += prob * chip.area_mm2
                tdp += prob * chip.tdp_watt
                energy += prob * chip.energy_per_bit_pj
            specs.append(
                DeviceSpec(
                    name=f"slot{slot}",
                    peak_flops=peak_flops,
                    peak_bw=peak_bw,
                    mem_size_gb=mem_size,
                    area_mm2=area,
                    tdp_watt=tdp,
                    energy_per_bit_pj=max(1e-6, energy),
                )
            )
        return specs


def chip_count_regularizer(alpha: torch.Tensor, lambda_chip: float) -> torch.Tensor:
    used_prob = 1.0 - alpha[:, 0]
    return lambda_chip * used_prob.sum()
