"""Chip type definitions."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChipType:
    name: str
    peak_flops: float
    peak_bw: float
    mem_size_gb: float
    area_mm2: float
    tdp_watt: float
    energy_per_bit_pj: float
