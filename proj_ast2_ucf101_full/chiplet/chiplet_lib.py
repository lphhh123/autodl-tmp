"""Chiplet library and Gumbel-softmax slots (SPEC §7)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


@dataclass
class ChipletType:
    name: str
    peak_flops_tflops: float
    peak_bw_gbps: float
    mem_gb: float
    die_area_mm2: float
    aspect_ratio: float
    tdp_w: float = 0.0

    @property
    def width_mm(self) -> float:
        return math.sqrt(self.die_area_mm2 * self.aspect_ratio)

    @property
    def height_mm(self) -> float:
        return math.sqrt(self.die_area_mm2 / self.aspect_ratio)


class ChipletLibrary:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        entries = data.get("chip_types", []) if isinstance(data, dict) else []
        self.types: Dict[str, ChipletType] = {}
        for cfg in entries:
            name = cfg["name"]
            self.types[name] = ChipletType(
                name=name,
                peak_flops_tflops=cfg["peak_flops"] / 1e12 if cfg["peak_flops"] > 1e6 else cfg["peak_flops_tflops"] if "peak_flops_tflops" in cfg else cfg["peak_flops"],
                peak_bw_gbps=cfg.get("peak_bw_gbps", cfg.get("peak_bw", 0.0) / 1e9 if "peak_bw" in cfg else 0.0),
                mem_gb=cfg["mem_gb"],
                die_area_mm2=cfg["area_mm2"],
                aspect_ratio=cfg.get("aspect_ratio", 1.0),
                tdp_w=cfg.get("tdp_w", 0.0),
            )

    def get(self, name: str) -> ChipletType:
        return self.types[name]

    def __getitem__(self, name: str) -> ChipletType:
        return self.get(name)


class ChipletSlots(nn.Module):
    def __init__(self, library: ChipletLibrary, candidate_names: List[str], num_slots: int, tau_init: float):
        super().__init__()
        self.library = library
        self.candidates = [library.get(n) for n in candidate_names]
        self.num_slots = num_slots
        self.num_types = len(self.candidates) + 1  # + empty
        self.logits = nn.Parameter(torch.zeros(num_slots, self.num_types))
        self.tau = tau_init

    def set_tau(self, tau: float):
        self.tau = tau

    def forward(self, hard: bool = False) -> Dict[str, torch.Tensor]:
        alpha = F.gumbel_softmax(self.logits, tau=self.tau, hard=hard, dim=-1)
        specs = self._expected_specs(alpha)
        return {"alpha": alpha, "eff_specs": specs}

    def _expected_specs(self, alpha: torch.Tensor) -> Dict[str, torch.Tensor]:
        device = alpha.device
        def stack_param(fn):
            vals = []
            for chip in self.candidates:
                vals.append(torch.tensor(fn(chip), device=device, dtype=torch.float32))
            vals.append(torch.tensor(0.0, device=device))  # empty
            return torch.stack(vals)

        flops = stack_param(lambda c: c.peak_flops_tflops * 1e12)
        bw = stack_param(lambda c: c.peak_bw_gbps * 1e9)
        mem = stack_param(lambda c: c.mem_gb)
        area = stack_param(lambda c: c.die_area_mm2)
        width = stack_param(lambda c: c.width_mm)
        height = stack_param(lambda c: c.height_mm)
        tdp = stack_param(lambda c: c.tdp_w)

        specs = {
            "peak_flops": alpha @ flops,
            "peak_bw": alpha @ bw,
            "mem_gb": alpha @ mem,
            "area_mm2": alpha @ area,
            "width_mm": alpha @ width,
            "height_mm": alpha @ height,
            "tdp_w": alpha @ tdp,
        }
        return specs
