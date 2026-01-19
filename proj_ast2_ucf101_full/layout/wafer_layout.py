"""Wafer layout model supporting discrete sites (SPEC v5.4 §7)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from mapping.segments import Segment


class WaferLayout(nn.Module):
    def __init__(self, num_slots: int, wafer_radius_mm: float, sites_xy: Optional[torch.Tensor] = None, assign: Optional[torch.Tensor] = None):
        super().__init__()
        self.wafer_radius_mm = wafer_radius_mm
        if sites_xy is None:
            # fallback to legacy continuous coordinates
            self.pos = nn.Parameter(torch.zeros(num_slots, 2))
            self.register_buffer("sites_xy", None)
            self.register_buffer("assign", None)
        else:
            self.register_buffer("sites_xy", sites_xy.float())
            if assign is None:
                assign = torch.arange(num_slots, device=sites_xy.device) % sites_xy.shape[0]
            self.register_buffer("assign", assign.long())
            self.pos = nn.Parameter(self.sites_xy[self.assign].clone())

    def current_pos_continuous(self) -> torch.Tensor:
        """Return learnable continuous positions for training-time optimization."""
        return self.pos

    def pos_from_assign(self, assign: torch.Tensor) -> torch.Tensor:
        """Return discrete positions from a given site assignment."""
        if self.sites_xy is None:
            raise ValueError("sites_xy is required to materialize discrete positions.")
        return self.sites_xy[assign]

    def project_pos_to_sites(self, pos: torch.Tensor) -> torch.Tensor:
        """Project continuous positions to nearest unique discrete sites."""
        if self.sites_xy is None:
            raise ValueError("sites_xy is required to project positions.")
        sites_xy = self.sites_xy
        S = pos.shape[0]
        Ns = sites_xy.shape[0]
        dists = torch.cdist(pos, sites_xy)
        min_dist, _ = dists.min(dim=1)
        slot_order = torch.argsort(min_dist)
        assign = torch.full((S,), -1, device=pos.device, dtype=torch.long)
        used = torch.zeros(Ns, device=pos.device, dtype=torch.bool)
        for slot_idx in slot_order:
            candidates = torch.argsort(dists[slot_idx])
            for site_idx in candidates:
                if not used[site_idx]:
                    assign[slot_idx] = site_idx
                    used[site_idx] = True
                    break
        if (assign < 0).any():
            raise ValueError("Failed to assign all slots to unique sites.")
        return assign

    @property
    def current_pos(self) -> torch.Tensor:
        """Backward-compatible access to continuous positions."""
        return self.current_pos_continuous()

    # SPEC §10.2
    def boundary_penalty(self, eff_specs: Dict[str, torch.Tensor], margin: float = 0.0):
        centers = self.current_pos_continuous()
        r_center = torch.sqrt((centers ** 2).sum(dim=1) + 1e-6)
        r_chip = torch.sqrt(eff_specs["area_mm2"] / math.pi + 1e-6)
        violation = torch.relu(r_center + r_chip + margin - self.wafer_radius_mm)
        return (violation ** 2).sum()

    # SPEC §10.3
    def overlap_penalty(self, eff_specs: Dict[str, torch.Tensor]):
        centers = self.current_pos_continuous()
        r_chip = torch.sqrt(eff_specs["area_mm2"] / math.pi + 1e-6)
        S = centers.shape[0]
        penalty = centers.new_tensor(0.0)
        for i in range(S):
            for j in range(i + 1, S):
                dist = torch.sqrt(((centers[i] - centers[j]) ** 2).sum() + 1e-6)
                min_dist = r_chip[i] + r_chip[j]
                overlap = torch.relu(min_dist - dist)
                penalty = penalty + overlap ** 2
        return penalty

    # SPEC §10.4
    def comm_loss(self, mapping: List[int], segments: List[Segment], eff_specs: Dict[str, torch.Tensor], distance_scale: float):
        centers = self.current_pos_continuous()
        comm_cost = centers.new_tensor(0.0)
        for k in range(len(segments) - 1):
            d1, d2 = mapping[k], mapping[k + 1]
            if d1 == d2:
                continue
            traffic = segments[k].traffic_out_bytes
            dist = torch.sqrt(((centers[d1] - centers[d2]) ** 2).sum() + 1e-6)
            comm_cost = comm_cost + traffic * dist * distance_scale
        return comm_cost

    # SPEC §10.5
    def thermal_penalty(self, eff_specs: Dict[str, torch.Tensor], T_ambient: float = 25.0, T_limit: float = 85.0, sigma_mm: float = 50.0, alpha: float = 0.01):
        centers = self.current_pos_continuous()
        power = eff_specs["tdp_w"]
        S = centers.shape[0]
        temps = []
        for i in range(S):
            r2 = ((centers - centers[i]) ** 2).sum(dim=1)
            K = torch.exp(-r2 / (2.0 * sigma_mm ** 2))
            T_i = T_ambient + alpha * (K * power).sum()
            temps.append(T_i)
        temps = torch.stack(temps)
        T_max = temps.max()
        return torch.relu(T_max - T_limit) ** 2

    # SPEC §10.6
    def forward(self, mapping: List[int], segments: List[Segment], eff_specs: Dict[str, torch.Tensor], lambda_boundary: float, lambda_overlap: float, lambda_comm: float, lambda_thermal: float, distance_scale: float):
        L_boundary = self.boundary_penalty(eff_specs)
        L_overlap = self.overlap_penalty(eff_specs)
        L_comm = self.comm_loss(mapping, segments, eff_specs, distance_scale)
        L_thermal = self.thermal_penalty(eff_specs)
        L_layout = lambda_boundary * L_boundary + lambda_overlap * L_overlap + lambda_comm * L_comm + lambda_thermal * L_thermal
        stats = {"boundary": L_boundary.detach(), "overlap": L_overlap.detach(), "comm": L_comm.detach(), "thermal": L_thermal.detach()}
        return L_layout, stats
