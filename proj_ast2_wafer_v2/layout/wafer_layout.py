from dataclasses import dataclass
from typing import List, Dict, Tuple

import math
import torch
from torch import nn


@dataclass
class ChipGeom:
    idx: int
    width: float  # um or mm (normalized)
    height: float
    power: float  # W


class WaferLayout(nn.Module):
    """Continuous wafer layout with learnable chip coordinates.

    Coordinates are normalized to [-1, 1] x [-1, 1] square;
    wafer is approximated by a circle of radius 1.
    """

    def __init__(self, chips: List[ChipGeom], init_scale: float = 0.5):
        super().__init__()
        self.chips = chips
        # positions: [num_chips, 2]
        init_pos = torch.zeros(len(chips), 2, dtype=torch.float32)
        # simple grid init
        side = math.ceil(len(chips) ** 0.5)
        xs = torch.linspace(-init_scale, init_scale, side)
        ys = torch.linspace(-init_scale, init_scale, side)
        idx = 0
        for y in ys:
            for x in xs:
                if idx >= len(chips):
                    break
                init_pos[idx, 0] = x
                init_pos[idx, 1] = y
                idx += 1
            if idx >= len(chips):
                break
        self.positions = nn.Parameter(init_pos)  # [N, 2]

    def forward(self) -> torch.Tensor:
        return self.positions


def hpwl_wirelength(
    positions: torch.Tensor,
    netlist: Dict[Tuple[int, int], float],
) -> torch.Tensor:
    """Half-perimeter wirelength weighted by traffic.

    Args:
        positions: [N, 2]
        netlist: (chip_u, chip_v) -> traffic_weight
    """
    loss = positions.new_tensor(0.0)
    for (u, v), w in netlist.items():
        pu = positions[u]
        pv = positions[v]
        dist = (pu - pv).abs().sum()  # L1
        loss = loss + w * dist
    return loss


def overlap_penalty(
    positions: torch.Tensor,
    chips: List[ChipGeom],
    margin: float = 0.02,
) -> torch.Tensor:
    """Soft penalty if two chips overlap or are too close."""
    N = len(chips)
    loss = positions.new_tensor(0.0)
    for i in range(N):
        wi = chips[i].width
        hi = chips[i].height
        for j in range(i + 1, N):
            wj = chips[j].width
            hj = chips[j].height
            dx = (positions[i, 0] - positions[j, 0]).abs()
            dy = (positions[i, 1] - positions[j, 1]).abs()
            min_dx = (wi + wj) / 2.0 + margin
            min_dy = (hi + hj) / 2.0 + margin
            loss = loss + torch.relu(min_dx - dx) + torch.relu(min_dy - dy)
    return loss


def boundary_penalty(
    positions: torch.Tensor,
    chips: List[ChipGeom],
    wafer_radius: float = 1.0,
    margin: float = 0.02,
) -> torch.Tensor:
    """Soft penalty when chip centers are too close to wafer boundary."""
    loss = positions.new_tensor(0.0)
    for i, chip in enumerate(chips):
        r = positions[i].norm(p=2)
        # approximate chip radius by max(width, height) / 2
        cr = max(chip.width, chip.height) / 2.0
        if r + cr > wafer_radius - margin:
            loss = loss + torch.relu(r + cr - (wafer_radius - margin))
    return loss


def thermal_penalty(
    positions: torch.Tensor,
    chips: List[ChipGeom],
    kernel_sigma: float = 0.3,
    temp_limit: float = 1.0,
) -> torch.Tensor:
    """Simple thermal proxy: each chip contributes heat decaying with distance.

    We approximate temperature at each chip as:
        T_i = sum_j power_j * exp(-||pos_i - pos_j||^2 / (2 sigma^2))

    Then penalize max(T_i) above a normalized temp_limit.
    """
    N = len(chips)
    power = positions.new_tensor([c.power for c in chips]).view(N, 1)  # [N,1]
    d2 = torch.cdist(positions, positions) ** 2  # [N,N]
    kernel = torch.exp(-d2 / (2 * kernel_sigma ** 2))  # [N,N]
    T = kernel @ power  # [N, 1]
    T_max = T.max()
    return torch.relu(T_max - temp_limit)


def layout_loss(
    layout: WaferLayout,
    netlist: Dict[Tuple[int, int], float],
    alpha_wire: float = 1.0,
    beta_overlap: float = 10.0,
    gamma_boundary: float = 10.0,
    delta_thermal: float = 1.0,
) -> torch.Tensor:
    positions = layout()
    chips = layout.chips
    loss_wire = hpwl_wirelength(positions, netlist)
    loss_overlap = overlap_penalty(positions, chips)
    loss_boundary = boundary_penalty(positions, chips)
    loss_thermal = thermal_penalty(positions, chips)

    return (alpha_wire * loss_wire
            + beta_overlap * loss_overlap
            + gamma_boundary * loss_boundary
            + delta_thermal * loss_thermal)
