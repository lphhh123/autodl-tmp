
from typing import List, Dict, Any, Tuple

import math
import torch


class WaferLayoutOptimizer:
    """Differentiable wafer layout optimizer (dummy but communication-aware).

    - Each chip is a square with area from gpu_data.yaml (approx).
    - We optimize chip centers (x, y) inside a circular wafer.
    - Objective:
        comm_loss = sum_ij traffic_ij * distance_ij
        + boundary / overlap penalties.

    This is still a simplified model, but it exposes the right hooks for
    later replacement by a more advanced floorplanner.
    """

    def __init__(self, wafer_radius_mm: float = 50.0, lr: float = 0.1, steps: int = 80,
                 lambda_comm: float = 1e-6):
        self.wafer_radius_mm = wafer_radius_mm
        self.lr = lr
        self.steps = steps
        self.lambda_comm = lambda_comm

    def optimize(
        self,
        device_instances: List[Dict[str, Any]],
        dev_edges: List[Tuple[int, int, float]],
        device_area_mm2: Dict[str, float],
    ) -> Dict[str, Any]:
        """Optimize positions.

        Parameters
        ----------
        device_instances: list of {id: str, chip_name: str}
        dev_edges: list of (i, j, traffic_bytes) over device indices
        device_area_mm2: dict device_id -> area_mm2

        Returns
        -------
        dict with:
          - positions: {device_id: {x, y}}
        """
        if not device_instances:
            return {"positions": {}}

        device_ids = [d["id"] for d in device_instances]
        n = len(device_ids)

        # Initial positions: place roughly on a circle
        angles = torch.linspace(0, 2 * math.pi, steps=n + 1)[:-1]
        radius = self.wafer_radius_mm * 0.6
        x = radius * torch.cos(angles)
        y = radius * torch.sin(angles)
        pos = torch.stack([x, y], dim=1)  # [n, 2]
        pos = torch.nn.Parameter(pos)

        optimizer = torch.optim.SGD([pos], lr=self.lr, momentum=0.9)

        # Chip half-sizes
        half_sizes = []
        for d in device_ids:
            area = device_area_mm2.get(d, 400.0)
            side = math.sqrt(area)
            half_sizes.append(side / 2.0)
        half_sizes = torch.tensor(half_sizes, dtype=torch.float32)

        for _ in range(self.steps):
            optimizer.zero_grad()
            loss = 0.0

            # 1) Communication distance loss
            if dev_edges:
                comm_loss = 0.0
                for i, j, traffic_bytes in dev_edges:
                    if i == j:
                        continue
                    pi = pos[i]
                    pj = pos[j]
                    dist = torch.norm(pi - pj) + 1e-6  # mm
                    comm_loss = comm_loss + float(traffic_bytes) * dist
                loss = loss + self.lambda_comm * comm_loss

            # 2) Wafer boundary penalty
            dist_from_center = torch.norm(pos, dim=1)  # [n]
            penalty_out = torch.relu(dist_from_center + half_sizes - self.wafer_radius_mm)
            loss = loss + (penalty_out ** 2).mean() * 1.0

            # 3) Overlap penalty
            for i in range(n):
                for j in range(i + 1, n):
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    center_dist = torch.sqrt(dx * dx + dy * dy + 1e-6)
                    min_allowed = half_sizes[i] + half_sizes[j]
                    overlap = torch.relu(min_allowed - center_dist)
                    loss = loss + (overlap ** 2) * 0.1

            loss.backward()
            optimizer.step()

        positions = {
            dev_id: dict(x=float(pos[i, 0].item()), y=float(pos[i, 1].item()))
            for i, dev_id in enumerate(device_ids)
        }
        return {"positions": positions}
