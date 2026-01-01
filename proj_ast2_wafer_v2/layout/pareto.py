"""Pareto set utilities for bi-objective optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


def dominates(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])


@dataclass
class ParetoPoint:
    comm_norm: float
    therm_norm: float
    total_scalar: float
    meta: dict


class ParetoSet:
    def __init__(self, eps_comm: float, eps_therm: float, max_points: int = 2000):
        self.eps_comm = eps_comm
        self.eps_therm = eps_therm
        self.max_points = max_points
        self.points: List[ParetoPoint] = []

    def _is_eps_dominated(self, obj: Tuple[float, float]) -> bool:
        for p in self.points:
            if obj[0] >= p.comm_norm - self.eps_comm and obj[1] >= p.therm_norm - self.eps_therm:
                return True
        return False

    def try_add(self, comm_norm: float, therm_norm: float, total_scalar: float, meta: dict) -> bool:
        obj = (comm_norm, therm_norm)
        if self._is_eps_dominated(obj):
            return False
        # remove dominated points
        self.points = [p for p in self.points if not dominates(obj, (p.comm_norm, p.therm_norm))]
        self.points.append(ParetoPoint(comm_norm, therm_norm, total_scalar, meta))
        if len(self.points) > self.max_points:
            # drop worst by total_scalar
            self.points.sort(key=lambda p: p.total_scalar)
            self.points = self.points[: self.max_points]
        return True

    def knee_point(self) -> ParetoPoint | None:
        if not self.points:
            return None
        pts = sorted(self.points, key=lambda p: (p.comm_norm, p.therm_norm))
        # heuristic: minimize distance to (min_comm, min_therm) line
        comms = np.array([p.comm_norm for p in pts])
        therms = np.array([p.therm_norm for p in pts])
        comm_min, comm_max = comms.min(), comms.max()
        therm_min, therm_max = therms.min(), therms.max()
        if comm_max == comm_min or therm_max == therm_min:
            return pts[0]
        comm_normed = (comms - comm_min) / (comm_max - comm_min + 1e-9)
        therm_normed = (therms - therm_min) / (therm_max - therm_min + 1e-9)
        dist = np.abs(comm_normed + therm_normed - 1)  # distance to knee line
        idx = int(np.argmin(dist))
        return pts[idx]

