"""Pareto front utilities (SPEC v4.3.2 §8/§11)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class ParetoPoint:
    comm_norm: float
    therm_norm: float
    payload: dict


class ParetoSet:
    def __init__(self, eps_comm: float = 0.0, eps_therm: float = 0.0, max_points: int = 2000):
        self.eps_comm = eps_comm
        self.eps_therm = eps_therm
        self.max_points = max_points
        self.points: List[ParetoPoint] = []

    def _dominates(self, a: ParetoPoint, b: ParetoPoint) -> bool:
        return (
            a.comm_norm <= b.comm_norm + self.eps_comm
            and a.therm_norm <= b.therm_norm + self.eps_therm
            and (a.comm_norm < b.comm_norm - 1e-9 or a.therm_norm < b.therm_norm - 1e-9)
        )

    def add(self, comm_norm: float, therm_norm: float, payload: dict) -> bool:
        candidate = ParetoPoint(comm_norm, therm_norm, payload)
        # remove dominated
        new_points: List[ParetoPoint] = []
        for p in self.points:
            if self._dominates(candidate, p):
                continue
            if self._dominates(p, candidate):
                return False
            new_points.append(p)
        new_points.append(candidate)
        if len(new_points) > self.max_points:
            # simple pruning by distance to origin
            new_points.sort(key=lambda x: x.comm_norm + x.therm_norm)
            new_points = new_points[: self.max_points]
        self.points = new_points
        return True

    def as_array(self) -> np.ndarray:
        return np.array([[p.comm_norm, p.therm_norm] for p in self.points], dtype=float)

    def knee_point(self) -> Tuple[float, float, dict]:
        if not self.points:
            return (float("inf"), float("inf"), {})
        arr = self.as_array()
        # knee: farthest from line between (min_comm,min_therm) and (max_comm,max_therm)
        min_vec = arr.min(axis=0)
        max_vec = arr.max(axis=0)
        if np.allclose(min_vec, max_vec):
            p = self.points[0]
            return p.comm_norm, p.therm_norm, p.payload
        line = max_vec - min_vec
        line_norm = np.linalg.norm(line)
        distances = []
        for idx, point in enumerate(arr):
            vec = point - min_vec
            proj = np.dot(vec, line) / (line_norm + 1e-9)
            proj_point = min_vec + proj * line / (line_norm + 1e-9)
            dist = np.linalg.norm(point - proj_point)
            distances.append((dist, idx))
        distances.sort(reverse=True)
        _, best_idx = distances[0]
        p = self.points[best_idx]
        return p.comm_norm, p.therm_norm, p.payload
