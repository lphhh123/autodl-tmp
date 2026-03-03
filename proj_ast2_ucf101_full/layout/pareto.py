"""Pareto front utilities (SPEC v5.4 §8/§11)."""
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

    def knee_point(self, normalize: bool = True) -> Tuple[float, float, dict]:
        if not self.points:
            return (float("inf"), float("inf"), {})
        arr = self.as_array()
        # anchor points on each axis (raw indices)
        anchor_comm_idx = int(arr[:, 0].argmin())
        anchor_therm_idx = int(arr[:, 1].argmin())
        a_raw = arr[anchor_comm_idx]
        b_raw = arr[anchor_therm_idx]

        if np.allclose(a_raw, b_raw):
            p = self.points[anchor_comm_idx]
            return p.comm_norm, p.therm_norm, p.payload

        # normalize to [0,1] per axis to avoid scale dominance (therm >> comm)
        if normalize:
            eps = 1e-12
            cmin, cmax = float(arr[:, 0].min()), float(arr[:, 0].max())
            tmin, tmax = float(arr[:, 1].min()), float(arr[:, 1].max())
            cden = max(eps, cmax - cmin)
            tden = max(eps, tmax - tmin)
            arr_use = np.stack([(arr[:, 0] - cmin) / cden, (arr[:, 1] - tmin) / tden], axis=1)
            a = arr_use[anchor_comm_idx]
            b = arr_use[anchor_therm_idx]
        else:
            arr_use = arr
            a = a_raw
            b = b_raw

        if np.allclose(a, b):
            p = self.points[anchor_comm_idx]
            return p.comm_norm, p.therm_norm, p.payload

        # line from a to b, compute perpendicular distance in arr_use space
        ab = b - a
        ab_norm = float(np.linalg.norm(ab)) + 1e-12
        max_dist = -1.0
        best_idx = anchor_comm_idx
        for idx, pt in enumerate(arr_use):
            ap = pt - a
            proj = float(np.dot(ap, ab)) / ab_norm
            closest = a + (proj / ab_norm) * ab
            dist = float(np.linalg.norm(pt - closest))
            if dist > max_dist:
                max_dist = dist
                best_idx = idx

        p = self.points[int(best_idx)]
        return p.comm_norm, p.therm_norm, p.payload

    def knee_point_raw(self) -> Tuple[float, float, dict]:
        # backward compatible raw knee (no normalization)
        return self.knee_point(normalize=False)

    def best_by_scalar(self, w_comm: float = 0.7, w_therm: float = 0.3) -> Tuple[float, float, dict]:
        """
        Pick the Pareto point that minimizes the scalar objective.
        Prefer payload['total_scalar'] if present; otherwise compute w_comm*comm + w_therm*therm.
        """
        if not self.points:
            return (float("inf"), float("inf"), {})
        best = None
        best_val = float("inf")
        for p in self.points:
            payload = p.payload or {}
            if "total_scalar" in payload and payload["total_scalar"] is not None:
                val = float(payload["total_scalar"])
            else:
                val = float(w_comm) * float(p.comm_norm) + float(w_therm) * float(p.therm_norm)
            if val < best_val:
                best_val = val
                best = p
        return best.comm_norm, best.therm_norm, best.payload
