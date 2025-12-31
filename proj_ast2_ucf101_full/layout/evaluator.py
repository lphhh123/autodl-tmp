"""Layout evaluator shared by training-time and offline stages (SPEC v4.3.2 §5)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class LayoutState:
    S: int
    Ns: int
    wafer_radius_mm: float
    sites_xy_mm: np.ndarray  # [Ns,2]
    assign: np.ndarray  # [S]
    chip_tdp_w: np.ndarray  # [S]
    traffic_bytes: np.ndarray  # [S,S]
    meta: Dict


class LayoutEvaluator:
    def __init__(self, sigma_mm: float, baseline: Dict[str, float], scalar_w: Dict[str, float]):
        self.sigma_mm = sigma_mm
        self.baseline = baseline
        self.scalar_w = scalar_w

    def _compute_comm(self, pos: np.ndarray, traffic_bytes: np.ndarray) -> float:
        t_sym = traffic_bytes + traffic_bytes.T
        total = 0.0
        for i in range(pos.shape[0]):
            for j in range(i + 1, pos.shape[0]):
                dist = float(np.linalg.norm(pos[i] - pos[j]))
                total += float(t_sym[i, j]) * dist
        return total

    def _compute_therm(self, pos: np.ndarray, chip_tdp_w: np.ndarray) -> float:
        total = 0.0
        for i in range(pos.shape[0]):
            for j in range(i + 1, pos.shape[0]):
                dist = float(np.linalg.norm(pos[i] - pos[j]))
                if dist <= 1e-9:
                    continue
                total += float(chip_tdp_w[i] * chip_tdp_w[j]) * math.exp(-dist / self.sigma_mm)
        return total

    def evaluate(self, st: LayoutState) -> Dict:
        pos = st.sites_xy_mm[st.assign]
        penalty_duplicate = max(0, len(st.assign) - len(np.unique(st.assign))) * float(
            self.scalar_w.get("w_penalty", 1.0)
        )
        boundary_mask = np.linalg.norm(pos, axis=1) > (st.wafer_radius_mm + 1e-6)
        penalty_boundary = float(boundary_mask.sum()) * self.scalar_w.get("w_penalty", 1.0)

        L_comm = self._compute_comm(pos, st.traffic_bytes)
        L_therm = self._compute_therm(pos, st.chip_tdp_w)
        comm_norm = L_comm / (self.baseline.get("L_comm_baseline", 1e-9) + 1e-9)
        therm_norm = L_therm / (self.baseline.get("L_therm_baseline", 1e-9) + 1e-9)
        total = (
            self.scalar_w.get("w_comm", 1.0) * comm_norm
            + self.scalar_w.get("w_therm", 1.0) * therm_norm
            + self.scalar_w.get("w_penalty", 1.0) * (penalty_duplicate + penalty_boundary)
        )
        return {
            "L_comm": L_comm,
            "L_therm": L_therm,
            "comm_norm": comm_norm,
            "therm_norm": therm_norm,
            "penalty": {"duplicate": penalty_duplicate, "boundary": penalty_boundary},
            "total_scalar": total,
        }
