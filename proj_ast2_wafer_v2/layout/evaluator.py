"""Layout evaluator shared by training-time and offline agent."""

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
    def __init__(self, sigma_mm: float, baseline: Dict, scalar_w: Dict):
        self.sigma_mm = float(sigma_mm)
        self.L_comm_baseline = float(baseline.get("L_comm_baseline", 1.0))
        self.L_therm_baseline = float(baseline.get("L_therm_baseline", 1.0))
        self.w_comm = float(scalar_w.get("w_comm", 1.0))
        self.w_therm = float(scalar_w.get("w_therm", 1.0))
        self.w_penalty = float(scalar_w.get("w_penalty", 1000.0))

    def evaluate(self, st: LayoutState) -> Dict:
        assign = st.assign
        sites = st.sites_xy_mm
        pos = sites[assign]

        traffic = st.traffic_bytes
        T_sym = traffic + traffic.T
        L_comm = 0.0
        L_therm = 0.0

        for i in range(st.S):
            for j in range(i + 1, st.S):
                d = float(np.linalg.norm(pos[i] - pos[j]))
                w_comm = float(T_sym[i, j])
                if w_comm:
                    L_comm += w_comm * d
                w_therm = float(st.chip_tdp_w[i] * st.chip_tdp_w[j])
                if w_therm:
                    L_therm += w_therm * math.exp(-d / max(self.sigma_mm, 1e-9))

        # penalties
        penalty_duplicate = 0.0
        penalty_boundary = 0.0
        unique, counts = np.unique(assign, return_counts=True)
        if np.any(counts > 1):
            penalty_duplicate = float(np.sum(counts[counts > 1] - 1))

        radii = np.linalg.norm(pos, axis=1)
        over_boundary = radii > st.wafer_radius_mm
        if np.any(over_boundary):
            penalty_boundary = float(np.sum(radii[over_boundary] - st.wafer_radius_mm))

        comm_norm = L_comm / (self.L_comm_baseline + 1e-9)
        therm_norm = L_therm / (self.L_therm_baseline + 1e-9)

        total_scalar = (
            self.w_comm * comm_norm
            + self.w_therm * therm_norm
            + self.w_penalty * (penalty_duplicate + penalty_boundary)
        )

        return {
            "L_comm": L_comm,
            "L_therm": L_therm,
            "comm_norm": comm_norm,
            "therm_norm": therm_norm,
            "penalty": {"duplicate": penalty_duplicate, "boundary": penalty_boundary},
            "total_scalar": total_scalar,
        }

