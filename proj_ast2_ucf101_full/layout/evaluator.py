"""Layout evaluator shared by training-time and offline stages (SPEC v5.4 §5)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from utils.stable_hash import stable_hash

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
        self.evaluator_calls = 0

    def objective_cfg_dict(self) -> dict:
        # Only include fields that affect objective value.
        return {
            "objective_version": "v5.4",
            "sigma_mm": float(self.sigma_mm),
            "baseline": {
                "L_comm_baseline": float(self.baseline.get("L_comm_baseline", 1.0)),
                "L_therm_baseline": float(self.baseline.get("L_therm_baseline", 1.0)),
            },
            "scalar_w": {
                "w_comm": float(self.scalar_w.get("w_comm", 0.0)),
                "w_therm": float(self.scalar_w.get("w_therm", 0.0)),
                "w_penalty": float(self.scalar_w.get("w_penalty", 0.0)),
            },
            "penalty_schema": "duplicate+boundary",
        }

    def objective_hash(self) -> str:
        return stable_hash(self.objective_cfg_dict())

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
        self.evaluator_calls += 1
        assign = np.asarray(st.assign, dtype=int)
        if assign.shape[0] != int(st.S):
            raise ValueError(f"[LayoutEvaluator] len(assign)={assign.shape[0]} != S={st.S}")

        bad = assign[(assign < 0) | (assign >= int(st.Ns))]
        if bad.size > 0:
            raise ValueError(
                f"[LayoutEvaluator] assign contains invalid site_id(s): {np.unique(bad)[:8].tolist()} (Ns={st.Ns})"
            )

        traffic = np.asarray(st.traffic_bytes)
        if traffic.shape != (int(st.S), int(st.S)):
            raise ValueError(
                f"[LayoutEvaluator] traffic_bytes shape {traffic.shape} != (S,S)=({st.S},{st.S}). "
                f"Fix exporter to inflate compact traffic to SxS before calling evaluate()."
            )

        chip_tdp = np.asarray(st.chip_tdp_w)
        if chip_tdp.shape[0] != int(st.S):
            raise ValueError(f"[LayoutEvaluator] len(chip_tdp_w)={chip_tdp.shape[0]} != S={st.S}")

        pos = st.sites_xy_mm[assign]
        penalty_duplicate = 0.0
        dup_count = len(st.assign) - len(np.unique(st.assign))
        if dup_count > 0:
            penalty_duplicate = float(dup_count) ** 2
        boundary_overflow = np.linalg.norm(pos, axis=1) - st.wafer_radius_mm
        penalty_boundary = float(np.sum(np.maximum(boundary_overflow, 0.0) ** 2))

        L_comm = self._compute_comm(pos, traffic)
        L_therm = self._compute_therm(pos, chip_tdp)
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

    @property
    def evaluate_calls(self) -> int:
        return int(self.evaluator_calls)
