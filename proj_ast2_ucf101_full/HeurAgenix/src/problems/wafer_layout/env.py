"""Wafer layout environment for HeurAgenix baseline."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np

from core import BaseEnv
from layout.evaluator import LayoutEvaluator, LayoutState
from problems.wafer_layout.components import WaferLayoutSolution


class WaferLayoutEnv(BaseEnv):
    def __init__(self, data_path: str, rng: random.Random | None = None):
        self.rng = rng or random.Random(0)
        super().__init__(data_path)
        self._evaluator = self._build_evaluator()

    def load_data(self, path: str) -> Dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def _build_evaluator(self) -> LayoutEvaluator:
        baseline = self.instance_data.get("baseline", {})
        objective_cfg = self.instance_data.get("objective_cfg", {})
        scalar = objective_cfg.get("scalar_weights", {})
        return LayoutEvaluator(
            sigma_mm=float(objective_cfg.get("sigma_mm", 20.0)),
            baseline={
                "L_comm_baseline": float(baseline.get("L_comm", 1.0)),
                "L_therm_baseline": float(baseline.get("L_therm", 1.0)),
            },
            scalar_w={
                "w_comm": float(scalar.get("w_comm", 0.7)),
                "w_therm": float(scalar.get("w_therm", 0.3)),
                "w_penalty": float(scalar.get("w_penalty", 1000.0)),
            },
        )

    def init_solution(self) -> WaferLayoutSolution:
        seed = self.instance_data.get("seed", {})
        baseline = self.instance_data.get("baseline", {})
        slots = int(self.instance_data["slots"]["S"])
        Ns = int(self.instance_data["sites"]["Ns"])
        if "assign_seed" in seed:
            assign = list(seed["assign_seed"])
        elif "assign_grid" in baseline:
            assign = list(baseline["assign_grid"])
        else:
            sites = list(range(Ns))
            self.rng.shuffle(sites)
            assign = sites[:slots]
        return WaferLayoutSolution(assign=assign, S=slots, Ns=Ns)

    def _build_state(self, assign: np.ndarray) -> LayoutState:
        return LayoutState(
            S=int(self.instance_data["slots"]["S"]),
            Ns=int(self.instance_data["sites"]["Ns"]),
            wafer_radius_mm=float(self.instance_data["wafer"]["radius_mm"]),
            sites_xy_mm=np.asarray(self.instance_data["sites"]["sites_xy"], dtype=np.float32),
            assign=assign,
            chip_tdp_w=np.asarray(self.instance_data["slots"]["tdp"], dtype=float),
            traffic_bytes=np.asarray(self.instance_data["mapping"]["traffic_matrix"], dtype=float),
            meta={},
        )

    def get_key_value(self, solution: WaferLayoutSolution) -> float:
        eval_out = self.evaluate(solution)
        return float(eval_out["total_scalar"])

    def evaluate(self, solution: WaferLayoutSolution) -> Dict[str, Any]:
        st = self._build_state(np.asarray(solution.assign, dtype=int))
        return self._evaluator.evaluate(st)
