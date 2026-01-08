"""Wafer layout environment for HeurAgenix baseline."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from core import BaseEnv
from problems.wafer_layout.components import WaferLayoutSolution


class WaferLayoutEnv(BaseEnv):
    def __init__(
        self,
        data_path: str,
        rng: random.Random | None = None,
        algorithm_data: Optional[Dict[str, Any]] = None,
    ):
        self.rng = rng or random.Random(0)
        self.algorithm_data: Dict[str, Any] = algorithm_data or {}
        super().__init__(data_path)
        self._fallback_evaluator: Optional[Callable[[np.ndarray], Dict[str, Any]]] = None

    def load_data(self, path: str) -> Dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def _ensure_fallback_eval(self) -> Callable[[np.ndarray], Dict[str, Any]]:
        if self._fallback_evaluator is None:
            from layout.evaluator import LayoutEvaluator, LayoutState

            baseline = self.instance_data.get("baseline", {})
            objective_cfg = self.instance_data.get("objective_cfg", {})
            scalar = objective_cfg.get("scalar_weights", {})
            sites_xy = np.asarray(self.instance_data["sites"]["sites_xy"], dtype=np.float32)
            slots = int(self.instance_data["slots"].get("S", len(self.instance_data["slots"].get("tdp", []))))
            Ns = int(self.instance_data["sites"].get("Ns", len(sites_xy)))
            evaluator = LayoutEvaluator(
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
            base_state = LayoutState(
                S=slots,
                Ns=Ns,
                wafer_radius_mm=float(self.instance_data["wafer"]["radius_mm"]),
                sites_xy_mm=sites_xy,
                assign=np.zeros(slots, dtype=int),
                chip_tdp_w=np.asarray(self.instance_data["slots"]["tdp"], dtype=float),
                traffic_bytes=np.asarray(self.instance_data["mapping"]["traffic_matrix"], dtype=float),
                meta={},
            )

            def _fallback(assign: np.ndarray) -> Dict[str, Any]:
                base_state.assign = np.asarray(assign, dtype=int)
                return evaluator.evaluate(base_state)

            self._fallback_evaluator = _fallback
        return self._fallback_evaluator

    def init_solution(self) -> WaferLayoutSolution:
        seed = self.instance_data.get("seed", {})
        baseline = self.instance_data.get("baseline", {})
        slots = int(self.instance_data["slots"].get("S", len(self.instance_data["slots"].get("tdp", []))))
        sites_xy = np.asarray(self.instance_data["sites"]["sites_xy"], dtype=np.float32)
        Ns = int(self.instance_data["sites"].get("Ns", len(sites_xy)))
        if "assign_seed" in seed:
            assign = list(seed["assign_seed"])
        elif "seed_assign" in seed:
            assign = list(seed["seed_assign"])
        elif "assign_grid" in baseline:
            assign = list(baseline["assign_grid"])
        else:
            if Ns <= 0:
                assign = [0 for _ in range(slots)]
            else:
                sites = list(range(Ns))
                self.rng.shuffle(sites)
                if slots <= Ns:
                    assign = sites[:slots]
                else:
                    assign = [sites[i % Ns] for i in range(slots)]
        return WaferLayoutSolution(assign=assign, S=slots, Ns=Ns)

    def get_key_value(self, solution: WaferLayoutSolution) -> float:
        eval_out = self.evaluate(solution)
        return float(eval_out.get("total_scalar", 0.0))

    def evaluate(self, solution: WaferLayoutSolution) -> Dict[str, Any]:
        eval_assign = self.algorithm_data.get("eval_assign")
        assign_arr = np.asarray(solution.assign, dtype=int)
        if callable(eval_assign):
            return eval_assign(assign_arr)
        fallback = self._ensure_fallback_eval()
        return fallback(assign_arr)


Env = WaferLayoutEnv
