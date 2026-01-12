from __future__ import annotations
import importlib
import json
import os
import time
import random
from typing import Any, Dict

import numpy as np

from src.problems.base.env import BaseEnv
from src.problems.wafer_layout.components import WaferLayoutSolution
from src.problems.wafer_layout.problem_state import (
    get_instance_problem_state,
    get_solution_problem_state,
    get_observation_problem_state,
)


def _load_layout_evaluator():
    if importlib.util.find_spec("layout.evaluator") is None:
        return None, None
    module = importlib.import_module("layout.evaluator")
    return module.LayoutEvaluator, module.LayoutState


class Env(BaseEnv):
    def load_data(self, data_path: str) -> dict:
        with open(data_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        assert "slots" in d and "sites" in d and "mapping" in d, f"bad instance: {data_path}"
        return d

    def __init__(self, data_name: str):
        super().__init__(data_name)

        self.problem = "wafer_layout"

        slots = self.instance_data.get("slots", {})
        sites = self.instance_data.get("sites", {})
        self.S = int(slots.get("S", len(slots.get("tdp", slots.get("slot_tdp_w", [])))))
        self.Ns = int(sites.get("Ns", len(sites.get("sites_xy", []))))
        self.problem_size = self.S
        self.construction_steps = self.S

        obj = self.instance_data.get("objective_cfg", {}) or self.instance_data.get("objective", {})
        self.sigma_mm = float(obj.get("sigma_mm", 20.0))
        w = obj.get("scalar_weights", {}) or {}
        self.w_comm = float(w.get("w_comm", 0.7))
        self.w_therm = float(w.get("w_therm", 0.3))
        self.w_penalty = float(w.get("w_penalty", w.get("w_dup", 0.1) + w.get("w_boundary", 0.2)))

        base = self.instance_data.get("baseline", {}) or {}
        self.base_comm = float(base.get("L_comm", 1.0)) if base.get("L_comm", 0.0) else 1.0
        self.base_therm = float(base.get("L_therm", 1.0)) if base.get("L_therm", 0.0) else 1.0

        self.temp_init = float(obj.get("temp_init", 0.05))
        self.temp_min = float(obj.get("temp_min", 0.005))
        self.temp_decay = float(obj.get("temp_decay", 0.999))

        self._rec_fp = None
        self._rec_path = None
        self._best_path = None

        self._LayoutEvaluator, self._LayoutState = _load_layout_evaluator()
        self._evaluator = None
        self._layout_state = None
        self._init_evaluator()

        self.problem_state = self.get_problem_state()

    def _init_evaluator(self):
        if self._LayoutEvaluator is None or self._LayoutState is None:
            self._evaluator = None
            self._layout_state = None
            return

        slots = self.instance_data.get("slots", {})
        sites = self.instance_data.get("sites", {})
        mapping = self.instance_data.get("mapping", {})
        wafer = self.instance_data.get("wafer", {})

        sites_xy = np.asarray(sites.get("sites_xy", []), dtype=np.float32)
        site_tdp_w = np.asarray(sites.get("site_tdp_w", np.zeros(len(sites_xy))), dtype=np.float32)
        traffic = np.asarray(mapping.get("traffic_matrix", mapping.get("traffic_bytes", [])), dtype=np.float32)
        slot_tdp = np.asarray(slots.get("tdp", slots.get("slot_tdp_w", [])), dtype=np.float32)
        if traffic.size == 0:
            traffic = np.zeros((self.S, self.S), dtype=np.float32)

        self._layout_state = self._LayoutState(
            S=int(self.S),
            Ns=int(self.Ns),
            wafer_radius_mm=float(wafer.get("radius_mm", 0.0)),
            sites_xy_mm=sites_xy,
            assign=np.zeros(self.S, dtype=np.int64),
            chip_tdp_w=slot_tdp,
            traffic_bytes=traffic,
            meta={"margin_mm": float(wafer.get("margin_mm", 0.0))},
        )
        self._evaluator = self._LayoutEvaluator(
            sigma_mm=self.sigma_mm,
            baseline={"L_comm_baseline": self.base_comm, "L_therm_baseline": self.base_therm},
            scalar_w={"w_comm": self.w_comm, "w_therm": self.w_therm, "w_penalty": self.w_penalty},
        )

    def init_solution(self) -> WaferLayoutSolution:
        seed = (self.instance_data.get("seed", {}) or {}).get("assign_seed", None)
        if seed is None:
            seed = [min(i, max(0, self.Ns - 1)) for i in range(self.S)]
        seed = list(seed)
        if len(seed) != self.S:
            seed = (seed + [0] * self.S)[: self.S]
        if self.Ns <= 0:
            seed = [0 for _ in range(self.S)]
        else:
            seed = [int(x) % self.Ns for x in seed]
        return WaferLayoutSolution(assign=np.asarray(seed, dtype=np.int64))

    def is_complete_solution(self) -> bool:
        return True

    def validate_solution(self, solution: WaferLayoutSolution) -> bool:
        if solution is None or not hasattr(solution, "assign"):
            return False
        a = list(solution.assign)
        if len(a) != self.S:
            return False
        for x in a:
            if not (0 <= int(x) < self.Ns):
                return False
        return True

    def compare(self, x, y):
        return y - x

    def get_key_value(self, solution: WaferLayoutSolution | None = None) -> float:
        sol = solution if solution is not None else self.current_solution
        if self._evaluator is None or self._layout_state is None:
            a = sol.assign
            dup = len(a) - len(set(a))
            return float(dup)
        self._layout_state.assign = np.asarray(sol.assign, dtype=np.int64)
        metrics = self._evaluator.evaluate(self._layout_state)
        return float(metrics["total_scalar"])

    def summarize_env(self) -> str:
        curr = float(self.key_value)
        best = float(getattr(self, "best_key_value", curr))
        return f"wafer_layout: S={self.S}, Ns={self.Ns}, step={self.current_steps}/{self.max_steps}, curr={curr:.6f}, best={best:.6f}"

    def get_problem_state(self) -> Dict[str, Any]:
        instance_state = get_instance_problem_state(self.instance_data)
        solution_state = get_solution_problem_state(self.instance_data, self.current_solution)
        state = {
            "instance_problem_state": instance_state,
            "solution_problem_state": solution_state,
        }
        state["observation_problem_state"] = get_observation_problem_state(state)
        return state

    def update_problem_state(self) -> None:
        self.problem_state = self.get_problem_state()

    def reset(self, output_dir: str = None):
        super().reset(output_dir=output_dir)

        self._rec_path = os.path.join(self.output_dir, "recordings.jsonl")
        self._best_path = os.path.join(self.output_dir, "best_solution.json")
        os.makedirs(self.output_dir, exist_ok=True)
        self._rec_fp = open(self._rec_path, "w", encoding="utf-8")

        self.key_value = self.get_key_value(self.current_solution)
        self.best_key_value = float(self.key_value)
        self.best_solution = WaferLayoutSolution(assign=list(self.current_solution.assign))

        self._dump_best()
        self.update_problem_state()

    def _dump_best(self):
        rec = {
            "best_assign": list(self.best_solution.assign),
            "best": {"assign": list(self.best_solution.assign)},
            "assign": list(self.best_solution.assign),
            "key_value": float(self.best_key_value),
            "problem": "wafer_layout",
            "S": int(self.S),
            "Ns": int(self.Ns),
        }
        with open(self._best_path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)

    def run_heuristic(self, heuristic, algorithm_data: dict = {}, record: bool = True, **kwargs):
        t0 = time.time()
        try:
            operator, algorithm_data = heuristic(self.problem_state, algorithm_data, **kwargs)
        except Exception as e:  # noqa: BLE001
            dt = (time.time() - t0) * 1000.0
            line = {"op": "error", "op_args": {"error": str(e)}, "accepted": 0, "time_ms": dt, "meta": {}}
            self._rec_fp.write(json.dumps(line, ensure_ascii=False) + "\n")
            self._rec_fp.flush()
            self.current_steps += 1
            return f"[heuristic_error] {e}"

        if not hasattr(operator, "run"):
            dt = (time.time() - t0) * 1000.0
            line = {"op": "invalid_operator", "op_args": {}, "accepted": 0, "time_ms": dt, "meta": {}}
            self._rec_fp.write(json.dumps(line, ensure_ascii=False) + "\n")
            self._rec_fp.flush()
            self.current_steps += 1
            return "[invalid_operator]"

        old_assign = np.asarray(self.current_solution.assign).copy()
        old_key = float(self.key_value)

        operator.run(self.current_solution)
        new_key = float(self.get_key_value(self.current_solution))

        accept = False
        if new_key <= old_key:
            accept = True
        else:
            delta = new_key - old_key
            temp = max(self.temp_min, self.temp_init * (self.temp_decay ** max(0, self.current_steps)))
            prob = float(np.exp(-delta / max(1e-9, temp)))
            if random.random() < prob:
                accept = True

        if not accept:
            self.current_solution.assign = old_assign
            new_key = old_key

        self.key_value = float(new_key)
        if float(new_key) < float(self.best_key_value):
            self.best_key_value = float(new_key)
            self.best_solution = WaferLayoutSolution(assign=list(self.current_solution.assign))
            self._dump_best()

        self.update_problem_state()

        dt = (time.time() - t0) * 1000.0
        op_args = operator.to_action() if hasattr(operator, "to_action") else {"op": type(operator).__name__}
        op = op_args.get("op", type(operator).__name__)

        meta = {"tabu_hit": 0, "inverse_hit": 0, "cooldown_hit": 0}
        line = {
            "op": op,
            "op_args": op_args,
            "accepted": 1 if accept else 0,
            "time_ms": dt,
            "assign": list(self.current_solution.assign),
            "meta": meta,
        }
        self._rec_fp.write(json.dumps(line, ensure_ascii=False) + "\n")
        self._rec_fp.flush()

        self.current_steps += 1
        return operator

    @property
    def continue_run(self) -> bool:
        if getattr(self, "max_steps", None) is None:
            return True
        return int(self.current_steps) < int(self.max_steps)

    def dump_result(self):
        if self._rec_fp is not None:
            try:
                self._rec_fp.flush()
                self._rec_fp.close()
            except Exception:
                pass
            self._rec_fp = None
        super().dump_result()
