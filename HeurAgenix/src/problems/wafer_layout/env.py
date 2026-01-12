import json
import time
from pathlib import Path

import numpy as np

from src.problems.base.env import BaseEnv

from .components import (
    WaferLayoutSolution,
    SwapOperator,
    RelocateOperator,
    ClusterMoveOperator,
    RandomKickOperator,
    NoOpOperator,
)

from layout.evaluator import LayoutEvaluator, LayoutState
from .problem_state import attach_key_value


class Env(BaseEnv):
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, problem="wafer_layout")

        self.key_item = "total_scalar"
        self.compare = lambda x, y: (y - x)
        self.construction_steps = int(self.problem_size)

        self.problem_state = self.get_problem_state()

    def load_data(self, data_path: str) -> dict:
        d = json.loads(Path(data_path).read_text(encoding="utf-8"))

        sites_xy = np.asarray(d["sites"]["sites_xy"], dtype=np.float32)
        tdp = np.asarray(d["chiplets"]["tdp"], dtype=np.float32)
        traffic = np.asarray(d["mapping"]["traffic_matrix"], dtype=np.float32)

        d["_sites_xy"] = sites_xy
        d["_tdp"] = tdp
        d["_traffic"] = traffic

        s_count = int(d.get("chiplets", {}).get("S", len(tdp)))
        ns_count = int(d.get("sites", {}).get("Ns", sites_xy.shape[0]))
        d["S"] = s_count
        d["Ns"] = ns_count

        weights = d.get("weights", {"w_comm": 0.5, "w_therm": 0.5})
        sigma = float(d.get("sigma", 1.0))
        baseline = d.get("baseline", {})
        baseline_dict = {
            "L_comm_baseline": float(baseline.get("L_comm", 1.0)),
            "L_therm_baseline": float(baseline.get("L_therm", 1.0)),
        }
        obj_cfg = {
            "w_comm": float(weights.get("w_comm", 0.5)),
            "w_therm": float(weights.get("w_therm", 0.5)),
            "sigma": sigma,
        }

        self._evaluator = LayoutEvaluator(objective_cfg=obj_cfg, baseline=baseline_dict)

        self._state_template = LayoutState(
            assign=np.zeros(s_count, dtype=np.int64),
            sites_xy=sites_xy,
            tdp=tdp,
            traffic_matrix=traffic,
            wafer_radius=float(d.get("wafer", {}).get("radius", 1.0)),
        )
        return d

    def init_solution(self, data_name: str) -> WaferLayoutSolution:
        s_count = int(self.instance_data["S"])
        ns_count = int(self.instance_data["Ns"])

        seed = int(self.instance_data.get("assign_seed", 1))
        rng = np.random.default_rng(seed)

        if ns_count >= s_count:
            assign = rng.choice(ns_count, size=s_count, replace=False).astype(np.int64)
        else:
            assign = rng.integers(low=0, high=ns_count, size=s_count, dtype=np.int64)

        return WaferLayoutSolution(assign=assign)

    def get_problem_size(self) -> int:
        return int(self.instance_data.get("S", 1))

    def get_key_value(self) -> float:
        metrics = self.evaluate_assign(self.current_solution.assign)
        return float(metrics["total_scalar"])

    def validation_solution(self, solution: WaferLayoutSolution) -> bool:
        return True

    def evaluate_assign(self, assign: np.ndarray) -> dict:
        self._state_template.assign = np.asarray(assign, dtype=np.int64)
        return self._evaluator.evaluate(self._state_template)

    def get_problem_state(self) -> dict:
        ps = {
            "instance_data": self.instance_data,
            "current_solution": self.current_solution,
            "key_item": self.key_item,
            "key_value": getattr(self, "key_value", None),
            "eval_assign": self.evaluate_assign,
        }
        return ps

    def run_heuristic(self, heuristic, **kwargs):
        t0 = time.time()

        self.problem_state = self.get_problem_state()
        if not hasattr(self, "key_value"):
            self.key_value = self.get_key_value()
        self.problem_state = attach_key_value(self.problem_state, self.key_value)
        op, delta = heuristic(self.problem_state, self.algorithm_data, **kwargs)
        if op is None:
            op = NoOpOperator()
            delta = {}

        new_sol = op.run(self.instance_data, self.current_solution)
        self.current_solution = new_sol
        self.is_valid_solution = self.validation_solution(self.current_solution)

        metrics = self.evaluate_assign(self.current_solution.assign)
        self.key_value = float(metrics["total_scalar"])
        self.is_complete_solution = True

        if (self.best_solution is None) or (self.compare(self.key_value, self.best_key_value) > 0):
            self.best_solution = self.current_solution.copy()
            self.best_key_value = float(self.key_value)

        rec = {
            "step_idx": int(self.current_steps),
            "heuristic": getattr(heuristic, "__name__", str(heuristic)),
            "op": getattr(op, "op_name", op.__class__.__name__),
            "op_args_json": op.to_json() if hasattr(op, "to_json") else {},
            "accepted": 1,
            "total_scalar": float(metrics["total_scalar"]),
            "comm_norm": float(metrics.get("comm_norm", 0.0)),
            "therm_norm": float(metrics.get("therm_norm", 0.0)),
            "objective_scalar": float(metrics["total_scalar"]),
            "assign": self.current_solution.to_list(),
            "time_ms": int((time.time() - t0) * 1000),
        }
        _ = delta
        self.recordings.append(rec)

        self.current_steps += 1
        return op

    def dump_result(self) -> None:
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rec_path = out_dir / "recordings.jsonl"
        with open(rec_path, "w", encoding="utf-8") as f:
            for r in self.recordings:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        best_assign = (
            self.best_solution.to_list()
            if self.best_solution is not None
            else self.current_solution.to_list()
        )
        best_eval = self.evaluate_assign(np.asarray(best_assign, dtype=np.int64))

        best_path = out_dir / "best_solution.json"
        best_path.write_text(
            json.dumps(
                {
                    "problem": "wafer_layout",
                    "seed_id": int(self.instance_data.get("assign_seed", 0)),
                    "best_assign": best_assign,
                    "best_key_value": float(self.best_key_value)
                    if self.best_solution is not None
                    else float(self.key_value),
                    "best_eval": best_eval,
                    "meta": {
                        "S": int(self.instance_data["S"]),
                        "Ns": int(self.instance_data["Ns"]),
                        "weights": self.instance_data.get("weights", {}),
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
