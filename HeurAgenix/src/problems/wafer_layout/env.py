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
        self.problem_size = self.get_problem_size()
        self.construction_steps = int(self.problem_size)
        self.current_steps = 0
        self.continue_run = True
        self.best_solution = None
        self.best_key_value = float("inf")
        self.key_value = self.get_key_value()

        self.problem_state = self.get_problem_state()

    def load_data(self, data_path: str) -> dict:
        d = json.loads(Path(data_path).read_text(encoding="utf-8"))

        slots = d.get("slots", {})
        sites = d.get("sites", {})
        mapping = d.get("mapping", {})
        wafer = d.get("wafer", {})
        baseline = d.get("baseline", {})
        objective_cfg = d.get("objective_cfg", {})
        scalar_weights = objective_cfg.get("scalar_weights", {})

        sites_xy = np.asarray(sites.get("sites_xy", []), dtype=np.float32)
        tdp = np.asarray(slots.get("tdp", []), dtype=np.float32)
        traffic = np.asarray(mapping.get("traffic_matrix", []), dtype=np.float32)

        s_count = int(slots.get("S", len(tdp)))
        ns_count = int(sites.get("Ns", sites_xy.shape[0]))
        if traffic.size == 0:
            traffic = np.zeros((s_count, s_count), dtype=np.float32)

        d["_sites_xy"] = sites_xy
        d["_tdp"] = tdp
        d["_traffic"] = traffic
        d["S"] = s_count
        d["Ns"] = ns_count

        baseline_dict = {
            "L_comm_baseline": float(baseline.get("L_comm", 1.0)),
            "L_therm_baseline": float(baseline.get("L_therm", 1.0)),
        }
        sigma = float(objective_cfg.get("sigma_mm", 20.0))
        scalar_w = {
            "w_comm": float(scalar_weights.get("w_comm", 0.7)),
            "w_therm": float(scalar_weights.get("w_therm", 0.3)),
            "w_penalty": float(scalar_weights.get("w_penalty", 1000.0)),
        }

        self._evaluator = LayoutEvaluator(
            sigma_mm=sigma,
            baseline=baseline_dict,
            scalar_w=scalar_w,
        )

        self._state_template = LayoutState(
            S=s_count,
            Ns=ns_count,
            wafer_radius_mm=float(wafer.get("radius_mm", 0.0)),
            sites_xy_mm=sites_xy,
            assign=np.zeros(s_count, dtype=np.int64),
            chip_tdp_w=tdp,
            traffic_bytes=traffic,
            meta={"margin_mm": float(wafer.get("margin_mm", 0.0))},
        )
        return d

    def init_solution(self) -> WaferLayoutSolution:
        s_count = int(self.instance_data["S"])
        ns_count = int(self.instance_data["Ns"])

        seed_payload = self.instance_data.get("seed", {}) or {}
        seed_assign = seed_payload.get("assign_seed")
        if isinstance(seed_assign, (list, tuple, np.ndarray)) and len(seed_assign) > 0:
            assign = np.asarray(seed_assign, dtype=np.int64)
        else:
            seed = int(seed_payload.get("assign_seed", 1))
            rng = np.random.default_rng(seed)
            if ns_count <= 0:
                assign = np.zeros(s_count, dtype=np.int64)
            elif ns_count >= s_count:
                assign = rng.choice(ns_count, size=s_count, replace=False).astype(np.int64)
            else:
                assign = rng.integers(low=0, high=ns_count, size=s_count, dtype=np.int64)

        return WaferLayoutSolution(assign=assign, S=s_count, Ns=ns_count)

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

    def reset(self, output_dir: str | None = None) -> None:
        super().reset(output_dir=output_dir)
        self.current_steps = 0
        self.continue_run = True
        self.best_solution = None
        self.best_key_value = float("inf")
        self.key_value = self.get_key_value()
        self.problem_state = self.get_problem_state()

    def run_heuristic(self, heuristic, **kwargs):
        self.problem_state = self.get_problem_state()
        if not hasattr(self, "key_value"):
            self.key_value = self.get_key_value()
        self.problem_state = attach_key_value(self.problem_state, self.key_value)
        op, delta = heuristic(self.problem_state, self.algorithm_data, **kwargs)
        if op is None:
            op = NoOpOperator()
            delta = {}

        return self.run_operator(
            op,
            inplace=True,
            meta=delta,
            stage=getattr(heuristic, "__name__", "heuristic"),
        )

    def run_operator(
        self,
        operator,
        inplace: bool = True,
        meta: dict | None = None,
        stage: str | None = None,
        time_ms: int = 0,
        new_solution: WaferLayoutSolution | None = None,
        **kwargs,
    ):
        t0 = time.time()

        accepted = bool(inplace)
        if accepted:
            if new_solution is not None:
                self.current_solution = new_solution
            else:
                self.current_solution = operator.run(self.instance_data, self.current_solution)
            self.is_valid_solution = self.validation_solution(self.current_solution)
            self.solution = self.current_solution

        metrics = self.evaluate_assign(self.current_solution.assign)
        self.key_value = float(metrics["total_scalar"])
        self.is_complete_solution = True

        if (self.best_solution is None) or (self.compare(self.key_value, self.best_key_value) > 0):
            self.best_solution = self.current_solution.copy()
            self.best_key_value = float(self.key_value)

        op_payload = operator.to_json() if hasattr(operator, "to_json") else {}
        op_name = op_payload.get("op", getattr(operator, "op_name", operator.__class__.__name__))
        op_args = {k: v for k, v in op_payload.items() if k != "op"}

        rec = {
            "op": op_name,
            "op_args": op_args,
            "accepted": int(accepted),
            "assign": self.current_solution.to_list(),
            "total_scalar": float(metrics["total_scalar"]),
            "comm_norm": float(metrics.get("comm_norm", 0.0)),
            "therm_norm": float(metrics.get("therm_norm", 0.0)),
            "time_ms": int(time_ms) if time_ms else int((time.time() - t0) * 1000),
        }
        if meta:
            rec["meta"] = meta
        if stage:
            rec["stage"] = stage

        self.recordings.append(rec)
        if self.output_dir:
            rec_path = Path(self.output_dir) / "recordings.jsonl"
            rec_path.parent.mkdir(parents=True, exist_ok=True)
            with rec_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        self.current_steps += 1
        return operator

    def dump_result(self) -> None:
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rec_path = out_dir / "recordings.jsonl"
        if not rec_path.exists():
            with rec_path.open("w", encoding="utf-8") as f:
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
                    "seed_id": self._resolve_seed_id(),
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

    def _resolve_seed_id(self) -> int:
        seed_payload = self.instance_data.get("seed", {}) or {}
        seed_value = seed_payload.get("assign_seed", 0)
        if isinstance(seed_value, (int, float, str)):
            try:
                return int(seed_value)
            except ValueError:
                return 0
        return 0
