from __future__ import annotations

import importlib.util
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from src.problems.base.env import BaseEnv
from src.problems.wafer_layout.components import NoOp, WaferLayoutOperator, WaferLayoutSolution

if importlib.util.find_spec("layout.evaluator") is not None:
    from layout.evaluator import LayoutEvaluator  # type: ignore
    from layout.evaluator import LayoutState  # type: ignore
else:
    LayoutEvaluator = None
    LayoutState = None

from src.problems.wafer_layout.problem_state import (
    get_instance_problem_state,
    get_observation_problem_state,
    get_solution_problem_state,
)


class Env(BaseEnv):
    def __init__(self, data_name: str, rng=None):
        self.rng = rng
        super().__init__(data_name, "wafer_layout")
        self.solution = self.current_solution
        self.best_solution = self.solution.copy()
        self.problem_size = int(self._infer_slot_count())
        self.construction_steps = 0
        self.max_steps = int(self.problem_size)
        self.algorithm_data: Dict[str, Any] = {}
        self._evaluator = None
        self._layout_state = None

    def _infer_slot_count(self) -> int:
        slots = self.instance_data.get("slots", [])
        if isinstance(slots, dict):
            return int(slots.get("S", len(slots.get("tdp", []) or [])))
        return int(len(slots))

    def _infer_site_count(self) -> int:
        sites_xy = self.instance_data.get("sites_xy")
        if sites_xy is None:
            sites_xy = self.instance_data.get("sites", {}).get("sites_xy", [])
        return int(self.instance_data.get("sites", {}).get("Ns", len(sites_xy or [])))

    def load_data(self, data_path: str) -> dict:
        with open(data_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def init_solution(self) -> WaferLayoutSolution:
        slots = self._infer_slot_count()
        sites_xy = self.instance_data.get("sites_xy")
        if sites_xy is None:
            sites_xy = self.instance_data.get("sites", {}).get("sites_xy", [])
        Ns = int(self.instance_data.get("sites", {}).get("Ns", len(sites_xy or [])))

        baseline = self.instance_data.get("baseline", {})
        seed = self.instance_data.get("seed", {})
        assign = None
        seed_assign = seed.get("assign_seed")
        if seed_assign is None:
            seed_assign = self.instance_data.get("seed_assign")
        baseline_assign = baseline.get("assign_grid")
        if baseline_assign is None:
            baseline_assign = self.instance_data.get("baseline_assign")
        if seed_assign is not None:
            assign = list(seed_assign)
        elif seed.get("seed_assign") is not None:
            assign = list(seed.get("seed_assign"))
        elif baseline_assign is not None:
            assign = list(baseline_assign)
        elif self.instance_data.get("baseline_assign_grid") is not None:
            assign = list(self.instance_data.get("baseline_assign_grid"))
        else:
            if Ns <= 0:
                assign = [0 for _ in range(slots)]
            else:
                assign = list(np.arange(slots, dtype=int) % max(1, Ns))
        return WaferLayoutSolution(assign=assign, S=slots, Ns=Ns)

    def validation_solution(self, solution: WaferLayoutSolution) -> bool:
        assign = solution.assign
        num_sites = self._infer_site_count()
        for a in assign:
            a = int(a)
            if a == -1:
                continue
            if a < 0 or a >= num_sites:
                return False
        return True

    def _ensure_evaluator(self) -> None:
        if self._evaluator is not None:
            return
        if LayoutEvaluator is None or LayoutState is None:
            return
        obj = self.instance_data.get("objective_cfg", {}) or {}
        scalar = obj.get("scalar_weights", obj) if isinstance(obj, dict) else {}
        baseline = self.instance_data.get("baseline", {}) or {}
        self._evaluator = LayoutEvaluator(
            sigma_mm=float(obj.get("sigma_mm", 20.0)),
            baseline={
                "L_comm_baseline": float(baseline.get("L_comm", obj.get("comm_baseline", 1.0))),
                "L_therm_baseline": float(baseline.get("L_therm", obj.get("therm_baseline", 1.0))),
            },
            scalar_w={
                "w_comm": float(scalar.get("w_comm", 0.7)),
                "w_therm": float(scalar.get("w_therm", 0.3)),
                "w_penalty": float(scalar.get("w_penalty", 1000.0)),
            },
        )
        sites_xy = self.instance_data.get("sites_xy")
        if sites_xy is None:
            sites_xy = self.instance_data.get("sites", {}).get("sites_xy", [])
        sites_xy = np.asarray(sites_xy, dtype=np.float32)
        slots = self._infer_slot_count()
        Ns = int(self.instance_data.get("sites", {}).get("Ns", len(sites_xy)))
        traffic = np.asarray(self.instance_data.get("mapping", {}).get("traffic_matrix", []), dtype=float)
        if traffic.size == 0:
            traffic = np.zeros((slots, slots), dtype=float)
        tdp = np.asarray(self.instance_data.get("slots", {}).get("tdp", []), dtype=float)
        self._layout_state = LayoutState(
            S=slots,
            Ns=Ns,
            wafer_radius_mm=float(self.instance_data.get("wafer", {}).get("radius_mm", 0.0)),
            sites_xy_mm=sites_xy,
            assign=np.zeros(slots, dtype=int),
            chip_tdp_w=tdp,
            traffic_bytes=traffic,
            meta={"margin_mm": float(self.instance_data.get("wafer", {}).get("margin_mm", 0.0))},
        )

    def _evaluate_full(self, assign: List[int]) -> Dict[str, float]:
        if LayoutEvaluator is None or LayoutState is None:
            from src.problems.wafer_layout.evaluator_copy import evaluate_layout

            return evaluate_layout(self.instance_data, assign)
        self._ensure_evaluator()
        if self._evaluator is None or self._layout_state is None:
            return {}
        self._layout_state.assign = np.asarray(assign, dtype=int)
        out = self._evaluator.evaluate(self._layout_state)
        return dict(out)

    def get_key_value(self, solution: WaferLayoutSolution) -> float:
        metrics = self._evaluate_full(solution.assign)
        return float(metrics.get("total_scalar", 0.0))

    @property
    def continue_run(self) -> bool:
        return int(self.construction_steps) < int(self.max_steps)

    @property
    def is_complete_solution(self) -> bool:
        return int(self.construction_steps) >= int(self.max_steps)

    def get_problem_state(self) -> Dict[str, Any]:
        instance_state = get_instance_problem_state(self.instance_data)
        solution_state = get_solution_problem_state(self.instance_data, self.solution)
        obs = get_observation_problem_state(solution_state)
        return {
            "instance_data": self.instance_data,
            "current_solution": self.solution,
            "observation": obs,
            "instance_state": instance_state,
            "solution_state": solution_state,
        }

    def reset(self, output_dir: Optional[str] = None) -> None:
        super().reset(output_dir)
        self.algorithm_data.update({"env": self, "rng": self.rng})
        self.initial_assign = list(getattr(self.solution, "assign", []) or [])
        self.construction_steps = 0

    def run_operator(
        self,
        operator: WaferLayoutOperator | None,
        inplace: bool = True,
        meta: Dict[str, Any] | None = None,
        new_solution: WaferLayoutSolution | None = None,
        stage: str | None = None,
        time_ms: int | None = None,
    ) -> bool:
        if not self.continue_run:
            return False
        if operator is None:
            operator = NoOp()
        pre_assign = list(self.current_solution.assign)
        before_metrics = self._evaluate_full(pre_assign)
        before_score = float(before_metrics.get("total_scalar", 0.0))
        accepted = bool(inplace)
        if inplace:
            if new_solution is None:
                new_solution = operator.run(self.current_solution)
            self.current_solution = new_solution
            self.solution = self.current_solution
        post_assign = list(self.current_solution.assign)
        after_metrics = self._evaluate_full(post_assign)
        after_score = float(after_metrics.get("total_scalar", 0.0))
        record: Dict[str, Any] = {
            "step": int(self.construction_steps),
            "operator": operator,
            "meta": meta or {},
            "accepted": accepted,
            "assign": post_assign,
            "total_scalar": float(after_score),
            "comm_norm": float(after_metrics.get("comm_norm", 0.0)),
            "therm_norm": float(after_metrics.get("therm_norm", 0.0)),
            "duplicate_penalty": float(after_metrics.get("penalty", {}).get("duplicate", 0.0)),
            "boundary_penalty": float(after_metrics.get("penalty", {}).get("boundary", 0.0)),
            "time_ms": int(time_ms or 0),
            "stage": stage or "",
            "pre_total_scalar": before_score,
        }
        self.recordings.append(record)
        if after_score < self.get_key_value(self.best_solution) - 1e-12:
            self.best_solution = WaferLayoutSolution(
                assign=post_assign,
                S=self.current_solution.S,
                Ns=self.current_solution.Ns,
            )
        self.construction_steps += 1
        return accepted

    @staticmethod
    def _operator_to_dict(operator: WaferLayoutOperator | None, pre_assign: list[int]) -> Dict[str, Any]:
        if operator is None:
            operator = NoOp()
        name = getattr(operator, "name", operator.__class__.__name__)
        op_args: Dict[str, Any] = {}
        if name == "swap" and hasattr(operator, "i") and hasattr(operator, "j"):
            op_args = {"i": int(operator.i), "j": int(operator.j)}
        elif name == "relocate" and hasattr(operator, "i") and hasattr(operator, "site_id"):
            slot = int(operator.i)
            from_site = int(pre_assign[slot]) if 0 <= slot < len(pre_assign) else None
            op_args = {"i": slot, "site_id": int(operator.site_id), "from_site": from_site}
        elif name == "cluster_move" and hasattr(operator, "slots"):
            op_args = {"slots": list(operator.slots), "target_sites": getattr(operator, "target_sites", [])}
        elif name == "random_kick" and hasattr(operator, "ops"):
            op_args = {"k": len(getattr(operator, "ops", []) or [])}
        return {
            "op": name,
            "op_args": op_args,
            "signature": operator.signature() if hasattr(operator, "signature") else name,
        }

    def dump_result(self) -> None:
        if not self.output_dir:
            return
        os.makedirs(self.output_dir, exist_ok=True)
        rec_path = os.path.join(self.output_dir, "recordings.jsonl")
        best_path = os.path.join(self.output_dir, "best_solution.json")
        finished_path = os.path.join(self.output_dir, "finished.txt")

        best_eval: Dict[str, Any] | None = None
        best_assign: list[int] | None = None
        prev_assign: list[int] = list(getattr(self, "initial_assign", self.current_solution.assign))

        with open(rec_path, "w", encoding="utf-8") as f:
            for rec in self.recordings:
                assign = list(rec.get("assign") or self.current_solution.assign)
                eval_out = self._evaluate_full(list(assign))
                if best_eval is None or eval_out.get("total_scalar", float("inf")) < best_eval.get(
                    "total_scalar", float("inf")
                ):
                    best_eval = dict(eval_out)
                    best_assign = list(assign)
                op_dict = self._operator_to_dict(rec.get("operator"), prev_assign)
                meta = rec.get("meta") or {}
                record = {
                    "step": int(rec.get("step", 0)),
                    "heuristic": meta.get("heuristic_name") or meta.get("heuristic") or op_dict["op"],
                    "op": op_dict["op"],
                    "op_args": op_dict["op_args"],
                    "signature": op_dict["signature"],
                    "assign": list(assign),
                    "key_value": float(eval_out.get("total_scalar", 0.0)),
                    "comm_norm": float(eval_out.get("comm_norm", 0.0)),
                    "therm_norm": float(eval_out.get("therm_norm", 0.0)),
                    "duplicate_penalty": float(eval_out.get("penalty", {}).get("duplicate", 0.0)),
                    "boundary_penalty": float(eval_out.get("penalty", {}).get("boundary", 0.0)),
                    "accepted": bool(rec.get("accepted", True)),
                    "time_ms": int(rec.get("time_ms", 0)),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                prev_assign = list(assign)

        best_payload = {
            "best_assign": best_assign or list(self.solution.assign),
            "best_metrics": best_eval or {},
        }
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_payload, f, ensure_ascii=False, indent=2)
        with open(finished_path, "w", encoding="utf-8") as f:
            f.write("ok\n")
