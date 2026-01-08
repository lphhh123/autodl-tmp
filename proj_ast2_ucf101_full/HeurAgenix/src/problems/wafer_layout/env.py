"""Wafer layout environment for HeurAgenix baseline."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.problems.base.env import BaseEnv
from src.problems.wafer_layout.components import NoOp, WaferLayoutOperator, WaferLayoutSolution
from src.problems.wafer_layout.problem_state import (
    get_instance_problem_state,
    get_observation_problem_state,
    get_solution_problem_state,
)


class Env(BaseEnv):
    def __init__(
        self,
        data_name: str,
        rng: random.Random | None = None,
        seed: int | None = None,
        algorithm_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.rng = rng or random.Random(0 if seed is None else seed)
        self._extra_algorithm_data: Dict[str, Any] = algorithm_data or {}
        self._evaluator = None
        self._layout_state = None
        self.last_eval: Dict[str, Any] = {}
        self.problem_size = 1
        self.construction_steps = 1
        self.key_item = "total_scalar"
        self.compare = min
        super().__init__(data_name, "wafer_layout")
        slots = int(self.instance_data["slots"].get("S", len(self.instance_data["slots"].get("tdp", []))))
        self.problem_size = max(1, slots)
        self.reset()

    def load_data(self, path: str) -> Dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def _ensure_evaluator(self) -> None:
        if self._evaluator is not None:
            return
        from layout.evaluator import LayoutEvaluator, LayoutState

        baseline = self.instance_data.get("baseline", {})
        objective_cfg = self.instance_data.get("objective_cfg", {})
        scalar = objective_cfg.get("scalar_weights", {})
        sites_xy = np.asarray(self.instance_data["sites"]["sites_xy"], dtype=np.float32)
        slots = int(self.instance_data["slots"].get("S", len(self.instance_data["slots"].get("tdp", []))))
        Ns = int(self.instance_data["sites"].get("Ns", len(sites_xy)))
        self._evaluator = LayoutEvaluator(
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
        self._layout_state = LayoutState(
            S=slots,
            Ns=Ns,
            wafer_radius_mm=float(self.instance_data["wafer"].get("radius_mm", 0.0)),
            sites_xy_mm=sites_xy,
            assign=np.zeros(slots, dtype=int),
            chip_tdp_w=np.asarray(self.instance_data["slots"].get("tdp", []), dtype=float),
            traffic_bytes=np.asarray(self.instance_data.get("mapping", {}).get("traffic_matrix", []), dtype=float),
            meta={"margin_mm": float(self.instance_data.get("wafer", {}).get("margin_mm", 0.0))},
        )

    def init_solution(self) -> WaferLayoutSolution:
        seed = self.instance_data.get("seed", {})
        baseline = self.instance_data.get("baseline", {})
        slots = int(self.instance_data["slots"].get("S", len(self.instance_data["slots"].get("tdp", []))))
        sites_xy = np.asarray(self.instance_data["sites"]["sites_xy"], dtype=np.float32)
        Ns = int(self.instance_data["sites"].get("Ns", len(sites_xy)))
        assign = None
        if seed.get("assign_seed") is not None:
            assign = list(seed["assign_seed"])
        elif seed.get("seed_assign") is not None:
            assign = list(seed["seed_assign"])
        elif baseline.get("assign_grid") is not None:
            assign = list(baseline["assign_grid"])
        elif self.instance_data.get("baseline_assign_grid") is not None:
            assign = list(self.instance_data["baseline_assign_grid"])
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

    def evaluate(self, solution: WaferLayoutSolution) -> Dict[str, Any]:
        self._ensure_evaluator()
        if self._evaluator is None or self._layout_state is None:
            return {}
        self._layout_state.assign = np.asarray(solution.assign, dtype=int)
        out = self._evaluator.evaluate(self._layout_state)
        self.last_eval = dict(out)
        return out

    def get_key_value(self, solution: WaferLayoutSolution) -> float:
        eval_out = self.evaluate(solution)
        return float(eval_out.get("total_scalar", 0.0))

    def validation_solution(self, new_solution: WaferLayoutSolution) -> bool:
        current_score = self.get_key_value(self.solution)
        return self.get_key_value(new_solution) <= current_score

    def is_complete_solution(self, solution: WaferLayoutSolution) -> bool:
        return False

    def get_problem_state(self) -> Dict[str, Any]:
        instance = get_instance_problem_state(self.instance_data)
        solution_state = get_solution_problem_state(self.instance_data, self.solution)
        eval_out = self.evaluate(self.solution)
        solution_state.update(
            {
                "total_scalar": eval_out.get("total_scalar", 0.0),
                "comm_norm": eval_out.get("comm_norm", 0.0),
                "therm_norm": eval_out.get("therm_norm", 0.0),
                "eval": eval_out,
            }
        )
        state = {"instance": instance, "solution": solution_state}
        return get_observation_problem_state(state)

    def reset(self, output_dir: Optional[str] = None) -> None:
        super().reset(output_dir)
        self.algorithm_data.update(self._extra_algorithm_data)
        self.algorithm_data["env"] = self
        self.algorithm_data["rng"] = self.rng
        self.initial_assign = list(getattr(self.solution, "assign", []) or [])

    def run_operator(
        self,
        operator: WaferLayoutOperator | None,
        accepted: bool,
        meta: Dict[str, Any] | None = None,
        stage: str | None = None,
        time_ms: int | None = None,
        new_solution: WaferLayoutSolution | None = None,
    ) -> None:
        if operator is None:
            operator = NoOp()
        if accepted:
            if new_solution is None:
                new_solution = operator.run(self.current_solution)
            self.current_solution = new_solution
            self.solution = self.current_solution
        record: Dict[str, Any] = {
            "step": int(self.step),
            "operator": operator,
            "meta": meta or {},
            "accepted": accepted,
            "assign": list(self.solution.assign),
        }
        if stage is not None:
            record["stage"] = stage
        if time_ms is not None:
            record["time_ms"] = int(time_ms)
        self.recordings.append(record)
        self.step += 1

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

    def dump_result(self, content: Dict[str, Any], output_folder: str) -> None:
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        recordings_path = output_path / "recordings.jsonl"
        best_path = output_path / "best_solution.json"

        best_eval: Dict[str, Any] | None = None
        best_assign: list[int] | None = None
        prev_assign: list[int] = list(getattr(self, "initial_assign", self.current_solution.assign))

        with recordings_path.open("w", encoding="utf-8") as f:
            for rec in self.recordings:
                assign = rec.get("assign") or list(self.solution.assign)
                eval_out = self.evaluate(WaferLayoutSolution(assign=list(assign), S=len(assign), Ns=self.solution.Ns))
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
            "meta": content or {},
        }
        best_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")
