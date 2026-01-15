"""Base environment for HeurAgenix problems."""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


class BaseEnv:
    def __init__(self, data_path: str, problem: Optional[str] = None):
        self.data_path = data_path
        self.problem = problem or ""
        self.instance_data = self.load_data(data_path)
        self.current_solution = self.init_solution()
        self.solution = self.current_solution
        self.recordings: List[Dict[str, Any]] = []
        self.step = 0
        self.algorithm_data: Dict[str, Any] = {}
        self.output_dir: Optional[str] = None
        self.current_steps = 0
        self.max_steps = None
        self.seed = 0
        self._rec_fp = None
        self._rec_path = None

    def load_data(self, path: str) -> Dict[str, Any]:
        raise NotImplementedError

    def init_solution(self):
        raise NotImplementedError

    def get_key_value(self, solution) -> float:
        raise NotImplementedError

    def get_problem_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    def summarize_env(self) -> Dict[str, Any]:
        if hasattr(self, "get_problem_state"):
            try:
                return dict(self.get_problem_state())
            except Exception:  # noqa: BLE001
                return {"instance_data": getattr(self, "instance_data", {}), "current_solution": self.solution}
        return {"instance_data": getattr(self, "instance_data", {}), "current_solution": self.solution}

    def reset(self, output_dir: Optional[str] = None) -> None:
        self.current_solution = self.init_solution()
        self.solution = self.current_solution
        self.recordings = []
        self.step = 0
        self.construction_steps = 0
        self.current_steps = 0
        self.algorithm_data = {}
        self.output_dir = output_dir
        if self._rec_fp is not None:
            try:
                self._rec_fp.close()
            except Exception:
                pass
        self._rec_fp = None
        self._rec_path = None
        self._ensure_rec_fp()

    def _ensure_rec_fp(self):
        if not self.output_dir:
            return None
        if self._rec_fp is not None:
            return self._rec_fp
        os.makedirs(self.output_dir, exist_ok=True)
        self._rec_path = os.path.join(self.output_dir, "recordings.jsonl")
        self._rec_fp = open(self._rec_path, "a", encoding="utf-8")
        return self._rec_fp

    def _write_record(self, line: Dict[str, Any]) -> None:
        fp = self._ensure_rec_fp()
        if fp is None:
            return
        fp.write(json.dumps(line, ensure_ascii=False) + "\n")
        fp.flush()

    def run_operator(
        self,
        operator,
        inplace: bool = True,
        meta: Optional[Dict[str, Any]] = None,
        stage: str = "heuragenix",
        time_ms: Optional[float] = None,
        heuristic_name: Optional[str] = None,
        add_record_item: Optional[Dict[str, Any]] = None,
        new_solution=None,
    ) -> bool:
        """Apply operator to current_solution and advance one construction step."""
        from ..base.operators import BaseOperator

        _ = (inplace, meta, stage, time_ms, heuristic_name, add_record_item, new_solution)

        if not isinstance(operator, BaseOperator):
            return False
        self.current_solution = operator.run(self.current_solution)
        self.problem_state = self.get_problem_state()
        if getattr(self, "construction_steps", None) is None:
            self.construction_steps = 0
        self.construction_steps += 1
        self.current_steps += 1
        record = {
            "iter": int(self.current_steps),
            "stage": stage,
            "op": getattr(operator, "name", operator.__class__.__name__),
            "heuristic_name": heuristic_name,
            "time_ms": float(time_ms) if time_ms is not None else None,
            "meta": meta or {},
        }
        if add_record_item:
            record.update(add_record_item)
        if not record.get("_skip_record"):
            try:
                record["key_value"] = float(self.get_key_value(self.current_solution))
            except Exception:
                record["key_value"] = None
            record["timestamp"] = float(time.time())
            self._write_record(record)
        return True

    def run_heuristic(self, heuristic, algorithm_data: Optional[Dict[str, Any]] = None, record: bool = True, **kwargs) -> bool:
        problem_state = {"instance_data": getattr(self, "instance_data", {}), "current_solution": self.solution}
        if hasattr(self, "get_problem_state"):
            try:
                problem_state = dict(self.get_problem_state())
            except Exception:  # noqa: BLE001
                problem_state = {"instance_data": getattr(self, "instance_data", {}), "current_solution": self.solution}
        if algorithm_data is None:
            algorithm_data = {"env": self, "rng": getattr(self, "rng", None)}
        operator, meta = heuristic(problem_state, algorithm_data=algorithm_data, **kwargs)
        if operator is None:
            return False
        if hasattr(self, "run_operator"):
            try:
                add_record_item = None if record else {"_skip_record": True}
                self.run_operator(
                    operator,
                    heuristic_name=getattr(heuristic, "__name__", "heuristic"),
                    meta=meta,
                    add_record_item=add_record_item,
                )
            except TypeError:
                self.run_operator(operator)
            return True
        return False

    def dump_result(self) -> None:
        return None

    def validate_solution(self, solution=None) -> bool:
        """
        Default validator.
        Most built-in problems implement `validation_solution(solution=None)`.
        wafer_layout implements `validate_solution(solution)`.
        """
        vs = getattr(self, "validation_solution", None)
        if callable(vs):
            try:
                return bool(vs(solution))
            except TypeError:
                return bool(vs())
        return True

    @property
    def is_valid_solution(self) -> bool:
        """
        Unified validity flag (bool property).
        It will call `validate_solution(...)` (possibly overridden), and fall back to True.
        """
        try:
            return bool(self.validate_solution(getattr(self, "current_solution", None)))
        except TypeError:
            return bool(self.validate_solution())
        except Exception:
            return True

    @property
    def continue_run(self) -> bool:
        ms = getattr(self, "max_steps", None)
        if ms is not None:
            try:
                if int(self.current_steps) >= int(ms):
                    return False
            except Exception:
                pass
        comp = getattr(self, "is_complete_solution", None)
        if comp is not None:
            try:
                done = comp() if callable(comp) else bool(comp)
                if done:
                    return False
            except Exception:
                pass
        return True
