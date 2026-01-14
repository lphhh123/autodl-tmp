"""Base environment for HeurAgenix problems."""
from __future__ import annotations

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
                self.run_operator(operator, heuristic_name=getattr(heuristic, "__name__", "heuristic"), meta=meta)
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
