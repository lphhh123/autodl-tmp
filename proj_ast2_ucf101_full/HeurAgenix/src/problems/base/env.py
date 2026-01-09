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

    def load_data(self, path: str) -> Dict[str, Any]:
        raise NotImplementedError

    def init_solution(self):
        raise NotImplementedError

    def get_key_value(self, solution) -> float:
        raise NotImplementedError

    def get_problem_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    def reset(self, output_dir: Optional[str] = None) -> None:
        self.current_solution = self.init_solution()
        self.solution = self.current_solution
        self.recordings = []
        self.step = 0
        self.algorithm_data = {}
        self.output_dir = output_dir

    def run_operator(
        self,
        operator,
        inplace: bool = True,
        meta: Dict[str, Any] | None = None,
        new_solution: Any | None = None,
        stage: str | None = None,
        time_ms: int | None = None,
    ) -> None:
        if inplace:
            if new_solution is None:
                new_solution = operator.run(self.current_solution)
            self.current_solution = new_solution
            self.solution = self.current_solution
        record = {
            "step": int(self.step),
            "operator": operator,
            "meta": meta or {},
            "accepted": bool(inplace),
        }
        if stage is not None:
            record["stage"] = stage
        if time_ms is not None:
            record["time_ms"] = int(time_ms)
        self.recordings.append(record)
        self.step += 1

    def dump_result(self) -> None:
        return None
