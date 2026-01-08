"""Base environment for HeurAgenix problems."""
from __future__ import annotations

from typing import Any, Dict, List


class BaseEnv:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.instance_data = self.load_data(data_path)
        self.current_solution = self.init_solution()
        self.solution = self.current_solution
        self.recordings: List[Dict[str, Any]] = []
        self.step = 0

    def load_data(self, path: str) -> Dict[str, Any]:
        raise NotImplementedError

    def init_solution(self):
        raise NotImplementedError

    def get_key_value(self, solution) -> float:
        raise NotImplementedError

    def get_problem_state(self) -> Dict[str, Any]:
        raise NotImplementedError

    def run_operator(self, operator, accepted: bool, meta: Dict[str, Any] | None = None) -> None:
        self.current_solution = operator.run(self.current_solution)
        self.solution = self.current_solution
        self.recordings.append(
            {
                "step": int(self.step),
                "operator": operator,
                "meta": meta or {},
                "accepted": accepted,
            }
        )
        self.step += 1

    def dump_result(self, content: Dict[str, Any], output_folder: str) -> None:
        return None
