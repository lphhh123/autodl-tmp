"""Minimal HeurAgenix core abstractions used by wafer_layout baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class BaseSolution:
    def copy(self) -> "BaseSolution":
        return type(self)(**self.__dict__)


class BaseOperator:
    def run(self, solution: BaseSolution) -> BaseSolution:  # pragma: no cover - interface stub
        raise NotImplementedError


class BaseEnv:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.instance_data = self.load_data(data_path)
        self.solution = self.init_solution()
        self.recordings: List[Dict[str, Any]] = []

    def load_data(self, path: str) -> Dict[str, Any]:
        raise NotImplementedError

    def init_solution(self) -> BaseSolution:
        raise NotImplementedError

    def get_key_value(self, solution: BaseSolution) -> float:
        raise NotImplementedError

    def is_complete_solution(self, solution: BaseSolution) -> bool:
        return True
