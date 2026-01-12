"""Base components for HeurAgenix problems."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseSolution:
    def copy(self):
        return type(self)(**self.__dict__)


class BaseOperator:
    name = "base"

    def run(self, solution: BaseSolution) -> BaseSolution:  # pragma: no cover - interface stub
        raise NotImplementedError
