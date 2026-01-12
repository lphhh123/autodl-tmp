from dataclasses import dataclass
from typing import Any, Dict, List

from src.problems.base.components import BaseSolution, BaseOperator


@dataclass
class WaferLayoutSolution(BaseSolution):
    """A full assignment solution: slot i -> site_id"""
    assign: List[int]


class SwapOperator(BaseOperator):
    def __init__(self, i: int, j: int):
        self.i = int(i)
        self.j = int(j)

    def run(self, solution: WaferLayoutSolution) -> WaferLayoutSolution:
        a = solution.assign
        a[self.i], a[self.j] = a[self.j], a[self.i]
        return solution

    def to_action(self) -> Dict[str, Any]:
        return {"op": "swap", "i": self.i, "j": self.j}


class RelocateOperator(BaseOperator):
    def __init__(self, i: int, site_id: int, from_site: int | None = None):
        self.i = int(i)
        self.site_id = int(site_id)
        self.from_site = None if from_site is None else int(from_site)
        self.degraded_swap_with: int | None = None

    def run(self, solution: WaferLayoutSolution) -> WaferLayoutSolution:
        a = solution.assign
        self.from_site = int(a[self.i])
        self.degraded_swap_with = None
        try:
            j = a.index(self.site_id)
        except ValueError:
            j = None
        if j is not None and j != self.i:
            a[self.i], a[j] = a[j], a[self.i]
            self.degraded_swap_with = int(j)
        else:
            a[self.i] = self.site_id
        return solution

    def to_action(self) -> Dict[str, Any]:
        return {
            "op": "relocate",
            "i": self.i,
            "site_id": self.site_id,
            "from_site": self.from_site,
            "degraded_swap_with": self.degraded_swap_with,
        }


class RandomKickOperator(BaseOperator):
    def __init__(self, idxs: List[int], site_ids: List[int]):
        self.idxs = [int(x) for x in idxs]
        self.site_ids = [int(x) for x in site_ids]

    def run(self, solution: WaferLayoutSolution) -> WaferLayoutSolution:
        a = solution.assign
        for i, s in zip(self.idxs, self.site_ids):
            a[i] = s
        return solution

    def to_action(self) -> Dict[str, Any]:
        return {"op": "random_kick", "idxs": self.idxs, "site_ids": self.site_ids}


class NoopOperator(BaseOperator):
    def __init__(self):
        pass

    def run(self, solution: WaferLayoutSolution) -> WaferLayoutSolution:
        return solution

    def to_action(self) -> Dict[str, Any]:
        return {"op": "noop"}
