"""Wafer layout solution and operators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from core import BaseOperator, BaseSolution


@dataclass
class WaferLayoutSolution(BaseSolution):
    assign: List[int]
    S: int
    Ns: int


class WaferLayoutOperator(BaseOperator):
    name = "base"

    def signature(self) -> str:
        return self.name


class SwapSlots(WaferLayoutOperator):
    name = "swap"

    def __init__(self, i: int, j: int) -> None:
        self.i = int(i)
        self.j = int(j)

    def run(self, solution: WaferLayoutSolution) -> WaferLayoutSolution:
        assign = list(solution.assign)
        assign[self.i], assign[self.j] = assign[self.j], assign[self.i]
        return WaferLayoutSolution(assign=assign, S=solution.S, Ns=solution.Ns)

    def signature(self) -> str:
        return f"swap:{self.i}<->{self.j}"


class RelocateSlot(WaferLayoutOperator):
    name = "relocate"

    def __init__(self, i: int, site_id: int) -> None:
        self.i = int(i)
        self.site_id = int(site_id)

    def run(self, solution: WaferLayoutSolution) -> WaferLayoutSolution:
        assign = list(solution.assign)
        if self.site_id in assign:
            j = assign.index(self.site_id)
            assign[self.i], assign[j] = assign[j], assign[self.i]
        else:
            assign[self.i] = self.site_id
        return WaferLayoutSolution(assign=assign, S=solution.S, Ns=solution.Ns)

    def signature(self) -> str:
        return f"relocate:{self.i}->{self.site_id}"


class ClusterMove(WaferLayoutOperator):
    name = "cluster_move"

    def __init__(self, slots: List[int], target_sites: List[int]) -> None:
        self.slots = [int(s) for s in slots]
        self.target_sites = [int(s) for s in target_sites]

    def run(self, solution: WaferLayoutSolution) -> WaferLayoutSolution:
        assign = list(solution.assign)
        for slot, site in zip(self.slots, self.target_sites):
            if site in assign:
                j = assign.index(site)
                assign[slot], assign[j] = assign[j], assign[slot]
            else:
                assign[slot] = site
        return WaferLayoutSolution(assign=assign, S=solution.S, Ns=solution.Ns)

    def signature(self) -> str:
        return f"cluster_move:{len(self.slots)}"


class RandomKick(WaferLayoutOperator):
    name = "random_kick"

    def __init__(self, ops: List[WaferLayoutOperator]) -> None:
        self.ops = ops

    def run(self, solution: WaferLayoutSolution) -> WaferLayoutSolution:
        out = solution
        for op in self.ops:
            out = op.run(out)
        return out

    def signature(self) -> str:
        return f"kick:k={len(self.ops)}"


class NoOp(WaferLayoutOperator):
    name = "noop"

    def run(self, solution: WaferLayoutSolution) -> WaferLayoutSolution:
        return WaferLayoutSolution(assign=list(solution.assign), S=solution.S, Ns=solution.Ns)

    def signature(self) -> str:
        return "noop"
