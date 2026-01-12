from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from src.problems.base.components import BaseOperator, BaseSolution


@dataclass
class WaferLayoutSolution(BaseSolution):
    assign: np.ndarray
    S: int
    Ns: int

    def copy(self):
        return WaferLayoutSolution(assign=self.assign.copy(), S=int(self.S), Ns=int(self.Ns))

    def to_list(self) -> List[int]:
        return [int(x) for x in self.assign.tolist()]

    def __str__(self) -> str:
        s_size = int(self.S)
        head = self.to_list()[: min(16, s_size)]
        return f"WaferLayoutSolution(S={s_size}, head_assign={head}...)"


class SwapOperator(BaseOperator):
    op_name = "swap"

    def __init__(self, i: int, j: int):
        self.i = int(i)
        self.j = int(j)

    def run(self, instance_data: dict, solution: WaferLayoutSolution) -> WaferLayoutSolution:
        sol = solution.copy()
        if self.i == self.j:
            return sol
        sol.assign[self.i], sol.assign[self.j] = sol.assign[self.j], sol.assign[self.i]
        return sol

    def to_json(self) -> Dict[str, Any]:
        return {"op": "swap", "i": self.i, "j": self.j}


class RelocateOperator(BaseOperator):
    op_name = "relocate"

    def __init__(self, i: int, site_id: int):
        self.i = int(i)
        self.site_id = int(site_id)

    def run(self, instance_data: dict, solution: WaferLayoutSolution) -> WaferLayoutSolution:
        sol = solution.copy()
        target = self.site_id
        pos = np.where(sol.assign == target)[0]
        if pos.size > 0:
            j = int(pos[0])
            sol.assign[self.i], sol.assign[j] = sol.assign[j], sol.assign[self.i]
        else:
            sol.assign[self.i] = target
        return sol

    def to_json(self) -> Dict[str, Any]:
        return {"op": "relocate", "i": self.i, "site_id": self.site_id}


class ClusterMoveOperator(BaseOperator):
    op_name = "cluster_move"

    def __init__(self, i_list: List[int], site_list: List[int]):
        self.i_list = [int(x) for x in i_list]
        self.site_list = [int(x) for x in site_list]

    def run(self, instance_data: dict, solution: WaferLayoutSolution) -> WaferLayoutSolution:
        sol = solution.copy()
        for i, site in zip(self.i_list, self.site_list):
            sol.assign[int(i)] = int(site)
        return sol

    def to_json(self) -> Dict[str, Any]:
        return {"op": "cluster_move", "i_list": self.i_list, "site_list": self.site_list}


class RandomKickOperator(BaseOperator):
    op_name = "random_kick"

    def __init__(self, moves: List[Dict[str, Any]]):
        self.moves = moves

    def run(self, instance_data: dict, solution: WaferLayoutSolution) -> WaferLayoutSolution:
        sol = solution.copy()
        for mv in self.moves:
            t = mv.get("type")
            if t == "swap":
                i = int(mv["i"])
                j = int(mv["j"])
                sol.assign[i], sol.assign[j] = sol.assign[j], sol.assign[i]
            elif t == "relocate":
                i = int(mv["i"])
                site = int(mv["site_id"])
                pos = np.where(sol.assign == site)[0]
                if pos.size > 0:
                    j = int(pos[0])
                    sol.assign[i], sol.assign[j] = sol.assign[j], sol.assign[i]
                else:
                    sol.assign[i] = site
        return sol

    def to_json(self) -> Dict[str, Any]:
        return {"op": "random_kick", "moves": self.moves}


class NoOpOperator(BaseOperator):
    op_name = "noop"

    def run(self, instance_data: dict, solution: WaferLayoutSolution) -> WaferLayoutSolution:
        return solution.copy()

    def to_json(self) -> Dict[str, Any]:
        return {"op": "noop"}
