"""Random hyper-heuristic baseline."""
from __future__ import annotations

import random
import time
from typing import Any, Callable, Dict, List, Tuple

from core import BaseEnv


class RandomHyperHeuristic:
    def __init__(
        self,
        env: BaseEnv,
        heuristics: List[Callable[..., Tuple[Any, Dict]]],
        rng: random.Random,
        selection_frequency: int = 1,
        sa_T0: float = 1.0,
        sa_alpha: float = 0.995,
    ) -> None:
        self.env = env
        self.heuristics = heuristics
        self.rng = rng
        self.selection_frequency = max(1, int(selection_frequency))
        self.sa_T0 = float(sa_T0)
        self.sa_alpha = float(sa_alpha)

    def run(self, max_steps: int) -> None:
        temperature = self.sa_T0
        current_solution = self.env.solution
        current_score = self.env.get_key_value(current_solution)
        for step in range(max_steps):
            step_start = time.perf_counter()
            heuristic = self.rng.choice(self.heuristics)
            operator, meta = heuristic({"solution": current_solution}, {"env": self.env, "rng": self.rng})
            new_solution = operator.run(current_solution)
            new_score = self.env.get_key_value(new_solution)
            delta = new_score - current_score
            accept = (delta < 0) or (self.rng.random() < pow(2.718281828, -delta / max(temperature, 1e-6)))
            if accept:
                current_solution = new_solution
                current_score = new_score
                self.env.solution = current_solution
            self.env.recordings.append(
                {
                    "step": step,
                    "operator": operator,
                    "meta": meta,
                    "accepted": accept,
                    "score": current_score,
                    "time_ms": int((time.perf_counter() - step_start) * 1000),
                }
            )
            temperature *= self.sa_alpha
