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
        stage_name: str = "heuragenix_random_hh",
    ) -> None:
        self.env = env
        self.heuristics = heuristics
        self.rng = rng
        self.selection_frequency = max(1, int(selection_frequency))
        self.sa_T0 = float(sa_T0)
        self.sa_alpha = float(sa_alpha)
        self.stage_name = stage_name
        self.usage_records: List[Dict[str, Any]] = []

    def run(self, max_steps: int) -> None:
        temperature = self.sa_T0
        current_solution = self.env.solution
        current_score = self.env.get_key_value(current_solution)
        selection_id = 0
        selected = None
        for step in range(max_steps):
            step_start = time.perf_counter()
            if step % self.selection_frequency == 0 or selected is None:
                selected = self.rng.choice(self.heuristics)
                self.usage_records.append(
                    {
                        "ok": True,
                        "reason": "random_pick",
                        "chosen_heuristic": getattr(selected, "__name__", "unknown"),
                        "candidates": [getattr(h, "__name__", "unknown") for h in self.heuristics],
                        "selection_id": selection_id,
                        "step": int(step),
                    }
                )
                selection_id += 1
            operator, meta = selected({"solution": current_solution}, {"env": self.env, "rng": self.rng})
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
                    "stage": self.stage_name,
                    "operator": operator,
                    "meta": meta,
                    "accepted": accept,
                    "score": current_score,
                    "time_ms": int((time.perf_counter() - step_start) * 1000),
                    "assign": list(current_solution.assign),
                }
            )
            temperature *= self.sa_alpha
