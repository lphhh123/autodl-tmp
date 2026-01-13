"""Single heuristic runner."""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple


class SingleHyperHeuristic:
    def __init__(
        self,
        heuristic: Callable[..., Tuple[Any, Dict]],
        iterations_scale_factor: float = 1.0,
        output_dir: str | None = None,
        seed: int = 0,
    ) -> None:
        self.heuristic = heuristic
        self.iterations_scale_factor = float(iterations_scale_factor)
        self.output_dir = output_dir
        self.seed = int(seed)

    def run(self, env: Any) -> bool:
        import random as _random

        algorithm_data = {
            "rng": _random.Random(int(getattr(self, "seed", 0))),
            "env": env,
        }

        while env.continue_run:
            env.run_heuristic(
                self.heuristic,
                algorithm_data=algorithm_data,
                add_record_item={"selection": "single"},
            )

        env.dump_result()
        return bool(getattr(env, "is_complete_solution", True)) and bool(getattr(env, "is_valid_solution", True))
