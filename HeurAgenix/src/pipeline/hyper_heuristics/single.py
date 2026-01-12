"""Single heuristic runner."""
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple


class SingleHyperHeuristic:
    def __init__(
        self,
        heuristic: Callable[..., Tuple[Any, Dict]],
        iterations_scale_factor: float = 1.0,
        output_dir: str | None = None,
    ) -> None:
        self.heuristic = heuristic
        self.iterations_scale_factor = float(iterations_scale_factor)
        self.output_dir = output_dir

    def run(self, env: Any) -> None:
        while getattr(env, "continue_run", True):
            problem_state = {"instance_data": getattr(env, "instance_data", {}), "current_solution": env.solution}
            if hasattr(env, "get_problem_state"):
                try:
                    problem_state = dict(env.get_problem_state())
                except Exception:  # noqa: BLE001
                    problem_state = {"instance_data": getattr(env, "instance_data", {}), "current_solution": env.solution}
            algorithm_data = {"env": env, "rng": getattr(env, "rng", None)}
            operator, meta = self.heuristic(problem_state, algorithm_data=algorithm_data)
            if operator is None:
                break
            if hasattr(env, "run_operator"):
                try:
                    env.run_operator(operator, heuristic_name=getattr(self.heuristic, "__name__", "heuristic"), meta=meta)
                except TypeError:
                    env.run_operator(operator)
            else:
                break
