"""Random hyper-heuristic baseline."""
from __future__ import annotations

import importlib.util
import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from src.util.get_heuristic import get_heuristic
from src.util.util import load_function


def _is_env_like(value: Any) -> bool:
    return hasattr(value, "get_key_value") and hasattr(value, "solution")


def _load_heuristic_from_path(path: Path) -> Callable[..., Tuple[Any, Dict]]:
    name = path.stem
    spec = importlib.util.spec_from_file_location(f"heuristics.{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load heuristic module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    if not hasattr(module, name):
        raise AttributeError(f"Missing heuristic function '{name}' in {path}")
    return getattr(module, name)


def _resolve_heuristics(heuristic_pool: Any, problem: str | None) -> List[Callable[..., Tuple[Any, Dict]]]:
    if heuristic_pool is None:
        raise ValueError("heuristic_pool is required")
    if isinstance(heuristic_pool, dict):
        return list(heuristic_pool.values())
    if isinstance(heuristic_pool, (list, tuple)):
        if heuristic_pool and callable(heuristic_pool[0]):
            return list(heuristic_pool)
        resolved: List[Callable[..., Tuple[Any, Dict]]] = []
        for item in heuristic_pool:
            if callable(item):
                resolved.append(item)
                continue
            path = Path(str(item))
            if not path.exists() and problem:
                heuristics_root = Path(__file__).resolve().parents[2] / "problems" / problem / "heuristics"
                candidates = list(heuristics_root.rglob(path.name)) + list(heuristics_root.rglob(f"{path.name}.py"))
                if candidates:
                    path = candidates[0]
            if not path.exists():
                raise FileNotFoundError(f"Heuristic file not found: {item}")
            resolved.append(_load_heuristic_from_path(path))
        return resolved
    return list(get_heuristic(str(heuristic_pool), problem or "").values())


class RandomHyperHeuristic:
    def __init__(
        self,
        *args,
        heuristic_pool: Any | None = None,
        problem: str | None = None,
        configs: Dict[str, Any] | None = None,
        heuristic_functions: Dict[str, Callable[..., Tuple[Any, Dict]]] | None = None,
        iterations_scale_factor: float = 1.0,
        heuristic_dir: str | None = None,
        selection_frequency: int = 1,
        rng: random.Random | None = None,
        seed: int = 0,
        sa_T0: float = 1.0,
        sa_alpha: float = 0.995,
        stage_name: str = "heuragenix_random_hh",
        **kwargs,
    ) -> None:
        self.legacy_mode = False
        if args and _is_env_like(args[0]):
            self.legacy_mode = True
            self.env = args[0]
            self.heuristics = args[1]
            self.rng = args[2]
            self.configs = dict(configs or {})
            self.heuristic_functions = heuristic_functions
            self.selection_frequency = max(1, int(selection_frequency))
            self.sa_T0 = float(sa_T0)
            self.sa_alpha = float(sa_alpha)
            self.stage_name = stage_name
            self.usage_records: List[Dict[str, Any]] = []
            return

        if args:
            heuristic_pool = heuristic_pool or args[0]
        if len(args) > 1 and problem is None:
            problem = args[1]

        self.configs = dict(configs or {})
        self.heuristic_functions = heuristic_functions
        self.heuristic_pool = heuristic_pool
        self.problem = problem
        self.iterations_scale_factor = float(iterations_scale_factor)
        self.heuristic_dir = heuristic_dir
        self.selection_frequency = max(1, int(selection_frequency))
        self.rng = rng or random.Random(0)
        self.seed = int(seed)
        self.sa_T0 = float(sa_T0)
        self.sa_alpha = float(sa_alpha)
        self.stage_name = stage_name
        self.usage_records: List[Dict[str, Any]] = []
        self._resolved_heuristics: List[Callable[..., Tuple[Any, Dict]]] | None = None
        self._resolved_heuristic_items: List[Tuple[str, Callable[..., Tuple[Any, Dict]]]] | None = None

    def _resolve_heuristics(self) -> List[Callable[..., Tuple[Any, Dict]]]:
        if self._resolved_heuristics is None:
            self._resolved_heuristics = _resolve_heuristics(self.heuristic_pool, self.problem)
        return self._resolved_heuristics

    def _resolve_heuristic_items(self) -> List[Tuple[str, Callable[..., Tuple[Any, Dict]]]]:
        if self._resolved_heuristic_items is not None:
            return self._resolved_heuristic_items
        if self.heuristic_functions:
            self._resolved_heuristic_items = list(self.heuristic_functions.items())
            return self._resolved_heuristic_items
        if isinstance(self.heuristic_pool, dict):
            self._resolved_heuristic_items = list(self.heuristic_pool.items())
            return self._resolved_heuristic_items
        heuristics = self._resolve_heuristics()
        self._resolved_heuristic_items = [(getattr(h, "__name__", "unknown"), h) for h in heuristics]
        return self._resolved_heuristic_items

    def _get_algorithm_data(self, env) -> Dict[str, Any]:
        data = dict(self.configs)
        data.update(dict(getattr(env, "algorithm_data", {}) or {}))
        data["env"] = env
        data["rng"] = getattr(env, "rng", None) or self.rng
        return data

    def _record_step(
        self,
        env,
        operator,
        meta: Dict[str, Any],
        accepted: bool,
        time_ms: int,
        new_solution,
    ) -> None:
        if hasattr(env, "run_operator"):
            env.run_operator(
                operator,
                inplace=accepted,
                meta=meta,
                stage=self.stage_name,
                time_ms=time_ms,
                new_solution=new_solution if accepted else None,
            )
            return
        env.recordings.append(
            {
                "step": getattr(env, "step", 0),
                "stage": self.stage_name,
                "operator": operator,
                "meta": meta,
                "accepted": accepted,
                "score": env.get_key_value(env.solution),
                "time_ms": int(time_ms),
                "assign": list(getattr(env.solution, "assign", [])),
            }
        )
        if hasattr(env, "step"):
            env.step += 1

    def _run_legacy(self, max_steps: int) -> None:
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
            problem_state = {"instance_data": getattr(self.env, "instance_data", {}), "current_solution": current_solution}
            if hasattr(self.env, "get_problem_state"):
                try:
                    problem_state = dict(self.env.get_problem_state())
                except Exception:  # noqa: BLE001
                    problem_state = {
                        "instance_data": getattr(self.env, "instance_data", {}),
                        "current_solution": current_solution,
                    }
            problem_state["current_solution"] = current_solution
            operator, meta = selected(problem_state, {"env": self.env, "rng": self.rng})
            new_solution = operator.run(current_solution)
            new_score = self.env.get_key_value(new_solution)
            delta = new_score - current_score
            if hasattr(self.env, "validation_solution") and not self.env.validation_solution(new_solution):
                accept = False
            else:
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

    def run(self, env: Any, max_steps: int | None = None) -> bool:
        import random as _random

        # legacy path (keep)
        if self.legacy_mode or isinstance(env, int):
            return self._run_legacy(int(env))
        if env is None:
            raise ValueError("env is required")

        # Make sure we have a pool of heuristic NAMES (strings)
        if not self.heuristic_pool:
            if self.problem and self.heuristic_dir:
                from src.util.util import get_heuristic_names

                self.heuristic_pool = get_heuristic_names(self.problem, self.heuristic_dir)
            if not self.heuristic_pool:
                return False

        # local rng for reproducibility
        rng = _random.Random(int(getattr(self, "seed", 0)))

        chosen_name = None
        while getattr(env, "continue_run", True):
            if chosen_name is None:
                chosen_name = rng.choice(self.heuristic_pool)

            heuristic = load_function(chosen_name, problem=self.problem)
            env.run_heuristic(
                heuristic,
                algorithm_data={"rng": rng, "env": env},
                add_record_item={"selection": "random_hh"},
            )

            if int(getattr(env, "current_steps", 0)) % max(1, int(getattr(self, "selection_frequency", 1))) == 0:
                chosen_name = rng.choice(self.heuristic_pool)

        if hasattr(env, "validate_solution"):
            return bool(env.validate_solution(getattr(env, "current_solution", None)))
        return True
