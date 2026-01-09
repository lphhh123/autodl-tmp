"""Random hyper-heuristic baseline."""
from __future__ import annotations

import importlib.util
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from util.get_heuristic import get_heuristic


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
        iterations_scale_factor: float = 1.0,
        selection_frequency: int = 1,
        rng: random.Random | None = None,
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

        self.heuristic_pool = heuristic_pool
        self.problem = problem
        self.iterations_scale_factor = float(iterations_scale_factor)
        self.selection_frequency = max(1, int(selection_frequency))
        self.rng = rng or random.Random(0)
        self.sa_T0 = float(sa_T0)
        self.sa_alpha = float(sa_alpha)
        self.stage_name = stage_name
        self.usage_records: List[Dict[str, Any]] = []
        self._resolved_heuristics: List[Callable[..., Tuple[Any, Dict]]] | None = None

    def _resolve_heuristics(self) -> List[Callable[..., Tuple[Any, Dict]]]:
        if self._resolved_heuristics is None:
            self._resolved_heuristics = _resolve_heuristics(self.heuristic_pool, self.problem)
        return self._resolved_heuristics

    def _get_algorithm_data(self, env) -> Dict[str, Any]:
        data = dict(getattr(env, "algorithm_data", {}) or {})
        data.setdefault("env", env)
        data.setdefault("rng", getattr(env, "rng", None) or self.rng)
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

    def run(self, env: Any, max_steps: int | None = None) -> None:
        if self.legacy_mode or isinstance(env, int):
            return self._run_legacy(int(env))
        if env is None:
            raise ValueError("env is required")
        rng = getattr(env, "rng", None) or self.rng
        temperature = self.sa_T0
        current_solution = env.solution
        current_score = env.get_key_value(current_solution)
        selection_id = 0
        selected = None
        heuristics = self._resolve_heuristics()

        steps = max_steps
        if steps is None:
            steps = max(1, int(self.iterations_scale_factor * max(1, getattr(env, "construction_steps", getattr(env, "problem_size", 1)))))
        if hasattr(env, "max_steps"):
            env.max_steps = int(steps)

        step = 0
        while True:
            if hasattr(env, "continue_run"):
                if not env.continue_run:
                    break
            elif step >= int(steps):
                break
            step_start = time.perf_counter()
            step_idx = int(getattr(env, "step_count", getattr(env, "step", step)))
            if step % self.selection_frequency == 0 or selected is None:
                selected = rng.choice(heuristics)
                self.usage_records.append(
                    {
                        "ok": True,
                        "reason": "random_pick",
                        "chosen_heuristic": getattr(selected, "__name__", "unknown"),
                        "candidates": [getattr(h, "__name__", "unknown") for h in heuristics],
                        "selection_id": selection_id,
                        "step": int(step_idx),
                    }
                )
                selection_id += 1
            algorithm_data = self._get_algorithm_data(env)
            problem_state = {"instance_data": getattr(env, "instance_data", {}), "current_solution": current_solution}
            if hasattr(env, "get_problem_state"):
                try:
                    problem_state = dict(env.get_problem_state())
                except Exception:  # noqa: BLE001
                    problem_state = {"instance_data": getattr(env, "instance_data", {}), "current_solution": current_solution}
            problem_state["current_solution"] = current_solution
            operator, meta = selected(problem_state, algorithm_data)
            new_solution = operator.run(current_solution)
            new_score = env.get_key_value(new_solution)
            delta = new_score - current_score
            if hasattr(env, "validation_solution") and not env.validation_solution(new_solution):
                accept = False
            else:
                accept = (delta < 0) or (rng.random() < pow(2.718281828, -delta / max(temperature, 1e-6)))
            if accept:
                current_solution = new_solution
                current_score = new_score
                env.solution = current_solution
            self._record_step(
                env,
                operator,
                meta,
                accept,
                int((time.perf_counter() - step_start) * 1000),
                new_solution,
            )
            temperature *= self.sa_alpha
            step += 1
