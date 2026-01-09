"""LLM-selection hyper-heuristic baseline."""
from __future__ import annotations

import importlib.util
import json
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


class LLMSelectionHyperHeuristic:
    def __init__(
        self,
        *args,
        llm_client: Any | None = None,
        heuristic_pool: Any | None = None,
        problem: str | None = None,
        iterations_scale_factor: float = 1.0,
        selection_frequency: int = 5,
        num_candidate_heuristics: int = 4,
        rollout_budget: int = 0,
        rng: random.Random | None = None,
        sa_T0: float = 1.0,
        sa_alpha: float = 0.995,
        timeout_sec: int = 90,
        max_retry: int = 1,
        llm_timeout_s: int = 30,
        max_llm_failures: int = 2,
        fallback_mode: str = "random",
        stage_name: str = "heuragenix_llm_hh",
        usage_path: str | None = None,
        llm_error: str | None = None,
        **kwargs,
    ) -> None:
        self.legacy_mode = False
        if args and _is_env_like(args[0]):
            self.legacy_mode = True
            self.env = args[0]
            self.heuristics = args[1]
            self.rng = args[2]
            self.selection_frequency = max(1, int(selection_frequency))
            self.num_candidate_heuristics = max(1, int(num_candidate_heuristics))
            self.sa_T0 = float(sa_T0)
            self.sa_alpha = float(sa_alpha)
            self.stage_name = stage_name
            self.llm = llm_client
            self.llm_timeout_s = int(llm_timeout_s)
            self.max_llm_failures = int(max_llm_failures)
            self.fallback_mode = fallback_mode
            self.fail_count = 0
            self.llm_disabled = False
            self.llm_error = llm_error
            self.usage_records: List[Dict[str, Any]] = []
            self.selection_records: List[Dict[str, Any]] = []
            self.usage_path = Path(usage_path) if usage_path else None
            return

        if args:
            llm_client = llm_client or args[0]
        if len(args) > 1 and heuristic_pool is None:
            heuristic_pool = args[1]
        if len(args) > 2 and problem is None:
            problem = args[2]

        self.llm = llm_client
        self.heuristic_pool = heuristic_pool
        self.problem = problem
        self.iterations_scale_factor = float(iterations_scale_factor)
        self.selection_frequency = max(1, int(selection_frequency))
        self.num_candidate_heuristics = max(1, int(num_candidate_heuristics))
        self.rollout_budget = int(rollout_budget)
        self.rng = rng or random.Random(0)
        self.sa_T0 = float(sa_T0)
        self.sa_alpha = float(sa_alpha)
        self.stage_name = stage_name
        self.timeout_sec = timeout_sec
        self.max_retry = max_retry
        self.llm_timeout_s = int(llm_timeout_s)
        self.max_llm_failures = int(max_llm_failures)
        self.fallback_mode = fallback_mode
        self.fail_count = 0
        self.llm_disabled = False
        self.llm_error = llm_error
        self.usage_records: List[Dict[str, Any]] = []
        self.selection_records: List[Dict[str, Any]] = []
        self.usage_path = Path(usage_path) if usage_path else None
        self._resolved_heuristics: List[Callable[..., Tuple[Any, Dict]]] | None = None

    def _append_usage(self, payload: Dict[str, Any]) -> None:
        if not self.usage_path:
            return
        self.usage_path.parent.mkdir(parents=True, exist_ok=True)
        with self.usage_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _llm_ready(self) -> bool:
        if self.llm is None:
            return False
        if hasattr(self.llm, "is_ready"):
            return bool(self.llm.is_ready())
        return bool(getattr(self.llm, "base_url", None) and getattr(self.llm, "model", None))

    def _resolve_heuristics(self) -> List[Callable[..., Tuple[Any, Dict]]]:
        if self._resolved_heuristics is None:
            self._resolved_heuristics = _resolve_heuristics(self.heuristic_pool, self.problem)
        return self._resolved_heuristics

    def _choose_candidate(
        self,
        state_summary: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        step_idx: int,
        selection_id: int,
        rng: random.Random,
    ) -> int:
        ok = True
        err = None
        chosen_idx: int

        if self.llm_disabled or not self._llm_ready():
            ok = False
            err = "llm_disabled" if self.llm_disabled else (self.llm_error or "llm_not_ready")
            self.fail_count += 1
            chosen_idx = int(rng.randint(0, len(candidates) - 1))
            if self.fail_count >= self.max_llm_failures:
                self.llm_disabled = True
        else:
            try:
                picks = self.llm.propose_pick(state_summary, 1, timeout_s=self.llm_timeout_s)
                if not picks:
                    raise RuntimeError("llm_empty_pick")
                chosen_idx = int(picks[0])
            except Exception as exc:  # noqa: BLE001
                ok = False
                err = repr(exc)
                self.fail_count += 1
                chosen_idx = int(rng.randint(0, len(candidates) - 1))
                if self.fail_count >= self.max_llm_failures:
                    self.llm_disabled = True

        chosen_idx = max(0, min(chosen_idx, len(candidates) - 1))
        usage = {
            "step": int(step_idx),
            "ok": bool(ok),
            "engine": "llm_hh",
            "candidate_num": int(len(candidates)),
            "chosen_idx": int(chosen_idx),
            "chosen_name": candidates[chosen_idx].get("name") if isinstance(candidates[chosen_idx], dict) else str(candidates[chosen_idx]),
            "error": err,
            "fail_count": int(self.fail_count),
            "llm_disabled": bool(self.llm_disabled),
            "selection_id": int(selection_id),
        }
        if hasattr(self.llm, "calls") and self.llm.calls:
            usage.update(self.llm.calls[-1])
        self.usage_records.append(usage)
        self._append_usage(usage)
        return chosen_idx

    def _get_algorithm_data(self, env) -> Dict[str, Any]:
        data = dict(getattr(env, "algorithm_data", {}) or {})
        data.setdefault("env", env)
        data.setdefault("rng", getattr(env, "rng", None) or self.rng)
        return data

    def _score_candidates(self, solution, env, rng: random.Random) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        current_score = env.get_key_value(solution)
        if self.legacy_mode:
            heuristics = list(self.heuristics)
        else:
            heuristics = list(self._resolve_heuristics())
        if len(heuristics) > self.num_candidate_heuristics:
            heuristics = rng.sample(heuristics, self.num_candidate_heuristics)
        algorithm_data = self._get_algorithm_data(env)
        for idx, heuristic in enumerate(heuristics):
            operator, meta = heuristic({"solution": solution}, algorithm_data)
            new_solution = operator.run(solution)
            new_score = env.get_key_value(new_solution)
            candidates.append(
                {
                    "id": idx,
                    "name": getattr(operator, "name", operator.__class__.__name__),
                    "d_total": float(new_score - current_score),
                    "operator": operator,
                    "meta": meta,
                }
            )
        return candidates

    def _accept(self, env, new_solution, delta: float, temperature: float, rng: random.Random) -> bool:
        if hasattr(env, "validation_solution"):
            try:
                return bool(env.validation_solution(new_solution))
            except Exception:  # noqa: BLE001
                pass
        return (delta < 0) or (rng.random() < pow(2.718281828, -delta / max(temperature, 1e-6)))

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
            try:
                env.run_operator(
                    operator,
                    accepted,
                    meta=meta,
                    stage=self.stage_name,
                    time_ms=time_ms,
                    new_solution=new_solution if accepted else None,
                )
                return
            except TypeError:
                env.run_operator(
                    operator,
                    accepted,
                    meta=meta,
                    new_solution=new_solution if accepted else None,
                )
                if env.recordings:
                    env.recordings[-1]["stage"] = self.stage_name
                    env.recordings[-1]["time_ms"] = int(time_ms)
                    if "assign" not in env.recordings[-1] and hasattr(env.solution, "assign"):
                        env.recordings[-1]["assign"] = list(env.solution.assign)
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
        selected_operator = None
        selected_meta: Dict[str, Any] = {}

        selection_id = 0
        for step in range(max_steps):
            step_start = time.perf_counter()
            if step % self.selection_frequency == 0 or selected_operator is None:
                candidates = self._score_candidates(current_solution, self.env, self.rng)
                cand_ids = [c["id"] for c in candidates]
                state_summary = {
                    "candidate_ids": cand_ids,
                    "forbidden_ids": [],
                    "candidates": [
                        {"id": c["id"], "type": c["name"], "d_total": c["d_total"]} for c in candidates
                    ],
                }
                chosen_idx = self._choose_candidate(state_summary, candidates, step, selection_id, self.rng)
                picked = next((c for c in candidates if c["id"] == chosen_idx), candidates[0])
                reason = "llm_pick"
                selected_operator = picked["operator"]
                selected_meta = {"heuristic_id": picked["id"], "heuristic_name": picked["name"], **picked.get("meta", {})}
                selection_id += 1

            new_solution = selected_operator.run(current_solution)
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
                    "operator": selected_operator,
                    "meta": selected_meta,
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
        selected_operator = None
        selected_meta: Dict[str, Any] = {}

        steps = max_steps
        if steps is None:
            steps = max(1, int(self.iterations_scale_factor * max(1, getattr(env, "problem_size", 1))))

        selection_id = 0
        for step in range(int(steps)):
            step_start = time.perf_counter()
            if step % self.selection_frequency == 0 or selected_operator is None:
                candidates = self._score_candidates(current_solution, env, rng)
                cand_ids = [c["id"] for c in candidates]
                state_summary = {
                    "candidate_ids": cand_ids,
                    "forbidden_ids": [],
                    "candidates": [
                        {"id": c["id"], "type": c["name"], "d_total": c["d_total"]} for c in candidates
                    ],
                }
                chosen_idx = self._choose_candidate(
                    state_summary,
                    candidates,
                    int(getattr(env, "step", step)),
                    selection_id,
                    rng,
                )
                picked = next((c for c in candidates if c["id"] == chosen_idx), candidates[0])
                reason = "llm_pick"
                selected_operator = picked["operator"]
                selected_meta = {"heuristic_id": picked["id"], "heuristic_name": picked["name"], **picked.get("meta", {})}
                selection_id += 1

            new_solution = selected_operator.run(current_solution)
            new_score = env.get_key_value(new_solution)
            delta = new_score - current_score
            accept = self._accept(env, new_solution, delta, temperature, rng)
            if accept:
                current_solution = new_solution
                current_score = new_score
                env.solution = current_solution
            self._record_step(
                env,
                selected_operator,
                selected_meta,
                accept,
                int((time.perf_counter() - step_start) * 1000),
                new_solution,
            )
            temperature *= self.sa_alpha
