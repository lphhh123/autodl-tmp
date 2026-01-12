from __future__ import annotations

import json
import os
import random
import time
import traceback
from typing import Any, Dict, List, Optional

from src.problems.base.env import BaseEnv
from src.util.llm_client.get_llm_client import get_llm_client
from src.util.util import get_heuristic_names, load_function


class LLMSelectionHyperHeuristic:
    """
    LLM chooses which heuristic to run (NOT an action).
    Must be robust: always log usage, fallback on failures, never infinite loop.
    """

    def __init__(
        self,
        heuristic_pool: Optional[List[str]],
        problem: str,
        heuristic_dir: str,
        llm_config_file: Optional[str],
        iterations_scale_factor: float = 2.0,
        selection_frequency: int = 5,
        num_candidate_heuristics: int = 1,
        rollout_budget: int = 0,
        output_dir: Optional[str] = None,
        llm_timeout_s: int = 30,
        max_llm_failures: int = 2,
        fallback_on_llm_failure: str = "random_hh",
        seed: int = 0,
    ):
        self.problem = problem
        self.heuristic_dir = heuristic_dir
        self.llm_config_file = llm_config_file

        self.iterations_scale_factor = float(iterations_scale_factor)
        self.selection_frequency = max(1, int(selection_frequency))
        self.num_candidate_heuristics = max(1, int(num_candidate_heuristics))
        self.rollout_budget = int(rollout_budget)

        self.heuristic_pool = list(heuristic_pool) if heuristic_pool else []

        self.output_dir = output_dir
        self.usage_path = os.path.join(output_dir, "llm_usage.jsonl") if output_dir else None

        self.llm_timeout_s = int(llm_timeout_s)
        self.max_llm_failures = int(max_llm_failures)
        self.fallback_on_llm_failure = str(fallback_on_llm_failure)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)

        if self.usage_path:
            os.makedirs(os.path.dirname(self.usage_path), exist_ok=True)
            open(self.usage_path, "a", encoding="utf-8").close()

    def _log_usage(self, rec: Dict[str, Any]) -> None:
        if not self.usage_path:
            return
        with open(self.usage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _get_algorithm_data(self, env: BaseEnv) -> Dict[str, Any]:
        data = dict(getattr(env, "algorithm_data", {}) or {})
        data["env"] = env
        data["rng"] = getattr(env, "rng", None) or self.rng
        return data

    def run(self, env: BaseEnv) -> bool:
        if not self.heuristic_pool:
            self.heuristic_pool = get_heuristic_names(self.problem, self.heuristic_dir)

        llm_ok = True
        llm_client = None
        try:
            if not self.llm_config_file:
                raise FileNotFoundError("llm_config_file is None/empty")
            llm_client = get_llm_client(self.llm_config_file)
            if llm_client is None:
                raise RuntimeError("get_llm_client returned None")
        except Exception as e:  # noqa: BLE001
            llm_ok = False
            self._log_usage({"ok": False, "reason": "llm_init_failed", "error": str(e)})

        fail_count = 0
        selection_round = 0

        max_steps = int(getattr(env, "max_steps", 0) or 0)
        if max_steps <= 0:
            ps = int(getattr(env, "problem_size", None) or getattr(env, "S", 1) or 1)
            max_steps = max(1, int(self.iterations_scale_factor * ps))

        while getattr(env, "continue_run", True) and int(getattr(env, "current_steps", 0)) < max_steps:
            pool = list(self.heuristic_pool)
            if not pool:
                self._log_usage({"ok": False, "reason": "empty_heuristic_pool"})
                break

            candidates = (
                pool if len(pool) <= self.num_candidate_heuristics else self.rng.sample(pool, self.num_candidate_heuristics)
            )
            chosen = self.rng.choice(candidates)
            ok = False
            err = None
            t0 = time.time()

            if llm_ok and llm_client is not None:
                try:
                    prompt = env.summarize_env()
                    chosen = llm_client.choose_heuristic(prompt, candidates, timeout_s=self.llm_timeout_s)
                    ok = True
                except Exception as e:  # noqa: BLE001
                    fail_count += 1
                    err = traceback.format_exc()
                    ok = False
                    chosen = self.rng.choice(candidates)

                    self._log_usage(
                        {
                            "ok": False,
                            "reason": "llm_call_failed",
                            "round": selection_round,
                            "error": str(e),
                            "traceback": err,
                            "fail_count": fail_count,
                            "fallback_pick": chosen,
                            "candidates": candidates,
                            "rollout_budget": self.rollout_budget,
                        }
                    )

                    if fail_count >= self.max_llm_failures:
                        llm_ok = False
                        self._log_usage({"ok": False, "reason": "llm_disabled_after_failures", "fail_count": fail_count})

            dt_ms = (time.time() - t0) * 1000.0
            self._log_usage(
                {
                    "ok": ok,
                    "round": selection_round,
                    "chosen_heuristic": chosen,
                    "candidates": candidates,
                    "time_ms": dt_ms,
                    "rollout_budget": self.rollout_budget,
                    "fallback_used": not ok,
                }
            )

            heuristic_fn = load_function(chosen, problem=self.problem)
            algo_data = self._get_algorithm_data(env)
            for _ in range(self.selection_frequency):
                if not getattr(env, "continue_run", True):
                    break
                if int(getattr(env, "current_steps", 0)) >= max_steps:
                    break
                env.run_heuristic(heuristic_fn, algorithm_data=algo_data, record=True)

            selection_round += 1

        env.dump_result()
        if hasattr(env, "validate_solution"):
            return bool(env.validate_solution(getattr(env, "current_solution", None)))
        return True
