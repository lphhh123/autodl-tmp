import json
import os
import random
from typing import List, Optional

from src.problems.base.env import BaseEnv

from src.util.llm_client.get_llm_client import get_llm_client
from src.util.util import load_function


class LLMSelectionHyperHeuristic:
    def __init__(
        self,
        heuristic_pool: List[str],
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
    ):
        self.problem = problem
        self.heuristic_dir = heuristic_dir
        self.llm_config_file = llm_config_file

        self.iterations_scale_factor = float(iterations_scale_factor)
        self.selection_frequency = int(selection_frequency)
        self.num_candidate_heuristics = int(num_candidate_heuristics)
        self.rollout_budget = int(rollout_budget)

        pool = []
        for h in heuristic_pool:
            name = h.split("/")[-1]
            if name.endswith(".py"):
                name = name[:-3]
            if name and not name.startswith("_"):
                pool.append(name)
        self.heuristic_pool = sorted(list(set(pool)))

        self.output_dir = output_dir
        self.usage_path = None
        if output_dir:
            self.usage_path = os.path.join(output_dir, "llm_usage.jsonl")

        self.llm_timeout_s = int(llm_timeout_s)
        self.max_llm_failures = int(max_llm_failures)
        self.fallback_on_llm_failure = str(fallback_on_llm_failure)

    def _log_usage(self, rec: dict):
        if not self.usage_path:
            return
        with open(self.usage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def run(self, env: BaseEnv) -> bool:
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            self.usage_path = self.usage_path or os.path.join(self.output_dir, "llm_usage.jsonl")
            if self.usage_path:
                with open(self.usage_path, "a", encoding="utf-8"):
                    pass

        fail_count, llm_ok = 0, True
        selection_round = 0
        while env.continue_run:
            selection_round += 1
            pool = self.heuristic_pool or []
            if not pool:
                self._log_usage(
                    {
                        "ok": False,
                        "selection_round": selection_round,
                        "candidates": [],
                        "chosen_heuristic": None,
                        "reason": "empty_pool",
                        "fallback_used": False,
                    }
                )
                break
            candidates = random.sample(pool, min(len(pool), self.num_candidate_heuristics))
            chosen = None
            ok = True
            prompt = None
            model = None
            provider = None
            reason = None
            error = None
            fallback_used = False
            try:
                if llm_ok:
                    prompt = env.summarize_env()
                    client = get_llm_client(self.llm_config_file)
                    if client is None:
                        raise FileNotFoundError("llm_config_file is None or invalid")
                    model = getattr(client, "model", None)
                    provider = getattr(client, "provider", None)
                    chosen = client.choose_heuristic(prompt, candidates, timeout_s=self.llm_timeout_s)
                else:
                    chosen = random.choice(candidates)
                    fallback_used = True
                    reason = "llm_disabled"
            except Exception as e:  # noqa: BLE001
                ok = False
                fail_count += 1
                reason = "llm_exception"
                error = str(e)
                chosen = random.choice(candidates)
                fallback_used = True
                if fail_count >= self.max_llm_failures:
                    llm_ok = False
                    self._log_usage(
                        {
                            "ok": False,
                            "selection_round": selection_round,
                            "candidates": candidates,
                            "chosen_heuristic": chosen,
                            "reason": "llm_disabled_after_failures",
                            "fallback_used": True,
                            "model": model,
                            "provider": provider,
                        }
                    )

            self._log_usage(
                {
                    "ok": ok,
                    "selection_round": selection_round,
                    "candidates": candidates,
                    "chosen_heuristic": chosen,
                    "reason": reason,
                    "error": error,
                    "fallback_used": fallback_used,
                    "llm_ok": llm_ok,
                    "prompt": prompt,
                    "model": model,
                    "provider": provider,
                    "fail_count": fail_count,
                }
            )
            fn = load_function(chosen, problem=self.problem)
            for _ in range(self.selection_frequency):
                if not env.continue_run:
                    break
                env.run_heuristic(fn)
        env.dump_result()
        return True
