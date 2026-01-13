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

    def _with_usage_base(self, env: BaseEnv, rec: Dict[str, Any]) -> Dict[str, Any]:
        base = {
            "case_name": str(getattr(env, "data_name", "")),
            "seed_id": int(getattr(env, "seed", 0)),
            "method": "llm_hh",
            "max_steps": int(getattr(env, "max_steps", 0)),
            "selection_frequency": int(self.selection_frequency),
            "num_candidate_heuristics": int(self.num_candidate_heuristics),
            "rollout_budget": int(self.rollout_budget),
        }
        for key, value in base.items():
            rec.setdefault(key, value)
        return rec

    def _get_algorithm_data(self, env: BaseEnv) -> Dict[str, Any]:
        data = dict(getattr(env, "algorithm_data", {}) or {})
        data["env"] = env
        data["rng"] = getattr(env, "rng", None) or self.rng
        return data

    def run(self, env: BaseEnv) -> bool:
        import os, json, time, random, traceback

        # ---------- paths ----------
        usage_path = None
        if getattr(self, "output_dir", None):
            usage_path = os.path.join(self.output_dir, "llm_usage.jsonl")

        def _log(rec: dict):
            if not usage_path:
                return
            with open(usage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # ---------- init llm ----------
        llm_ok = True
        llm_client = None
        try:
            llm_client = get_llm_client(
                self.llm_config_file,
                timeout_sec=self.llm_timeout_s,
                max_retry=1,
            )
            if llm_client is None:
                llm_ok = False
                _log(
                    {
                        "ok": False,
                        "reason": "missing_llm_config_or_client_none",
                        "engine_used": "random_hh",
                        "fallback_used": True,
                        "llm_config_file": self.llm_config_file,
                    }
                )
        except Exception as e:
            llm_ok = False
            _log(
                {
                    "ok": False,
                    "engine_used": "random_hh",
                    "reason": "llm_client_init_failed",
                    "error": str(e),
                }
            )

        fail_count = 0
        selection_round = 0
        K = max(1, int(getattr(self, "selection_frequency", 1)))
        max_fail = int(getattr(self, "max_llm_failures", 2))
        fallback_mode = str(getattr(self, "fallback_on_llm_failure", "random_hh"))  # "random_hh" or "stop"
        timeout_s = int(getattr(self, "llm_timeout_s", 30))

        # ---------- run loop ----------
        while getattr(env, "continue_run", True):
            heuristic_pool = get_heuristic_names(self.problem, self.heuristic_dir)
            if not heuristic_pool:
                _log({"ok": False, "engine_used": "random_hh", "reason": "empty_heuristic_pool"})
                break

            # candidate set
            C = int(getattr(self, "num_candidate_heuristics", 1))
            C = max(1, C)
            if len(heuristic_pool) <= C:
                candidates = list(heuristic_pool)
            else:
                candidates = self.rng.sample(list(heuristic_pool), C)

            selection_round += 1
            chosen = None
            stage_name = "random_hh"
            ok = False
            reason = "random_pick_default"
            err = None

            # --- if llm disabled and policy says stop, stop early ---
            if (not llm_ok) and fallback_mode == "stop":
                _log(
                    {
                        "ok": False,
                        "engine_used": "stop",
                        "reason": "llm_unavailable_and_stop",
                        "fail_count": fail_count,
                        "selection_round": selection_round,
                    }
                )
                break

            # --- try llm choose ---
            if llm_ok and llm_client is not None:
                try:
                    state_summary = env.summarize_env()
                    chosen = llm_client.choose_heuristic(
                        state_summary,
                        candidates,
                        timeout_s=timeout_s,
                    )
                    stage_name = "llm_hh"
                    ok = True
                    reason = "llm_pick"
                    if chosen not in candidates:
                        ok = False
                        reason = "llm_pick_not_in_candidates"
                        chosen = self.rng.choice(candidates)
                        stage_name = "random_hh"
                except Exception as e:
                    fail_count += 1
                    err = traceback.format_exc()
                    chosen = self.rng.choice(candidates)
                    stage_name = "random_hh"
                    ok = False
                    reason = "llm_call_failed_fallback_random"
                    _log(
                        {
                            "ok": False,
                            "engine_used": stage_name,
                            "reason": reason,
                            "error": str(e),
                            "fail_count": fail_count,
                            "max_llm_failures": max_fail,
                            "selection_round": selection_round,
                            "fallback_pick": chosen,
                        }
                    )
                    if fail_count >= max_fail:
                        llm_ok = False
                        _log(
                            {
                                "ok": False,
                                "reason": "llm_disabled_after_failures",
                                "fail_count": fail_count,
                                "engine_used": "random_hh",
                                "fallback_used": True,
                            }
                        )

            if chosen is None:
                chosen = self.rng.choice(candidates)
                stage_name = "random_hh"
                ok = False
                reason = "llm_disabled_random_pick"

            # always record one usage line for THIS selection
            _log(
                {
                    "ok": ok,
                    "engine_used": stage_name,
                    "reason": reason,
                    "chosen_heuristic": chosen,
                    "candidates": candidates,
                    "selection_round": selection_round,
                    "selection_frequency": K,
                    "fail_count": fail_count,
                    "llm_enabled": bool(llm_ok and llm_client is not None),
                }
            )

            heuristic = load_function(chosen, problem=self.problem)

            for _ in range(K):
                if not getattr(env, "continue_run", True):
                    break
                env.run_heuristic(
                    heuristic,
                    selection=stage_name,
                    algorithm_data={"env": env, "rng": getattr(env, "rng", None)},
                    add_record_item={
                        # ★关键：让 recordings 的 stage/selection 反映真实 engine
                        "selection": stage_name,
                        "chosen_heuristic": chosen,
                        "candidates": candidates,
                        "selection_round": selection_round,
                        "llm_enabled": bool(llm_ok and llm_client is not None),
                        "llm_fail_count": int(fail_count),
                        "fallback_mode": fallback_mode,
                    },
                )

        # finish
        ok_valid = env.is_valid_solution(env.current_solution) if hasattr(env, "is_valid_solution") else True
        ok_done = (
            env.is_complete_solution()
            if callable(getattr(env, "is_complete_solution", None))
            else bool(getattr(env, "is_complete_solution", True))
        )
        return bool(ok_done) and bool(ok_valid)
