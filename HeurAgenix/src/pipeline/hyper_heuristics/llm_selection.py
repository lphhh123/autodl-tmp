import json
import os
import random
import time
import traceback
from typing import List, Optional

from src.problems.base.env import BaseEnv

from src.util.llm_client.get_llm_client import get_llm_client
from src.util.util import load_function, search_file, extract_function_with_short_docstring
from src.util.function_to_tool import convert_function_to_tool


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
        llm_ok = True
        llm_client = None
        try:
            if self.llm_config_file is None:
                raise FileNotFoundError("llm_config_file is None")
            llm_client = get_llm_client(self.llm_config_file)
        except Exception as e:  # noqa: BLE001
            llm_ok = False
            self._log_usage({"ok": False, "reason": "llm_init_failed", "error": str(e)})

        max_steps = getattr(env, "max_steps", None)
        if max_steps is None:
            max_steps = max(1, int(env.construction_steps * self.iterations_scale_factor))

        fail_count = 0
        selection_round = 0

        while getattr(env, "continue_run", True) and getattr(env, "current_steps", 0) < int(max_steps):
            if not self.heuristic_pool:
                self._log_usage({"ok": False, "reason": "empty_heuristic_pool"})
                break

            if len(self.heuristic_pool) <= self.num_candidate_heuristics:
                candidates = list(self.heuristic_pool)
            else:
                candidates = random.sample(self.heuristic_pool, self.num_candidate_heuristics)

            chosen = random.choice(candidates)
            tool_calls = None
            resp_text = None
            ok = False
            err = None

            if llm_ok:
                try:
                    tools = []
                    tools_map = {}
                    for h in candidates:
                        fn = load_function(h, problem=self.problem)
                        short_doc = extract_function_with_short_docstring(fn)
                        tools.append(convert_function_to_tool(fn, description=short_doc))
                        tools_map[h] = fn

                    problem_desc = open(
                        search_file(os.path.join("prompt", "problem_description.txt"), self.problem),
                        "r", encoding="utf-8"
                    ).read()
                    special_remind_path = search_file(os.path.join("prompt", "special_remind.txt"), self.problem)
                    special_remind = ""
                    if special_remind_path and os.path.exists(special_remind_path):
                        special_remind = open(special_remind_path, "r", encoding="utf-8").read()

                    user_prompt = (
                        f"{problem_desc}\n\n"
                        f"Current state:\n{env.summarize_env()}\n\n"
                        f"Choose ONE heuristic to apply next from these candidates:\n"
                        + "\n".join([f"- {h}" for h in candidates]) + "\n\n"
                        f"{special_remind}\n"
                        f"Call exactly one tool (one heuristic)."
                    )

                    t0 = time.time()
                    resp_text, tool_calls = llm_client.chat_with_tools(user_prompt, tools, tool_choice="auto")
                    dt_ms = (time.time() - t0) * 1000.0

                    if tool_calls and len(tool_calls) > 0:
                        fn_name, fn_args = tool_calls[0]
                        if fn_name in tools_map:
                            chosen = fn_name
                            ok = True
                        else:
                            ok = False
                    else:
                        ok = False

                    self._log_usage({
                        "ok": ok,
                        "round": selection_round,
                        "candidates": candidates,
                        "chosen": chosen,
                        "response": resp_text,
                        "tool_calls": tool_calls,
                        "time_ms": dt_ms,
                    })

                except Exception as e:  # noqa: BLE001
                    fail_count += 1
                    err = traceback.format_exc()
                    self._log_usage({
                        "ok": False,
                        "reason": "llm_call_failed",
                        "round": selection_round,
                        "candidates": candidates,
                        "fallback_pick": chosen,
                        "fail_count": fail_count,
                        "error": str(e),
                        "traceback": err,
                    })
                    if fail_count >= self.max_llm_failures:
                        if self.fallback_on_llm_failure == "random_hh":
                            llm_ok = False
                            self._log_usage({"ok": False, "reason": "llm_disabled", "fail_count": fail_count})
                        else:
                            break

            heuristic_fn = load_function(chosen, problem=self.problem)
            for _ in range(self.selection_frequency):
                if getattr(env, "current_steps", 0) >= int(max_steps):
                    break
                if not getattr(env, "continue_run", True):
                    break
                r = env.run_heuristic(heuristic_fn, algorithm_data={}, record=True)
                _ = r

            selection_round += 1

        env.dump_result()
        return bool(getattr(env, "is_valid_solution", True))
