import os
import time
import json
import random
import traceback
from typing import List, Optional

import numpy as np

from src.problems.base.env import BaseEnv
from src.util.util import load_function, extract, find_closest_match


select_heuristic_tool = [
    {
        "type": "function",
        "function": {
            "name": "select_heuristic",
            "description": "Select a heuristic name from the candidate list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "heuristic": {"type": "string"},
                },
                "required": ["heuristic"],
            },
        },
    }
]


def generate_heuristic_pool_introduction(candidate_heuristics: List[str], problem: str) -> str:
    lines = [f"Heuristic pool for problem: {problem}"]
    for idx, name in enumerate(candidate_heuristics, start=1):
        lines.append(f"{idx}. {name}")
    return "\n".join(lines)


def filter_dict_to_str(items: List[dict]) -> str:
    return json.dumps(items, ensure_ascii=False, default=str)


class LLMSelectionHyperHeuristic:
    def __init__(
        self,
        llm_client,
        heuristic_pool: List[str],
        problem: str,
        tool_calling: bool = False,
        iterations_scale_factor: float = 1.0,
        selection_frequency: int = 5,
        num_candidate_heuristics: int = 1,
        rollout_budget: int = 0,
        output_dir: Optional[str] = None,
        seed: int = 0,
        max_llm_failures: int = 2,
    ):
        self.llm_client = llm_client
        self.heuristic_pool = list(heuristic_pool)
        self.problem = problem

        self.tool_calling = bool(tool_calling)
        self.iterations_scale_factor = float(iterations_scale_factor)
        self.selection_frequency = int(selection_frequency)
        self.num_candidate_heuristics = int(num_candidate_heuristics)
        self.rollout_budget = int(rollout_budget)

        self.output_dir = output_dir
        self.seed = int(seed)
        self.max_llm_failures = int(max_llm_failures)
        self.heuristic_dir = "basic_heuristics"

        self.tools = None
        if self.tool_calling:
            # convert each heuristic into a "tool"
            from src.util.util import search_file, convert_function_to_tool
            tools = []
            for h in self.heuristic_pool:
                path = search_file(h + ".py", problem)
                with open(path, "r", encoding="utf-8") as f:
                    code = f.read()
                tools.append(convert_function_to_tool(code))
            self.tools = tools

    def _usage_path(self):
        if not self.output_dir:
            return None
        return os.path.join(self.output_dir, "llm_usage.jsonl")

    def _log_usage(self, rec: dict):
        p = self._usage_path()
        if not p:
            return
        rec = dict(rec)
        rec.setdefault("ts", time.time())
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _choose_candidates(self, rng: random.Random):
        if self.num_candidate_heuristics <= 0 or self.num_candidate_heuristics >= len(self.heuristic_pool):
            return list(self.heuristic_pool)
        return rng.sample(self.heuristic_pool, self.num_candidate_heuristics)

    def _llm_pick(self, candidates: List[str]):
        """
        Return heuristic_name.
        tool_calling: use chat_with_tools -> [(fn_name, params)]
        text: parse "Selected heuristic: <name>"
        """
        if self.tool_calling:
            calls = self.llm_client.chat_with_tools(self.tools)
            if not calls:
                raise RuntimeError("empty tool calls")
            # pick first tool name
            return calls[0][0]

        # text mode
        action = self.llm_client.chat()
        picked = extract(action, "Selected heuristic")
        if picked is None:
            # try fuzzy match from whole response
            picked = find_closest_match(action, candidates)

        if picked not in candidates:
            picked = find_closest_match(str(picked), candidates)
        return picked

    def run(self, env: BaseEnv) -> bool:
        usage_path = None
        if getattr(self, "output_dir", None):
            usage_path = os.path.join(self.output_dir, "llm_usage.jsonl")

        def _log_usage(rec: dict):
            if usage_path:
                rec["timestamp"] = time.time()
                with open(usage_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        out_dir = getattr(self.llm_client, "output_dir", None)
        usage_path_client = os.path.join(out_dir, "llm_usage.jsonl") if out_dir else None

        def log_usage(rec: dict):
            if usage_path_client:
                with open(usage_path_client, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if not self.heuristic_pool:
            try:
                cand_dir = os.path.join(
                    "src",
                    "problems",
                    self.problem,
                    "heuristics",
                    self.heuristic_dir,
                )
                self.heuristic_pool = [
                    x.split(".")[0]
                    for x in os.listdir(cand_dir)
                    if x.endswith(".py") and not x.startswith("__")
                ]
            except Exception:
                pass

        if not self.heuristic_pool:
            log_usage({"ok": False, "reason": "empty_heuristic_pool"})
            _log_usage({"ok": False, "reason": "empty_heuristic_pool"})
            return False

        max_steps = int(env.construction_steps * self.iterations_scale_factor)
        max_rounds = int(np.ceil(max_steps / max(1, self.selection_frequency)))

        fail_count = 0
        max_fail = getattr(self, "max_llm_failures", 2)

        selection_round = 0
        while selection_round < max_rounds and env.current_steps < max_steps and env.continue_run:
            selection_round += 1

            if len(self.heuristic_pool) <= self.num_candidate_heuristics:
                candidate_heuristics = list(self.heuristic_pool)
            else:
                candidate_heuristics = random.sample(
                    self.heuristic_pool, self.num_candidate_heuristics
                )

            t0 = time.time()
            chosen = None
            ok = False
            err = None

            try:
                heuristic_intros = generate_heuristic_pool_introduction(
                    candidate_heuristics, self.problem
                )

                background_info = {
                    "problem": self.problem,
                    "problem_description": env.prompt,
                }
                is_cop = self.llm_client.load(
                    background_info, background_file="background_without_code.txt"
                )
                if isinstance(is_cop, str) and "is_cop:no" in is_cop:
                    raise RuntimeError("LLM says is_cop:no")

                problem_state = env.get_problem_state()
                prompt_dict = {
                    "heuristic_pool_introduction": heuristic_intros,
                    "task": f"Select a heuristic for {self.problem}.",
                    "problem_state": filter_dict_to_str([problem_state]),
                    "candidate_heuristics": str(candidate_heuristics),
                }

                if self.llm_client.support_tool_calling:
                    resp = self.llm_client.chat_with_tools(
                        prompt_dict, tools=select_heuristic_tool
                    )
                    name = resp.get("heuristic", None) if isinstance(resp, dict) else None
                    if not name:
                        raise RuntimeError(f"tool_call_empty: {resp}")
                    chosen = find_closest_match(name, candidate_heuristics)
                else:
                    resp = self.llm_client.chat(prompt_dict, text_file="select_heuristic.txt")
                    chosen = find_closest_match(resp, candidate_heuristics)

                ok = True

            except Exception as e:
                fail_count += 1
                err = traceback.format_exc()
                chosen = random.choice(candidate_heuristics)
                ok = False
                _log_usage({"ok": False, "reason": "llm_call_failed", "error": str(e)})
                log_usage(
                    {
                        "ok": False,
                        "reason": "llm_choose_failed",
                        "round": selection_round,
                        "error": str(e),
                        "fail_count": fail_count,
                        "candidates": candidate_heuristics,
                        "fallback_pick": chosen,
                    }
                )

            if (not ok) and fail_count >= max_fail:
                _log_usage({"ok": False, "reason": "llm_call_failed", "error": err or ""})
                log_usage(
                    {
                        "ok": False,
                        "reason": "llm_disabled_after_failures",
                        "fail_count": fail_count,
                    }
                )
                while env.current_steps < max_steps and env.continue_run:
                    name = random.choice(self.heuristic_pool)
                    h = load_function(name, problem=self.problem)
                    env.run_heuristic(h)
                env.dump_result()
                return env.is_complete_solution and env.is_valid_solution

            log_usage(
                {
                    "ok": ok,
                    "round": selection_round,
                    "chosen": chosen,
                    "candidates": candidate_heuristics,
                    "time_ms": int((time.time() - t0) * 1000),
                }
            )
            _log_usage({"ok": True, "chosen": chosen})

            heuristic = load_function(chosen, problem=self.problem)
            for _ in range(self.selection_frequency):
                if env.current_steps >= max_steps or (not env.continue_run):
                    break
                env.run_heuristic(heuristic)

        env.dump_result()
        return env.is_complete_solution and env.is_valid_solution
