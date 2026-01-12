import os
import time
import json
import random
import traceback
from typing import List, Optional

from src.problems.base.env import BaseEnv
from src.util.util import load_function, extract, find_closest_match


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
        rng = random.Random(self.seed)
        algorithm_data = {"rng": rng, "env": env}

        # init LLM background (best-effort)
        llm_enabled = True
        fail_count = 0
        try:
            self.llm_client.reset()
            # keep "without_code" to reduce tokens
            self.llm_client.load_background("background_without_code.txt")
            if self.tool_calling:
                self.llm_client.load("heuristic_selection_with_tool.txt")
            else:
                self.llm_client.load("heuristic_selection.txt")
        except Exception as e:
            llm_enabled = False
            self._log_usage({"ok": False, "reason": "llm_init_failed", "error": str(e)})

        selection_round = 0
        while env.continue_run:
            candidates = self._choose_candidates(rng)

            t0 = time.time()
            ok = False
            chosen = None
            err = None

            # rollout_budget>0 才允许用 tts_bon（懒加载，避免 dill/依赖炸）
            if self.rollout_budget > 0 and len(candidates) > 1:
                try:
                    from src.util.tts_bon import tts_bon
                    candidates = tts_bon(
                        env,
                        candidates,
                        rollout_budget=self.rollout_budget,
                        problem=self.problem,
                        iterations_scale_factor=self.iterations_scale_factor,
                    )
                except Exception as e:
                    # rollout 失败不影响主流程
                    self._log_usage({"ok": False, "reason": "rollout_failed", "error": str(e)})

            if llm_enabled:
                try:
                    prompt = env.summarize_env() if hasattr(env, "summarize_env") else str(env.get_problem_state())
                    # 将候选集合写入对话上下文
                    self.llm_client.chat(prompt + "\n\nCandidates:\n" + "\n".join([f"- {x}" for x in candidates]))
                    chosen = self._llm_pick(candidates)
                    ok = True
                except Exception as e:
                    fail_count += 1
                    err = traceback.format_exc()
                    chosen = rng.choice(candidates)
                    ok = False
                    self._log_usage({
                        "ok": False,
                        "reason": "llm_call_failed",
                        "fail_count": fail_count,
                        "error": str(e),
                        "fallback_pick": chosen,
                    })
                    if fail_count >= self.max_llm_failures:
                        llm_enabled = False
                        self._log_usage({"ok": False, "reason": "llm_disabled", "fail_count": fail_count})

            if chosen is None:
                chosen = rng.choice(candidates)

            dt_ms = int((time.time() - t0) * 1000)
            self._log_usage({
                "ok": ok,
                "selection_round": selection_round,
                "chosen_heuristic": chosen,
                "candidates": candidates,
                "tool_calling": self.tool_calling,
                "time_ms": dt_ms,
                "error": err,
            })

            heuristic = load_function(chosen, problem=self.problem)

            # execute selection_frequency steps
            for _ in range(self.selection_frequency):
                if not env.continue_run:
                    break
                env.run_heuristic(
                    heuristic,
                    algorithm_data=algorithm_data,
                    add_record_item={
                        "selection_round": selection_round,
                        "chosen_heuristic": chosen,
                        "candidates": candidates,
                        "llm_ok": ok,
                    },
                )

            selection_round += 1

        return bool(env.is_valid_solution(env.current_solution))
