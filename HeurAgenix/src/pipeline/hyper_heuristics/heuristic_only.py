from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Optional

from src.problems.base.env import BaseEnv
from src.util.util import get_heuristic_names, load_function


class HeuristicOnlyHyperHeuristic:
    """
    Deterministic heuristic selector (round-robin).
    Does not call any LLM.
    """

    def __init__(
        self,
        heuristic_pool: Optional[List[str]],
        problem: str,
        heuristic_dir: str,
        iterations_scale_factor: float = 2.0,
        selection_frequency: int = 5,
        output_dir: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        self.problem = problem
        self.heuristic_dir = heuristic_dir
        self.iterations_scale_factor = float(iterations_scale_factor)
        self.selection_frequency = max(1, int(selection_frequency))
        self.heuristic_pool = list(heuristic_pool) if heuristic_pool else []
        self.output_dir = output_dir
        self.usage_path = os.path.join(output_dir, "llm_usage.jsonl") if output_dir else None
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

    def run(self, env: BaseEnv) -> bool:
        selection_round = 0
        K = max(1, int(self.selection_frequency))
        idx = 0

        while getattr(env, "continue_run", True):
            heuristic_pool = self.heuristic_pool or get_heuristic_names(self.problem, self.heuristic_dir)
            if not heuristic_pool:
                self._log_usage({"ok": False, "engine_used": "heuristic_only", "reason": "empty_heuristic_pool"})
                break
            heuristic_pool = sorted(list(heuristic_pool))
            chosen = heuristic_pool[idx % len(heuristic_pool)]
            idx += 1
            selection_round += 1

            self._log_usage(
                {
                    "ok": True,
                    "engine_used": "heuristic_only",
                    "reason": "heuristic_only_pick",
                    "chosen_heuristic": chosen,
                    "candidates": heuristic_pool,
                    "selection_round": selection_round,
                    "selection_frequency": K,
                }
            )

            heuristic = load_function(chosen, problem=self.problem)
            for _ in range(K):
                if not getattr(env, "continue_run", True):
                    break
                env.run_heuristic(
                    heuristic,
                    selection="heuristic_only",
                    algorithm_data={"env": env, "rng": getattr(env, "rng", None) or self.rng},
                    add_record_item={
                        "selection": "heuristic_only",
                        "chosen_heuristic": chosen,
                        "candidates": heuristic_pool,
                        "selection_round": selection_round,
                    },
                )

        ok_valid = env.is_valid_solution(env.current_solution) if hasattr(env, "is_valid_solution") else True
        ok_done = (
            env.is_complete_solution()
            if callable(getattr(env, "is_complete_solution", None))
            else bool(getattr(env, "is_complete_solution", True))
        )
        return bool(ok_done) and bool(ok_valid)
