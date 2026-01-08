"""LLM-selection hyper-heuristic baseline."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from core import BaseEnv


class LLMSelectionHyperHeuristic:
    def __init__(
        self,
        env: BaseEnv,
        heuristics: List[Callable[..., Tuple[Any, Dict]]],
        rng: random.Random,
        selection_frequency: int = 5,
        num_candidate_heuristics: int = 4,
        sa_T0: float = 1.0,
        sa_alpha: float = 0.995,
        timeout_sec: int = 90,
        max_retry: int = 1,
        stage_name: str = "heuragenix_llm_hh",
        llm_client: Any | None = None,
        usage_path: str | None = None,
    ) -> None:
        self.env = env
        self.heuristics = heuristics
        self.rng = rng
        self.selection_frequency = max(1, int(selection_frequency))
        self.num_candidate_heuristics = max(1, int(num_candidate_heuristics))
        self.sa_T0 = float(sa_T0)
        self.sa_alpha = float(sa_alpha)
        self.stage_name = stage_name
        self.llm = llm_client
        self.usage_records: List[Dict[str, Any]] = []
        self.selection_records: List[Dict[str, Any]] = []
        self.usage_path = Path(usage_path) if usage_path else None

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

    def _score_candidates(self, solution) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        current_score = self.env.get_key_value(solution)
        heuristics = list(self.heuristics)
        if len(heuristics) > self.num_candidate_heuristics:
            heuristics = self.rng.sample(heuristics, self.num_candidate_heuristics)
        for idx, heuristic in enumerate(heuristics):
            operator, meta = heuristic({"solution": solution}, {"env": self.env, "rng": self.rng})
            new_solution = operator.run(solution)
            new_score = self.env.get_key_value(new_solution)
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

    def run(self, max_steps: int) -> None:
        if not self._llm_ready():
            raise RuntimeError("LLM not configured")
        temperature = self.sa_T0
        current_solution = self.env.solution
        current_score = self.env.get_key_value(current_solution)
        selected_operator = None
        selected_meta: Dict[str, Any] = {}

        selection_id = 0
        for step in range(max_steps):
            step_start = time.perf_counter()
            if step % self.selection_frequency == 0 or selected_operator is None:
                candidates = self._score_candidates(current_solution)
                cand_ids = [c["id"] for c in candidates]
                state_summary = {
                    "candidate_ids": cand_ids,
                    "forbidden_ids": [],
                    "candidates": [
                        {"id": c["id"], "type": c["name"], "d_total": c["d_total"]} for c in candidates
                    ],
                }
                picks = self.llm.propose_pick(state_summary, 1)
                usage = dict(getattr(self.llm, "last_usage") or {})
                usage["step"] = int(step)
                usage["n_pick"] = len(picks)
                self.usage_records.append(usage)
                if not picks or not usage.get("ok", True):
                    failure = {
                        "ok": False,
                        "reason": usage.get("error") or "llm_empty_pick",
                        "chosen_heuristic": None,
                        "candidates": [
                            {"id": c["id"], "type": c["name"], "d_total": c["d_total"]} for c in candidates
                        ],
                        "selection_id": selection_id,
                        "step": int(step),
                        **usage,
                    }
                    self.selection_records.append(failure)
                    self._append_usage(failure)
                    raise RuntimeError(usage.get("error") or "LLM selection failed")
                picked = next((c for c in candidates if c["id"] == picks[0]), candidates[0])
                reason = "llm_pick"
                selected_operator = picked["operator"]
                selected_meta = {"heuristic_id": picked["id"], "heuristic_name": picked["name"], **picked.get("meta", {})}
                selection_payload = {
                    "ok": True,
                    "reason": reason,
                    "chosen_heuristic": picked["name"],
                    "candidates": [{"id": c["id"], "type": c["name"], "d_total": c["d_total"]} for c in candidates],
                    "selection_id": selection_id,
                    "step": int(step),
                    **usage,
                }
                self.selection_records.append(selection_payload)
                self._append_usage(selection_payload)
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
