"""LLM planner interface and Volc Ark implementation (SPEC v4.3.2 §9)."""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict

import requests


class LLMProvider(ABC):
    @abstractmethod
    def propose_actions(self, state_summary: Dict, k: int) -> List[Dict]:
        ...


class HeuristicProvider(LLMProvider):
    def propose_actions(self, state_summary: Dict, k: int) -> List[Dict]:
        actions: List[Dict] = []
        hot_pairs = state_summary.get("top_hot_pairs", [])
        for pair in hot_pairs[:k]:
            actions.append({"op": "swap", "i": pair.get("i"), "j": pair.get("j")})
        return actions


class VolcArkProvider(LLMProvider):
    def __init__(self, timeout_sec: int = 30, max_retry: int = 2):
        self.endpoint = os.getenv("VOLC_ARK_ENDPOINT")
        self.api_key = os.getenv("VOLC_ARK_API_KEY")
        self.model = os.getenv("VOLC_ARK_MODEL")
        self.timeout_sec = timeout_sec
        self.max_retry = max_retry
        self.last_usage = None

    def _build_payload(self, state_summary: Dict, k: int) -> Dict:
        prompt = (
            "You are a placement planner. Return JSON with actions."
            " Only output JSON with key actions (list)."
        )
        content = json.dumps({"hint": prompt, "state": state_summary, "k": k})
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_object"},
        }

    def propose_actions(self, state_summary: Dict, k: int) -> List[Dict]:
        if not self.endpoint or not self.api_key or not self.model:
            self.last_usage = {"ok": False, "skipped": True, "reason": "VOLC_ARK credentials missing"}
            return []
        payload = self._build_payload(state_summary, k)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_error = None
        for _ in range(self.max_retry + 1):
            try:
                resp = requests.post(
                    f"{self.endpoint}/chat/completions",
                    json=payload,
                    timeout=self.timeout_sec,
                )
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                self.last_usage = {
                    "model": self.model,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "request_bytes": len(json.dumps(payload)),
                    "response_bytes": len(resp.content),
                }
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                parsed = json.loads(content)
                actions = parsed.get("actions", [])
                if isinstance(actions, list):
                    self.last_usage["ok"] = True
                    return actions
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        self.last_usage = {"ok": False, "error": str(last_error) if last_error else "unknown"}
        return []
