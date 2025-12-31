"""LLM planner interface and Volc Ark implementation (SPEC v4.3.2 §9)."""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
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
    def __init__(self, timeout_sec: int = 30, max_retry: int = 2, usage_log: str | None = None):
        self.endpoint = os.getenv("VOLC_ARK_ENDPOINT")
        self.api_key = os.getenv("VOLC_ARK_API_KEY")
        self.model = os.getenv("VOLC_ARK_MODEL")
        self.timeout_sec = timeout_sec
        self.max_retry = max_retry
        self.usage_log = Path(usage_log) if usage_log else None
        if not self.endpoint or not self.api_key or not self.model:
            raise RuntimeError("VOLC Ark credentials missing in environment variables")

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
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                parsed = json.loads(content)
                actions = parsed.get("actions", [])
                self._log_usage(data)
                if isinstance(actions, list):
                    return actions
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(f"Volc Ark call failed after retries: {last_error}")

    def _log_usage(self, data: Dict):
        if not self.usage_log:
            return
        usage = data.get("usage", {})
        entry = {
            "ts": datetime.utcnow().isoformat(),
            "model": self.model,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
        }
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        with self.usage_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
