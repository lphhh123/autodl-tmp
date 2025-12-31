"""LLM provider abstraction (Volc Ark stub + heuristic fallback)."""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List

import requests


class LLMProvider(ABC):
    @abstractmethod
    def propose_actions(self, state_summary: Dict, k: int) -> List[Dict]:
        ...


class HeuristicProvider(LLMProvider):
    def propose_actions(self, state_summary: Dict, k: int) -> List[Dict]:
        actions = []
        hot_pairs = state_summary.get("top_hot_pairs", [])
        for pair in hot_pairs[:k]:
            actions.append({"op": "swap", "i": pair.get("i"), "j": pair.get("j")})
        while len(actions) < k:
            actions.append({"op": "relocate", "i": 0, "site_id": None})
        return actions


class VolcArkProvider(LLMProvider):
    def __init__(self, timeout_sec: int = 30, max_retry: int = 2):
        self.api_key = os.getenv("VOLC_ARK_API_KEY", "")
        self.endpoint = os.getenv("VOLC_ARK_ENDPOINT", "")
        self.model = os.getenv("VOLC_ARK_MODEL", "")
        self.timeout = timeout_sec
        self.max_retry = max_retry

    def propose_actions(self, state_summary: Dict, k: int) -> List[Dict]:
        if not self.api_key or not self.endpoint or not self.model:
            return HeuristicProvider().propose_actions(state_summary, k)

        prompt = (
            "Propose layout actions as JSON. Only respond with an object containing "
            '"actions" list. Each action has op (swap/relocate/cluster_move) and fields.'
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps({"state": state_summary, "k": k})},
            ],
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        last_err = None
        for _ in range(self.max_retry):
            try:
                resp = requests.post(
                    f"{self.endpoint}/chat/completions",
                    json=payload,
                    timeout=self.timeout,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = json.loads(content)
                actions = parsed.get("actions", [])
                return actions[:k]
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        # fallback
        return HeuristicProvider().propose_actions(state_summary, k)

