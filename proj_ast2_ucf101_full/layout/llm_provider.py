"""LLM planner interface and Volc Ark implementation (SPEC v4.3.2 §9)."""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional

import requests
from requests.exceptions import ReadTimeout


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
    def __init__(self, timeout_sec: int = 30, max_retry: int = 2, max_tokens: int = 256):
        self.endpoint = (os.getenv("VOLC_ARK_ENDPOINT") or "https://ark.cn-beijing.volces.com/api/v3").strip()
        self.model = (os.getenv("VOLC_ARK_MODEL") or "").strip()
        self.api_key = (os.getenv("VOLC_ARK_API_KEY") or os.getenv("ARK_API_KEY") or "").strip()
        self.timeout_sec = timeout_sec
        self.max_retry = max_retry
        self.max_tokens = int(max_tokens)
        # allow users to pass a full /chat/completions url and normalize to base
        if self.endpoint.rstrip("/").endswith("/chat/completions"):
            self.endpoint = self.endpoint.rstrip("/")[:-len("/chat/completions")]
        self.endpoint = self.endpoint.rstrip("/")
        # handle keys provided as "Bearer xxx"
        if self.api_key.lower().startswith("bearer "):
            self.api_key = self.api_key[7:].strip()
        self.last_usage = None

    def _build_payload(self, state_summary: Dict, k: int) -> Dict:
        system_prompt = (
            "You are a placement planner. Return ONLY valid JSON without explanations. "
            "Output either a JSON array of actions or {\"actions\":[...]}. Each action must be one of: "
            "swap(i,j), relocate(i,site_id), cluster_move(cluster_id,region_id)."
        )
        user_content = json.dumps({"state": state_summary, "k": k})
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_completion_tokens": int(self.max_tokens),
            "temperature": 0.2,
        }

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        if not text:
            return None
        start_candidates = [idx for idx in (text.find("[") , text.find("{")) if idx != -1]
        if not start_candidates:
            return None
        start = min(start_candidates)
        end_bracket = text.rfind("]")
        end_brace = text.rfind("}")
        end_candidates = [idx for idx in (end_bracket, end_brace) if idx != -1 and idx > start]
        if not end_candidates:
            return None
        end = max(end_candidates)
        snippet = text[start : end + 1]
        return snippet if snippet else None

    def propose_actions(self, state_summary: Dict, k: int) -> List[Dict]:
        url = f"{self.endpoint}/chat/completions"
        if not self.endpoint or not self.api_key or not self.model:
            self.last_usage = {
                "ok": False,
                "skipped": True,
                "reason": "VOLC_ARK credentials missing",
                "endpoint": self.endpoint,
                "url": url,
                "model": self.model,
                "key_len": len(self.api_key),
            }
            return []

        payload = self._build_payload(state_summary, k)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_error = None
        raw_preview: str = ""
        for _ in range(self.max_retry + 1):
            try:
                resp = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=(5, self.timeout_sec),
                )
                self.last_usage = {
                    "endpoint": self.endpoint,
                    "url": url,
                    "model": self.model,
                    "key_len": len(self.api_key),
                    "request_bytes": len(json.dumps(payload)),
                    "status_code": resp.status_code,
                }
                if resp.status_code != 200:
                    self.last_usage.update({"ok": False, "resp": resp.text[:200]})
                    last_error = Exception(f"HTTP {resp.status_code}")
                    continue

                data = resp.json()
                usage = data.get("usage", {})
                self.last_usage.update(
                    {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                        "response_bytes": len(resp.content),
                    }
                )
                message = data.get("choices", [{}])[0].get("message", {})
                text = (message.get("content") or "").strip()
                if not text:
                    text = (message.get("reasoning_content") or "").strip()
                raw_preview = text
                extracted = self._extract_json(text)
                if not extracted:
                    self.last_usage.update(
                        {
                            "ok": False,
                            "error": "no_json_fragment",
                            "raw_preview": raw_preview[:200],
                        }
                    )
                    return []

                try:
                    parsed = json.loads(extracted)
                except json.JSONDecodeError as exc:  # noqa: BLE001
                    self.last_usage.update(
                        {
                            "ok": False,
                            "error": f"json_decode:{exc.msg}",
                            "raw_preview": extracted[:200],
                        }
                    )
                    return []

                actions = parsed.get("actions") if isinstance(parsed, dict) else parsed
                if isinstance(actions, list):
                    self.last_usage.update({"ok": True})
                    return actions

                self.last_usage.update(
                    {"ok": False, "error": "no_actions_list", "raw_preview": str(parsed)[:200]}
                )
                return []
            except ReadTimeout as exc:
                last_error = exc
                self.last_usage = {
                    "ok": False,
                    "endpoint": self.endpoint,
                    "url": url,
                    "model": self.model,
                    "key_len": len(self.api_key),
                    "error": "ReadTimeout",
                }
                return []
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                resp_text = resp.text[:200] if "resp" in locals() else None
                self.last_usage = {
                    "ok": False,
                    "endpoint": self.endpoint,
                    "url": url,
                    "model": self.model,
                    "key_len": len(self.api_key),
                    "error": repr(exc),
                }
                if resp_text is not None:
                    self.last_usage["resp"] = resp_text
                if raw_preview:
                    self.last_usage["raw_preview"] = raw_preview[:200]
                continue
        if self.last_usage is None:
            self.last_usage = {"ok": False}
        if last_error:
            self.last_usage.setdefault("error", repr(last_error))
        return []
