"""LLM planner interface and Volc Ark implementation (SPEC v4.3.2 §9)."""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple

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
    def __init__(self, timeout_sec: int = 90, max_retry: int = 1, max_tokens: int = 192):
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
            "You are a placement-planning assistant.\n\n"
            "ABSOLUTE OUTPUT CONTRACT (NO EXCEPTIONS):\n"
            "1) You MUST output ONLY one JSON object wrapped EXACTLY as:\n"
            "BEGIN_JSON\n"
            "{...}\n"
            "END_JSON\n"
            "2) Output MUST contain NOTHING outside the wrapper. No markdown. No explanation. No reasoning. No code fences.\n"
            "3) The JSON MUST be an object with exactly one top-level key: \"actions\".\n"
            "4) \"actions\" MUST be an array of 0..K objects.\n\n"
            "Allowed action schemas (choose one):\n"
            "- {\"op\":\"swap\",\"i\":INT,\"j\":INT}\n"
            "- {\"op\":\"relocate\",\"i\":INT,\"site_id\":INT}\n"
            "- {\"op\":\"cluster_move\",\"cluster_id\":INT,\"region_id\":INT}\n\n"
            "FORBIDDEN:\n"
            "- Any extra keys (e.g., \"why\", \"score\", \"thoughts\", \"analysis\", \"reasoning\", \"comment\").\n"
            "- Any strings not inside JSON.\n"
            "- Any output without BEGIN_JSON/END_JSON.\n\n"
            "If you cannot comply, output:\n"
            "BEGIN_JSON\n"
            "{\"actions\":[]}\n"
            "END_JSON\n\n"
            "BAD EXAMPLES (DO NOT DO THIS):\n"
            "- Here are actions: [{\"op\":\"swap\",\"i\":0,\"j\":1}]\n"
            "- ```json {\"actions\":[...]} ```\n"
            "- {\"actions\":[...]}           (missing wrapper)\n"
            "- BEGIN_JSON {\"actions\":[...]} END_JSON  (must be on separate lines)\n"
        )
        state_json = json.dumps(state_summary, separators=(",", ":"))
        S = int(state_summary.get("S", 0))
        Ns = int(state_summary.get("Ns", 0))
        num_clusters = int(state_summary.get("num_clusters", 0))
        num_regions = int(state_summary.get("num_regions", 0))
        user_content = (
            "TASK:\n"
            "Propose up to K actions to reduce total_scalar and improve comm_norm/therm_norm.\n\n"
            f"K = {int(k)}\n\n"
            "STATE_JSON:\n"
            f"{state_json}\n\n"
            "VALID RANGES:\n"
            f"- slot index i,j in [0, {S - 1}]\n"
            f"- site_id in [0, {Ns - 1}]\n"
            f"- cluster_id in [0, {num_clusters - 1}]\n"
            f"- region_id in [0, {num_regions - 1}]\n\n"
            "HARD RULES:\n"
            "- Use only allowed ops: swap / relocate / cluster_move.\n"
            "- Do NOT output any text outside BEGIN_JSON/END_JSON.\n"
            "- Do NOT output markdown or code fences.\n"
            "- Do NOT include extra keys.\n"
            "- If you are unsure, return an empty list.\n\n"
            "NOW OUTPUT ONLY THE WRAPPED JSON."
        )
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_completion_tokens": int(min(self.max_tokens, 256)),
            "temperature": 0,
            "top_p": 1,
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

    @staticmethod
    def extract_wrapped_json(raw: str) -> Tuple[Optional[dict], Optional[str]]:
        b = raw.find("BEGIN_JSON")
        e = raw.find("END_JSON")
        if b < 0 or e < 0 or e <= b:
            return None, "missing_wrapper"
        inner = raw[b + len("BEGIN_JSON") : e].strip()
        try:
            obj = json.loads(inner)
        except Exception as ex:  # noqa: BLE001
            return None, f"bad_json_in_wrapper:{ex!r}"
        return obj, None

    @staticmethod
    def _validate_actions(
        actions: Optional[List[Dict]],
        S: int,
        Ns: int,
        num_clusters: int,
        num_regions: int,
    ) -> List[Dict]:
        if not isinstance(actions, list):
            return []
        valid: List[Dict] = []
        for action in actions:
            if not isinstance(action, dict):
                continue
            op = action.get("op")
            if op == "swap":
                try:
                    i = int(action.get("i", -1))
                    j = int(action.get("j", -1))
                except Exception:
                    continue
                if 0 <= i < S and 0 <= j < S:
                    valid.append({"op": "swap", "i": i, "j": j})
            elif op == "relocate":
                try:
                    i = int(action.get("i", -1))
                    site_id = int(action.get("site_id", -1))
                except Exception:
                    continue
                if 0 <= i < S and 0 <= site_id < Ns:
                    valid.append({"op": "relocate", "i": i, "site_id": site_id})
            elif op == "cluster_move":
                try:
                    cluster_id = int(action.get("cluster_id", -1))
                    region_id = int(action.get("region_id", -1))
                except Exception:
                    continue
                if 0 <= cluster_id < num_clusters and 0 <= region_id < num_regions:
                    valid.append({"op": "cluster_move", "cluster_id": cluster_id, "region_id": region_id})
        return valid

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
        request_bytes = len(json.dumps(payload))
        S = int(state_summary.get("S", 0))
        Ns = int(state_summary.get("Ns", 0))
        num_clusters = int(state_summary.get("num_clusters", 0))
        num_regions = int(state_summary.get("num_regions", 0))

        for _ in range(self.max_retry + 1):
            try:
                resp = None
                resp = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout_sec,
                )
                base_usage = {
                    "endpoint": self.endpoint,
                    "url": url,
                    "model": self.model,
                    "key_len": len(self.api_key),
                    "request_bytes": request_bytes,
                    "status_code": resp.status_code,
                }
                if resp.status_code != 200:
                    self.last_usage = {**base_usage, "ok": False, "resp": resp.text[:200]}
                    last_error = Exception(f"HTTP {resp.status_code}")
                    continue

                data = resp.json()
                usage = data.get("usage", {})
                msg = (data.get("choices") or [{}])[0].get("message", {})
                raw = (msg.get("content") or "").strip()
                if not raw:
                    raw = (msg.get("reasoning_content") or "").strip()
                raw_preview = raw[:200]

                parsed, parse_error = self.extract_wrapped_json(raw)
                if parse_error:
                    last_error = Exception(parse_error)
                    self.last_usage = {
                        **base_usage,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                        "response_bytes": len(resp.content),
                        "ok": False,
                        "error": parse_error,
                        "raw_preview": raw_preview,
                    }
                    continue

                actions_raw = parsed.get("actions") if isinstance(parsed, dict) else []
                valid_actions = self._validate_actions(actions_raw, S, Ns, num_clusters, num_regions)
                ok = len(valid_actions) > 0
                self.last_usage = {
                    **base_usage,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "response_bytes": len(resp.content),
                    "ok": ok,
                    "raw_preview": raw_preview,
                    "n_actions": len(valid_actions),
                    "error": None if ok else "empty_or_invalid_actions",
                }
                if ok:
                    return valid_actions
                continue
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
                resp_text = resp.text[:200] if resp is not None else None
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
            continue
        if self.last_usage is None:
            self.last_usage = {"ok": False}
        if last_error:
            self.last_usage.setdefault("error", repr(last_error))
        return []
