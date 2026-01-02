"""LLM planner interface and Volc Ark implementation (SPEC v4.3.2 §9)."""
from __future__ import annotations

import json
import os
import re
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
    def __init__(self, timeout_sec: int = 90, max_retry: int = 1, max_tokens: int = 96):
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

    def _build_payload(self, state_summary: Dict, k: int, repair_raw: Optional[str] = None) -> Dict:
        system_prompt = (
            "STRICT MODE. OUTPUT MUST START IMMEDIATELY.\n\n"
            "You MUST reply with ONLY the following 3 lines:\n"
            "LINE1: BEGIN_JSON\n"
            "LINE2: {\"actions\":[...]}   (a single JSON object, no extra keys)\n"
            "LINE3: END_JSON\n\n"
            "HARD RULES:\n"
            "- The very first characters of your reply MUST be \"BEGIN_JSON\".\n"
            "- No greetings, no analysis, no explanation, no markdown, no code fences.\n"
            "- JSON top-level MUST be exactly {\"actions\": [...]}\n"
            "- Each action MUST be one of:\n"
            "  {\"op\":\"swap\",\"i\":INT,\"j\":INT}\n"
            "  {\"op\":\"relocate\",\"i\":INT,\"site_id\":INT}\n"
            "  {\"op\":\"cluster_move\",\"cluster_id\":INT,\"region_id\":INT}\n"
            "- Return at most K actions; if none, use {\"actions\":[]}.\n\n"
            "If you violate any rule, you have FAILED. Output the empty list."
        )
        S = int(state_summary.get("S", 0))
        Ns = int(state_summary.get("Ns", 0))
        num_clusters = int(state_summary.get("num_clusters", 0))
        num_regions = int(state_summary.get("num_regions", 0))

        if repair_raw is None:
            state_json = json.dumps(state_summary, separators=(",", ":"))
            user_content = (
                f"K={int(k)}\n"
                "VALID RANGES:\n"
                f"i,j in [0,{S - 1}], site_id in [0,{Ns - 1}], cluster_id in [0,{num_clusters - 1}], region_id in [0,{num_regions - 1}]\n\n"
                "STATE_JSON:\n"
                f"{state_json}\n\n"
                "Return ONLY the 3-line wrapped JSON. Start with BEGIN_JSON."
            )
        else:
            user_content = (
                "REPAIR TASK:\n"
                "Convert the following text into the required 3-line wrapped JSON format.\n"
                "Do NOT add any extra text.\n\n"
                "RAW_TEXT:\n<<<\n"
                f"{repair_raw}\n"
                ">>>"
            )

        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_completion_tokens": int(min(self.max_tokens, 96)),
            "temperature": 0,
            "top_p": 1,
            "stop": ["END_JSON"],
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
    def _recover_json(raw: str) -> Tuple[Optional[dict], Optional[str]]:
        obj_match = re.search(r"\{\s*\"actions\"\s*:\s*\[.*?\]\s*\}", raw, re.S)
        if obj_match:
            try:
                return json.loads(obj_match.group(0)), None
            except Exception:  # noqa: BLE001
                pass

        arr_match = re.search(r"\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]", raw, re.S)
        if arr_match:
            try:
                return {"actions": json.loads(arr_match.group(0))}, None
            except Exception:  # noqa: BLE001
                pass
        return None, "no_json_found"

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

    def _parse_actions(
        self,
        raw: str,
        S: int,
        Ns: int,
        num_clusters: int,
        num_regions: int,
    ) -> Tuple[List[Dict], Optional[str]]:
        parsed, parse_error = self.extract_wrapped_json(raw)
        if parsed is None:
            parsed, parse_error = self._recover_json(raw)

        if parsed is None:
            return [], parse_error or "no_json_found"

        actions_raw = parsed.get("actions") if isinstance(parsed, dict) else []
        valid_actions = self._validate_actions(actions_raw, S, Ns, num_clusters, num_regions)
        if valid_actions:
            return valid_actions, None
        return [], "empty_or_invalid_actions"

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

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        last_error = None
        raw_preview: str = ""
        S = int(state_summary.get("S", 0))
        Ns = int(state_summary.get("Ns", 0))
        num_clusters = int(state_summary.get("num_clusters", 0))
        num_regions = int(state_summary.get("num_regions", 0))
        attempts = self.max_retry + 1
        repair_attempted = False
        payload = self._build_payload(state_summary, k)

        def do_request(body: Dict) -> Tuple[Optional[requests.Response], Dict]:
            resp = requests.post(
                url,
                json=body,
                headers=headers,
                timeout=self.timeout_sec,
            )
            base_usage = {
                "endpoint": self.endpoint,
                "url": url,
                "model": self.model,
                "key_len": len(self.api_key),
                "request_bytes": len(json.dumps(body)),
                "status_code": resp.status_code,
            }
            return resp, base_usage

        for attempt in range(attempts):
            try:
                resp = None
                resp, base_usage = do_request(payload)
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

                valid_actions, parse_error = self._parse_actions(raw, S, Ns, num_clusters, num_regions)
                ok = len(valid_actions) > 0 and parse_error is None
                self.last_usage = {
                    **base_usage,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "response_bytes": len(resp.content),
                    "ok": ok,
                    "raw_preview": raw_preview,
                    "n_actions": len(valid_actions),
                    "error": parse_error,
                }
                if ok:
                    return valid_actions

                if (parse_error in {"missing_wrapper", "no_json_found"}) and not repair_attempted:
                    repair_attempted = True
                    repair_raw = raw[:1000]
                    payload = self._build_payload(state_summary, k, repair_raw=repair_raw)
                    continue
                last_error = Exception(parse_error or "empty_or_invalid_actions")
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
