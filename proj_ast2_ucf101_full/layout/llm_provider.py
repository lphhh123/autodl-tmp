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
    def propose_pick(self, state_summary: Dict, k: int) -> List[int]:
        ...

    def propose_picks(self, state_summary: Dict, k: int) -> List[int]:
        return self.propose_pick(state_summary, k)


class HeuristicProvider(LLMProvider):
    def propose_pick(self, state_summary: Dict, k: int) -> List[int]:
        cands = state_summary.get("candidates", [])
        cands_sorted = sorted(cands, key=lambda x: float(x.get("d_total", 0)))
        picks: List[int] = []
        non_swap_ids: List[int] = []
        for cand in cands_sorted:
            cid = int(cand.get("id", -1))
            if cid in state_summary.get("forbidden_ids", []):
                continue
            if cand.get("type") not in ("swap", None):
                non_swap_ids.append(cid)
            picks.append(cid)
            if len(picks) >= k:
                break
        if picks and non_swap_ids and all(
            (next((c for c in cands if int(c.get("id", -1)) == pid), {}).get("type") == "swap")
            for pid in picks
        ):
            picks[-1] = non_swap_ids[0]
        return picks


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
            "STRICT PICK MODE.\n"
            "You MUST output EXACTLY 3 lines and NOTHING ELSE.\n"
            "LINE1: BEGIN_JSON\n"
            "LINE2: {\"pick\":[ID1,ID2,...]}\n"
            "LINE3: END_JSON\n\n"
            "HARD RULES:\n"
            "- The FIRST characters MUST be \"BEGIN_JSON\".\n"
            "- Output JSON must have ONLY key \"pick\".\n"
            "- Each ID must be an integer.\n"
            "- IDs MUST be chosen ONLY from candidate_ids.\n"
            "- IDs MUST NOT appear in forbidden_ids.\n"
            "- IDs must be UNIQUE. Max K IDs.\n"
            "- If you cannot comply, output {\"pick\":[]}.\n\n"
            "OPTIMIZATION GOAL:\n"
            "- Prefer IDs with more negative d_total (best improvement).\n"
            "- Secondary: improve d_comm and d_therm (more negative is better).\n"
            "- DIVERSITY: If candidates include relocate/cluster_move, ensure at least ONE picked ID is NOT swap.\n\n"
            "ANTI-TEMPLATE:\n"
            "- Do NOT always pick the smallest IDs.\n"
            "- Do NOT always pick the same pattern (e.g., 0-1,1-2,2-3 swaps).\n"
            "- Use the provided candidate list only.\n"
        )

        if repair_raw is None:
            state_json = json.dumps(state_summary, separators=(",", ":"))
            user_content = (
                f"K={int(k)}\n"
                f"STATE_JSON={state_json}\n"
                "REMINDER:\n"
                "- Output ONLY 3 lines wrapper.\n"
                "- Use ONLY candidate_ids and avoid forbidden_ids.\n"
                "- Pick up to K IDs, unique.\n"
            )
        else:
            user_content = (
                "REPAIR: Convert RAW text to the 3-line wrapper. Do not change ids.\n"
                "RAW_TEXT:\nBEGIN_RAW\n"
                f"{repair_raw}\nEND_RAW"
            )

        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_completion_tokens": int(min(self.max_tokens, 128)),
            "temperature": 0.15,
            "top_p": 0.9,
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

    def _validate_picks(self, picks: Optional[List], candidate_ids: List[int], forbidden_ids: List[int], k: int) -> List[int]:
        if not isinstance(picks, list):
            return []
        allowed = set(int(x) for x in candidate_ids)
        forb = set(int(x) for x in forbidden_ids)
        clean: List[int] = []
        for p in picks:
            try:
                pid = int(p)
            except Exception:
                continue
            if pid in forb or pid not in allowed or pid in clean:
                continue
            clean.append(pid)
            if len(clean) >= k:
                break
        return clean

    def _parse_picks(
        self,
        raw: str,
        candidate_ids: List[int],
        forbidden_ids: List[int],
        k: int,
    ) -> Tuple[List[int], Optional[str]]:
        parsed, parse_error = self.extract_wrapped_json(raw)
        if parsed is None:
            snippet = self._extract_json(raw)
            if snippet:
                try:
                    parsed = json.loads(snippet)
                    parse_error = None
                except Exception:  # noqa: BLE001
                    return [], parse_error or "missing_wrapper"
            else:
                return [], parse_error or "missing_wrapper"
        picks_raw = parsed.get("pick") if isinstance(parsed, dict) else []
        valid_picks = self._validate_picks(picks_raw, candidate_ids, forbidden_ids, k)
        if valid_picks:
            return valid_picks, None
        return [], "empty_or_invalid_picks"

    def propose_pick(self, state_summary: Dict, k: int) -> List[int]:
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
        candidate_ids = state_summary.get("candidate_ids", [])
        forbidden_ids = state_summary.get("forbidden_ids", [])
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

                valid_picks, parse_error = self._parse_picks(raw, candidate_ids, forbidden_ids, k)
                ok = len(valid_picks) > 0 and parse_error is None
                self.last_usage = {
                    **base_usage,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "response_bytes": len(resp.content),
                    "ok": ok,
                    "raw_preview": raw_preview,
                    "n_pick": len(valid_picks),
                    "error": parse_error,
                    "status_code": resp.status_code,
                }
                if ok:
                    return valid_picks

                if (parse_error in {"missing_wrapper", "no_json_found", "bad_json_in_wrapper"}) and not repair_attempted:
                    repair_attempted = True
                    repair_raw = raw[:1000]
                    payload = self._build_payload(state_summary, k, repair_raw=repair_raw)
                    continue
                last_error = Exception(parse_error or "empty_or_invalid_picks")
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
