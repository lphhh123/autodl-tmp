"""Load LLM clients from config files compatible with HeurAgenix README."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


class OpenAICompatibleClient:
    def __init__(
        self,
        provider: str,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        timeout_sec: int = 90,
        max_retry: int = 1,
        include_model: bool = True,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.timeout_sec = timeout_sec
        self.max_retry = max_retry
        self.include_model = include_model
        self.last_usage: Optional[Dict] = None
        self.calls: List[Dict] = []

    def is_ready(self) -> bool:
        return bool(self.base_url and self.model)

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _build_payload(self, state_summary: Dict, k: int) -> Dict:
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
        state_json = json.dumps(state_summary, separators=(",", ":"))
        user_content = (
            f"K={int(k)}\n"
            f"STATE_JSON={state_json}\n"
            "REMINDER:\n"
            "- Output ONLY 3 lines wrapper.\n"
            "- Use ONLY candidate_ids and avoid forbidden_ids.\n"
            "- Pick up to K IDs, unique.\n"
        )
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 96,
            "temperature": 0.15,
            "top_p": 0.9,
        }
        if self.include_model:
            payload["model"] = self.model
        return payload

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        if not text:
            return None
        start_candidates = [idx for idx in (text.find("["), text.find("{")) if idx != -1]
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
    def _extract_wrapped_json(raw: str) -> Tuple[Optional[dict], Optional[str]]:
        b = raw.find("BEGIN_JSON")
        e = raw.find("END_JSON")
        if b < 0 or e < 0 or e <= b:
            return None, "missing_wrapper"
        inner = raw[b + len("BEGIN_JSON") : e].strip()
        try:
            obj = json.loads(inner)
        except Exception as exc:  # noqa: BLE001
            return None, f"bad_json_in_wrapper:{exc!r}"
        return obj, None

    @staticmethod
    def _validate_picks(picks: Optional[List], candidate_ids: List[int], forbidden_ids: List[int], k: int) -> List[int]:
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

    def _parse_picks(self, raw: str, candidate_ids: List[int], forbidden_ids: List[int], k: int) -> Tuple[List[int], Optional[str]]:
        parsed, parse_error = self._extract_wrapped_json(raw)
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
        if not self.is_ready():
            self.last_usage = {"ok": False, "reason": "client_not_ready", "provider": self.provider}
            return []
        url = self.base_url
        headers = self._headers()
        payload = self._build_payload(state_summary, k)
        candidate_ids = state_summary.get("candidate_ids", [])
        forbidden_ids = state_summary.get("forbidden_ids", [])

        last_error = None
        for attempt in range(self.max_retry + 1):
            start = time.perf_counter()
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout_sec)
                duration_ms = int((time.perf_counter() - start) * 1000)
                if resp.status_code != 200:
                    self.last_usage = {
                        "ok": False,
                        "provider": self.provider,
                        "model": self.model,
                        "status_code": resp.status_code,
                        "resp": resp.text[:200],
                        "duration_ms": duration_ms,
                    }
                    last_error = Exception(f"HTTP {resp.status_code}")
                    continue
                data = resp.json()
                msg = (data.get("choices") or [{}])[0].get("message", {})
                raw = (msg.get("content") or "").strip()
                usage = data.get("usage", {}) or {}
                picks, parse_error = self._parse_picks(raw, candidate_ids, forbidden_ids, k)
                ok = bool(picks) and parse_error is None
                self.last_usage = {
                    "ok": ok,
                    "provider": self.provider,
                    "model": self.model,
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "duration_ms": duration_ms,
                    "status_code": resp.status_code,
                    "error": parse_error,
                }
                self.calls.append(dict(self.last_usage))
                if ok:
                    return picks
                last_error = Exception(parse_error or "empty_or_invalid_picks")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self.last_usage = {
                    "ok": False,
                    "provider": self.provider,
                    "model": self.model,
                    "error": repr(exc),
                }
        if last_error and self.last_usage is not None:
            self.last_usage.setdefault("error", repr(last_error))
        return []


def get_llm_client(
    config_path: str,
    timeout_sec: int = 90,
    max_retry: int = 1,
    prompt_dir: str | None = None,
    output_dir: str | None = None,
) -> Optional[OpenAICompatibleClient]:
    _ = prompt_dir, output_dir
    path = Path(config_path)
    if not path.exists():
        return None
    config = json.loads(path.read_text(encoding="utf-8"))
    provider = (config.get("provider") or config.get("type") or config.get("mode") or "api_model").lower()
    model = config.get("model") or config.get("model_name") or config.get("deployment") or ""
    api_key = config.get("api_key") or config.get("key") or config.get("token")

    if provider in {"azure", "azure_gpt"}:
        endpoint = (config.get("endpoint") or config.get("base_url") or "").rstrip("/")
        deployment = config.get("deployment") or model
        api_version = config.get("api_version", "2024-02-15-preview")
        if endpoint and deployment:
            url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
            return OpenAICompatibleClient(
                provider="azure_gpt",
                base_url=url,
                model=str(deployment),
                api_key=api_key,
                timeout_sec=timeout_sec,
                max_retry=max_retry,
                include_model=False,
            )
        return None

    if provider in {"local_model", "vllm"}:
        base_url = (config.get("base_url") or config.get("endpoint") or "http://localhost:8000/v1").rstrip("/")
        url = f"{base_url}/chat/completions"
        return OpenAICompatibleClient(
            provider=provider,
            base_url=url,
            model=str(model or "local-model"),
            api_key=api_key,
            timeout_sec=timeout_sec,
            max_retry=max_retry,
        )

    base_url = (config.get("base_url") or config.get("endpoint") or "").rstrip("/")
    if base_url:
        url = f"{base_url}/chat/completions"
        return OpenAICompatibleClient(
            provider=provider,
            base_url=url,
            model=str(model or "api-model"),
            api_key=api_key,
            timeout_sec=timeout_sec,
            max_retry=max_retry,
        )
    return None
