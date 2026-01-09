"""Load LLM clients from config files compatible with HeurAgenix README."""
from __future__ import annotations

import json
import time
import os
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
        temperature: float = 0.15,
        top_p: float = 0.9,
        max_tokens: int = 96,
    ) -> None:
        self.provider = provider
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or ""
        self.timeout_sec = timeout_sec
        self.max_retry = max_retry
        self.include_model = include_model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
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
            "max_tokens": int(self.max_tokens),
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
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

    def propose_pick(self, state_summary: Dict, k: int, timeout_s: int | None = None) -> List[int]:
        if not self.is_ready():
            self.last_usage = {"ok": False, "reason": "client_not_ready", "provider": self.provider}
            return []
        url = self.base_url
        headers = self._headers()
        payload = self._build_payload(state_summary, k)
        candidate_ids = state_summary.get("candidate_ids", [])
        forbidden_ids = state_summary.get("forbidden_ids", [])

        timeout = self.timeout_sec if timeout_s is None else int(timeout_s)
        last_error = None
        for attempt in range(self.max_retry + 1):
            start = time.perf_counter()
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
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
    model = config.get("model") or config.get("model_name") or config.get("deployment") or config.get("model_path") or ""
    max_attempts = int(config.get("max_attempts", max_retry + 1))
    max_retry = max(0, max_attempts - 1)
    temperature = float(config.get("temperature", 0.15))
    top_p = float(config.get("top-p", config.get("top_p", 0.9)))
    max_tokens = int(config.get("max_tokens", 96))

    def _resolve_api_key(env_name: str) -> str:
        api_key = config.get("api_key") or config.get("key") or config.get("token")
        if api_key:
            return str(api_key)
        env_key = os.getenv(env_name)
        if env_key:
            return env_key
        raise ValueError(f"missing api_key ({env_name})")

    if provider in {"azure", "azure_gpt"}:
        endpoint = (config.get("azure_endpoint") or config.get("api_base") or config.get("endpoint") or "").rstrip("/")
        api_version = config.get("api_version", "2024-02-15-preview")
        if not endpoint:
            raise ValueError("missing azure_endpoint")
        if not model:
            raise ValueError("missing model")
        api_key = _resolve_api_key("AZURE_OPENAI_API_KEY")
        url = f"{endpoint}/openai/deployments/{model}/chat/completions?api-version={api_version}"
        return OpenAICompatibleClient(
            provider="azure_gpt",
            base_url=url,
            model=str(model),
            api_key=api_key,
            timeout_sec=timeout_sec,
            max_retry=max_retry,
            include_model=False,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    if provider in {"local_model", "vllm"}:
        base_url = (config.get("base_url") or config.get("endpoint") or "http://localhost:8000/v1").rstrip("/")
        api_key = config.get("api_key", "")
        url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
        return OpenAICompatibleClient(
            provider=provider,
            base_url=url,
            model=str(model or "local-model"),
            api_key=str(api_key),
            timeout_sec=timeout_sec,
            max_retry=max_retry,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    if provider in {"api_model"}:
        base_url = (config.get("url") or config.get("base_url") or config.get("endpoint") or "").rstrip("/")
        if not base_url:
            raise ValueError("missing url")
        api_key = _resolve_api_key("OPENAI_API_KEY")
        url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
        return OpenAICompatibleClient(
            provider=provider,
            base_url=url,
            model=str(model or "api-model"),
            api_key=api_key,
            timeout_sec=timeout_sec,
            max_retry=max_retry,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    base_url = (config.get("base_url") or config.get("endpoint") or "").rstrip("/")
    if base_url:
        url = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
        api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY", "")
        return OpenAICompatibleClient(
            provider=provider,
            base_url=url,
            model=str(model or "api-model"),
            api_key=str(api_key),
            timeout_sec=timeout_sec,
            max_retry=max_retry,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    return None
