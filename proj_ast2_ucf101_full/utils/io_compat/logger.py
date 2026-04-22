from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

_ORIGINAL_OPEN: Callable[..., Any] | None = None
_LOG_PATH: Path | None = None


def configure_logger(log_path: Path, original_open: Callable[..., Any]) -> None:
    global _ORIGINAL_OPEN, _LOG_PATH
    _ORIGINAL_OPEN = original_open
    _LOG_PATH = log_path


def clear_logger() -> None:
    global _ORIGINAL_OPEN, _LOG_PATH
    _ORIGINAL_OPEN = None
    _LOG_PATH = None


def write_jsonl_event(event: dict[str, Any]) -> None:
    if _ORIGINAL_OPEN is None or _LOG_PATH is None:
        return
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        with _ORIGINAL_OPEN(_LOG_PATH, "a", encoding="utf-8") as fp:
            fp.write(payload)
    except Exception:
        return
