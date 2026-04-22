from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from . import logger, settings


class UtilsWriteHelper:
    _instance: "UtilsWriteHelper | None" = None
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._local = threading.local()
        self.active = True

    @classmethod
    def get_instance(cls) -> "UtilsWriteHelper":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def configure(self, active: bool = True) -> None:
        with self._lock:
            self.active = active

    def _in_reentry(self) -> bool:
        return bool(getattr(self._local, "reentrancy", False))

    def _set_reentry(self, value: bool) -> None:
        self._local.reentrancy = value

    def _should_record(self, file_path: str | None) -> bool:
        if not self.active or file_path is None:
            return False
        cfg = settings.SETTINGS
        base = Path(file_path).name
        if cfg.record_all_writes:
            return Path(base).suffix.lower() in cfg.include_exts
        return base in cfg.file_list_full

    def _caller_info(self) -> dict[str, Any]:
        stack = traceback.extract_stack()
        for frame in reversed(stack[:-1]):
            file_name = frame.filename.replace("\\", "/")
            if any(token in file_name for token in settings.FILTER_FRAMES):
                continue
            return {"file": frame.filename, "line": frame.lineno, "func": frame.name}
        return {"file": "<unknown>", "line": 0, "func": "<unknown>"}

    def _emit(self, file_path: str | None, mode: str, op_kind: str, bytes_written: int, success: bool, error: str | None = None) -> None:
        event: dict[str, Any] = {
            "ts_ms": int(time.time() * 1000),
            "file_path": file_path,
            "basename": Path(file_path).name if file_path else None,
            "op_kind": op_kind,
            "mode": mode,
            "bytes_written": int(bytes_written),
            "success": bool(success),
            "caller": self._caller_info(),
        }
        if error:
            event["error"] = error
        logger.write_jsonl_event(event)

    def process_write(self, file_path: str | None, mode: str, op_kind: str, bytes_written: int, native_call: Callable[[], Any]) -> Any:
        if self._in_reentry() or not self.active:
            return native_call()
        if not self._should_record(file_path):
            return native_call()
        with self._lock:
            self._set_reentry(True)
            try:
                result = native_call()
                self._emit(file_path, mode, op_kind, bytes_written, True)
                return result
            except Exception as exc:
                self._emit(file_path, mode, op_kind, bytes_written, False, error=str(exc))
                raise
            finally:
                self._set_reentry(False)


def get_helper() -> UtilsWriteHelper:
    return UtilsWriteHelper.get_instance()
