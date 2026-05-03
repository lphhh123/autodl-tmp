from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .helper import get_helper
from .mode import is_binary_mode


class _BaseFileProxy:
    __io_compat_proxy__ = True

    def __init__(self, orig: Any, file_path: str | None, mode: str) -> None:
        self._orig = orig
        self._file_path = str(Path(file_path).resolve()) if file_path else None
        self._mode = mode or ""
        self._helper = get_helper()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)

    def __iter__(self):
        return iter(self._orig)

    def __enter__(self):
        self._orig.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._orig.__exit__(exc_type, exc, tb)

    @property
    def name(self):
        return getattr(self._orig, "name", None)

    @property
    def encoding(self):
        return getattr(self._orig, "encoding", None)

    @property
    def closed(self):
        return getattr(self._orig, "closed", None)

    def fileno(self):
        return self._orig.fileno()

    def flush(self):
        return self._orig.flush()

    def close(self):
        return self._orig.close()

    def tell(self):
        return self._orig.tell()

    def seek(self, offset: int, whence: int = 0):
        return self._orig.seek(offset, whence)


class TextFileProxy(_BaseFileProxy):
    def write(self, s: str) -> Any:
        size = len((s or "").encode("utf-8"))
        return self._helper.process_write(
            self._file_path,
            self._mode,
            "file.write",
            size,
            native_call=lambda: self._orig.write(s),
        )

    def writelines(self, lines: Iterable[str]) -> Any:
        cached = list(lines)
        size = sum(len((item or "").encode("utf-8")) for item in cached)
        return self._helper.process_write(
            self._file_path,
            self._mode,
            "file.writelines",
            size,
            native_call=lambda: self._orig.writelines(cached),
        )


class BinaryFileProxy(_BaseFileProxy):
    def write(self, b: bytes) -> Any:
        data = bytes(b or b"")
        size = len(data)
        return self._helper.process_write(
            self._file_path,
            self._mode,
            "file.write",
            size,
            native_call=lambda: self._orig.write(b),
        )

    def writelines(self, lines: Iterable[bytes]) -> Any:
        cached = list(lines)
        size = sum(len(bytes(item or b"")) for item in cached)
        return self._helper.process_write(
            self._file_path,
            self._mode,
            "file.writelines",
            size,
            native_call=lambda: self._orig.writelines(cached),
        )


def wrap_file_object(orig: Any, file_path: str | None, mode: str) -> Any:
    if getattr(orig, "__io_compat_proxy__", False):
        return orig
    if is_binary_mode(mode):
        return BinaryFileProxy(orig, file_path, mode)
    return TextFileProxy(orig, file_path, mode)
