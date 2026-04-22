from __future__ import annotations

import builtins
import io
from pathlib import Path
from typing import Any

from .helper import get_helper
from .logger import clear_logger, configure_logger
from .mode import is_write_mode
from .proxy import wrap_file_object
from .settings import update_settings
from . import settings

_ORIGINAL_BUILTINS_OPEN = builtins.open
_ORIGINAL_IO_OPEN = io.open
_ORIGINAL_PATH_OPEN = Path.open
_ORIGINAL_PATH_WRITE_TEXT = Path.write_text
_ORIGINAL_PATH_WRITE_BYTES = Path.write_bytes
_PATCHED = False


def _normalize_path(file_arg: Any) -> str | None:
    try:
        if file_arg is None:
            return None
        return str(Path(file_arg).resolve())
    except Exception:
        return str(file_arg)


def _patched_builtins_open(file, mode="r", *args, **kwargs):
    obj = _ORIGINAL_BUILTINS_OPEN(file, mode, *args, **kwargs)
    if not is_write_mode(mode):
        return obj
    return wrap_file_object(obj, _normalize_path(file), mode)


def _patched_io_open(file, mode="r", *args, **kwargs):
    obj = _ORIGINAL_IO_OPEN(file, mode, *args, **kwargs)
    if not is_write_mode(mode):
        return obj
    return wrap_file_object(obj, _normalize_path(file), mode)




def _patched_path_open(self: Path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
    obj = _ORIGINAL_PATH_OPEN(self, mode=mode, buffering=buffering, encoding=encoding, errors=errors, newline=newline)
    if not is_write_mode(mode):
        return obj
    return wrap_file_object(obj, str(self.resolve()), mode)


def _patched_write_text(self: Path, data: str, encoding=None, errors=None, newline=None):
    helper = get_helper()
    enc = encoding or "utf-8"
    size = len((data or "").encode(enc, errors="strict"))
    return helper.process_write(
        str(self.resolve()),
        "w",
        "path.write_text",
        size,
        native_call=lambda: _ORIGINAL_PATH_WRITE_TEXT(
            self, data, encoding=encoding, errors=errors, newline=newline
        ),
    )


def _patched_write_bytes(self: Path, data: bytes):
    helper = get_helper()
    size = len(data or b"")
    return helper.process_write(
        str(self.resolve()),
        "wb",
        "path.write_bytes",
        size,
        native_call=lambda: _ORIGINAL_PATH_WRITE_BYTES(self, data),
    )


def install_patches(config: dict[str, Any] | None = None) -> None:
    global _PATCHED
    update_settings(config)
    configure_logger(settings.SETTINGS.log_path, _ORIGINAL_BUILTINS_OPEN)
    get_helper().configure(active=True)
    if _PATCHED:
        return
    builtins.open = _patched_builtins_open
    io.open = _patched_io_open
    Path.open = _patched_path_open
    Path.write_text = _patched_write_text
    Path.write_bytes = _patched_write_bytes
    _PATCHED = True


def uninstall_patches() -> None:
    global _PATCHED
    get_helper().configure(active=False)
    if not _PATCHED:
        clear_logger()
        return
    builtins.open = _ORIGINAL_BUILTINS_OPEN
    io.open = _ORIGINAL_IO_OPEN
    Path.open = _ORIGINAL_PATH_OPEN
    Path.write_text = _ORIGINAL_PATH_WRITE_TEXT
    Path.write_bytes = _ORIGINAL_PATH_WRITE_BYTES
    clear_logger()
    _PATCHED = False
