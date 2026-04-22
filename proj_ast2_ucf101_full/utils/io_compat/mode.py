from __future__ import annotations


def is_write_mode(mode: str | None) -> bool:
    m = (mode or "r")
    return any(flag in m for flag in ("w", "a", "+", "x"))


def is_binary_mode(mode: str | None) -> bool:
    return "b" in (mode or "")


def is_append_mode(mode: str | None) -> bool:
    return "a" in (mode or "")
