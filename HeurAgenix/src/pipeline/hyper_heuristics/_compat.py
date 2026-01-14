from __future__ import annotations

from typing import Any


def env_continue_run(env: Any) -> bool:
    v = getattr(env, "continue_run", True)
    try:
        return bool(v() if callable(v) else v)
    except Exception:
        return True


def env_is_complete(env: Any) -> bool:
    v = getattr(env, "is_complete_solution", True)
    try:
        return bool(v() if callable(v) else v)
    except Exception:
        return True


def env_is_valid(env: Any) -> bool:
    v = getattr(env, "is_valid_solution", True)
    if callable(v):
        try:
            return bool(v(getattr(env, "current_solution", None)))
        except TypeError:
            return bool(v())
        except Exception:
            return True
    try:
        return bool(v)
    except Exception:
        return True
