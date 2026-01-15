from __future__ import annotations

from typing import Any


def env_continue_run(env: Any) -> bool:
    val = getattr(env, "continue_run", None)
    if callable(val):
        try:
            return bool(val())
        except TypeError:
            return bool(val)
    if isinstance(val, bool):
        return val

    cur = getattr(env, "current_steps", None)
    mx = getattr(env, "max_steps", None)
    if isinstance(cur, int) and isinstance(mx, int) and mx > 0:
        return cur < mx

    return False


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
