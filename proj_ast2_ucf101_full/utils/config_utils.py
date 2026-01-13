from __future__ import annotations

from typing import Any


def get_nested(cfg: Any, path: str, default: Any = None) -> Any:
    """Safely read cfg.xxx.yyy with dot-path; works for dict / SimpleNamespace / OmegaConf-like."""
    cur = cfg
    for key in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
        else:
            cur = getattr(cur, key, default)
    return cur


def set_nested(cfg: Any, path: str, value: Any) -> None:
    """Set cfg.xxx.yyy for dict / namespace. Create intermediate dicts when needed."""
    keys = path.split(".")
    cur = cfg
    for k in keys[:-1]:
        if isinstance(cur, dict):
            if k not in cur or cur[k] is None:
                cur[k] = {}
            cur = cur[k]
        else:
            nxt = getattr(cur, k, None)
            if nxt is None:
                setattr(cur, k, {})
                nxt = getattr(cur, k)
            cur = nxt
    last = keys[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        setattr(cur, last, value)
