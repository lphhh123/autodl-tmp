"""Fallback rollout selector that tolerates missing dill."""
from __future__ import annotations

from typing import Any, List

try:
    import dill

    dill.settings["recurse"] = True
except Exception:  # noqa: BLE001
    dill = None


def tts_bon(candidate_heuristics: List[Any], *args, **kwargs):
    if dill is None:
        return candidate_heuristics[0] if candidate_heuristics else None
    return candidate_heuristics[0] if candidate_heuristics else None
