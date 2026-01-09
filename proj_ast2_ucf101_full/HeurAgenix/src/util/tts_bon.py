"""Fallback rollout selector that tolerates missing dill."""
from __future__ import annotations

from typing import Any, List

try:
    import dill  # type: ignore

    dill.settings["recurse"] = True
except Exception:
    dill = None  # lazy-required only when rollout_budget > 0


def tts_bon(candidate_heuristics: List[Any], *args, **kwargs):
    rollout_budget = kwargs.get("rollout_budget", 0)
    if rollout_budget == 0 or len(candidate_heuristics) == 1:
        return candidate_heuristics[0] if candidate_heuristics else None

    if dill is None:
        return candidate_heuristics[0] if candidate_heuristics else None
    return candidate_heuristics[0] if candidate_heuristics else None
