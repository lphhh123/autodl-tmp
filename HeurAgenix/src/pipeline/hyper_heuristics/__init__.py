"""
Hyper-heuristic implementations.

We provide lazy exports to avoid importing heavy deps at module import time.
This also keeps compatibility with:
  from pipeline.hyper_heuristics import RandomHyperHeuristic
when HeurAgenix/src is on sys.path.
"""
from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # only for type checkers; no runtime heavy import
    from .random import RandomHyperHeuristic
    from .llm_selection import LLMSelectionHyperHeuristic
    from .single import SingleHyperHeuristic

__all__ = [
    "RandomHyperHeuristic",
    "LLMSelectionHyperHeuristic",
    "SingleHyperHeuristic",
]


def __getattr__(name: str):
    if name == "RandomHyperHeuristic":
        return import_module(".random", __name__).RandomHyperHeuristic
    if name == "LLMSelectionHyperHeuristic":
        return import_module(".llm_selection", __name__).LLMSelectionHyperHeuristic
    if name == "SingleHyperHeuristic":
        return import_module(".single", __name__).SingleHyperHeuristic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
