"""Heuristic loader for HeurAgenix problems."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Callable, Dict


def _resolve_heuristic_dir(heuristic_dir: str, problem: str) -> Path:
    candidate = Path(heuristic_dir)
    if candidate.exists():
        return candidate
    base_dir = Path(__file__).resolve().parents[1] / "problems" / problem / "heuristics"
    return base_dir / heuristic_dir


def get_heuristic(heuristic_dir: str, problem: str) -> Dict[str, Callable]:
    base_dir = _resolve_heuristic_dir(heuristic_dir, problem)
    if not base_dir.exists():
        raise FileNotFoundError(f"Heuristic directory not found: {base_dir}")
    heuristics: Dict[str, Callable] = {}
    for py_file in sorted(base_dir.glob("*.py")):
        if py_file.name.startswith("__"):
            continue
        name = py_file.stem
        spec = importlib.util.spec_from_file_location(f"heuristics.{problem}.{name}", py_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load heuristic module: {py_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        if not hasattr(module, name):
            raise AttributeError(f"Missing heuristic function '{name}' in {py_file}")
        heuristics[name] = getattr(module, name)
    if not heuristics:
        raise RuntimeError(f"No heuristics found in {base_dir}")
    return heuristics
