"""Utility helpers for locating files in HeurAgenix."""
from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def search_file(name: str, problem: str) -> str:
    """Search for data/config/heuristic assets within the HeurAgenix repo."""
    path = Path(name)
    if path.exists():
        return str(path)

    split_dirs = {
        "train_data",
        "validation_data",
        "test_data",
        "smoke_data",
        "evolution_data",
        "llm_config",
    }
    if name in split_dirs:
        candidate_dir = _REPO_ROOT / "data" / problem / name
        if candidate_dir.exists():
            return str(candidate_dir)

    roots = [
        _REPO_ROOT / "src" / "problems" / problem,
        _REPO_ROOT / "src" / "problems" / problem / "heuristics" / "basic_heuristics",
        _REPO_ROOT / "src" / "problems" / problem / "heuristics" / "evolution_heuristics",
        _REPO_ROOT / "src" / "problems" / problem / "prompt",
        _REPO_ROOT / "data" / problem / "test_data",
        _REPO_ROOT / "data" / problem,
    ]

    if name.endswith(".json"):
        roots = [
            _REPO_ROOT / "data" / problem / "test_data",
            _REPO_ROOT / "data" / problem,
        ] + roots

    target = name
    for root in roots:
        if not root.exists():
            continue
        direct = root / target
        if direct.exists():
            return str(direct)
        for hit in root.rglob("*"):
            if hit.is_file() and hit.name == target:
                return str(hit)

    raise FileNotFoundError(f"search_file failed: name={name}, problem={problem}")

def load_heuristic_functions(problem: str, heuristic_dir: str):
    """
    Return dict[name] = function, searching:
      src/problems/{problem}/heuristics/{heuristic_dir}/*.py
    """
    base = _REPO_ROOT / "src" / "problems" / problem / "heuristics" / heuristic_dir
    if not base.exists():
        raise FileNotFoundError(f"heuristic_dir not found: {base}")

    funcs = {}
    for py in sorted(base.glob("*.py")):
        if py.name.startswith("__"):
            continue
        name = py.stem
        spec = importlib.util.spec_from_file_location(f"{problem}.{heuristic_dir}.{name}", str(py))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, name):
            raise RuntimeError(f"{py} missing function {name}")
        funcs[name] = getattr(mod, name)
    return funcs


def load_hyper_heuristic_prompt(problem: str):
    """
    Minimal prompt loader used by llm_selection.
    Prefer base prompt heuristic_selection.txt + problem-specific description.
    """
    base_prompt = (_REPO_ROOT / "src" / "problems" / "base" / "prompt" / "heuristic_selection.txt").read_text(encoding="utf-8")
    prob_desc = (_REPO_ROOT / "src" / "problems" / problem / "prompt" / "problem_description.txt").read_text(encoding="utf-8")
    return prob_desc + "\n\n" + base_prompt


def load_function(name: str, problem: str):
    """Load a heuristic function by name from problems/{problem}/heuristics."""
    heuristics_root = _REPO_ROOT / "src" / "problems" / problem / "heuristics"
    candidates = list(heuristics_root.rglob(f"{name}.py")) + list(heuristics_root.rglob(name))
    if not candidates:
        raise FileNotFoundError(f"heuristic not found: {name} in {heuristics_root}")
    path = candidates[0]
    spec = importlib.util.spec_from_file_location(f"{problem}.heuristics.{path.stem}", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load heuristic module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, path.stem):
        raise AttributeError(f"{path} missing function {path.stem}")
    return getattr(module, path.stem)
