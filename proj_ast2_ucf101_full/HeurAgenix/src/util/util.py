"""Utility helpers for locating files in HeurAgenix."""
from __future__ import annotations

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
