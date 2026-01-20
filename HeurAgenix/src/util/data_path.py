from __future__ import annotations

from pathlib import Path
import os


def _repo_root() -> Path:
    # this file: HeurAgenix/src/util/data_path.py
    # parents[0]=util, [1]=src, [2]=HeurAgenix
    # NOTE: Path.parents is 1-based in meaning; here:
    #   data_path.py -> util -> src -> HeurAgenix
    return Path(__file__).resolve().parents[2]


def get_data_path() -> Path:
    """
    Official HeurAgenix expects data under:
      <repo_root>/data/{problem}/(train_data, validation_data, test_data, smoke_data)

    Allow overriding by env var AMLT_DATA_DIR (set by wrapper).
    """
    base = os.environ.get("AMLT_DATA_DIR")
    if base:
        return Path(base).expanduser().resolve()

    repo_root = _repo_root()
    cand = repo_root / "data"
    if cand.exists():
        return cand.resolve()

    # Fallback: search upward from CWD
    cwd = Path.cwd().resolve()
    for p in [cwd] + list(cwd.parents):
        d = p / "data"
        if d.exists():
            return d.resolve()

    # Last resort: return the canonical path even if not created yet
    return cand.resolve()
