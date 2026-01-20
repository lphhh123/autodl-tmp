from __future__ import annotations

from pathlib import Path
import os


def _repo_root() -> Path:
    # this file: HeurAgenix/src/util/output_dir.py
    # output_dir.py -> util -> src -> HeurAgenix
    return Path(__file__).resolve().parents[2]


def get_output_dir() -> Path:
    """
    Official HeurAgenix outputs under:
      <output_base>/output/{problem}/{test_data}/{result_dir}/{engine}/...

    Wrapper will set AMLT_OUTPUT_DIR=<base_dir>. We append /output unless it already ends with /output.
    This keeps compatibility with guides that refer to ./output as the canonical root.
    """
    base = os.environ.get("AMLT_OUTPUT_DIR")
    if base:
        p = Path(base).expanduser().resolve()
        return p if p.name == "output" else (p / "output")

    repo_root = _repo_root()
    return (repo_root / "output").resolve()
