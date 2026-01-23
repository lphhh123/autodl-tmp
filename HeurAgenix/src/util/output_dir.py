from __future__ import annotations

from pathlib import Path
import os


def _repo_root() -> Path:
    # this file: HeurAgenix/src/util/output_dir.py
    # output_dir.py -> util -> src -> HeurAgenix
    return Path(__file__).resolve().parents[2]


def get_output_root(repo_root: Path) -> Path:
    """Get output root directory.

    Official HeurAgenix outputs under:
      <output_root>/{problem}/{test_data}/{result_dir}/{engine}/...

    If AMLT_OUTPUT_DIR is set, treat it as the output root directory.
    """
    base = os.environ.get("AMLT_OUTPUT_DIR", None)
    if base:
        return Path(base).expanduser().resolve()
    return (repo_root / "output").resolve()


def get_output_dir() -> Path:
    return get_output_root(_repo_root())
