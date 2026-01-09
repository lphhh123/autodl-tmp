# HeurAgenix/src/util/output_dir.py
from pathlib import Path
import os


def get_output_dir() -> Path:
    """
    Output base dir. Wrapper will set AMLT_OUTPUT_DIR=<some_dir>.
    We always append /output to avoid nested output/output.
    """
    base = os.environ.get("AMLT_OUTPUT_DIR", None)
    if base:
        return Path(base) / "output"
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "output"
