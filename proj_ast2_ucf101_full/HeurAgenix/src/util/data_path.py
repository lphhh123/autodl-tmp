# HeurAgenix/src/util/data_path.py
from pathlib import Path
import os


def get_data_path() -> Path:
    """
    Official README expects data under ./data/{problem}/...
    Allow overriding by env var AMLT_DATA_DIR (if set).
    """
    base = os.environ.get("AMLT_DATA_DIR", None)
    if base:
        return Path(base)
    # repo_root/HeurAgenix
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "data"
