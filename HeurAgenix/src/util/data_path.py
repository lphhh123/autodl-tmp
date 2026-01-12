from pathlib import Path
import os


def get_data_path() -> Path:
    """
    Resolve data path for all tasks:
      1. If AMLT_DATA_DIR is set, use it.
      2. Else search ../../data relative to this file.
    """
    base = os.environ.get("AMLT_DATA_DIR")
    if base:
        return Path(base).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    data_path = repo_root / "data"
    if not data_path.exists():
        data_path = Path.cwd() / "data"
    return data_path.resolve()
