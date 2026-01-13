from pathlib import Path
import os


def get_output_dir() -> Path:
    """
    Resolve output path:
      1. Prefer AMLT_OUTPUT_DIR (used by wrapper)
      2. Fallback to ../../output
      3. Auto deduplicate 'output/output'
    """
    base = os.environ.get("AMLT_OUTPUT_DIR")
    if base:
        p = Path(base).resolve()
        if p.name == "output":
            return p
        return p / "output"
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / "output").resolve()
