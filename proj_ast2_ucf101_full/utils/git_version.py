from __future__ import annotations

import subprocess
from pathlib import Path


def get_git_commit_or_version(repo_root: str, fallback: str = "v5.4") -> str:
    root = Path(repo_root)
    try:
        # If git is unavailable or .git missing, fall back.
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL)
        s = out.decode("utf-8").strip()
        if s:
            return s
    except Exception:
        pass
    return fallback
