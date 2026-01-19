import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any


def _sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def _try_git_info(repo_root: Path) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root)).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_root)).decode().strip()
        return {"git_commit": commit, "dirty": bool(dirty)}
    except Exception:
        return {"git_commit": None, "dirty": None}


def write_run_manifest(
    *,
    out_dir: str | Path,
    cfg_resolved_text: str,
    cfg_path: str | None,
    argv: list[str],
    seed: int,
    spec_version: str = "v5.4",
    extra: dict[str, Any] | None = None,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    cfg_sha256 = _sha256_text(cfg_resolved_text)

    repo_root = out_dir
    # heuristic: project root is two levels above scripts; allow caller to override via extra
    if extra and extra.get("repo_root"):
        repo_root = Path(extra["repo_root"])

    git_info = _try_git_info(repo_root)

    manifest = {
        "run_id": run_id,
        "spec_version": spec_version,
        "cfg_path": cfg_path,
        "cfg_sha256": cfg_sha256,
        "command": argv,
        "seed": int(seed),
        **git_info,
        "locked_acc_ref": {},
    }

    if extra:
        manifest.update(extra)

    with (out_dir / "run_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest
