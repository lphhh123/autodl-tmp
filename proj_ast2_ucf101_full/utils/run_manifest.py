from __future__ import annotations

import json
import time
import uuid
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from .git_version import get_git_commit_or_version


def _sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _code_snapshot_hash(root: Path) -> str:
    """
    Lightweight snapshot hash: sha256 over (relpath|size|mtime) of *.py/*.yaml/*.yml/*.json/*.md
    Avoids reading full contents; stable enough for drift detection.
    """
    pats = ["**/*.py", "**/*.yaml", "**/*.yml", "**/*.json", "**/*.md"]
    items = []
    for pat in pats:
        for fp in root.glob(pat):
            if not fp.is_file():
                continue
            if any(x in fp.parts for x in ("outputs", "output", "data", "__pycache__")):
                continue
            st = fp.stat()
            items.append(f"{fp.relative_to(root)}|{st.st_size}|{int(st.st_mtime)}")
    items.sort()
    return _sha256_text("\n".join(items))


def write_run_manifest(
    out_dir: str,
    cfg_path: str,
    cfg_hash: str,
    seed: int,
    stable_hw_state: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    spec_version: str = "v5.4",
    command: Optional[str] = None,
    code_root: Optional[str] = None,
) -> Dict[str, Any]:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir_p / "run_manifest.json"

    rid = str(run_id or uuid.uuid4())

    resolved_cfg = out_dir_p / "config_resolved.yaml"
    if resolved_cfg.exists():
        cfg_sha256 = _sha256_file(resolved_cfg)
    else:
        cfg_sha256 = _sha256_file(Path(cfg_path))

    repo_root = Path(code_root).resolve() if code_root else out_dir_p.resolve()
    try:
        for parent in [repo_root] + list(repo_root.parents):
            if (parent / "utils").exists() and (parent / "scripts").exists():
                repo_root = parent
                break
    except Exception:
        pass

    git_commit = get_git_commit_or_version(str(repo_root), fallback="code_only")
    snapshot = _code_snapshot_hash(repo_root)

    locked = {
        "acc_ref_value": stable_hw_state.get("acc_ref", None),
        "acc_ref_source": stable_hw_state.get("acc_ref_source", None),
        "acc_ref_from_run": stable_hw_state.get("acc_ref_from_run", None),
        "epsilon_drop": stable_hw_state.get("epsilon_drop", None),
    }

    units = (extra or {}).get("units", {"time_unit": "ms", "dist_unit": "mm"})

    manifest: Dict[str, Any] = {
        "schema_version": str(spec_version),
        "run_id": rid,
        "timestamp": int(time.time()),
        "cfg_path": str(cfg_path),
        "cfg_hash": str(cfg_hash),
        "cfg_sha256": str(cfg_sha256),
        "seed": int(seed),
        "git_commit": str(git_commit),
        "code_snapshot_hash": str(snapshot),
        "command": str(command or ""),
        "locked_acc_ref": locked,
        "stable_hw_state": stable_hw_state or {},
        "units": units,
        "extra": extra or {},
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
