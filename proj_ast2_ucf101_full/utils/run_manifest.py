from __future__ import annotations

import json
import os
import platform
import sys
import time
import uuid
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional

from .git_version import get_git_commit_or_version
from .stable_hash import stable_hash


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

    run_id = str(run_id or uuid.uuid4())
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # ---- compute hashes ----
    cfg_text = Path(cfg_path).read_text(encoding="utf-8")
    cfg_sha256 = hashlib.sha256(cfg_text.encode("utf-8")).hexdigest()
    cfg_hash_legacy = stable_hash(cfg_text)

    repo_root = Path(code_root).resolve() if code_root else out_dir_p.resolve()
    try:
        for parent in [repo_root] + list(repo_root.parents):
            if (parent / "utils").exists() and (parent / "scripts").exists():
                repo_root = parent
                break
    except Exception:
        pass

    git_info = {"commit": get_git_commit_or_version(str(repo_root), fallback="code_only")}

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "created_at": created_at,
        "cmd": command or " ".join(sys.argv),
        "cwd": os.getcwd(),
        "git": git_info,
        "python": sys.version,
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "seed": int(seed),
        "cfg_path": str(cfg_path),
        # v5.4 canonical: sha256 must be real sha256
        "cfg_sha256": cfg_sha256,
        # keep legacy stable-hash for backward compatibility
        "cfg_hash_legacy": cfg_hash_legacy,
        # keep explicit alias for resolved yaml content (same input here)
        "cfg_sha256_resolved": cfg_sha256,
        "stable_hw_state": stable_hw_state or {},
        "locked_acc_ref_value": (stable_hw_state or {}).get("acc_ref"),
        "locked_acc_ref_source": (stable_hw_state or {}).get("acc_ref_source"),
        "extra_meta": extra or {},
    }

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
