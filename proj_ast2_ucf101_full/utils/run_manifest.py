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
    Lightweight snapshot hash: sha256 over (relpath|sha256(file_content)) of *.py/*.yaml/*.yml/*.json/*.md
    """
    pats = ["**/*.py", "**/*.yaml", "**/*.yml", "**/*.json", "**/*.md"]
    items = []
    for pat in pats:
        for fp in root.glob(pat):
            if not fp.is_file():
                continue
            if any(x in fp.parts for x in ("outputs", "output", "data", "__pycache__")):
                continue
            items.append(f"{fp.relative_to(root)}|{_sha256_file(fp)}")
    items.sort()
    return _sha256_text("\n".join(items))


def write_run_manifest(
    out_dir: str,
    cfg_path: str,
    cfg_hash: str,
    seed: Optional[int] = None,
    stable_hw_state: Optional[Dict[str, Any]] = None,
    cfg: Optional[Any] = None,
    extra: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
    spec_version: str = "v5.4",
    command: Optional[str] = None,
    code_root: Optional[str] = None,
    seed_id: Optional[int] = None,
    git_sha: Optional[str] = None,
    metrics_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir_p / "run_manifest.json"

    if seed is None:
        seed = seed_id if seed_id is not None else 0
    if seed_id is None:
        seed_id = int(seed)

    run_id = str(run_id or uuid.uuid4())
    created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    cfg_text = Path(cfg_path).read_text(encoding="utf-8")
    cfg_sha256 = hashlib.sha256(cfg_text.encode("utf-8")).hexdigest()

    repo_root = Path(code_root).resolve() if code_root else out_dir_p.resolve()
    try:
        for parent in [repo_root] + list(repo_root.parents):
            if (parent / "utils").exists() and (parent / "scripts").exists():
                repo_root = parent
                break
    except Exception:
        pass

    commit = None
    try:
        commit = get_git_commit_or_version(str(repo_root), fallback="code_only")
    except Exception:
        commit = None
    if (not commit) and git_sha:
        commit = git_sha
    if not commit:
        commit = "unknown"

    code_snapshot = None
    try:
        if commit in ("code_only", "unknown"):
            code_snapshot = _code_snapshot_hash(repo_root)
    except Exception:
        code_snapshot = None

    manifest: Dict[str, Any] = {
        "spec_version": spec_version,
        "run_id": run_id,
        "created_at": created_at,
        "cmd": command or " ".join(sys.argv),
        "cwd": os.getcwd(),
        "python": sys.version,
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
        "git": {"commit": commit},
        "code_snapshot_hash": code_snapshot,
        "seed": int(seed),
        "seed_id": int(seed_id),
        "cfg_path": str(cfg_path),
        "cfg_hash": str(cfg_hash),
        "cfg_sha256": cfg_sha256,
        "stable_hw_state": stable_hw_state or {},
        "locked_acc_ref_value": (stable_hw_state or {}).get("acc_ref"),
        "locked_acc_ref_source": (stable_hw_state or {}).get("acc_ref_source"),
        "metrics_summary": metrics_summary or {},
        "extra_meta": extra or {},
    }
    st = stable_hw_state or {}
    manifest["stable_hw_acc_ref"] = st.get("acc_ref", None)
    manifest["stable_hw_acc_ref_source"] = st.get("acc_ref_source", None)
    manifest["stable_hw_acc_ref_locked"] = bool(st.get("acc_ref_locked", False))
    manifest["stable_hw_epsilon_drop"] = float(st.get("epsilon_drop", 0.0) or 0.0)
    no_drift_requested = bool(st.get("no_drift_requested", False))
    no_drift_effective = bool(st.get("no_drift_effective", False))
    if "no_drift_requested" not in st and cfg is not None:
        no_drift_requested = bool(getattr(getattr(cfg, "no_drift", None), "enabled", False))
    if "no_drift_effective" not in st and cfg is not None:
        no_drift_effective = bool(getattr(getattr(cfg, "no_drift", None), "enabled", False))
    manifest["stable_hw_no_drift_requested"] = no_drift_requested
    manifest["stable_hw_no_drift_effective"] = no_drift_effective
    manifest["stable_hw_ref_update_mode"] = st.get("ref_update_mode", "DISABLED")
    manifest["stable_hw_hw_ref_source"] = st.get("hw_ref_source", None)

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
