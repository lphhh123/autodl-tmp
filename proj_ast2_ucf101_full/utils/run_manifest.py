from __future__ import annotations

import json
from pathlib import Path


def write_run_manifest(
    out_dir: Path,
    cfg_path: str,
    cfg_hash: str,
    run_id: str,
    seed: int,
    command: str = None,
    stable_hw_state: dict = None,
    extra_meta: dict = None,
):
    """
    v5.4 run_manifest per SPEC_D:
      - cfg_sha256 from resolved config text
      - git_commit/dirty or code_snapshot_sha256
      - command, seed, budget_main_axis (if provided), dataset_id (if provided)
    """
    import hashlib, os, subprocess, sys

    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer resolved config if present
    resolved_cfg = out_dir / "config_resolved.yaml"
    cfg_file = resolved_cfg if resolved_cfg.exists() else Path(cfg_path)

    cfg_text = cfg_file.read_text(encoding="utf-8") if cfg_file.exists() else ""
    cfg_sha256 = hashlib.sha256(cfg_text.encode("utf-8")).hexdigest() if cfg_text else None

    # git info (best effort)
    project_root = Path(__file__).resolve().parents[1]
    git_commit = None
    git_dirty = None
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(project_root)).decode().strip()
        dirty_out = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(project_root)).decode().strip()
        git_dirty = bool(dirty_out)
    except Exception:
        git_commit = None
        git_dirty = None

    # code snapshot hash (fallback when git missing)
    def _snapshot_sha256(root: Path) -> str:
        h = hashlib.sha256()
        ex_dirs = {"__pycache__", ".git", "outputs", "output", "data", "datasets", "checkpoints"}
        ex_ext = {".pyc", ".pth", ".pt", ".ckpt"}
        files = []
        for p in root.rglob("*"):
            if p.is_dir() and p.name in ex_dirs:
                continue
            if not p.is_file():
                continue
            if p.suffix in ex_ext:
                continue
            if p.suffix.lower() not in {".py", ".yaml", ".yml", ".md", ".json", ".txt"}:
                continue
            files.append(p)
        for p in sorted(files, key=lambda x: str(x)):
            rel = str(p.relative_to(root)).encode("utf-8")
            h.update(rel)
            h.update(b"\n")
            h.update(p.read_bytes())
            h.update(b"\n")
        return h.hexdigest()

    code_snapshot_sha256 = None if git_commit else _snapshot_sha256(project_root)

    if command is None:
        command = " ".join(sys.argv)

    meta = extra_meta or {}
    manifest = {
        "spec_version": "v5.4",
        "run_id": run_id,
        "seed": int(seed),
        "command": command,
        "cfg_path": str(cfg_file),
        "cfg_hash_legacy": cfg_hash,
        "cfg_sha256": cfg_hash,
        "cfg_sha256_resolved": cfg_sha256,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "code_snapshot_sha256": code_snapshot_sha256,
        "budget_main_axis": meta.get("budget_main_axis"),
        "dataset_id": meta.get("dataset_id"),
        "stable_hw_state": stable_hw_state or {},
        "locked_acc_ref_value": (stable_hw_state or {}).get("acc_ref"),
        "locked_acc_ref_source": (stable_hw_state or {}).get("acc_ref_source"),
    }

    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
