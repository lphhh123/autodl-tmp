from __future__ import annotations

import json
import os
import platform
import time
import uuid
from typing import Any, Dict, Optional


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
):
    os.makedirs(out_dir, exist_ok=True)
    rid = str(run_id or uuid.uuid4().hex)
    meta = {
        "run_id": rid,
        "spec_version": spec_version,
        "timestamp_unix": int(time.time()),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "cfg": {
            "cfg_path": str(cfg_path),
            "cfg_hash": str(cfg_hash),
            "seed": int(seed),
        },
        "command": command or "",
        "stable_hw_state": stable_hw_state or {},
    }
    meta["hw_refs"] = {
        "ref_T": stable_hw_state.get("ref_T"),
        "ref_E": stable_hw_state.get("ref_E"),
        "ref_M": stable_hw_state.get("ref_M"),
        "ref_C": stable_hw_state.get("ref_C"),
        "hw_ref_source": stable_hw_state.get("hw_ref_source"),
    }
    meta["stability"] = {
        "no_drift_enabled": stable_hw_state.get("no_drift_enabled"),
        "no_double_scale_enabled": stable_hw_state.get("no_double_scale_enabled"),
    }
    meta["signature"] = stable_hw_state.get("run_signature")
    if extra:
        meta["extra"] = extra
    manifest_path = os.path.join(out_dir, "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return rid
