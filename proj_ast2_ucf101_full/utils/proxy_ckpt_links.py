from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List


# Mapping: device_name -> trained ckpt folder name in project root
DEVICE_TO_SRC_DIR: Dict[str, str] = {
    "RTX3090_FP16": "proxy_ckpts_3090",
    "RTX4090_FP16": "proxy_ckpts_4090",
    "RTX2080Ti_FP16": "proxy_ckpts_2080ti",
}

REQUIRED_FILES: List[str] = [
    "proxy_ms.pt",
    "proxy_peak_mem_mb.pt",
    "proxy_energy_mj.pt",
]


def ensure_proxy_ckpts_dir(project_root: Path, device_name: str, dst_dir: Path) -> None:
    """
    Ensure dst_dir exists and contains REQUIRED_FILES.
    If missing, copy from project_root/{DEVICE_TO_SRC_DIR[device_name]}.
    Fail-fast if src folder or files are missing.
    """
    project_root = Path(project_root).resolve()
    dst_dir = Path(dst_dir).resolve()

    if device_name not in DEVICE_TO_SRC_DIR:
        raise RuntimeError(
            f"[ProxyCkptResolveError] device_name={device_name!r} not supported. "
            f"Supported: {sorted(DEVICE_TO_SRC_DIR.keys())}"
        )

    src_dir = (project_root / DEVICE_TO_SRC_DIR[device_name]).resolve()
    if not src_dir.is_dir():
        raise RuntimeError(
            f"[ProxyCkptResolveError] src_dir not found: {src_dir}\n"
            f"Expected a trained proxy directory at: {project_root}/{DEVICE_TO_SRC_DIR[device_name]}"
        )

    # verify src files
    missing_src = [f for f in REQUIRED_FILES if not (src_dir / f).is_file()]
    if missing_src:
        raise RuntimeError(
            f"[ProxyCkptResolveError] missing files in {src_dir}: {missing_src}\n"
            f"Please re-run proxy training to generate {REQUIRED_FILES}."
        )

    # ensure dst
    dst_dir.mkdir(parents=True, exist_ok=True)

    # copy if missing
    for f in REQUIRED_FILES:
        src = src_dir / f
        dst = dst_dir / f
        if not dst.is_file():
            shutil.copy2(src, dst)

    # final verify
    missing_dst = [f for f in REQUIRED_FILES if not (dst_dir / f).is_file()]
    if missing_dst:
        raise RuntimeError(
            f"[ProxyCkptResolveError] failed to materialize files in {dst_dir}: {missing_dst}"
        )
