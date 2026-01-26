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
    Ensure dst_dir exists and contains REQUIRED_FILES for THIS device_name.
    Always guarantees correctness:
      - uses per-device dst_dir (proxy_ckpts/<device_name>/)
      - uses marker to detect mismatch
      - overwrites files from src_dir if mismatch or missing
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

    missing_src = [f for f in REQUIRED_FILES if not (src_dir / f).is_file()]
    if missing_src:
        raise RuntimeError(
            f"[ProxyCkptResolveError] missing files in {src_dir}: {missing_src}\n"
            f"Please re-run proxy training to generate {REQUIRED_FILES}."
        )

    dst_dir.mkdir(parents=True, exist_ok=True)

    marker = dst_dir / "_DEVICE_NAME.txt"
    if marker.is_file():
        old = marker.read_text(encoding="utf-8").strip()
        if old != device_name:
            # mismatch => clear and recreate
            shutil.rmtree(dst_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)

    # Always copy (overwrite) to guarantee correct device content
    for f in REQUIRED_FILES:
        shutil.copy2(src_dir / f, dst_dir / f)

    marker.write_text(device_name, encoding="utf-8")

    missing_dst = [f for f in REQUIRED_FILES if not (dst_dir / f).is_file()]
    if missing_dst:
        raise RuntimeError(
            f"[ProxyCkptResolveError] failed to materialize files in {dst_dir}: {missing_dst}"
        )
