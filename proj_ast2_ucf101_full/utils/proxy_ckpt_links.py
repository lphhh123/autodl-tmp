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

    Supported source layouts (priority):
      (A) <project_root>/proxy_ckpts/<device_name>/*.pt           (full/proxy_retrain output style)
      (B) <project_root>/<proxy_ckpts_3090|4090|2080ti>/*.pt      (code-only bundled style)
      (C) <project_root>/proxy_retrain/full/proxy_ckpts/<device_name>/*.pt
      (D) <project_root>/full/proxy_ckpts/<device_name>/*.pt

    dst_dir is expected to be <some_root>/proxy_ckpts/<device_name>/  (materialized standard layout).
    """
    project_root = Path(project_root).resolve()
    dst_dir = Path(dst_dir).resolve()

    if device_name not in DEVICE_TO_SRC_DIR:
        raise RuntimeError(
            f"[ProxyCkptResolveError] device_name={device_name!r} not supported. "
            f"Supported: {sorted(DEVICE_TO_SRC_DIR.keys())}"
        )

    candidates = []
    candidates.append(project_root / "proxy_ckpts" / device_name)
    candidates.append(project_root / "proxy_retrain" / "full" / "proxy_ckpts" / device_name)
    candidates.append(project_root / "full" / "proxy_ckpts" / device_name)
    candidates.append(project_root / DEVICE_TO_SRC_DIR[device_name])

    src_dir = None
    for c in candidates:
        if c.is_dir() and all((c / f).is_file() for f in REQUIRED_FILES):
            src_dir = c.resolve()
            break

    if src_dir is None:
        msg = [
            f"[ProxyCkptResolveError] cannot find trained proxy ckpts for device={device_name}",
            f"project_root={project_root}",
            "searched candidate source dirs (must contain REQUIRED_FILES):",
        ]
        for c in candidates:
            msg.append(f"  - {c}  (exists={c.is_dir()})")
        msg.append(f"REQUIRED_FILES={REQUIRED_FILES}")
        raise RuntimeError("\n".join(msg))

    dst_dir.mkdir(parents=True, exist_ok=True)

    marker = dst_dir / "_DEVICE_NAME.txt"
    if marker.is_file():
        old = marker.read_text(encoding="utf-8").strip()
        if old != device_name:
            shutil.rmtree(dst_dir)
            dst_dir.mkdir(parents=True, exist_ok=True)

    if src_dir != dst_dir:
        for f in REQUIRED_FILES:
            shutil.copy2(src_dir / f, dst_dir / f)

    marker.write_text(device_name, encoding="utf-8")

    missing_dst = [f for f in REQUIRED_FILES if not (dst_dir / f).is_file()]
    if missing_dst:
        raise RuntimeError(
            f"[ProxyCkptResolveError] failed to materialize files in {dst_dir}: {missing_dst}"
        )
