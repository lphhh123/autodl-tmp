"""Torch backend knobs (TF32 etc.).

Centralizes low-level CUDA backend settings so experiment entrypoints can
enable/disable them in a controlled, auditable way.

Rules:
- Opt-in via env var ENABLE_TF32 ("1"/"true"/"yes")
- Safe on older torch versions (feature-gated)
"""

from __future__ import annotations

import os
from typing import Any, Optional


def _env_flag(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    if val is None:
        return False
    val = str(val).strip().lower()
    return val in ("1", "true", "yes", "y", "on")


def maybe_enable_tf32(logger: Optional[Any] = None) -> bool:
    """Enable TF32 if requested via ENABLE_TF32.

    Returns True if TF32 was enabled.
    """
    if not _env_flag("ENABLE_TF32", default="0"):
        return False

    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # PyTorch 2.x: controls float32 matmul precision on Ampere+.
        if hasattr(torch, "set_float32_matmul_precision"):
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        if logger is not None:
            logger.info("[TF32] ENABLED (allow_tf32 matmul/cudnn).")
        else:
            print("[TF32] ENABLED (allow_tf32 matmul/cudnn).")
        return True
    except Exception as exc:
        if logger is not None:
            logger.warning("[TF32] failed to enable: %s", exc)
        else:
            print(f"[TF32] failed to enable: {exc}")
        return False
