"""Config validation helpers for Version-C."""
from __future__ import annotations

from typing import Any, Dict


def _get_attr(cfg: Any, path: str) -> Any:
    cur = cfg
    for key in path.split("."):
        if not hasattr(cur, key):
            raise KeyError(f"Missing required config key: {path}")
        cur = getattr(cur, key)
    return cur


def validate_cfg(cfg: Any) -> None:
    required = [
        "hw.device_name",
        "hw.gpu_yaml",
        "hw.proxy_weight_dir",
        "layout.sigma_mm",
        "layout.scalar_weights",
        "layout.wafer_radius_mm",
        "layout.margin_mm",
        "mapping.strategy",
        "mapping.mem_limit_factor",
        "train.seed",
        "train.out_dir",
    ]
    for path in required:
        value = _get_attr(cfg, path)
        if value is None:
            raise ValueError(f"Required config value is None: {path}")


def cfg_to_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: cfg_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cfg_to_dict(v) for v in obj]
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, int, float, bool)):
        return {k: cfg_to_dict(v) for k, v in obj.__dict__.items()}
    return obj
