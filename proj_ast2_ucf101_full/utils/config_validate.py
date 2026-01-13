from __future__ import annotations

from typing import Any, Dict, List

from .config_utils import get_nested, set_nested


def _require(cfg: Any, key: str, errs: List[str]) -> None:
    v = get_nested(cfg, key, None)
    if v is None:
        errs.append(f"Missing required config key: {key}")


def validate_and_fill_defaults(cfg: Any) -> Dict[str, Any]:
    """
    Validate config with mode-aware rules.
    - Hard errors only for truly mandatory keys.
    - Fill reasonable defaults for layout/hw fields when enabled.
    Returns a dict with warnings + resolved flags.
    """
    warnings: List[str] = []
    errors: List[str] = []

    # ---- Always required (all modes) ----
    # allow model.type or model.name
    if get_nested(cfg, "model.type", None) is None and get_nested(cfg, "model.name", None) is None:
        errors.append("Missing required config key: model.type (or model.name)")
    if get_nested(cfg, "data.dataset", None) is None and get_nested(cfg, "dataset.name", None) is None:
        warnings.append("dataset not found at data.dataset; will rely on dataset.* keys if present")

    # training basics (accept train.* or training.*)
    if get_nested(cfg, "train.device", None) is None and get_nested(cfg, "training.device", None) is None:
        warnings.append("train.device missing; defaulting to cuda if available")
        set_nested(cfg, "train.device", "cuda")

    # lr/batch/epochs (either path is acceptable)
    if get_nested(cfg, "train.epochs", None) is None and get_nested(cfg, "training.epochs", None) is None:
        warnings.append("train.epochs missing; defaulting to 1 for smoke")
        set_nested(cfg, "train.epochs", 1)
    if get_nested(cfg, "train.batch_size", None) is None and get_nested(cfg, "training.batch_size", None) is None:
        warnings.append("train.batch_size missing; defaulting to 1 for smoke")
        set_nested(cfg, "train.batch_size", 1)
    if get_nested(cfg, "train.lr", None) is None and get_nested(cfg, "training.lr", None) is None:
        warnings.append("train.lr missing; defaulting to 1e-4")
        set_nested(cfg, "train.lr", 1e-4)

    # ---- HW enabled? ----
    lambda_hw = get_nested(cfg, "hw.lambda_hw", None)
    if lambda_hw is None:
        # compat old
        lambda_hw = get_nested(cfg, "loss.lambda_hw", 0.0)
        set_nested(cfg, "hw.lambda_hw", float(lambda_hw))
        warnings.append("loss.lambda_hw found; normalized to hw.lambda_hw")

    hw_enabled = float(get_nested(cfg, "hw.lambda_hw", 0.0)) > 0.0 or str(get_nested(cfg, "hw.mode", "none")) != "none"
    if hw_enabled:
        _require(cfg, "hw.gpu_yaml", errors)
        _require(cfg, "hw.proxy_weight_dir", errors)
        _require(cfg, "hw.device_name", errors)

    # ---- Layout enabled? ----
    layout_enabled = bool(get_nested(cfg, "layout.optimize_layout", False)) or bool(get_nested(cfg, "version_c.update_layout", False))
    if layout_enabled:
        # fill defaults rather than error
        if get_nested(cfg, "layout.wafer_radius_mm", None) is None:
            set_nested(cfg, "layout.wafer_radius_mm", 150.0)
            warnings.append("layout.wafer_radius_mm missing; defaulting to 150.0")
        if get_nested(cfg, "layout.margin_mm", None) is None:
            set_nested(cfg, "layout.margin_mm", 0.0)
            warnings.append("layout.margin_mm missing; defaulting to 0.0")
        if get_nested(cfg, "layout.sigma_mm", None) is None:
            set_nested(cfg, "layout.sigma_mm", 1.0)
            warnings.append("layout.sigma_mm missing; defaulting to 1.0")
        if get_nested(cfg, "layout.scalar_weights", None) is None:
            # reasonable defaults
            set_nested(cfg, "layout.scalar_weights", {"comm": 1.0, "therm": 1.0, "boundary": 1.0, "duplicate": 0.0})
            warnings.append("layout.scalar_weights missing; defaulting to comm/therm/boundary=1.0")

    if errors:
        msg = "Config validation failed:\n" + "\n".join(f"- {e}" for e in errors)
        raise ValueError(msg)

    return {"warnings": warnings, "hw_enabled": hw_enabled, "layout_enabled": layout_enabled}
