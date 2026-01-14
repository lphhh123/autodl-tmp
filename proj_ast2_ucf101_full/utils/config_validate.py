from __future__ import annotations

from typing import Any, Dict

from .config_utils import get_nested, set_nested

# 仅对“必需”字段做强制默认，避免过度约束
REQ_VERSION_C_HW_DEFAULTS = {
    # wafer / site
    "hw.num_slots": 64,
    "hw.wafer_radius_mm": 150.0,
    "hw.site_margin_mm": 5.0,

    # objective weights (layout + comm/thermal)
    "hw.lambda_boundary": 1.0,
    "hw.lambda_overlap": 1.0,
    "hw.lambda_comm_extra": 1.0,
    "hw.lambda_thermal": 1.0,

    # mapping/comm
    "hw.distance_scale_ms": 0.0,
    "hw.mapping_strategy": "greedy_local",

    # hw loss weight
    "hw.lambda_hw": 0.0,

    # device/proxy
    "hw.device_name": "RTX4090_FP16",
    "hw.gpu_yaml": "configs/gpu_data.yaml",
    "hw.proxy_weight_dir": "proxy_weights",

    # memory constraints
    "mapping.mem_limit_factor": 1.0,
    "mapping.strategy": "greedy_local",
}

REQ_TRAIN_DEFAULTS = {
    "train.seed": 0,
    "train.device": "cuda",
    "train.amp": True,
    "train.lr": 3e-4,
    "train.weight_decay": 0.05,
    "loss.lambda_AST": 1.0,
}

REQ_VERSION_C_TRAINING_DEFAULTS = {
    "training.outer_epochs": 1,
    "training.inner_steps_ast": 50,
    "training.inner_steps_alpha": 20,
    "training.inner_steps_layout": 20,
    "training.model_type": "video",  # or "video_audio"
    "training.twostage": False,
    "training.mapping_only": False,
    "training.layout_only": False,
}

REQ_CHIPLET_DEFAULTS = {
    "chiplet.candidate_types": ["RTX4090_FP16"],
    "chiplet.tau_init": 1.0,
    "chiplet.tau_decay": 0.98,
    "chiplet.tau_min": 0.2,
}


def _apply_defaults(cfg: Any, defaults: Dict[str, Any]) -> None:
    for k, v in defaults.items():
        if get_nested(cfg, k, None) is None:
            set_nested(cfg, k, v)


def _sync_layout_to_hw(cfg: Any) -> None:
    """
    兼容旧配置：如果用户写了 layout.*，同步到 hw.*。
    """
    pairs = [
        ("layout.num_slots", "hw.num_slots"),
        ("layout.wafer_radius_mm", "hw.wafer_radius_mm"),
        ("layout.site_margin_mm", "hw.site_margin_mm"),
        ("layout.lambda_boundary", "hw.lambda_boundary"),
        ("layout.lambda_overlap", "hw.lambda_overlap"),
        ("layout.lambda_comm_extra", "hw.lambda_comm_extra"),
        ("layout.lambda_thermal", "hw.lambda_thermal"),
    ]
    for src, dst in pairs:
        v = get_nested(cfg, src, None)
        if v is not None and get_nested(cfg, dst, None) is None:
            set_nested(cfg, dst, v)


def validate_and_fill_defaults(cfg: Any, mode: str = "version_c") -> Any:
    """
    mode:
      - "version_c": Version-C training/eval (needs cfg.hw.*)
      - "layout":    layout-only agent scripts (optional; keep minimal)
      - "single":    single-device pruning baseline
    """
    # always: common train defaults
    _apply_defaults(cfg, REQ_TRAIN_DEFAULTS)

    # compat: layout.* -> hw.*
    _sync_layout_to_hw(cfg)

    if mode == "version_c":
        _apply_defaults(cfg, REQ_VERSION_C_HW_DEFAULTS)
        _apply_defaults(cfg, REQ_VERSION_C_TRAINING_DEFAULTS)
        _apply_defaults(cfg, REQ_CHIPLET_DEFAULTS)
    elif mode == "single":
        # single-device baseline: only need hw.device_name/gpu_yaml/proxy_weight_dir/lambda_hw
        _apply_defaults(
            cfg,
            {
                "hw.device_name": "RTX4090_FP16",
                "hw.gpu_yaml": "configs/gpu_data.yaml",
                "hw.proxy_weight_dir": "proxy_weights",
                "hw.lambda_hw": 0.0,
            },
        )
    else:
        # layout mode: keep minimal; still allow reading hw.wafer_radius_mm etc
        _apply_defaults(
            cfg,
            {
                "hw.num_slots": 64,
                "hw.wafer_radius_mm": 150.0,
                "hw.site_margin_mm": 5.0,
                "hw.lambda_boundary": 1.0,
                "hw.lambda_overlap": 1.0,
                "hw.lambda_comm_extra": 1.0,
                "hw.lambda_thermal": 1.0,
            },
        )

    return cfg
