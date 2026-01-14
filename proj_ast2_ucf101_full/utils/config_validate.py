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

    # ---- stable_hw defaults ----
    if "stable_hw" not in cfg:
        cfg["stable_hw"] = {}

    stable_hw = cfg["stable_hw"]
    hw = cfg.get("hw", {})

    if "enabled" not in stable_hw:
        stable_hw["enabled"] = bool(
            float(hw.get("lambda_hw_max", hw.get("lambda_hw", 0.0))) > 0.0
        )

    stable_hw.setdefault("lambda_hw_schedule", {})
    sched = stable_hw["lambda_hw_schedule"]
    if "lambda_hw_max" not in sched or sched["lambda_hw_max"] is None:
        base_lam = hw.get("lambda_hw_max", None)
        if base_lam is None:
            base_lam = hw.get("lambda_hw", 0.0)
        try:
            sched["lambda_hw_max"] = float(base_lam)
        except Exception:
            sched["lambda_hw_max"] = 0.0
    sched.setdefault("enabled", True)
    sched.setdefault("warmup_epochs", 5)
    sched.setdefault("ramp_epochs", 10)
    sched.setdefault("phase_name", "warmup_ramp")
    sched.setdefault("clamp_min", 0.0)
    sched.setdefault("clamp_max", float(sched.get("lambda_hw_max", 0.0)))
    if stable_hw.get("enabled") and float(sched.get("lambda_hw_max", 0.0)) <= 0.0:
        print(
            "[WARN] stable_hw.enabled=True but lambda_hw_max<=0; using cfg.hw.lambda_hw as fallback."
        )
        try:
            sched["lambda_hw_max"] = float(hw.get("lambda_hw", 0.0))
        except Exception:
            sched["lambda_hw_max"] = 0.0
    if sched.get("clamp_max") is None or float(sched.get("clamp_max", 0.0)) <= 0.0:
        sched["clamp_max"] = float(sched.get("lambda_hw_max", 0.0))

    stable_hw.setdefault("normalize", {})
    norm = stable_hw["normalize"]
    missing_targets = [
        name
        for name in ("target_ratio_T", "target_ratio_E", "target_ratio_M", "target_ratio_C")
        if name not in norm
    ]
    norm.setdefault("mode", "hinge_log_ratio")
    norm.setdefault("eps", 1e-6)
    norm.setdefault("clip_term_max", 10.0)
    norm.setdefault("mem_hinge_only", True)
    norm.setdefault("abs_ratio", False)
    norm.setdefault("wT", 0.2)
    norm.setdefault("wE", 0.2)
    norm.setdefault("wM", 0.4)
    norm.setdefault("wC", 0.2)
    norm.setdefault("target_ratio_T", 0.9)
    norm.setdefault("target_ratio_E", 0.9)
    norm.setdefault("target_ratio_M", 0.9)
    norm.setdefault("target_ratio_C", 0.9)
    norm.setdefault("ref_source", "ema")
    norm.setdefault("baseline_stats_path", None)

    if stable_hw.get("enabled") and missing_targets:
        print(
            "[WARN] stable_hw.enabled=True but missing normalize target ratios "
            f"{missing_targets}; using spec defaults (0.9)."
        )

    if stable_hw.get("enabled"):
        target_ratios = [
            ("target_ratio_T", norm.get("target_ratio_T")),
            ("target_ratio_E", norm.get("target_ratio_E")),
            ("target_ratio_M", norm.get("target_ratio_M")),
            ("target_ratio_C", norm.get("target_ratio_C")),
        ]
        for name, value in target_ratios:
            try:
                ratio = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"stable_hw.normalize.{name} must be a float in (0, 1].") from exc
            if ratio <= 0.0 or ratio > 1.0:
                raise ValueError(f"stable_hw.normalize.{name} must be in (0, 1].")

        weights = [
            ("wT", norm.get("wT")),
            ("wE", norm.get("wE")),
            ("wM", norm.get("wM")),
            ("wC", norm.get("wC")),
        ]
        weight_sum = 0.0
        for name, value in weights:
            try:
                weight = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"stable_hw.normalize.{name} must be a non-negative float.") from exc
            if weight < 0.0:
                raise ValueError(f"stable_hw.normalize.{name} must be non-negative.")
            weight_sum += weight
        if weight_sum <= 0.0:
            raise ValueError("stable_hw.normalize weights must sum to a positive value.")

        mode_value = str(norm.get("mode", ""))
        allowed_modes = {"ratio", "log_ratio", "hinge_log_ratio"}
        if mode_value not in allowed_modes:
            raise ValueError(
                f"stable_hw.normalize.mode must be one of {sorted(allowed_modes)}; got {mode_value!r}."
            )

    stable_hw.setdefault("discrete_isolation", {})
    iso = stable_hw["discrete_isolation"]
    iso.setdefault("use_cached_mapping_for_inner_steps", True)
    iso.setdefault("use_cached_layout_for_inner_steps", True)
    iso.setdefault("mapping_update_every_epochs", 1)
    iso.setdefault("layout_update_every_epochs", 1)
    iso.setdefault("track_live_segments", False)

    # accuracy_guard defaults (Version-C trainer uses attribute access)
    guard = stable_hw.get("accuracy_guard", {}) or {}
    stable_hw["accuracy_guard"] = guard
    guard.setdefault("enabled", True)
    guard.setdefault("use_ema", True)
    guard.setdefault("ema_beta", 0.8)
    guard.setdefault("epsilon_drop", 0.01)

    on_violate = guard.get("on_violate", {}) or {}
    guard["on_violate"] = on_violate
    on_violate.setdefault("scale_lambda_hw", 0.5)
    on_violate.setdefault("max_consecutive", 3)

    stable_hw.setdefault("ema_beta", 0.9)
    cfg["stable_hw"] = stable_hw

    return cfg
