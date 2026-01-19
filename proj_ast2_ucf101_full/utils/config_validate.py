from __future__ import annotations

from typing import Any, Dict

from .config_utils import get_nested, set_nested
from .config import AttrDict

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


def _migrate_stable_hw_to_v5(cfg: Any) -> None:
    stable_hw = get_nested(cfg, "stable_hw", {}) or {}
    if not isinstance(stable_hw, dict):
        return
    guard = stable_hw.get("accuracy_guard", {}) or {}
    controller = stable_hw.get("controller", {}) or {}

    metric = guard.get("metric")
    if guard.get("metric_key") is None and metric is not None:
        if str(metric) in {"acc1", "val_acc1", "val_acc"}:
            guard["metric_key"] = "val_acc1"
        else:
            guard["metric_key"] = str(metric)

    on_violate = guard.get("on_violate", {}) or {}
    recover = on_violate.get("recover", {}) if isinstance(on_violate, dict) else {}
    cand_epochs = []
    if isinstance(on_violate, dict) and on_violate.get("freeze_rho_epochs") is not None:
        cand_epochs.append(int(on_violate.get("freeze_rho_epochs") or 0))
    if isinstance(recover, dict) and recover.get("patience_epochs") is not None:
        cand_epochs.append(int(recover.get("patience_epochs") or 0))
    if cand_epochs and controller.get("recovery_min_epochs") is None:
        controller["recovery_min_epochs"] = max(1, max(cand_epochs))

    guard.pop("on_violate", None)
    guard.pop("max_consecutive", None)
    guard.pop("metric", None)

    stable_hw.pop("hw_refs_update", None)
    stable_hw["accuracy_guard"] = guard
    stable_hw["controller"] = controller
    cfg["stable_hw"] = stable_hw


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
    # ---- legacy compat: layout.optimize_layout -> hw.optimize_layout ----
    try:
        layout_opt = get_nested(cfg, "layout.optimize_layout", None)
        hw_opt = get_nested(cfg, "hw.optimize_layout", None)
        if hw_opt is None and layout_opt is not None:
            set_nested(cfg, "hw.optimize_layout", bool(layout_opt))
            print(
                "[WARN] Detected legacy config key layout.optimize_layout. "
                "Please move it to hw.optimize_layout (SPEC). "
                f"Auto-synced hw.optimize_layout={bool(layout_opt)} for this run."
            )
    except Exception:
        pass

    if mode == "version_c":
        _apply_defaults(cfg, REQ_VERSION_C_HW_DEFAULTS)
        _apply_defaults(cfg, REQ_VERSION_C_TRAINING_DEFAULTS)
        _apply_defaults(cfg, REQ_CHIPLET_DEFAULTS)
    elif mode == "ast2":
        # minimal defaults for reproducibility
        if not hasattr(cfg, "train"):
            cfg.train = AttrDict({})
        if getattr(cfg.train, "seed", None) is None:
            cfg.train.seed = 2024
        return cfg
    elif mode == "layout":
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

        if not hasattr(cfg, "objective"):
            cfg.objective = AttrDict({})
        if getattr(cfg.objective, "sigma_mm", None) is None:
            cfg.objective.sigma_mm = 2.0

        if not hasattr(cfg.objective, "scalar_weights"):
            cfg.objective.scalar_weights = AttrDict({})
        sw = cfg.objective.scalar_weights
        if getattr(sw, "w_comm", None) is None:
            sw.w_comm = 1.0
        if getattr(sw, "w_therm", None) is None:
            sw.w_therm = 1.0
        if getattr(sw, "w_penalty", None) is None:
            sw.w_penalty = 1.0

        # optional oscillation metric defaults
        if not hasattr(cfg, "oscillation"):
            cfg.oscillation = AttrDict({})
        if getattr(cfg.oscillation, "window", None) is None:
            cfg.oscillation.window = 10
        if getattr(cfg.oscillation, "eps_flat", None) is None:
            cfg.oscillation.eps_flat = 1e-6

        return cfg
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

    # ---- stable_hw defaults (v5 canonical) ----
    if "stable_hw" not in cfg:
        cfg["stable_hw"] = {}

    _migrate_stable_hw_to_v5(cfg)

    stable_hw = get_nested(cfg, "stable_hw", {}) or {}
    stable_hw_enabled = bool(stable_hw.get("enabled", False))
    stable_hw["enabled"] = stable_hw_enabled

    # legacy -> v5.1 canonical mapping
    if isinstance(stable_hw, dict):
        locked = stable_hw.get("locked_acc_ref")
        if locked is None:
            locked = {}
        locked.setdefault("baseline_stats_path", stable_hw.get("baseline_stats_path"))
        locked.setdefault("freeze_epoch", 0)
        locked.setdefault("prefer_dense_baseline", True)
        stable_hw["locked_acc_ref"] = locked
        ag = stable_hw.get("accuracy_guard")
        if isinstance(ag, dict):
            on_violate = ag.get("on_violate")
            if isinstance(on_violate, dict):
                on_violate["scale_lambda_hw"] = float(on_violate.get("scale_lambda_hw", 0.0))
    else:
        if getattr(stable_hw, "locked_acc_ref", None) is None:
            stable_hw.locked_acc_ref = type("Obj", (), {})()
            stable_hw.locked_acc_ref.baseline_stats_path = getattr(stable_hw, "baseline_stats_path", None)
            stable_hw.locked_acc_ref.freeze_epoch = 0
            stable_hw.locked_acc_ref.prefer_dense_baseline = True
        ag = getattr(stable_hw, "accuracy_guard", None)
        if ag is not None and getattr(ag, "on_violate", None) is not None:
            setattr(ag.on_violate, "scale_lambda_hw", float(getattr(ag.on_violate, "scale_lambda_hw", 0.0)))

    controller = stable_hw.get("controller", {}) or {}
    controller.setdefault("freeze_schedule_in_recovery", True)
    controller.setdefault("recovery_min_epochs", 1)
    controller.setdefault("margin_exit", 0.0)
    controller.setdefault("k_exit", 1)
    stable_hw["controller"] = controller

    guard = stable_hw.get("accuracy_guard", {}) or {}
    guard.setdefault("enabled", stable_hw_enabled)
    guard.setdefault("metric_key", "val_acc1")
    guard.setdefault("epsilon_drop", 0.01)
    guard.setdefault("use_ema", True)
    guard.setdefault("ema_beta", 0.9)
    guard.setdefault("baseline_stats_path", str(get_nested(cfg, "paths.baseline_stats_path", "") or ""))
    stable_hw["accuracy_guard"] = guard

    sched = stable_hw.get("lambda_hw_schedule", {}) or {}
    stable_hw["lambda_hw_schedule"] = sched
    sched.setdefault("enabled", stable_hw_enabled)
    sched.setdefault("lambda_hw_max", 0.0)
    sched.setdefault("warmup_epochs", 5)
    sched.setdefault("ramp_epochs", 10)

    norm = stable_hw.get("normalize", {}) or {}
    stable_hw["normalize"] = norm
    norm.setdefault("enabled", True)
    norm.setdefault("method", "hinge_log_ratio")
    norm.setdefault("clip_eps", 1e-6)
    norm.setdefault("clip_term_max", 10.0)
    norm.setdefault("mem_hinge_only", True)
    norm.setdefault("abs_ratio", False)
    weights = norm.get("weights", {}) or {}
    weights.setdefault("wT", 0.2)
    weights.setdefault("wE", 0.2)
    weights.setdefault("wM", 0.4)
    weights.setdefault("wC", 0.2)
    norm["weights"] = weights
    ref = norm.get("ref", {}) or {}
    ref.setdefault("target_ratio_T", 0.9)
    ref.setdefault("target_ratio_E", 0.9)
    ref.setdefault("target_ratio_M", 0.9)
    ref.setdefault("target_ratio_C", 0.9)
    norm["ref"] = ref
    norm.setdefault("eps", float(norm.get("clip_eps", 1e-6)))
    norm.setdefault("wT", float(weights["wT"]))
    norm.setdefault("wE", float(weights["wE"]))
    norm.setdefault("wM", float(weights["wM"]))
    norm.setdefault("wC", float(weights["wC"]))
    norm.setdefault("target_ratio_T", float(ref["target_ratio_T"]))
    norm.setdefault("target_ratio_E", float(ref["target_ratio_E"]))
    norm.setdefault("target_ratio_M", float(ref["target_ratio_M"]))
    norm.setdefault("target_ratio_C", float(ref["target_ratio_C"]))

    # ---------- discrete_isolation ----------
    iso = stable_hw.get("discrete_isolation", {}) or {}
    if "track_live_in_inner_steps" in iso and "track_live_segments" not in iso:
        iso["track_live_segments"] = bool(iso.get("track_live_in_inner_steps"))
        print(
            "[WARN] stable_hw.discrete_isolation.track_live_in_inner_steps is deprecated; "
            "use track_live_segments instead."
        )
    stable_hw["discrete_isolation"] = iso
    iso.setdefault("use_cached_mapping_for_inner_steps", True)
    iso.setdefault("use_cached_layout_for_inner_steps", True)
    iso.setdefault("mapping_update_every_epochs", 1)
    iso.setdefault("layout_update_every_epochs", 1)
    iso.setdefault("track_live_segments", False)  # optional debug mode
    iso.setdefault("track_live_every_steps", 1)

    # validate ints
    for k in ["mapping_update_every_epochs", "layout_update_every_epochs", "track_live_every_steps"]:
        v = int(iso.get(k, 1))
        if v < 1:
            raise ValueError(f"stable_hw.discrete_isolation.{k} must be >=1")
        iso[k] = v

    cfg["stable_hw"] = stable_hw

    # ---- guardrail: HW loss enabled but lambda is effectively zero ----
    try:
        use_hw_loss = bool(get_nested(cfg, "hw.use_hw_loss", True))
        stable_en = bool(get_nested(cfg, "stable_hw.enabled", False))
        lam_hw = float(get_nested(cfg, "hw.lambda_hw", 0.0) or 0.0)
        lam_max = float(get_nested(cfg, "stable_hw.lambda_hw_schedule.lambda_hw_max", 0.0) or 0.0)
        sched_en = bool(get_nested(cfg, "stable_hw.lambda_hw_schedule.enabled", False))

        if use_hw_loss:
            if stable_en and sched_en and lam_max <= 0.0:
                print(
                    "[WARN] stable_hw is enabled but stable_hw.lambda_hw_schedule.lambda_hw_max <= 0. "
                    "HW loss may have no effect. Please set a positive lambda_hw_max."
                )
            if (not stable_en) and lam_hw <= 0.0:
                print(
                    "[WARN] hw.use_hw_loss=true but stable_hw.enabled=false and hw.lambda_hw<=0. "
                    "HW loss weight is zero; HW term will be ineffective. "
                    "Enable stable_hw or set hw.lambda_hw>0."
                )
    except Exception:
        pass

    # ===== v5.4 NoDoubleScale: when stable_hw is enabled, legacy lambda_hw fields are ignored =====
    try:
        stable_hw = getattr(cfg, "stable_hw", None)
        if stable_hw is not None and bool(getattr(stable_hw, "enabled", False)):
            if hasattr(cfg, "loss") and hasattr(cfg.loss, "lambda_hw"):
                cfg.loss.lambda_hw = 0.0
            if hasattr(cfg, "hw") and hasattr(cfg.hw, "lambda_hw"):
                cfg.hw.lambda_hw = 0.0
            if hasattr(cfg, "stable_hw") and not hasattr(cfg.stable_hw, "no_double_scale"):
                cfg.stable_hw.no_double_scale = True
    except Exception:
        pass

    return cfg
