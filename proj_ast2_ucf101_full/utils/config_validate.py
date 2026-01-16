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

    # ---- stable_hw defaults (CLEAN + SPEC-ALIGNED) ----
    if "stable_hw" not in cfg:
        cfg["stable_hw"] = {}

    stable_hw = get_nested(cfg, "stable_hw", {}) or {}
    stable_hw_enabled = bool(stable_hw.get("enabled", False))
    stable_hw["enabled"] = stable_hw_enabled

    # ---------- lambda schedule ----------
    sched = stable_hw.get("lambda_hw_schedule", {}) or {}
    stable_hw["lambda_hw_schedule"] = sched
    sched.setdefault("enabled", stable_hw_enabled)
    sched.setdefault("warmup_epochs", 5)
    sched.setdefault("ramp_epochs", 10)
    sched.setdefault("stabilize_epochs", 0)
    sched.setdefault("lambda_hw_min", 0.0)
    sched.setdefault("clamp_min", 0.0)

    # IMPORTANT: DO NOT silently bind lambda_hw_max to hw.lambda_hw unless allow_legacy=true
    if "lambda_hw_max" not in sched or sched.get("lambda_hw_max") is None:
        sched["lambda_hw_max"] = 0.0

    # if stable_hw enabled, require lambda_hw_max > 0 unless allow_legacy_lambda_hw
    if stable_hw_enabled and bool(sched.get("enabled", True)):
        lam_max = float(sched.get("lambda_hw_max", 0.0) or 0.0)
        if lam_max <= 0.0:
            allow_legacy = bool(stable_hw.get("allow_legacy_lambda_hw", False))
            legacy = float(get_nested(cfg, "hw.lambda_hw", 0.0) or 0.0)
            if allow_legacy and legacy > 0.0:
                print(
                    "[WARN] stable_hw enabled but stable_hw.lambda_hw_schedule.lambda_hw_max missing/<=0; "
                    f"using legacy hw.lambda_hw={legacy} because allow_legacy_lambda_hw=true."
                )
                sched["lambda_hw_max"] = float(legacy)
            else:
                raise ValueError(
                    "stable_hw enabled but stable_hw.lambda_hw_schedule.lambda_hw_max missing/<=0.\n"
                    "Please set stable_hw.lambda_hw_schedule.lambda_hw_max: <positive float>\n"
                    "Or set stable_hw.allow_legacy_lambda_hw=true to reuse hw.lambda_hw."
                )
    sch = cfg["stable_hw"]["lambda_hw_schedule"]
    if float(sch.get("lambda_hw_max", 0.0) or 0.0) <= 0.0:
        print("[WARN][stable_hw] lambda_hw_schedule.lambda_hw_max is 0. HW schedule will be ineffective.")

    sched.setdefault("clamp_max", float(sched.get("lambda_hw_max", 0.0) or 0.0))

    # ---------- normalize ----------
    norm = stable_hw.get("normalize", {}) or {}
    stable_hw["normalize"] = norm
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

    # ref source for normalization (SPEC: baseline_stats / ema)
    norm.setdefault("ref_source", "ema")
    norm.setdefault("baseline_stats_path", str(get_nested(cfg, "paths.baseline_stats_path", "") or ""))

    # ---------- hw_refs_update ----------
    ref_up = stable_hw.get("hw_refs_update", {}) or {}
    stable_hw["hw_refs_update"] = ref_up
    ref_up.setdefault("enabled", stable_hw_enabled)
    ref_up.setdefault("ref_source", str(ref_up.get("ref_source", "ema")))  # "ema" or "baseline_stats"
    ref_up.setdefault("ema_beta", float(ref_up.get("ema_beta", 0.95) or 0.95))
    ref_up.setdefault("update_only_when_better", bool(ref_up.get("update_only_when_better", False)))
    ref_up.setdefault("better_key", str(ref_up.get("better_key", "total_scalar")))
    ref_up.setdefault("min_delta_ratio", float(ref_up.get("min_delta_ratio", 0.01) or 0.01))
    ref_up.setdefault(
        "baseline_stats_path",
        str(ref_up.get("baseline_stats_path", "")) or str(get_nested(cfg, "paths.baseline_stats_path", "")),
    )

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

    # ---------- accuracy_guard (SPEC keys) ----------
    guard = stable_hw.get("accuracy_guard", {}) or {}
    stable_hw["accuracy_guard"] = guard
    guard.setdefault("enabled", stable_hw_enabled)
    guard.setdefault("metric", "acc1")
    guard.setdefault("use_ema", True)
    guard.setdefault("ema_beta", 0.98)
    guard.setdefault("epsilon_drop", 0.02)
    guard.setdefault("max_consecutive", int(guard.get("max_consecutive", 3) or 3))
    guard.setdefault("guard_val_batches", int(guard.get("guard_val_batches", 0) or 0))  # 0=use full val

    onv = guard.get("on_violate", {}) or {}
    guard["on_violate"] = onv
    onv.setdefault("scale_lambda_hw", 0.5)
    onv.setdefault("freeze_rho_epochs", int(onv.get("freeze_rho_epochs", 1) or 1))
    onv.setdefault("max_violations", int(onv.get("max_violations", guard.get("max_consecutive", 3)) or 3))
    onv.setdefault("disable_hw_after_max_violations", False)
    recover = onv.get("recover", {}) or {}
    onv["recover"] = recover
    recover.setdefault("enable", True)
    recover.setdefault("patience_epochs", 1)
    recover.setdefault("restore_lambda_hw", True)

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

    return cfg
