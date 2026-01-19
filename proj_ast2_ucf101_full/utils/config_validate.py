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
    """
    v5.4 canonical:
      stable_hw.locked_acc_ref.*
      stable_hw.lambda_hw_schedule.*
      stable_hw.accuracy_guard.controller.*   (AUTHORITATIVE)
    Back-compat:
      stable_hw.controller.*   (alias; deprecated)
      stable_hw.accuracy_guard.metric_key / epsilon_drop etc. (alias; deprecated)
    """
    stable_hw = get_nested(cfg, "stable_hw", {}) or {}
    if not isinstance(stable_hw, dict):
        return

    guard = stable_hw.get("accuracy_guard", {}) or {}
    if not isinstance(guard, dict):
        guard = {}

    # old: stable_hw.controller -> new: stable_hw.accuracy_guard.controller
    old_ctrl = stable_hw.get("controller", {}) or {}
    if not isinstance(old_ctrl, dict):
        old_ctrl = {}

    ctrl = guard.get("controller", {}) or {}
    if not isinstance(ctrl, dict):
        ctrl = {}

    # merge: guard.controller has higher priority
    merged = dict(old_ctrl)
    merged.update(ctrl)
    guard["controller"] = merged

    # legacy metric -> controller.metric
    metric = guard.get("metric") or guard.get("metric_key") or merged.get("metric")
    if metric is None:
        metric = "val_acc1"
    guard["metric"] = str(metric)

    # ---- v5.4 hard migrations (must not crash) ----

    # 1) locked_acc_ref: allow bool -> dict
    locked = stable_hw.get("locked_acc_ref", None)
    if isinstance(locked, bool):
        stable_hw["locked_acc_ref"] = {
            "enabled": bool(locked),
            "freeze_epoch": int(stable_hw.get("warmup_epochs", 0) or 0),
            "prefer_dense_baseline": True,
            "baseline_stats_path": "",
        }

    # 2) normalize: keep SPEC key "mode", but also allow legacy "method"
    norm = stable_hw.get("normalize", None)
    if isinstance(norm, dict):
        if "mode" not in norm and "method" in norm:
            norm["mode"] = norm["method"]
        if "method" not in norm and "mode" in norm:
            norm["method"] = norm["mode"]
        if "eps" not in norm and "clip_eps" in norm:
            norm["eps"] = norm["clip_eps"]

    # 3) accuracy_guard schema: accept legacy on_violate -> controller
    guard = stable_hw.get("accuracy_guard", None)
    if isinstance(guard, dict):
        # metric_key default
        if "metric_key" not in guard and "metric" in guard:
            # metric="acc1" is display; internal default should be val_acc1
            guard["metric_key"] = "val_acc1"
        if "metric" not in guard:
            guard["metric"] = "acc1"

        # move on_violate -> controller if controller missing
        if "controller" not in guard and isinstance(guard.get("on_violate", None), dict):
            ov = guard["on_violate"]
            guard["controller"] = {
                "metric": "val_acc1",
                "freeze_schedule_in_recovery": True,
                "freeze_discrete_updates_in_recovery": bool(ov.get("freeze_discrete_updates", True)),
                "cut_hw_loss_on_violate": True,
                "scale_lambda_hw": float(ov.get("scale_lambda_hw", 0.0)),
                "recovery_epochs": int(ov.get("recovery_epochs", 1)),
                "recovery_min_epochs": int(ov.get("min_recovery_epochs", 1)),
                "k_exit": 1,
                "margin_exit": 0.0,
            }

    stable_hw["accuracy_guard"] = guard
    cfg["stable_hw"] = stable_hw
    return stable_hw


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

    # v5.4: if stable_hw block exists but user didn't specify stable_hw.enabled, treat as enabled by default.
    if isinstance(stable_hw, dict):
        if "enabled" not in stable_hw:
            stable_hw_enabled = True
            stable_hw["enabled"] = True
        else:
            stable_hw_enabled = bool(stable_hw.get("enabled", False))
            stable_hw["enabled"] = stable_hw_enabled

        # canonical: stable_hw.accuracy_guard.controller.* (SPEC) -> mirror into legacy stable_hw.controller.*
        ag = stable_hw.get("accuracy_guard", {}) or {}
        if isinstance(ag, dict):
            ctl = ag.get("controller")
            if isinstance(ctl, dict) and ctl:
                legacy_ctl = stable_hw.get("controller", {}) or {}
                if not isinstance(legacy_ctl, dict):
                    legacy_ctl = {}
                legacy_ctl = {**legacy_ctl, **ctl}
                stable_hw["controller"] = legacy_ctl
                # metric alias
                if "metric" in ctl and "metric_key" not in ag:
                    ag["metric_key"] = str(ctl["metric"])
                stable_hw["accuracy_guard"] = ag
    else:
        # object style
        if not hasattr(stable_hw, "enabled"):
            stable_hw.enabled = True
            stable_hw_enabled = True
        else:
            stable_hw_enabled = bool(getattr(stable_hw, "enabled", False))

    # ---- v5.4 stable_hw schema migration (must accept bool -> dict) ----
    locked = stable_hw.get("locked_acc_ref", None)
    if isinstance(locked, bool):
        locked = {"enabled": bool(locked)}
    elif locked is None:
        locked = {}
    elif not isinstance(locked, dict):
        # tolerate AttrDict/OmegaConf nodes
        try:
            locked = dict(locked)
        except Exception:
            locked = {}

    stable_hw["locked_acc_ref"] = locked
    locked.setdefault("enabled", True)
    locked.setdefault("prefer_dense_baseline", True)
    locked.setdefault("baseline_stats_path", stable_hw.get("baseline_stats_path", None))
    locked.setdefault("freeze_epoch", 0)

    # accuracy_guard: ensure baseline path parity to avoid drift
    guard = stable_hw.get("accuracy_guard", None)
    if isinstance(guard, bool):
        guard = {"enabled": bool(guard)}
    elif guard is None:
        guard = {}
    elif not isinstance(guard, dict):
        try:
            guard = dict(guard)
        except Exception:
            guard = {}
    stable_hw["accuracy_guard"] = guard
    guard.setdefault("enabled", True)
    guard.setdefault("epsilon_drop", 0.01)
    guard.setdefault("prefer_val_acc1", True)
    guard.setdefault("allow_train_ema_fallback", False)  # v5.4: default forbid silent downgrade
    guard.setdefault("baseline_stats_path", locked.get("baseline_stats_path", None))
    guard.setdefault("delta_below_thr", 0.005)  # for restart trigger (spec §12B.4.2)

    # lambda_hw_schedule: MUST implement clamp_min/max per spec
    sched = stable_hw.get("lambda_hw_schedule", None)
    if isinstance(sched, bool):
        sched = {"enabled": bool(sched)}
    elif sched is None:
        sched = {}
    elif not isinstance(sched, dict):
        try:
            sched = dict(sched)
        except Exception:
            sched = {}
    stable_hw["lambda_hw_schedule"] = sched
    sched.setdefault("enabled", True)
    sched.setdefault("warmup_epochs", 0)
    sched.setdefault("ramp_epochs", 1)
    sched.setdefault("lambda_hw_min", 0.0)
    sched.setdefault("lambda_hw_max", 0.2)
    sched.setdefault("clamp_min", 0.0)
    sched.setdefault("clamp_max", float(sched.get("lambda_hw_max", 0.2)))

    # controller: restart window anti-starvation per spec
    controller = stable_hw.get("controller", None)
    if isinstance(controller, bool):
        controller = {"enabled": bool(controller)}
    elif controller is None:
        controller = {}
    elif not isinstance(controller, dict):
        try:
            controller = dict(controller)
        except Exception:
            controller = {}
    stable_hw["controller"] = controller
    controller.setdefault("enabled", True)
    controller.setdefault("recovery_patience_epochs", 3)
    controller.setdefault("restart_window_epochs", 1)
    controller.setdefault("lr_restart_mul", 2.0)
    controller.setdefault("min_epochs_between_restarts", 1)
    # -------------------------------------------------------------------

    # -------- v5.4 canonical accuracy_guard.controller --------
    guard = stable_hw.get("accuracy_guard", {}) or {}
    if not isinstance(guard, dict):
        guard = {}
    guard.setdefault("enabled", stable_hw_enabled)

    controller = guard.get("controller", {}) or stable_hw.get("controller", {}) or {}
    if not isinstance(controller, dict):
        controller = {}

    # v5.4 required controller defaults
    controller.setdefault("mode", "accuracy_first_hard_gating")
    controller.setdefault("metric", str(guard.get("metric", "val_acc1")))
    controller.setdefault("epsilon_drop", 0.01)
    controller.setdefault("epsilon_drop_type", "abs")  # abs drop in accuracy
    controller.setdefault("freeze_rho_on_violate", True)
    controller.setdefault("cut_hw_loss_on_violate", True)
    controller.setdefault("recovery_min_epochs", 1)
    controller.setdefault("freeze_schedule_in_recovery", True)
    # naming alias: spec uses freeze_discrete_updates
    controller.setdefault("freeze_discrete_updates", True)
    # keep legacy name for internal callers
    controller.setdefault("freeze_discrete_on_violate", controller.get("freeze_discrete_updates", True))
    controller.setdefault("k_exit", 2)
    controller.setdefault("margin_exit", 0.002)

    guard["controller"] = controller

    # Back-compat aliases (do NOT treat as authoritative)
    stable_hw["controller"] = controller  # deprecated alias
    guard.setdefault("metric_key", str(controller["metric"]))          # deprecated alias
    guard.setdefault("epsilon_drop", float(controller["epsilon_drop"]))# deprecated alias

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
    norm.setdefault("mode", str(norm.get("method", "hinge_log_ratio")))
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
                loss_lambda_hw = float(getattr(cfg.loss, "lambda_hw", 0.0) or 0.0)
                if loss_lambda_hw not in (0.0, 1.0):
                    print(
                        "[WARN] stable_hw is enabled; loss.lambda_hw is ignored by v5.4 NoDoubleScale. "
                        "Set loss.lambda_hw to 0/1 or remove it to avoid confusion."
                    )
                cfg.loss.lambda_hw = 0.0
            if hasattr(cfg, "hw") and hasattr(cfg.hw, "lambda_hw"):
                cfg.hw.lambda_hw = 0.0
            if hasattr(cfg, "stable_hw") and not hasattr(cfg.stable_hw, "no_double_scale"):
                cfg.stable_hw.no_double_scale = True
    except Exception:
        pass

    # ---- v5.4: sanitize hw refs early (avoid silent instability) ----
    for k, default in [
        ("latency_ref_ms", 1.0),
        ("energy_ref_mj", 1.0),
        ("mem_ref_mb", 1.0),
        ("comm_ref_ms", 1.0),
    ]:
        try:
            v = float(getattr(cfg.hw, k, default))
        except Exception:
            v = default
        if v <= 0.0:
            setattr(cfg.hw, k, float(default))

    return cfg
