from __future__ import annotations

from typing import Any, Dict
from types import SimpleNamespace

from omegaconf import OmegaConf

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


def _ensure_namespace(cfg: Any, key: str) -> Any:
    if not hasattr(cfg, key) or getattr(cfg, key) is None:
        setattr(cfg, key, AttrDict({}))
    return getattr(cfg, key)


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

    # ---- v5.4 FIX: accept DictConfig / AttrDict as mapping-like ----
    try:
        from omegaconf import DictConfig  # type: ignore
        mapping_types = (dict, AttrDict, DictConfig)
    except Exception:
        mapping_types = (dict, AttrDict)

    if not isinstance(stable_hw, mapping_types):
        return

    guard = stable_hw.get("accuracy_guard", {}) or {}
    if not isinstance(guard, mapping_types):
        guard = {}

    nds = stable_hw.get("no_double_scale", None)
    if isinstance(nds, bool):
        stable_hw["no_double_scale"] = {"enabled": bool(nds)}
    elif nds is None:
        stable_hw["no_double_scale"] = {"enabled": True}
    elif isinstance(nds, dict) and "enabled" not in nds:
        nds["enabled"] = True

    # old: stable_hw.controller -> new: stable_hw.accuracy_guard.controller
    old_ctrl = stable_hw.get("controller", {}) or {}
    if not isinstance(old_ctrl, mapping_types):
        old_ctrl = {}

    ctrl = guard.get("controller", {}) or {}
    if not isinstance(ctrl, mapping_types):
        ctrl = {}

    # merge: guard.controller has higher priority
    merged = dict(old_ctrl)
    merged.update(dict(ctrl))
    guard["controller"] = merged

    # alias migration for top-level guard params
    for k in ["metric_key", "epsilon_drop", "acc_margin", "allow_train_ema_fallback"]:
        if k in stable_hw and k not in guard:
            guard[k] = stable_hw[k]

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
    # ---- v5.4: infer train.mode for backward compatibility ----
    train_mode = get_nested(cfg, "train.mode", None)
    if train_mode is None:
        # accept legacy knobs but normalize into train.mode
        legacy_training_mode = get_nested(cfg, "training.mode", None)
        legacy_hw_mode = get_nested(cfg, "hw.mode", None)

        def _is_vc(x):
            if x is None:
                return False
            s = str(x).lower()
            return ("version_c" in s) or (s == "vc") or (s == "version-c")

        if _is_vc(legacy_training_mode) or _is_vc(legacy_hw_mode) or (mode == "version_c"):
            set_nested(cfg, "train.mode", "version_c")
        else:
            # keep None for non-version_c pipelines
            pass
    # always: common train defaults
    _apply_defaults(cfg, REQ_TRAIN_DEFAULTS)

    # compat: layout.* -> hw.*
    _sync_layout_to_hw(cfg)
    # ---- promote export keys: cfg.version_c.export_* -> cfg.export_* ----
    vc = getattr(cfg, "version_c", None)
    if vc is not None:
        if getattr(cfg, "export_layout_input", None) is None and getattr(vc, "export_layout_input", None) is not None:
            cfg.export_layout_input = bool(vc.export_layout_input)
        if getattr(cfg, "export_dir", None) is None and getattr(vc, "export_dir", None) is not None:
            cfg.export_dir = str(vc.export_dir)
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

    def _ensure(obj, key: str, value):
        if getattr(obj, key, None) is None:
            if isinstance(value, dict):
                setattr(obj, key, OmegaConf.create(value))
            else:
                setattr(obj, key, value)

    # -----------------------------
    # StableHW defaults for v5.4 (CONTRACT-SEALED)
    # -----------------------------
    _ensure(cfg, "stable_hw", {})
    stable_hw = cfg.stable_hw

    # v5.4 is holistic semantics: missing enabled MUST NOT silently disable.
    # Default policy: version_c => enabled defaults True; otherwise => defaults False.
    stable_hw_enabled = bool(get_nested(cfg, "stable_hw.enabled", (mode == "version_c")))
    set_nested(cfg, "stable_hw.enabled", stable_hw_enabled)
    stable_hw.enabled = stable_hw_enabled

    # helper: if submodule enabled missing -> inherit parent enabled
    def _inherit_enabled(path: str) -> bool:
        v = get_nested(cfg, path, None)
        if v is None:
            set_nested(cfg, path, stable_hw_enabled)
            v = stable_hw_enabled
        return bool(v)

    # forbid ambiguous configs: parent off but child explicitly on
    if not stable_hw_enabled:
        for p in (
            "stable_hw.normalize.enabled",
            "stable_hw.lambda_hw_schedule.enabled",
            "stable_hw.accuracy_guard.enabled",
            "stable_hw.locked_acc_ref.enabled",
            "stable_hw.no_drift.enabled",
            "stable_hw.no_double_scale.enabled",
        ):
            if get_nested(cfg, p, False) is True:
                raise ValueError(
                    f"[SPEC v5.4] Ambiguous StableHW config: {p}=True while stable_hw.enabled=False. "
                    f"Fix: set stable_hw.enabled=True or disable the submodule explicitly."
                )

    # --- normalize ---
    _ensure(stable_hw, "normalize", {})
    _ensure(stable_hw.normalize, "enabled", _inherit_enabled("stable_hw.normalize.enabled"))
    _ensure(stable_hw.normalize, "kind", get_nested(cfg, "stable_hw.normalize.kind", "log_ratio_hinge"))
    _ensure(stable_hw.normalize, "hinge_tau", float(get_nested(cfg, "stable_hw.normalize.hinge_tau", 0.02)))
    _ensure(stable_hw.normalize, "eps", float(get_nested(cfg, "stable_hw.normalize.eps", 1e-8)))

    # --- lambda schedule ---
    _ensure(stable_hw, "lambda_hw_schedule", {})
    _ensure(stable_hw.lambda_hw_schedule, "enabled", _inherit_enabled("stable_hw.lambda_hw_schedule.enabled"))
    _ensure(stable_hw.lambda_hw_schedule, "kind", get_nested(cfg, "stable_hw.lambda_hw_schedule.kind", "linear_warmup_hold"))
    _ensure(stable_hw.lambda_hw_schedule, "start_step", int(get_nested(cfg, "stable_hw.lambda_hw_schedule.start_step", 0)))
    _ensure(stable_hw.lambda_hw_schedule, "warmup_steps", int(get_nested(cfg, "stable_hw.lambda_hw_schedule.warmup_steps", 200)))
    _ensure(stable_hw.lambda_hw_schedule, "lambda_hw_min", float(get_nested(cfg, "stable_hw.lambda_hw_schedule.lambda_hw_min", 0.0)))
    _ensure(
        stable_hw.lambda_hw_schedule,
        "lambda_hw_max",
        float(get_nested(cfg, "stable_hw.lambda_hw_schedule.lambda_hw_max", float(get_nested(cfg, "hw.lambda_hw", 0.0)))),
    )

    # --- accuracy guard (HardGating) ---
    _ensure(stable_hw, "accuracy_guard", {})
    _ensure(stable_hw.accuracy_guard, "enabled", _inherit_enabled("stable_hw.accuracy_guard.enabled"))
    _ensure(stable_hw.accuracy_guard, "metric", get_nested(cfg, "stable_hw.accuracy_guard.metric", "acc1"))
    _ensure(
        stable_hw.accuracy_guard,
        "acc_drop_max",
        float(get_nested(cfg, "stable_hw.accuracy_guard.acc_drop_max", float(get_nested(cfg, "train.acc_drop_max", 0.0)))),
    )

    # --- locked acc ref ---
    _ensure(stable_hw, "locked_acc_ref", {})
    _ensure(stable_hw.locked_acc_ref, "enabled", _inherit_enabled("stable_hw.locked_acc_ref.enabled"))
    _ensure(stable_hw.locked_acc_ref, "source", get_nested(cfg, "stable_hw.locked_acc_ref.source", "auto"))
    _ensure(
        stable_hw.locked_acc_ref,
        "expected_acc1",
        float(get_nested(cfg, "stable_hw.locked_acc_ref.expected_acc1", float(get_nested(cfg, "train.expected_acc1", 0.0)))),
    )
    _ensure(stable_hw.locked_acc_ref, "path", get_nested(cfg, "stable_hw.locked_acc_ref.path", get_nested(cfg, "train.acc_ref_path", None)))

    # --- no drift ---
    _ensure(stable_hw, "no_drift", {})
    _ensure(stable_hw.no_drift, "enabled", _inherit_enabled("stable_hw.no_drift.enabled"))
    _ensure(stable_hw.no_drift, "mode", get_nested(cfg, "stable_hw.no_drift.mode", "frozen"))

    # --- no double scale ---
    _ensure(stable_hw, "no_double_scale", {})
    _ensure(stable_hw.no_double_scale, "enabled", _inherit_enabled("stable_hw.no_double_scale.enabled"))

    if stable_hw_enabled and not bool(get_nested(cfg, "stable_hw.no_double_scale.enabled", True)):
        raise ValueError("[SPEC v5.4] stable_hw.no_double_scale.enabled MUST be True when stable_hw.enabled=True.")

    # NoDoubleScale contract: legacy lambdas must be exactly 0 when stable_hw is enabled
    if stable_hw_enabled:
        if abs(float(get_nested(cfg, "hw.lambda_hw", 0.0))) > 1e-12:
            raise ValueError("[SPEC v5.4] hw.lambda_hw MUST be 0 when stable_hw.enabled=True (NoDoubleScale).")
        if abs(float(get_nested(cfg, "loss.lambda_hw", 0.0))) > 1e-12:
            raise ValueError("[SPEC v5.4] loss.lambda_hw MUST be 0 when stable_hw.enabled=True (NoDoubleScale).")

    if mode == "version_c":
        _apply_defaults(cfg, REQ_VERSION_C_HW_DEFAULTS)
        _apply_defaults(cfg, REQ_VERSION_C_TRAINING_DEFAULTS)
        _apply_defaults(cfg, REQ_CHIPLET_DEFAULTS)
        # ---- v5.4 NoDoubleScale contract (Version-C must enable) ----
        if get_nested(cfg, "stable_hw.enabled", True):
            nds = get_nested(cfg, "stable_hw.no_double_scale", None)
            if nds is None:
                set_nested(cfg, "stable_hw.no_double_scale", True)
            elif bool(nds) is False:
                raise ValueError(
                    "[SPEC v5.4] stable_hw.no_double_scale must be True for Version-C runs "
                    "(NoDoubleScale / LockedAccRef / NoDrift contract)."
                )
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

        # ------------------------------------------------------------
        # Layout-mode default: still emit StableHW signature knobs explicitly
        # (so trace signature can be strict per SPEC_E without silent defaults).
        # ------------------------------------------------------------
        stable_hw = _ensure_namespace(cfg, "stable_hw")
        stable_hw.enabled = bool(getattr(stable_hw, "enabled", False))

        ag = _ensure_namespace(stable_hw, "accuracy_guard")
        ag.enabled = bool(getattr(ag, "enabled", False))
        if getattr(ag, "metric", None) is None:
            ag.metric = "top1"
        if getattr(ag, "acc_threshold_rel_to_ref", None) is None:
            ag.acc_threshold_rel_to_ref = 1.0

        lar = _ensure_namespace(stable_hw, "locked_acc_ref")
        lar.enabled = bool(getattr(lar, "enabled", False))
        if getattr(lar, "source", None) is None:
            lar.source = "none"

        nd = _ensure_namespace(stable_hw, "no_drift")
        nd.enabled = bool(getattr(nd, "enabled", False))

        if not hasattr(stable_hw, "no_double_scale"):
            stable_hw.no_double_scale = False

        # ---- v5.4: unify budget fields for layout pipelines ----
        if not hasattr(cfg, "budget") or cfg.budget is None:
            cfg.budget = AttrDict({})
        budget = cfg.budget

        if getattr(budget, "total_eval_budget", None) is None:
            max_eval_calls = getattr(getattr(cfg, "baseline", None), "max_eval_calls", None)
            steps = getattr(getattr(cfg, "detailed_place", None), "steps", None)
            if max_eval_calls is not None:
                budget.total_eval_budget = int(max_eval_calls)
            elif steps is not None:
                budget.total_eval_budget = int(steps) + 2
            else:
                budget.total_eval_budget = 0

        if getattr(budget, "max_wallclock_sec", None) is None:
            mw = getattr(getattr(cfg, "baseline", None), "max_wallclock_sec", None)
            mr = getattr(getattr(cfg, "detailed_place", None), "max_runtime_sec", None)
            if mw is not None:
                budget.max_wallclock_sec = float(mw)
            elif mr is not None:
                budget.max_wallclock_sec = float(mr)
            else:
                budget.max_wallclock_sec = 0.0

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
    # IMPORTANT:
    #   1) Do NOT silently override schedule fields twice.
    #   2) Default stable_hw.enabled depends on mode to avoid "ast2 configs silently changed".
    #   3) NoDoubleScale ONLY enforced when stable_hw.enabled=True.

    stable_hw = getattr(cfg, "stable_hw", None)
    if stable_hw is None:
        from omegaconf import OmegaConf
        stable_hw = OmegaConf.create({})
        cfg.stable_hw = stable_hw

    # migrate legacy keys -> v5 structure (safe no-op if already v5)
    _migrate_stable_hw_to_v5(cfg)
    stable_hw = cfg.stable_hw

    # default enable policy
    default_enabled = True if str(mode) in ("version_c", "single_device") else False
    if getattr(stable_hw, "enabled", None) is None:
        stable_hw.enabled = bool(default_enabled)
    stable_hw_enabled = bool(stable_hw.enabled)

    # allow legacy alias: stable_hw.allow_train_ema_fallback -> stable_hw.accuracy_guard.allow_train_ema_fallback
    if getattr(stable_hw, "allow_train_ema_fallback", None) is not None:
        if getattr(getattr(stable_hw, "accuracy_guard", None), "allow_train_ema_fallback", None) is None:
            if getattr(stable_hw, "accuracy_guard", None) is None:
                from omegaconf import OmegaConf
                stable_hw.accuracy_guard = OmegaConf.create({})
            stable_hw.accuracy_guard.allow_train_ema_fallback = bool(stable_hw.allow_train_ema_fallback)

    # ---- locked acc ref (v5) ----
    # v5.4 contract: locked_acc_ref is defined EITHER at root (preferred) OR under stable_hw (legacy), not both.
    root_locked = getattr(cfg, "locked_acc_ref", None)
    nested_locked = getattr(stable_hw, "locked_acc_ref", None)

    if root_locked is not None and nested_locked is not None:
        raise ValueError(
            "locked_acc_ref must be defined only once (root preferred). Remove one of: "
            "locked_acc_ref OR stable_hw.locked_acc_ref"
        )

    # If neither exists, create the legacy nested container (so downstream defaults can still be applied)
    if root_locked is None and nested_locked is None:
        from omegaconf import OmegaConf

        stable_hw.locked_acc_ref = OmegaConf.create({})
        nested_locked = stable_hw.locked_acc_ref

    # Apply defaults onto whichever container is actually used
    locked = root_locked if root_locked is not None else nested_locked

    if getattr(locked, "enabled", None) is None:
        locked.enabled = True
    locked.enabled = bool(locked.enabled)
    locked.setdefault("freeze_epoch", 0)
    locked.setdefault("warmup_epochs", 1)
    locked.setdefault("ref_source", "best_warmup_val")
    locked.setdefault("baseline_stats_path", None)
    locked.setdefault("prefer_dense_baseline", True)
    locked.setdefault("acc_margin", 0.0)
    locked.setdefault("min_acc_ref", 0.0)
    # ---- ensure locked_acc_ref.source exists (stable_hw reads `source`, not `ref_source`) ----
    if getattr(locked, "source", None) is None:
        rs = getattr(locked, "ref_source", None)
        locked.source = str(rs) if rs is not None else "warmup_best"

    # ---- accuracy guard (v5) ----
    if getattr(stable_hw, "accuracy_guard", None) is None:
        from omegaconf import OmegaConf
        stable_hw.accuracy_guard = OmegaConf.create({})
    guard = stable_hw.accuracy_guard
    if getattr(guard, "enabled", None) is None:
        guard.enabled = True
    guard.enabled = bool(guard.enabled)
    guard.setdefault("epsilon_drop", 0.002)
    guard.setdefault("guard_mode", "hard")
    guard.setdefault("freeze_hw_on_drop", True)
    guard.setdefault("freeze_discrete_on_drop", True)
    guard.setdefault("freeze_alpha_on_drop", False)
    guard.setdefault("prefer_val_metric", True)
    guard.setdefault("allow_train_ema_fallback", False)

    if getattr(guard, "controller", None) is None:
        from omegaconf import OmegaConf
        guard.controller = OmegaConf.create({})
    ctrl = guard.controller
    ctrl.setdefault("max_bad_epochs", 1)
    ctrl.setdefault("lr_restart_mul", 2.0)
    ctrl.setdefault("cooldown_epochs", 1)
    ctrl.setdefault("recovery_min_epochs", 1)
    ctrl.setdefault("recovery_mode", "freeze_discrete_and_hw")
    ctrl.setdefault("resume_hw_after", "val_recovers")
    ctrl.setdefault("acc_ema_beta", 0.9)
    ctrl.setdefault("train_ema_gate_eps", 0.001)
    ctrl.setdefault("log_prefix", "StableHW")

    # ---- lambda schedule (v5) ----
    if getattr(stable_hw, "lambda_hw_schedule", None) is None:
        from omegaconf import OmegaConf
        stable_hw.lambda_hw_schedule = OmegaConf.create({})
    sched = stable_hw.lambda_hw_schedule
    if getattr(sched, "enabled", None) is None:
        sched.enabled = True
    sched.enabled = bool(sched.enabled)

    # defaults MUST match v5.4 intent (NOT 0.0)
    sched.setdefault("warmup_epochs", 5)
    sched.setdefault("ramp_epochs", 10)

    # ---- alias bridge: do NOT override user configs written in older keys ----
    # If user provided max_lambda/min_lambda but not lambda_hw_max/min, map them first.
    if getattr(sched, "lambda_hw_max", None) is None and getattr(sched, "max_lambda", None) is not None:
        sched.lambda_hw_max = float(sched.max_lambda)
    if getattr(sched, "lambda_hw_min", None) is None and getattr(sched, "min_lambda", None) is not None:
        sched.lambda_hw_min = float(sched.min_lambda)

    # Now apply true defaults only when still missing.
    if getattr(sched, "lambda_hw_min", None) is None:
        sched.lambda_hw_min = 0.0
    if getattr(sched, "lambda_hw_max", None) is None:
        sched.lambda_hw_max = 0.2

    # clamp defaults
    sched.setdefault("clamp_min", float(sched.lambda_hw_min))
    sched.setdefault("clamp_max", float(sched.lambda_hw_max))

    # ---- normalization defaults (v5) ----
    if getattr(stable_hw, "normalize", None) is None:
        from omegaconf import OmegaConf
        stable_hw.normalize = OmegaConf.create({})
    norm = stable_hw.normalize
    if getattr(norm, "enabled", None) is None:
        norm.enabled = True
    norm.enabled = bool(norm.enabled)
    norm.setdefault("mode", "hinge_log_ratio")
    norm.setdefault("wT", 1.0)
    norm.setdefault("wE", 0.0)
    norm.setdefault("wM", 0.0)
    norm.setdefault("wC", 0.0)
    norm.setdefault("clip_term_max", 2.0)
    norm.setdefault("eps", 1e-6)

    # ===== v5.4 defaults: NoDrift + frozen HW refs =====
    if getattr(cfg, "stable_hw", None) is not None and bool(getattr(cfg.stable_hw, "enabled", False)):
        # v5.4 contract: no_drift is defined only once (root preferred).
        root_no_drift = getattr(cfg, "no_drift", None)
        nested_no_drift = getattr(cfg.stable_hw, "no_drift", None)

        if root_no_drift is not None and nested_no_drift is not None:
            raise ValueError(
                "no_drift must be defined only once (root preferred). Remove one of: "
                "no_drift OR stable_hw.no_drift"
            )

        # Only create nested default when root is absent AND nested is absent
        if root_no_drift is None and nested_no_drift is None:
            cfg.stable_hw.no_drift = SimpleNamespace(enabled=True)
            nested_no_drift = cfg.stable_hw.no_drift

        # Determine enabled flag from root (dict-style) or nested (bool/dict-style)
        no_drift_enabled = True
        if root_no_drift is not None:
            no_drift_enabled = (
                bool(getattr(root_no_drift, "enabled", True))
                if hasattr(root_no_drift, "enabled")
                else bool(root_no_drift)
            )
        else:
            if hasattr(nested_no_drift, "enabled"):
                no_drift_enabled = bool(nested_no_drift.enabled)
            else:
                no_drift_enabled = bool(nested_no_drift)

        if not hasattr(cfg.stable_hw, "normalize") or cfg.stable_hw.normalize is None:
            cfg.stable_hw.normalize = SimpleNamespace(enabled=True)

        if not hasattr(cfg.stable_hw.normalize, "ref_update") or cfg.stable_hw.normalize.ref_update is None:
            cfg.stable_hw.normalize.ref_update = "frozen"

        # NoDrift priority: enforce frozen even if user mistakenly sets ema
        if no_drift_enabled:
            cfg.stable_hw.normalize.ref_update = "frozen"

        # Canonicalize: always expose dict-style stable_hw.no_drift with .enabled for downstream code.
        _cur_nd = getattr(cfg.stable_hw, "no_drift", None)
        if _cur_nd is None or not hasattr(_cur_nd, "enabled"):
            cfg.stable_hw.no_drift = SimpleNamespace(enabled=bool(no_drift_enabled))
        else:
            _cur_nd.enabled = bool(no_drift_enabled)

        v = str(cfg.stable_hw.normalize.ref_update).lower()
        if v not in ("frozen", "ema"):
            raise ValueError(
                "stable_hw.normalize.ref_update must be 'frozen' or 'ema', got: "
                f"{cfg.stable_hw.normalize.ref_update}"
            )

        print(
            "[v5.4][cfg] locked_acc_ref source = "
            f"{'root' if getattr(cfg, 'locked_acc_ref', None) is not None else 'stable_hw'}"
        )
        print(
            "[v5.4][cfg] no_drift source = "
            f"{'root' if getattr(cfg, 'no_drift', None) is not None else 'stable_hw'}"
        )

    # --- v5.4 contract: metric name must be consistent ---
    try:
        if getattr(cfg, "stable_hw", None) is not None and bool(getattr(cfg.stable_hw, "enabled", False)):
            g = getattr(cfg.stable_hw, "accuracy_guard", None)
            l = getattr(cfg.stable_hw, "locked_acc_ref", None)
            eval_cfg = getattr(cfg, "eval", None)

            guard_metric = None
            if g is not None:
                ctrl = getattr(g, "controller", None)
                guard_metric = getattr(ctrl, "metric", None) if ctrl is not None else getattr(g, "metric_name", None)

            locked_metric = getattr(l, "metric_name", None) if l is not None else None
            eval_metric = getattr(eval_cfg, "acc_metric_name", None) if eval_cfg is not None else None

            metrics = [m for m in [guard_metric, locked_metric, eval_metric] if m is not None]
            if len(set(metrics)) > 1:
                raise ValueError(
                    f"[v5.4] metric mismatch: guard={guard_metric} locked_acc_ref={locked_metric} "
                    f"eval={eval_metric}. These must be identical to satisfy LockedAccRef/NoDrift semantics."
                )
    except Exception:
        raise

    # ---- discrete isolation defaults (v5) ----
    if getattr(stable_hw, "discrete_isolation", None) is None:
        from omegaconf import OmegaConf
        stable_hw.discrete_isolation = OmegaConf.create({})
    iso = stable_hw.discrete_isolation
    iso.setdefault("enabled", True)
    iso.setdefault("mapping_update_every_epochs", 1)
    iso.setdefault("layout_update_every_epochs", 1)
    iso.setdefault("cache_mapping_layout", True)
    iso.setdefault("track_live_segments", False)
    iso.setdefault("allow_cache_fallback", True)

    # ===== v5.4 NoDoubleScale: stable_hw enabled => legacy lambda_hw MUST be 0 (warn + override) =====
    stable_en = bool(get_nested(cfg, "stable_hw.enabled", False))
    legacy_hw_lam = float(getattr(getattr(cfg, "hw", {}), "lambda_hw", 0.0) or 0.0)
    legacy_loss_lam = float(getattr(getattr(cfg, "loss", {}), "lambda_hw", 0.0) or 0.0)

    if stable_en and (legacy_hw_lam != 0.0 or legacy_loss_lam != 0.0):
        print(
            f"[WARN] NoDoubleScale(v5.4): overriding legacy lambdas to 0.0 "
            f"(hw.lambda_hw={legacy_hw_lam}, loss.lambda_hw={legacy_loss_lam})."
        )
        if hasattr(cfg, "hw"):
            cfg.hw.lambda_hw = 0.0
        if hasattr(cfg, "loss"):
            cfg.loss.lambda_hw = 0.0

    if (not stable_en) and (legacy_hw_lam > 0.0 or legacy_loss_lam > 0.0):
        raise ValueError(
            "v5.4 Acc-First requires stable_hw.enabled=true for any HW optimization. "
            f"Found stable_hw.enabled=false but hw.lambda_hw={legacy_hw_lam}, loss.lambda_hw={legacy_loss_lam}. "
            "Fix: set both lambdas to 0 for pure-accuracy runs OR enable stable_hw and use lambda_hw_effective."
        )

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

    # =========================
    # v5.4 CONTRACT ENFORCEMENT
    # =========================
    # v5.4 原则：未显式写 enabled ≠ False；并且在 version_c 训练语义下，不允许悄悄退化
    # 统一以 cfg.stable_hw.* 为单一事实源，并同步 root-level shim，避免“读到不同开关”的漂移

    # ---- mirror stable_hw -> root-level shims (single source of truth) ----
    if hasattr(cfg, "stable_hw") and cfg.stable_hw is not None:
        _nd = getattr(cfg.stable_hw, "no_drift", None)
        _nds = getattr(cfg.stable_hw, "no_double_scale", None)
        _lar = getattr(cfg.stable_hw, "locked_acc_ref", None)

        # keep shims present + consistent
        if not hasattr(cfg, "no_drift") or cfg.no_drift is None:
            cfg.no_drift = OmegaConf.create({})
        cfg.no_drift.enabled = bool(getattr(_nd, "enabled", bool(_nd)) if _nd is not None else False)

        if not hasattr(cfg, "no_double_scale") or cfg.no_double_scale is None:
            cfg.no_double_scale = OmegaConf.create({})
        cfg.no_double_scale.enabled = bool(getattr(_nds, "enabled", bool(_nds)) if _nds is not None else False)

        if not hasattr(cfg, "locked_acc_ref") or cfg.locked_acc_ref is None:
            cfg.locked_acc_ref = OmegaConf.create({})
        cfg.locked_acc_ref.enabled = bool(getattr(_lar, "enabled", bool(_lar)) if _lar is not None else False)

    # ---- hard contract: version_c + stable_hw.enabled => all core submodules must be enabled
    #      unless stable_hw.force_disable_ok=true (explicit ablation escape hatch)
    train_mode_now = str(get_nested(cfg, "train.mode", "baseline") or "baseline")
    stable_hw_enabled_now = bool(get_nested(cfg, "stable_hw.enabled", True))
    force_disable_ok_now = bool(get_nested(cfg, "stable_hw.force_disable_ok", False))

    if train_mode_now == "version_c" and stable_hw_enabled_now and not force_disable_ok_now:
        required = {
            "stable_hw.normalize.enabled": bool(get_nested(cfg, "stable_hw.normalize.enabled", True)),
            "stable_hw.lambda_hw_schedule.enabled": bool(get_nested(cfg, "stable_hw.lambda_hw_schedule.enabled", True)),
            "stable_hw.accuracy_guard.enabled": bool(get_nested(cfg, "stable_hw.accuracy_guard.enabled", True)),
            "stable_hw.locked_acc_ref.enabled": bool(get_nested(cfg, "stable_hw.locked_acc_ref.enabled", True)),
            "stable_hw.no_drift.enabled": bool(get_nested(cfg, "stable_hw.no_drift.enabled", True)),
            "stable_hw.no_double_scale.enabled": bool(get_nested(cfg, "stable_hw.no_double_scale.enabled", True)),
        }
        bad = [k for k, v in required.items() if not v]
        if bad:
            raise ValueError(
                "v5.4 contract violation: version_c requires StableHW core submodules enabled. "
                f"Missing/disabled: {bad}. "
                "If you really want to disable (ablation), set stable_hw.force_disable_ok=true explicitly."
            )

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
    # ---- v5.4: ensure min_latency_ms is positive when stable_hw enabled ----
    try:
        if bool(getattr(getattr(cfg, "stable_hw", None), "enabled", False)):
            v = float(getattr(cfg.stable_hw, "min_latency_ms", 0.0) or 0.0)
            if v <= 0.0:
                cfg.stable_hw.min_latency_ms = 1e-3
    except Exception:
        pass

    # ---- v5.4 strict contracts for Version-C mode ----
    mode = str(get_nested(cfg, "train.mode", "") or "")
    if mode == "version_c":
        # Escape hatch ONLY for explicit ablations (default False).
        force_disable_ok = bool(get_nested(cfg, "stable_hw.force_disable_ok", False))

        stable_en = bool(get_nested(cfg, "stable_hw.enabled", False))
        # v5.4 allows locked_acc_ref at root-level OR under stable_hw (legacy), but not both.
        locked_root = get_nested(cfg, "locked_acc_ref", None)
        if locked_root is not None:
            locked_en = bool(locked_root.get("enabled", False))
        else:
            locked_nested = get_nested(cfg, "stable_hw.locked_acc_ref", {}) or {}
            locked_en = bool(locked_nested.get("enabled", False))

        if (not stable_en) and (not force_disable_ok):
            raise ValueError(
                "SPEC v5.4 contract violation: train.mode=version_c requires stable_hw.enabled=true. "
                "If you are intentionally running an ablation, set stable_hw.force_disable_ok=true explicitly."
            )

        if (not locked_en) and (not force_disable_ok):
            raise ValueError(
                "SPEC v5.4 contract violation: train.mode=version_c requires stable_hw.locked_acc_ref.enabled=true. "
                "If you are intentionally running an ablation, set stable_hw.force_disable_ok=true explicitly."
            )

    # ---- v5.4 Addendum: signature must be assign-only ----
    if bool(get_nested(cfg, "signature.allow_pos_signature", False)):
        raise ValueError(
            "SPEC v5.4 signature must be assign-only. "
            "Please set signature.allow_pos_signature=false."
        )

    return cfg
