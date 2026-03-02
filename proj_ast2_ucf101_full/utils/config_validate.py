from __future__ import annotations

from typing import Any, Dict
import json
import os
from types import SimpleNamespace
from pathlib import Path

from omegaconf import OmegaConf, DictConfig

from .config_utils import get_nested, set_nested
from .config import AttrDict

# NOTE(v5.4): strict=True is the ONLY valid behavior for new runs.
# non-strict exists solely for legacy compatibility and MUST write evidence into _contract.overrides.

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
    # Align repository default (configs/*.yaml) to avoid pointing to missing ckpts.
    "hw.device_name": "RTX3090_FP16",
    "hw.gpu_yaml": "configs/gpu_data.yaml",
    "hw.proxy_weight_dir": "./proxy_ckpts",

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


def _strip_contract(container: Any) -> Any:
    """
    Produce LEGAL config snapshots for evidence-chain:
      - remove all meta keys starting with "_contract"
      - remove top-level key "contract" (meta, not training semantics)
    """
    if not isinstance(container, dict):
        return container
    out = {}
    for k, v in container.items():
        if isinstance(k, str) and (k.startswith("_contract") or k == "contract"):
            continue
        if isinstance(v, dict):
            out[k] = _strip_contract(v)
        else:
            out[k] = v
    return out


def _flatten_diff(path: str, req: Any, eff: Any, out: list[tuple[str, Any, Any]]) -> None:
    # req 缺失：用哨兵 "__MISSING__" 记录（门禁可审计）
    if isinstance(req, dict) and isinstance(eff, dict):
        keys = sorted(set(req.keys()) | set(eff.keys()))
        for k in keys:
            _flatten_diff(
                f"{path}.{k}" if path else str(k),
                req.get(k, "__MISSING__"),
                eff.get(k, "__MISSING__"),
                out,
            )
        return
    if isinstance(req, list) and isinstance(eff, list):
        n = max(len(req), len(eff))
        for i in range(n):
            _flatten_diff(
                f"{path}[{i}]",
                req[i] if i < len(req) else "__MISSING__",
                eff[i] if i < len(eff) else "__MISSING__",
                out,
            )
        return
    # 叶子：值不同则记录
    if req != eff:
        out.append((path, req, eff))


def _augment_contract_overrides(contract: Any) -> None:
    if isinstance(contract, dict):
        req = contract.get("requested_config_snapshot", None) or {}
        eff = contract.get("effective_config_snapshot", None) or {}
        overrides = contract.setdefault("overrides", [])
    else:
        req = getattr(contract, "requested_config_snapshot", None) or {}
        eff = getattr(contract, "effective_config_snapshot", None) or {}
        if getattr(contract, "overrides", None) is None:
            contract.overrides = []
        overrides = contract.overrides
    diffs: list[tuple[str, Any, Any]] = []
    _flatten_diff("", req, eff, diffs)

    existing = set()
    for o in overrides or []:
        if isinstance(o, dict):
            p = o.get("path")
            if isinstance(p, str) and p.strip():
                existing.add(p.strip())

    for (p, r, e) in diffs:
        # 防止根 diff 产生空 path
        if not isinstance(p, str) or not p.strip():
            continue
        p = p.strip()
        if p in existing:
            continue
        overrides.append(
            {
                "path": p,
                "requested": r,
                "effective": e,
                "reason": "auto_diff_requested_vs_effective",
            }
        )


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


def _new_empty_ns(parent: Any):
    """
    Create an empty mapping node compatible with the parent container.
    - If parent is OmegaConf DictConfig: return OmegaConf.create({})
    - Else (AttrDict or others): return AttrDict({})
    """
    try:
        if isinstance(parent, DictConfig):
            return OmegaConf.create({})
    except Exception:
        pass
    return AttrDict({})


def _ensure_namespace(cfg: Any, key: str) -> Any:
    if not hasattr(cfg, key) or getattr(cfg, key) is None:
        setattr(cfg, key, _new_empty_ns(cfg))
    return getattr(cfg, key)


def _record_resolved_aliases(cfg: Any, alias_pairs: list[tuple[str, str]]) -> None:
    resolved = get_nested(cfg, "_resolved_aliases", None)
    if resolved is None or not isinstance(resolved, dict):
        resolved = {}
        set_nested(cfg, "_resolved_aliases", resolved)
    for old_path, new_path in alias_pairs:
        val = get_nested(cfg, old_path, None)
        if val is not None:
            resolved[old_path] = {"mapped_to": new_path, "value": val}
            set_nested(cfg, old_path, None)
    set_nested(cfg, "_resolved_aliases", resolved)


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
            "baseline_stats_path": None,
            "strict": True,
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
    cfg_contract = get_nested(cfg, "_contract", None)
    if cfg_contract is None:
        set_nested(cfg, "_contract", {})
    if get_nested(cfg, "_contract.requested_config_snapshot", None) is None:
        requested_snapshot_raw = OmegaConf.to_container(cfg, resolve=False)
        set_nested(cfg, "_contract.requested_config_snapshot", _strip_contract(requested_snapshot_raw))
    if get_nested(cfg, "_contract.overrides", None) is None:
        set_nested(cfg, "_contract.overrides", [])

    # ---- v5.4 strict gate (Hard Gate A) ----
    # v5.4 strict 默认开启；若显式关闭必须进入证据链 overrides
    if get_nested(cfg, "_contract.strict", None) is None:
        set_nested(cfg, "_contract.strict", True)
        cfg._contract.overrides.append(
            {
                "path": "_contract.strict",
                "requested": None,
                "effective": True,
                "reason": "default_strict_v54",
            }
        )

    STRICT = bool(get_nested(cfg, "_contract.strict", False))
    # -------------------------------------------------------------------------
    # v5.4 convenience: many configs reference ${out_dir}. If the config only
    # defines train.out_dir, provide a root-level alias to avoid interpolation
    # failures during OmegaConf resolve (e.g., smoke_check_config_no_drift).
    # -------------------------------------------------------------------------
    if get_nested(cfg, "out_dir", None) is None:
        _train_out = get_nested(cfg, "train.out_dir", None)
        if isinstance(_train_out, str) and _train_out.strip():
            set_nested(cfg, "out_dir", _train_out)
            try:
                cfg._contract.overrides.append(
                    {
                        "path": "out_dir",
                        "requested": None,
                        "effective": _train_out,
                        "reason": "alias_train_out_dir_for_interpolation",
                    }
                )
            except Exception:
                pass
    # ---- v5.4 helper: ensure NEW tabular proxy ckpts dir exists (no command change required) ----
    try:
        hw_proxy_dir = get_nested(cfg, "hw.proxy_weight_dir", None)
        hw_device = get_nested(cfg, "hw.device_name", None)

        if isinstance(hw_proxy_dir, str) and isinstance(hw_device, str):
            # normalize "./proxy_ckpts" / "proxy_ckpts" / "./proxy_ckpts/"
            norm = hw_proxy_dir.rstrip("/")
            if norm in ("./proxy_ckpts", "proxy_ckpts"):
                from pathlib import Path as _Path
                from .proxy_ckpt_links import ensure_proxy_ckpts_dir

                project_root = _Path(__file__).resolve().parents[1]
                # IMPORTANT: per-device subdir
                dst_dir = (project_root / norm / hw_device).resolve()
                ensure_proxy_ckpts_dir(project_root, hw_device, dst_dir)
    except Exception as _e:
        raise RuntimeError(f"[v5.4][ProxyCkptResolveError] {_e}") from _e
    if not STRICT:
        cfg._contract.overrides.append(
            {
                "path": "_contract.strict",
                "requested": False,
                "effective": False,
                "reason": "explicit_strict_disable",
            }
        )

    # --- v5.4 strict hard gate A: forbid legacy/root-level keys (single source of truth) ---
    FORBIDDEN_LEGACY_ROOT_KEYS_V54 = [
        "locked_acc_ref",
        "no_drift",
        "no_double_scale",
        "hard_gating",
        "accuracy_guard",
        "no_drift_enabled",
        "locked_acc_ref_enabled",
        "hard_gating_enabled",
        "acc_first_enabled",
        "no_double_scale_enabled",
        "acc_first_hard_gating_enabled",
    ]

    # -----------------------------------------------------------------------------

    def _fail_on_legacy(path: str, hint: str):
        v = get_nested(cfg, path, None)
        if v is not None:
            raise ValueError(f"v5.4 strict: legacy key '{path}' is forbidden. {hint}")

    if STRICT:
        for p in [
            "no_drift",
            "locked_acc_ref",
            "no_double_scale",
            "accuracy_guard",
            "hard_gating",
            "no_drift_enabled",
            "locked_acc_ref_enabled",
            "hard_gating_enabled",
            "acc_first_enabled",
            "no_double_scale_enabled",
            "acc_first_hard_gating_enabled",
            "layout.optimize_layout",
            "layout.num_slots",
            "layout.wafer_radius_mm",
            "layout.site_margin_mm",
            "layout.lambda_boundary",
            "layout.lambda_overlap",
            "layout.lambda_comm_extra",
            "layout.lambda_thermal",
        ]:
            _fail_on_legacy(p, "Use canonical cfg.stable_hw.* and cfg.hw.* only.")

    def _stamp_contract(cfg_to_stamp: Any) -> Any:
        # ===== v5.4 contract stamp (MUST exist for any v5.4 run) =====
        try:
            OmegaConf.set_struct(cfg_to_stamp, False)
        except Exception:
            pass
        if not hasattr(cfg_to_stamp, "contract") or getattr(cfg_to_stamp, "contract") is None:
            cfg_to_stamp.contract = _new_empty_ns(cfg_to_stamp)
        cfg_to_stamp.contract.version = "v5.4"
        cfg_to_stamp.contract.validated = True
        cfg_to_stamp.contract.validated_by = "validate_and_fill_defaults"
        cfg_to_stamp.contract.strict = bool(get_nested(cfg_to_stamp, "_contract.strict", False))
        from .trace_contract_v54 import compute_effective_cfg_digest_v54
        # LEGAL: seal_digest = sha256(effective_config_snapshot) with meta stripped
        effective = get_nested(cfg_to_stamp, "_contract.effective_config_snapshot", None)
        if effective is None:
            effective = _strip_contract(OmegaConf.to_container(cfg_to_stamp, resolve=True))
        cfg_to_stamp.contract.seal_digest = compute_effective_cfg_digest_v54(effective)
        set_nested(cfg_to_stamp, "_contract.stamped_v54", True)
        return cfg_to_stamp
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

    # ============================
    # v5.4 Contract: SINGLE SOURCE OF TRUTH
    # - locked_acc_ref / no_drift / no_double_scale MUST exist in ONLY ONE place
    # - If user provides legacy top-level, migrate -> cfg.stable_hw and DELETE top-level
    # - If both provided, HARD FAIL (P0)
    # ============================
    for k in ("locked_acc_ref", "no_drift", "no_double_scale"):
        root_obj = None
        try:
            root_obj = cfg.get(k, None)
        except Exception:
            root_obj = getattr(cfg, k, None)
        nested_obj = None
        if getattr(cfg, "stable_hw", None) is not None:
            try:
                nested_obj = cfg.stable_hw.get(k, None)
            except Exception:
                nested_obj = getattr(cfg.stable_hw, k, None)

        if root_obj is not None and nested_obj is not None:
            raise ValueError(
                f"v5.4 contract violation: both '{k}' and 'stable_hw.{k}' are set. "
                f"Keep ONLY one. (SPEC_E smoke requires this)"
            )

        if root_obj is not None and nested_obj is None:
            if STRICT:
                raise ValueError(
                    f"v5.4 strict: legacy root key '{k}' is forbidden. "
                    f"Move it to 'stable_hw.{k}' explicitly."
                )
            # non-strict only: migrate legacy -> stable_hw
            try:
                cfg.stable_hw[k] = root_obj
            except Exception:
                setattr(cfg.stable_hw, k, root_obj)
            # delete top-level to avoid ambiguity
            try:
                del cfg[k]
            except Exception:
                try:
                    cfg.pop(k)
                except Exception:
                    pass

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
    stable_hw_force_disable_ok = bool(get_nested(cfg, "stable_hw.force_disable_ok", False))
    if not stable_hw_enabled:
        bad = []
        for p in (
            "stable_hw.normalize.enabled",
            "stable_hw.lambda_hw_schedule.enabled",
            "stable_hw.accuracy_guard.enabled",
            "stable_hw.locked_acc_ref.enabled",
            "stable_hw.no_drift.enabled",
            "stable_hw.no_double_scale.enabled",
        ):
            if get_nested(cfg, p, False) is True:
                bad.append(p)
        if bad:
            if stable_hw_force_disable_ok:
                for p in bad:
                    try:
                        OmegaConf.update(cfg, p, False, merge=False)
                    except Exception:
                        pass
                print(
                    "[v5.4 contract][WARN] stable_hw.enabled=False with force_disable_ok=True, "
                    f"auto-disabled residual submodules: {bad}"
                )
            else:
                raise ValueError(
                    f"[SPEC v5.4] Ambiguous StableHW config: {bad} are enabled while stable_hw.enabled=False. "
                    "Fix: set stable_hw.enabled=True or disable the submodule explicitly. "
                    "If this is an ablation/baseline, set stable_hw.force_disable_ok=true explicitly."
                )

    # --- [v5.4 CONTRACT] twostage must not silently disable StableHW in version_c ---
    if stable_hw_enabled and bool(get_nested(cfg, "training.twostage", False)):
        raise ValueError(
            "[v5.4 P0] training.twostage=True would nullify HW loss while StableHW is enabled. "
            "This is a silent semantic degradation. Disable twostage for version_c (StableHW) runs."
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
    _ensure(stable_hw.lambda_hw_schedule, "lambda_hw_min", float(get_nested(cfg, "stable_hw.lambda_hw_schedule.lambda_hw_min", 0.0)))
    _ensure(
        stable_hw.lambda_hw_schedule,
        "lambda_hw_max",
        float(get_nested(cfg, "stable_hw.lambda_hw_schedule.lambda_hw_max", 0.2)),
    )
    if bool(get_nested(cfg, "stable_hw.enabled", False)):
        v = float(get_nested(cfg, "stable_hw.lambda_hw_schedule.lambda_hw_max", 0.0) or 0.0)
        if v <= 0.0:
            raise ValueError(
                "[V5.4 CONTRACT] stable_hw.enabled=True but lambda_hw_max<=0. "
                "Refuse silent HW-loss disable. Set stable_hw.lambda_hw_schedule.lambda_hw_max explicitly if ablation."
            )

    # --- accuracy guard (HardGating) ---
    _ensure(stable_hw, "accuracy_guard", {})
    _ensure(stable_hw.accuracy_guard, "enabled", _inherit_enabled("stable_hw.accuracy_guard.enabled"))
    _ensure(stable_hw.accuracy_guard, "metric", get_nested(cfg, "stable_hw.accuracy_guard.metric", "acc1"))
    if float(get_nested(cfg, "train.acc_drop_max", 0.0) or 0.0) <= 0.0:
        set_nested(cfg, "train.acc_drop_max", 0.002)
        if mode in ("version_c", "ast2"):
            print("[v5.4 contract] stable_hw.acc_drop_max missing/0; force to 0.002")
    default_acc_drop_max = float(get_nested(cfg, "train.acc_drop_max", 0.002) or 0.002)
    _ensure(
        stable_hw.accuracy_guard,
        "acc_drop_max",
        float(get_nested(cfg, "stable_hw.accuracy_guard.acc_drop_max", default_acc_drop_max) or default_acc_drop_max),
    )
    # keep controller.epsilon_drop aligned with contract field acc_drop_max (do not diverge silently)
    _ensure(stable_hw.accuracy_guard, "controller", {})
    _ensure(
        stable_hw.accuracy_guard.controller,
        "epsilon_drop",
        float(
            get_nested(
                cfg,
                "stable_hw.accuracy_guard.controller.epsilon_drop",
                get_nested(cfg, "stable_hw.accuracy_guard.acc_drop_max", default_acc_drop_max),
            )
            or default_acc_drop_max
        ),
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
    _ensure(stable_hw.locked_acc_ref, "allow_placeholder", False)

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
    # (override + trace in later block to avoid silent mismatch).

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
        if not hasattr(cfg, "train") or getattr(cfg, "train") is None:
            cfg.train = _new_empty_ns(cfg)
        if getattr(cfg.train, "seed", None) is None:
            cfg.train.seed = 2024
        return _stamp_contract(cfg)
    elif mode in ("layout", "layout_heuragenix"):
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

        if not hasattr(cfg, "objective") or getattr(cfg, "objective") is None:
            cfg.objective = _new_empty_ns(cfg)
        if getattr(cfg.objective, "sigma_mm", None) is None:
            cfg.objective.sigma_mm = 2.0

        if not hasattr(cfg.objective, "scalar_weights") or getattr(cfg.objective, "scalar_weights") is None:
            cfg.objective.scalar_weights = _new_empty_ns(cfg.objective)
        sw = cfg.objective.scalar_weights
        if getattr(sw, "w_comm", None) is None:
            sw.w_comm = 1.0
        if getattr(sw, "w_therm", None) is None:
            sw.w_therm = 1.0
        if getattr(sw, "w_penalty", None) is None:
            sw.w_penalty = 1.0

        # optional oscillation metric defaults
        if not hasattr(cfg, "oscillation") or getattr(cfg, "oscillation") is None:
            cfg.oscillation = _new_empty_ns(cfg)
        if getattr(cfg.oscillation, "window", None) is None:
            cfg.oscillation.window = 10
        if getattr(cfg.oscillation, "eps_flat", None) is None:
            cfg.oscillation.eps_flat = 1e-6

        # ------------------------------------------------------------
        # Layout-mode default: still emit StableHW signature knobs explicitly
        # (so trace signature can be strict per SPEC_E without silent defaults).
        # ------------------------------------------------------------
        stable_hw = _ensure_namespace(cfg, "stable_hw")
        stable_hw.enabled = bool(getattr(stable_hw, "enabled", True))

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
        nd.enabled = bool(getattr(nd, "enabled", True))

        if not hasattr(stable_hw, "no_double_scale"):
            stable_hw.no_double_scale = True

        # ---- v5.4: unify budget fields for layout pipelines ----
        if not hasattr(cfg, "budget") or cfg.budget is None:
            cfg.budget = _new_empty_ns(cfg)
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

        # ---- v5.4 contract: make cache_key_schema_version explicit in resolved_config ----
        if get_nested(cfg, "detailed_place.policy_switch.cache_key_schema_version", None) is None:
            set_nested(cfg, "detailed_place.policy_switch.cache_key_schema_version", "v5.4")
            cfg._contract.overrides.append(
                {
                    "path": "detailed_place.policy_switch.cache_key_schema_version",
                    "requested": None,
                    "effective": "v5.4",
                    "reason": "default_for_auditability_v5.4",
                }
            )

        baseline_cfg = getattr(cfg, "baseline", None)
        if baseline_cfg is None and isinstance(cfg, dict):
            baseline_cfg = cfg.get("baseline", {})
        method = None
        llm_config_file = None
        if isinstance(baseline_cfg, dict):
            method = baseline_cfg.get("method")
            llm_config_file = baseline_cfg.get("llm_config_file")
        elif baseline_cfg is not None:
            method = getattr(baseline_cfg, "method", None)
            llm_config_file = getattr(baseline_cfg, "llm_config_file", None)
        if str(method) == "llm_hh" and not str(llm_config_file or "").strip():
            raise ValueError(
                "[v5.4 P1] layout_heuragenix requires baseline.llm_config_file to be explicitly set "
                "when method=llm_hh to avoid ambiguous defaults."
            )

        return _stamp_contract(cfg)
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

    # v5.4 contract: once filled, STOP. Do not apply legacy defaults.

    # ------------------------------------------------------------------
    # [v5.4 CONTRACT HARDEN] NoDrift ⇔ normalize.ref_update must be frozen
    # - strict: mismatch => FAIL-FAST (no silent fix)
    # - non-strict: fix allowed BUT MUST be recorded in _contract.overrides
    # ------------------------------------------------------------------
    if getattr(cfg, "stable_hw", None) is not None and bool(getattr(cfg.stable_hw, "enabled", False)):
        STRICT = bool(get_nested(cfg, "_contract.strict", True))

        nd_enabled = bool(get_nested(cfg, "stable_hw.no_drift.enabled", True))
        ref_update_eff = str(get_nested(cfg, "stable_hw.normalize.ref_update", "frozen") or "frozen").lower()

        # Requested value (for evidence chain)
        req_snap = get_nested(cfg, "_contract.requested_config_snapshot", {}) or {}

        def _req_get(path, default=None):
            cur = req_snap
            for k in path.split("."):
                if not isinstance(cur, dict) or k not in cur:
                    return default
                cur = cur[k]
            return cur

        ref_update_req = _req_get("stable_hw.normalize.ref_update", None)
        if ref_update_req is not None:
            ref_update_req = str(ref_update_req).lower()

        if nd_enabled and ref_update_eff != "frozen":
            if STRICT:
                raise ValueError(
                    "[v5.4 P0][NoDrift] stable_hw.no_drift.enabled=True requires "
                    "stable_hw.normalize.ref_update='frozen'. "
                    f"Got effective={ref_update_eff}, requested={ref_update_req}."
                )
            # non-strict: force + record
            set_nested(cfg, "stable_hw.normalize.ref_update", "frozen")
            cfg._contract.overrides.append(
                {
                    "path": "stable_hw.normalize.ref_update",
                    "requested": ref_update_req,
                    "effective": "frozen",
                    "reason": "no_drift_requires_frozen_ref_update",
                }
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
        stable_hw.discrete_isolation = OmegaConf.create({})
    iso = stable_hw.discrete_isolation
    iso.setdefault("enabled", True)
    iso.setdefault("mapping_update_every_epochs", 1)
    iso.setdefault("layout_update_every_epochs", 1)
    iso.setdefault("cache_mapping_layout", True)
    iso.setdefault("track_live_segments", False)
    iso.setdefault("use_cached_hw_mats", False)

    # ===== v5.4 NoDoubleScale: stable_hw enabled => legacy lambda_hw MUST be 0 (warn + override) =====
    stable_en = bool(get_nested(cfg, "stable_hw.enabled", False))
    legacy_hw_lam = float(getattr(getattr(cfg, "hw", {}), "lambda_hw", 0.0) or 0.0)
    legacy_loss_lam = float(getattr(getattr(cfg, "loss", {}), "lambda_hw", 0.0) or 0.0)

    if stable_en and (legacy_hw_lam != 0.0 or legacy_loss_lam != 0.0):
        print(
            f"[WARN] NoDoubleScale(v5.4): overriding legacy lambdas to 0.0 "
            f"(hw.lambda_hw={legacy_hw_lam}, loss.lambda_hw={legacy_loss_lam})."
        )
        overrides = get_nested(cfg, "_contract.overrides", [])
        if legacy_hw_lam != 0.0:
            overrides.append(
                {
                    "path": "hw.lambda_hw",
                    "requested": float(legacy_hw_lam),
                    "effective": 0.0,
                    "reason": "v5.4 NoDoubleScale under stable_hw.enabled",
                }
            )
        if legacy_loss_lam != 0.0:
            overrides.append(
                {
                    "path": "loss.lambda_hw",
                    "requested": float(legacy_loss_lam),
                    "effective": 0.0,
                    "reason": "v5.4 NoDoubleScale under stable_hw.enabled",
                }
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
    # 统一以 cfg.stable_hw.* 为单一事实源

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

        locked_cfg = get_nested(cfg, "stable_hw.locked_acc_ref", {}) or {}
        if bool(locked_cfg.get("enabled", False)) and (not force_disable_ok):
            strict = bool(locked_cfg.get("strict", True))
            src = str(locked_cfg.get("source", "baseline_stats"))
            p = locked_cfg.get("baseline_stats_path", None)
            if strict and src == "baseline_stats":
                if not p:
                    raise ValueError(
                        "SPEC v5.4 violation: locked_acc_ref.baseline_stats_path is required (strict=true)."
                    )
                if not Path(p).exists():
                    raise ValueError(f"SPEC v5.4 violation: baseline_stats_path not found: {p}")
                lk = str(p)

                # ★ v5.4 P0: disallow placeholder baseline stats under strict mode
                try:
                    with open(lk, "r", encoding="utf-8") as f:
                        _bs = json.load(f)
                except Exception as e:
                    raise ValueError(f"[v5.4 P0] failed to read baseline_stats_path: {lk}: {e}")

                _note = str(_bs.get("note", "")).lower()
                if _bs.get("is_placeholder") is True or ("placeholder" in _note):
                    raise ValueError(
                        "[v5.4 P0] baseline_stats_path points to a PLACEHOLDER baseline_stats. "
                        "This would silently corrupt LockedAccRef/NoDrift semantics. "
                        "Run a real baseline and regenerate baseline_stats."
                    )

                # ★ v5.4 P0: enforce single baseline source when using baseline_stats for hw refs
                hw_bs = str(get_nested(cfg, "stable_hw.baseline_stats_path", "") or "").strip()
                hw_src = str(get_nested(cfg, "stable_hw.hw_ref_source", "") or "").strip()

                if hw_src == "baseline_stats":
                    if not hw_bs:
                        raise ValueError(
                            "[v5.4 P0] stable_hw.hw_ref_source=baseline_stats but stable_hw.baseline_stats_path is empty."
                        )
                    if os.path.abspath(hw_bs) != os.path.abspath(lk):
                        raise ValueError(
                            "[v5.4 P0] baseline split detected: "
                            "stable_hw.baseline_stats_path != stable_hw.locked_acc_ref.baseline_stats_path. "
                            "v5.4 requires a single baseline source for auditability."
                        )

    # === v5.4 contract enforcement (P0 hard fail) ===
    mode_value = str(get_nested(cfg, "train.mode", "") or "")
    if mode_value in ("version_c", "version_c_train", "vc_train"):
        shw = get_nested(cfg, "stable_hw", None)
        if shw is None:
            raise ValueError("v5.4 contract: missing stable_hw section (would silently degrade)")

        stable_enabled = bool(get_nested(cfg, "stable_hw.enabled", False))
        if not stable_enabled:
            if not bool(get_nested(cfg, "stable_hw.force_disable_ok", False)):
                raise ValueError(
                    "v5.4 contract: stable_hw.enabled must not be False in version_c mode. "
                    "If you are intentionally running an ablation (e.g., two-stage), set "
                    "stable_hw.force_disable_ok=true explicitly."
                )
            print("[WARN] v5.4 contract: stable_hw disabled with force_disable_ok=true (ablation/two-stage).")

            if bool(get_nested(cfg, "hw.use_hw_loss", False)):
                hw_l = float(get_nested(cfg, "hw.lambda_hw", 0.0) or 0.0)
                loss_l = float(get_nested(cfg, "loss.lambda_hw", 0.0) or 0.0)
                if hw_l <= 0.0 and loss_l <= 0.0:
                    print(
                        "[WARN] hw.use_hw_loss=true but hw.lambda_hw/loss.lambda_hw are <=0 while stable_hw is disabled. "
                        "HW loss will be ineffective. Set hw.lambda_hw>0 (or loss.lambda_hw>0) if intended."
                    )
        else:
            if get_nested(cfg, "stable_hw.locked_acc_ref", None) is None:
                raise ValueError("v5.4 contract: missing stable_hw.locked_acc_ref")
            if get_nested(cfg, "stable_hw.locked_acc_ref.enabled", True) is False:
                raise ValueError("v5.4 contract: locked_acc_ref must not be disabled")

            if get_nested(cfg, "stable_hw.no_drift", None) is None:
                raise ValueError("v5.4 contract: missing stable_hw.no_drift")
            if get_nested(cfg, "stable_hw.no_drift.enabled", True) is False:
                raise ValueError("v5.4 contract: no_drift must not be disabled")

            if (
                get_nested(cfg, "stable_hw.accuracy_guard", None) is None
                and get_nested(cfg, "stable_hw.guard", None) is None
            ):
                raise ValueError("v5.4 contract: missing stable_hw.accuracy_guard/guard")

            if get_nested(cfg, "stable_hw.lambda_hw_schedule", None) is None:
                raise ValueError("v5.4 contract: missing stable_hw.lambda_hw_schedule (avoid silent hw-loss=0)")

            if float(get_nested(cfg, "hw.lambda_hw", 0.0) or 0.0) != 0.0:
                raise ValueError("v5.4 contract: hw.lambda_hw must be 0 (use stable_hw.lambda_hw_schedule)")
            if float(get_nested(cfg, "loss.lambda_hw", 0.0) or 0.0) != 0.0:
                raise ValueError("v5.4 contract: loss.lambda_hw must be 0 (use stable_hw.lambda_hw_schedule)")

    # ---- v5.4 Addendum: signature must be assign-only ----
    if bool(get_nested(cfg, "signature.allow_pos_signature", False)):
        raise ValueError(
            "SPEC v5.4 signature must be assign-only. "
            "Please set signature.allow_pos_signature=false."
        )

    _record_resolved_aliases(
        cfg,
        [
            ("stable_hw.controller", "stable_hw.accuracy_guard.controller"),
            ("stable_hw.metric_key", "stable_hw.accuracy_guard.metric_key"),
            ("stable_hw.epsilon_drop", "stable_hw.accuracy_guard.epsilon_drop"),
            ("stable_hw.acc_margin", "stable_hw.accuracy_guard.acc_margin"),
            ("stable_hw.allow_train_ema_fallback", "stable_hw.accuracy_guard.allow_train_ema_fallback"),
            ("stable_hw.accuracy_guard.on_violate", "stable_hw.accuracy_guard.controller"),
            ("stable_hw.normalize.method", "stable_hw.normalize.mode"),
            ("stable_hw.normalize.clip_eps", "stable_hw.normalize.eps"),
            ("layout.optimize_layout", "hw.optimize_layout"),
        ],
    )

    def _get_dict_path(d: dict, path: str):
        cur = d
        for k in path.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return "__MISSING__"
            cur = cur[k]
        return cur

    def _auto_record_contract_diffs(cfg_to_record, paths):
        req = get_nested(cfg_to_record, "_contract.requested_config_snapshot", {}) or {}
        # Contract diff recording should not hard-crash on missing interpolations.
        # Some entry scripts inject aliases like cfg.out_dir at runtime, but config-only
        # smoke checks may not. Fall back to a non-resolved snapshot if resolve fails.
        try:
            eff = OmegaConf.to_container(cfg_to_record, resolve=True)
        except Exception:
            eff = OmegaConf.to_container(cfg_to_record, resolve=False)
        existing = set()
        for it in (get_nested(cfg_to_record, "_contract.overrides", []) or []):
            if isinstance(it, dict) and "path" in it:
                existing.add(str(it["path"]))

        for p in paths:
            r = _get_dict_path(req, p)
            e = _get_dict_path(eff, p)
            if r != e and p not in existing:
                cfg_to_record._contract.overrides.append(
                    {
                        "path": p,
                        "requested": None if r == "__MISSING__" else r,
                        "effective": None if e == "__MISSING__" else e,
                        "reason": "auto_contract_diff_v5.4",
                    }
                )

    _contract_sensitive_paths = [
        # hard gate A/B/C核心语义
        "contract.strict",
        "stable_hw.enabled",
        "stable_hw.normalize.enabled",
        "stable_hw.normalize.kind",
        "stable_hw.normalize.ref_update",
        "stable_hw.lambda_hw_schedule.enabled",
        "stable_hw.lambda_hw_schedule.kind",
        "stable_hw.lambda_hw_schedule.lambda_hw_min",
        "stable_hw.lambda_hw_schedule.lambda_hw_max",
        "stable_hw.accuracy_guard.enabled",
        "stable_hw.accuracy_guard.acc_drop_max",
        "stable_hw.locked_acc_ref.enabled",
        "stable_hw.locked_acc_ref.source",
        "stable_hw.locked_acc_ref.expected_acc1",
        "stable_hw.locked_acc_ref.path",
        "stable_hw.no_drift.enabled",
        "stable_hw.no_drift.mode",
        "stable_hw.no_double_scale",          # bool 或 dict
        "stable_hw.no_double_scale.enabled",  # 若存在
        # 训练侧关键门
        "train.acc_drop_max",
        "training.twostage",
    ]
    _auto_record_contract_diffs(cfg, _contract_sensitive_paths)

    # ===========================
    # v5.4 CONTRACT: auto-record requested vs effective for CRITICAL paths
    # ===========================
    try:
        contract = cfg.setdefault("_contract", {})
        requested = contract.get("requested_config_snapshot", {}) or {}
        overrides = contract.setdefault("overrides", []) or []
        # ---- SPEC_E: sanitize contract overrides to auditable schema ----
        raw_overrides = contract.get("overrides", []) or []
        sanitized = []

        def _push(path, requested, effective, reason):
            if not isinstance(path, str):
                return
            path = path.strip()
            if not path:
                return
            if not isinstance(reason, str):
                return
            reason = reason.strip()
            if not reason:
                return
            sanitized.append(
                {
                    "path": path,
                    "requested": requested,
                    "effective": effective,
                    "reason": reason,
                }
            )

        for it in raw_overrides:
            if not isinstance(it, dict):
                continue

            # compliant shape
            if all(k in it for k in ("path", "requested", "effective", "reason")):
                _push(it.get("path"), it.get("requested"), it.get("effective"), it.get("reason"))
                continue

            # legacy shape
            if all(k in it for k in ("key_path", "old", "new")):
                _push(
                    it.get("key_path"),
                    it.get("old"),
                    it.get("new"),
                    it.get("reason", "legacy_override"),
                )
                continue

            # unknown shape -> drop
            continue

        # IMPORTANT: write back
        contract["overrides"] = sanitized
        overrides = contract["overrides"]

        # prevent duplicates
        existing = set()
        for o in overrides:
            p = o.get("path", "")
            if p:
                existing.add(p)

        def _get_requested(path: str):
            cur = requested
            for k in path.split("."):
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(k, None)
            return cur

        def _get_effective(path: str):
            return get_nested(cfg, path, None)

        CRITICAL = [
            # v5.4 semantic anchors
            "stable_hw.enabled",
            "stable_hw.accuracy_guard.enabled",
            "stable_hw.accuracy_guard.controller.guard_mode",
            "stable_hw.locked_acc_ref.enabled",
            "stable_hw.locked_acc_ref.source",
            "stable_hw.no_drift.enabled",
            "stable_hw.no_double_scale.enabled",
            # trace/audit anchors
            "trace.enabled",
            "hw_proxy.sanitize.enabled",
        ]

        for path in CRITICAL:
            if path in existing:
                continue
            req = _get_requested(path)
            eff = _get_effective(path)
            if req != eff:
                overrides.append(
                    {
                        "path": path,
                        "requested": req,
                        "effective": eff,
                        "reason": "auto_default_or_mirror_v5p4",
                    }
                )
                existing.add(path)

        contract["overrides"] = overrides
        cfg["_contract"] = contract
    except Exception as exc:
        raise RuntimeError(
            "[P0][v5.4] Failed to compute contract_overrides; trace would become non-auditable."
        ) from exc

    effective_snapshot_raw = OmegaConf.to_container(cfg, resolve=True)
    cfg._contract.effective_config_snapshot = _strip_contract(effective_snapshot_raw)
    from utils.trace_guard import _sha256_json
    cfg._contract.requested_config_sha256 = _sha256_json(cfg._contract.requested_config_snapshot)
    cfg._contract.effective_config_sha256 = _sha256_json(cfg._contract.effective_config_snapshot)
    _augment_contract_overrides(cfg._contract)

    return _stamp_contract(cfg)
