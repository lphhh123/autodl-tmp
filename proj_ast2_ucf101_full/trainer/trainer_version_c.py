"""Version-C full trainer (SPEC §12.2)."""
from __future__ import annotations

import json
import math
import random
import os
import itertools
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from chiplet.chiplet_lib import ChipletLibrary, ChipletSlots
from utils.data_ucf101 import UCF101Dataset
from hw_proxy.hw_loss import compute_hw_loss
from hw_proxy.layer_hw_proxy import LayerHwProxy
from hw_proxy.multi_device_oracle import MultiDeviceHwOracle
from layout.candidate_pool import signature_for_assign
from layout.evaluator import LayoutEvaluator, LayoutState
from layout.sites import build_sites
from layout.wafer_layout import WaferLayout
from mapping.mapping_solver import MappingSolver
from mapping.partitioner import PartitionPlanner
from models.video_vit import VideoViT, VideoAudioAST
from utils.distributed_utils import get_device
from utils.eval_utils import eval_acc1
from utils.logging_utils import setup_logger, log_stats
from utils.seed import seed_everything
from utils.stable_hash import stable_hash
from utils.safe_json import safe_dump, safe_dumps


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DataParallel/DDP wrappers."""
    return getattr(model, "module", model)


def _oc_select(cfg, key: str, default=None):
    """OmegaConf-safe nested key access with a plain default."""
    try:
        value = OmegaConf.select(cfg, key)
        return default if value is None else value
    except Exception:
        return default


def _to_pyfloat(x, default: float = 0.0) -> float:
    """Robust scalar conversion with tensor detach to avoid grad->scalar warnings."""
    try:
        if torch.is_tensor(x):
            return float(x.detach().cpu().item())
    except Exception:
        pass
    try:
        import numpy as _np
        if isinstance(x, (_np.generic,)):
            return float(x)
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return float(default)


def _ast_warm_eff(cfg) -> int:
    """Return the effective dense warmup length used by the AST schedule."""
    sched = getattr(getattr(cfg, "ast", None), "schedule", None)
    if sched is None:
        return 0
    warm = int(getattr(sched, "warmup_epochs", 0) or 0)
    force_dense = int(getattr(sched, "force_dense_epochs", warm) or warm)
    warm_eff = max(warm, force_dense)
    try:
        warm_eff = int(os.environ.get("AST_WARMUP_EPOCHS", str(warm_eff)))
    except Exception:
        pass
    return int(max(0, warm_eff))


def _resolve_amp_settings(cfg, device_type: str, logger=None):
    """
    Resolve AMP settings:
      - cfg.train.amp: bool
      - cfg.train.amp_dtype: 'fp16'|'bf16' (default: 'fp16')
    Behavior:
      - autocast dtype = bf16 when requested and supported
      - GradScaler enabled ONLY for fp16 (bf16 => scaler disabled)
    """
    train_cfg = getattr(cfg, "train", cfg)
    amp_enabled = bool(getattr(train_cfg, "amp", False))
    amp_dtype_str = str(getattr(train_cfg, "amp_dtype", "fp16") or "fp16").lower()
    if amp_dtype_str in ("bf16", "bfloat16"):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    # Safety fallback: if bf16 requested on cuda but not supported, fallback to fp16.
    if amp_enabled and device_type == "cuda" and amp_dtype == torch.bfloat16:
        bf16_ok = True
        try:
            # torch >= 2.0 usually has this
            if hasattr(torch.cuda, "is_bf16_supported"):
                bf16_ok = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_ok = False
        if not bf16_ok:
            if logger is not None:
                logger.warning("[AMP] bf16 requested but torch.cuda.is_bf16_supported() is False; falling back to fp16.")
            amp_dtype = torch.float16

    # GradScaler is only needed for fp16 autocast
    use_scaler = bool(amp_enabled and (amp_dtype == torch.float16))
    return amp_enabled, amp_dtype, use_scaler, amp_dtype_str


def _optimizer_has_any_grad(opt: torch.optim.Optimizer) -> bool:
    """Return True if any parameter in optimizer has a non-None gradient.

    This avoids calling GradScaler.step() on optimizers with no tracked grads,
    which can raise: "No inf checks were recorded for this optimizer."
    """
    try:
        for group in opt.param_groups:
            for p in group.get("params", []):
                if p is not None and getattr(p, "grad", None) is not None:
                    return True
    except Exception:
        return False
    return False


_BACKBONE_PARAM_PREFIXES = (
    "patch_embed.",
    "blocks.",
    "norm.",
    "pos_embed",
    "cls_token",
)


def _is_backbone_param_name(name: str) -> bool:
    name = str(name or "")
    # DataParallel prefixes param names with "module.".
    if name.startswith("module."):
        name = name[len("module."):]
    return any(name.startswith(prefix) for prefix in _BACKBONE_PARAM_PREFIXES)


def _build_model_param_groups(model: torch.nn.Module, lr: float, backbone_lr_scale: float) -> tuple[list[dict], dict]:
    backbone_params = []
    head_params = []
    seen = set()
    for name, param in model.named_parameters():
        if param is None:
            continue
        pid = id(param)
        if pid in seen:
            continue
        seen.add(pid)
        if _is_backbone_param_name(name):
            backbone_params.append(param)
        else:
            head_params.append(param)

    groups = []
    if backbone_params:
        groups.append({
            "params": backbone_params,
            "lr": float(lr) * float(backbone_lr_scale),
            "lr_scale": float(backbone_lr_scale),
            "group_name": "backbone",
        })
    if head_params:
        groups.append({
            "params": head_params,
            "lr": float(lr),
            "lr_scale": 1.0,
            "group_name": "head",
        })

    meta = {
        "backbone_tensors": int(len(backbone_params)),
        "head_tensors": int(len(head_params)),
        "backbone_params": int(sum(int(p.numel()) for p in backbone_params)),
        "head_params": int(sum(int(p.numel()) for p in head_params)),
    }
    return groups, meta


def _set_backbone_trainable(model: torch.nn.Module, trainable: bool) -> dict:
    tensors = 0
    params = 0
    for name, param in model.named_parameters():
        if param is None or (not _is_backbone_param_name(name)):
            continue
        param.requires_grad = bool(trainable)
        tensors += 1
        params += int(param.numel())
    return {"tensors": int(tensors), "params": int(params), "trainable": bool(trainable)}


def _ast_interp(a: float, b: float, t: float, curve: str = "linear") -> float:
    t = float(max(0.0, min(1.0, t)))
    curve = str(curve or "linear").lower()
    if curve == "cosine":
        # cosine ease-in-out
        tt = 0.5 - 0.5 * math.cos(math.pi * t)
    else:
        tt = t
    return float(a + (b - a) * tt)


def _compute_ch_keep_target_stepwise(cfg, epoch: int) -> float:
    """Return channel keep target with stepwise post-warmup semantics."""
    ast = getattr(cfg, "ast", None)
    sched = getattr(ast, "schedule", None) if ast is not None else None
    if sched is None:
        return 1.0

    warmup_epochs = int(getattr(sched, "warmup_epochs", 0) or 0)
    force_dense_epochs = int(getattr(sched, "force_dense_epochs", warmup_epochs) or warmup_epochs)
    warm_eff = max(int(warmup_epochs), int(force_dense_epochs))

    ch_keep_start = float(getattr(sched, "ch_keep_start", 1.0) or 1.0)
    ch_keep_end = float(getattr(sched, "ch_keep_end", 1.0) or 1.0)
    ch_ramp = int(getattr(sched, "ch_ramp_epochs", getattr(sched, "ramp_epochs", 0)) or getattr(sched, "ramp_epochs", 0) or 0)
    curve = str(getattr(sched, "curve", "cosine") or "cosine")

    if int(epoch) < int(warm_eff):
        return 1.0
    if ch_ramp <= 0:
        return float(ch_keep_end)

    step_idx = int(epoch) - int(warm_eff) + 1
    t_ch = float(step_idx) / float(ch_ramp)
    t_ch = float(max(0.0, min(1.0, t_ch)))
    return float(_ast_interp(ch_keep_start, ch_keep_end, t_ch, curve=curve))


def compute_ast_schedule_effective(cfg, epoch: int) -> dict:
    """Compute per-epoch AST (token gating) schedule without mutating cfg.

    Returns dict with:
      - force_dense (bool)
      - rho_token (float)
      - token_temperature (float)
      - lambda_ast (float)  (multiplier for info['L_AST'])
      - phase (str): warmup|ramp|stabilize|disabled
      - t (float): ramp progress in [0,1]
    """
    ast = getattr(cfg, "ast", None)
    sched = getattr(ast, "schedule", None) if ast is not None else None
    if sched is None or (not bool(getattr(sched, "enabled", False))):
        return {"phase": "disabled"}

    warm = int(getattr(sched, "warmup_epochs", 0) or 0)
    # v5.4+: allow extending the dense warmup beyond warmup_epochs.
    # This is crucial when channel/structural gates exist: we often want to keep them fully open
    # until LockedAccRef is ready (e.g. freeze_epoch=15), otherwise the model can lose accuracy
    # before the guard becomes active.
    force_dense_epochs = int(getattr(sched, "force_dense_epochs", warm) or warm)
    warm_eff = max(int(warm), int(force_dense_epochs))
    ramp = int(getattr(sched, "ramp_epochs", 0) or 0)
    curve = str(getattr(sched, "curve", "cosine") or "cosine")

    rho_end = float(getattr(sched, "rho_end", getattr(ast, "rho_token_target", 1.0)) or getattr(ast, "rho_token_target", 1.0))
    rho_start = float(getattr(sched, "rho_start", 1.0) or 1.0)

    temp_end = float(getattr(sched, "temp_end", getattr(ast, "token_temperature", 0.1)) or getattr(ast, "token_temperature", 0.1))
    temp_start = float(getattr(sched, "temp_start", 1.0) or 1.0)

    loss_cfg = getattr(cfg, "loss", None)
    lam_end = float(getattr(sched, "lambda_ast_end", getattr(loss_cfg, "lambda_AST", 1.0) if loss_cfg is not None else 1.0) or (getattr(loss_cfg, "lambda_AST", 1.0) if loss_cfg is not None else 1.0))
    lam_start = float(getattr(sched, "lambda_ast_start", 0.0) or 0.0)

    # NOTE: use warm_eff (not warm) for force_dense gating.
    if epoch < warm_eff:
        return {
            "phase": "warmup",
            "t": 0.0,
            "force_dense": True,
            "rho_token": 1.0,
            "token_temperature": temp_start,
            "lambda_ast": lam_start,
            # Channel keep target (MorphNet-style controlled pruning).
            "ch_keep_target": 1.0,
        }

    # Channel keep target schedule (defaults to "no channel pruning" if ch_keep_end==1.0)
    ch_keep_start = float(getattr(sched, "ch_keep_start", 1.0) or 1.0)
    ch_keep_end = float(getattr(sched, "ch_keep_end", 1.0) or 1.0)
    ch_ramp = int(getattr(sched, "ch_ramp_epochs", ramp) or ramp)

    if ramp <= 0:
        ch_keep_target = _compute_ch_keep_target_stepwise(cfg, int(epoch))
        return {
            "phase": "stabilize",
            "t": 1.0,
            "force_dense": False,
            "rho_token": rho_end,
            "token_temperature": temp_end,
            "lambda_ast": lam_end,
            "ch_keep_target": float(ch_keep_target),
        }

    # Ramp progress for this epoch.
    # Use t=0 at the first ramp epoch (epoch == warm_eff) so
    # rho/temp/lambda start from *_start without an immediate jump.
    t = float(epoch - warm_eff) / float(ramp)
    t = float(max(0.0, min(1.0, t)))
    phase = "ramp" if t < 1.0 else "stabilize"

    # Channel keep schedule uses strict stepwise post-warmup semantics.
    ch_keep_target = _compute_ch_keep_target_stepwise(cfg, int(epoch))
    return {
        "phase": phase,
        "t": t,
        "force_dense": False,
        "rho_token": _ast_interp(rho_start, rho_end, t, curve=curve),
        "token_temperature": _ast_interp(temp_start, temp_end, t, curve=curve),
        "lambda_ast": _ast_interp(lam_start, lam_end, t, curve=curve),
        "ch_keep_target": float(ch_keep_target),
    }


# -------------------------
# AST schedule freeze (StableHW recovery)
# -------------------------
def _stablehw_freeze_ast_now(stable_state: Dict[str, Any]) -> bool:
    """Return True when StableHW requests freezing/pinning the AST pruning schedule."""
    if not isinstance(stable_state, dict):
        return False
    # freeze_schedule is the canonical flag (set by accuracy_guard); keep legacy alias too.
    return bool(stable_state.get("freeze_schedule", False)) or bool(stable_state.get("freeze_ast_schedule", False)) or bool(
        stable_state.get("in_recovery", False)
    )


def _stable_hw_guard_controls_enabled(cfg) -> bool:
    if bool(_oc_select(cfg, "suite_cleanup.disable_stable_hw_guard_controls", False)):
        return False
    return bool(_oc_select(cfg, "stable_hw.enable_guard_controls", True))


def _apply_ast_runtime_overrides_to_model(model: torch.nn.Module, cfg, ast_sched: dict) -> Optional[dict]:
    """Apply AST schedule to a model's pruner (if present) without mutating cfg."""
    model_u = unwrap_model(model)
    pruner = getattr(model_u, "ast_pruner", None)
    if pruner is None or not hasattr(pruner, "set_runtime_overrides"):
        return None

    ast_cfg = getattr(cfg, "ast", None)
    force_dense = bool(ast_sched.get("force_dense", False))
    rho_token = float(ast_sched.get("rho_token", getattr(ast_cfg, "rho_token_target", 1.0) if ast_cfg is not None else 1.0))
    token_temperature = float(
        ast_sched.get("token_temperature", getattr(ast_cfg, "token_temperature", 0.1) if ast_cfg is not None else 0.1)
    )

    ch_keep_target = ast_sched.get("ch_keep_target", None)
    pruner.set_runtime_overrides(
        force_dense=force_dense,
        rho_token=rho_token,
        token_temperature=token_temperature,
        ch_keep_target=(float(ch_keep_target) if ch_keep_target is not None else None),
    )
    return {
        "force_dense": force_dense,
        "rho_token": float(rho_token),
        "token_temperature": float(token_temperature),
        "ch_keep_target": float(ch_keep_target) if ch_keep_target is not None else None,
    }


def compute_ast_schedule_effective_with_stable_hw_freeze(cfg, stable_state: Dict[str, Any], outer: int) -> Tuple[dict, int]:
    """Compute AST schedule using a virtual epoch that pauses while StableHW is in recovery.

    - When not frozen: advance stable_state['ast_sched_virtual_epoch'] by 1 per outer.
    - When frozen: reuse the last applied schedule and DO NOT advance the virtual epoch
      (prevents token_keep from continuing to drop while StableHW is trying to recover accuracy).
    Returns: (ast_sched, ast_epoch_used)
    """
    if not isinstance(stable_state, dict):
        stable_state = {}

    # virtual epoch defaults to outer index (fresh run) or resume start_outer
    v_epoch = stable_state.get("ast_sched_virtual_epoch", None)
    try:
        v_epoch = int(outer if v_epoch is None else v_epoch)
    except Exception:
        v_epoch = int(outer)

    freeze_now = _stablehw_freeze_ast_now(stable_state)

    # Trust-region limits for channel keep target (prevents overshoot and makes recovery easier).
    # Defaults are conservative; configs can override.
    tr_delta_down = float(_oc_select(cfg, "stable_hw.accuracy_guard.controller.trust_region.delta_down", 0.0) or 0.0)
    tr_delta_up = float(_oc_select(cfg, "stable_hw.accuracy_guard.controller.trust_region.delta_up", 0.02) or 0.02)
    tr_delta_down = max(0.0, float(tr_delta_down))
    tr_delta_up = max(0.0, float(tr_delta_up))
    affect_ch_keep_target = bool(_oc_select(cfg, "stable_hw.affect_ch_keep_target", True))

    # Soft recovery: instead of hard force-dense, gently roll back pruning target when violating accuracy.
    sr_enabled = bool(_oc_select(cfg, "stable_hw.accuracy_guard.controller.soft_recovery.enabled", True))
    sr_step = float(_oc_select(cfg, "stable_hw.accuracy_guard.controller.soft_recovery.rollback_step", 0.03) or 0.03)
    sr_hold = int(_oc_select(cfg, "stable_hw.accuracy_guard.controller.soft_recovery.hold_epochs", 2) or 2)
    sr_max = int(_oc_select(cfg, "stable_hw.accuracy_guard.controller.soft_recovery.max_rollbacks", 3) or 3)
    sr_fallback_force_dense = bool(_oc_select(cfg, "stable_hw.accuracy_guard.controller.soft_recovery.fallback_force_dense", False))
    sr_step = max(0.0, float(sr_step))
    sr_hold = max(0, int(sr_hold))
    sr_max = max(0, int(sr_max))

    # If we just exited RECOVERY/VIOLATE, back off the virtual epoch a bit to avoid
    # immediately re-entering violation with the same (too aggressive) sparsity target.
    was_frozen = bool(stable_state.get("_ast_freeze_prev", False))
    backoff = int(_oc_select(cfg, "stable_hw.accuracy_guard.controller.sched_backoff_epochs", 0) or 0)
    if was_frozen and (not freeze_now) and backoff > 0:
        try:
            v_epoch = max(0, int(v_epoch) - int(backoff))
        except Exception:
            pass

    if freeze_now:
        # Prefer a pinned snapshot; fall back to last applied schedule.
        frozen = stable_state.get("ast_sched_frozen") or stable_state.get("ast_sched_last_applied") or {}
        if not isinstance(frozen, dict) or not frozen:
            # If StableHW requests freezing before we've recorded ast_sched_last_applied
            # (e.g., entering RECOVERY right after HW enables), do NOT jump back to dense.
            # Instead, compute the schedule for the *current* virtual epoch and pin it.
            cand = None
            try:
                cand = compute_ast_schedule_effective(cfg, int(v_epoch))
            except Exception:
                cand = None
            if isinstance(cand, dict) and cand:
                frozen = dict(cand)
                # Record so subsequent frozen steps stay consistent/auditable.
                stable_state["ast_sched_last_applied"] = dict(frozen)
                stable_state["ast_sched_last_epoch"] = int(v_epoch)
            else:
                # Last-resort fail-safe: dense behavior
                ast_cfg = getattr(cfg, "ast", None)
                token_temperature = float(getattr(ast_cfg, "token_temperature", 0.1) if ast_cfg is not None else 0.1)
                frozen = {
                    "phase": "warmup",
                    "t": 0.0,
                    "force_dense": True,
                    "rho_token": 1.0,
                    "token_temperature": token_temperature,
                    "lambda_ast": 0.0,
                }
        else:
            frozen = dict(frozen)

        # During recovery: PAUSE schedule progression.
        # Optionally force full dense to quickly recover accuracy (recommended for channel-gate pruning).
        ast_cfg = getattr(cfg, "ast", None)

        # Keep previous settings if present; otherwise fall back to dense safely.
        rho_prev = float(frozen.get("rho_token", 1.0))
        temp_prev = float(frozen.get("token_temperature", getattr(ast_cfg, "token_temperature", 0.1) if ast_cfg is not None else 0.1))

        # Optional: tiny relaxation to help recovery, but keep it very small to avoid gaming.
        # Set to 0.0 by default (no behavior change unless you later add cfg knob).
        relax = 0.0
        try:
            relax = float(getattr(getattr(cfg, "stable_hw", None), "recovery_rho_relax", 0.0) or 0.0)
        except Exception:
            relax = 0.0

        rho_hold = min(1.0, max(0.0, rho_prev + relax))

        guard_mode = str(stable_state.get("guard_mode", "")).upper()
        base_ch_keep = _compute_ch_keep_target_stepwise(cfg, int(outer))
        last_keep = float(stable_state.get("ch_keep_target_last", frozen.get("ch_keep_target", 1.0) if isinstance(frozen, dict) else 1.0) or 1.0)

        if sr_enabled and guard_mode in ("VIOLATE", "RECOVERY"):
            # Enter / continue soft recovery: raise ch_keep_target gradually and hold.
            entering = (not was_frozen)
            rollbacks = int(stable_state.get("soft_recovery_rollbacks", 0) or 0)
            hold_until = int(stable_state.get("soft_recovery_hold_until", -1) or -1)
            if entering:
                rollbacks = 0
                hold_until = int(outer) + int(sr_hold)

            do_rb = bool(entering or (int(outer) >= int(hold_until) and (sr_max <= 0 or int(rollbacks) < int(sr_max))))
            if do_rb:
                rollbacks = int(rollbacks) + 1
                hold_until = int(outer) + int(sr_hold)
                desired = min(1.0, float(last_keep) + float(sr_step))
                stable_state["soft_recovery_last_rollback_outer"] = int(outer)
            else:
                desired = float(last_keep)

            if affect_ch_keep_target:
                # Trust region clamp (avoid oscillation): allow limited up/down adjustments.
                lo = float(last_keep) - float(tr_delta_down)
                hi = float(last_keep) + float(tr_delta_up)
                new_keep = float(max(lo, min(hi, float(desired))))
                new_keep = float(max(0.0, min(1.0, new_keep)))
            else:
                new_keep = float(base_ch_keep)

            stable_state["soft_recovery_rollbacks"] = int(rollbacks)
            stable_state["soft_recovery_hold_until"] = int(hold_until)
            stable_state["soft_recovery_desired"] = float(desired)
            stable_state["ch_keep_target_last"] = float(new_keep)

            frozen["force_dense"] = False
            frozen["rho_token"] = float(rho_hold)
            frozen["token_temperature"] = float(temp_prev)
            frozen["ch_keep_target"] = float(new_keep)

            # Optional last-resort safety (off by default): hard force-dense if too many rollbacks.
            if sr_fallback_force_dense and sr_max > 0 and int(rollbacks) >= int(sr_max) and int(outer) >= int(hold_until):
                frozen["force_dense"] = True
                frozen["rho_token"] = 1.0
                frozen["token_temperature"] = float(temp_prev)
                if affect_ch_keep_target:
                    frozen["ch_keep_target"] = 1.0
        else:
            # Legacy behavior: optionally hard force-dense on VIOLATE/RECOVERY.
            force_dense_on_violate = bool(_oc_select(cfg, "stable_hw.accuracy_guard.controller.force_dense_on_violate", False))
            if force_dense_on_violate and guard_mode in ("VIOLATE", "RECOVERY"):
                frozen["force_dense"] = True
                frozen["rho_token"] = 1.0
                frozen["token_temperature"] = float(temp_prev)
                if affect_ch_keep_target:
                    frozen["ch_keep_target"] = 1.0
                    stable_state["ch_keep_target_last"] = 1.0
            else:
                frozen["force_dense"] = bool(frozen.get("force_dense", False))
                frozen["rho_token"] = float(rho_hold)
                frozen["token_temperature"] = float(temp_prev)

        # Disable AST auxiliary loss during recovery to reduce extra instability.
        frozen["lambda_ast"] = 0.0
        if not affect_ch_keep_target:
            frozen["ch_keep_target"] = float(base_ch_keep)
            stable_state["ch_keep_target_last"] = float(base_ch_keep)

        stable_state["ast_sched_frozen"] = dict(frozen)
        stable_state["ast_sched_frozen_outer"] = int(outer)
        # Do NOT advance virtual epoch.
        stable_state["_ast_freeze_prev"] = True
        return dict(frozen), int(v_epoch)

    # not frozen => advance virtual epoch
    stable_state["_ast_freeze_prev"] = False
    ast_epoch_used = int(v_epoch)
    ast_sched = compute_ast_schedule_effective(cfg, ast_epoch_used)
    if not isinstance(ast_sched, dict):
        ast_sched = {"phase": "disabled"}

    # Apply trust-region to ch_keep_target to prevent abrupt pruning jumps.
    if isinstance(ast_sched, dict) and ("ch_keep_target" in ast_sched) and (ast_sched.get("ch_keep_target") is not None):
        if affect_ch_keep_target:
            base_keep = float(ast_sched.get("ch_keep_target", 1.0) or 1.0)
            prev_keep = float(stable_state.get("ch_keep_target_last", base_keep) or base_keep)
            lo = float(prev_keep) - float(tr_delta_down)
            hi = float(prev_keep) + float(tr_delta_up)
            keep_eff = float(max(lo, min(hi, float(base_keep))))
            keep_eff = float(max(0.0, min(1.0, keep_eff)))
            ast_sched["ch_keep_target"] = float(keep_eff)
            stable_state["ch_keep_target_last"] = float(keep_eff)
        else:
            base_keep = _compute_ch_keep_target_stepwise(cfg, int(outer))
            ast_sched["ch_keep_target"] = float(base_keep)
            stable_state["ch_keep_target_last"] = float(base_keep)

    # Leaving recovery: clear soft-recovery counters (schedule backoff already handled above).
    stable_state.pop("soft_recovery_hold_until", None)
    stable_state.pop("soft_recovery_desired", None)
    stable_state["soft_recovery_rollbacks"] = 0
    stable_state["ast_sched_last_applied"] = dict(ast_sched)
    stable_state["ast_sched_last_epoch"] = int(ast_epoch_used)
    stable_state["ast_sched_virtual_epoch"] = int(ast_epoch_used) + 1
    # clear frozen snapshot when leaving recovery
    stable_state.pop("ast_sched_frozen", None)
    stable_state.pop("ast_sched_frozen_outer", None)
    return dict(ast_sched), int(ast_epoch_used)

def _atomic_torch_save(obj, dst: Path) -> None:
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dst.with_suffix(dst.suffix + f".tmp.{os.getpid()}")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, dst)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.ema = deepcopy(unwrap_model(model)).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            msd = unwrap_model(model).state_dict()
            for k, v in self.ema.state_dict().items():
                if k in msd:
                    v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))


def _topk_correct_frac(logits: torch.Tensor, target: torch.Tensor, k: int) -> torch.Tensor:
    topk = logits.topk(k, dim=1).indices
    return topk.eq(target.view(-1, 1)).any(dim=1).float()


def save_checkpoint_version_c(
    out_dir: Path,
    tag: str,
    *,
    model: torch.nn.Module,
    ema_model: Optional[ModelEMA],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    epoch: int,
    best_acc1: Optional[float],
    seal_digest: str,
    run_id: str,
) -> Path:
    ckpt_dir = Path(out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{tag}.pth"
    payload = {
        "epoch": int(epoch),
        "best_acc1": (float(best_acc1) if best_acc1 is not None else None),
        "seal_digest": str(seal_digest),
        "run_id": str(run_id),
        "model": unwrap_model(model).state_dict(),
        "ema": (ema_model.ema.state_dict() if ema_model is not None else None),
        "optimizer": optimizer.state_dict(),
        "scaler": (scaler.state_dict() if scaler is not None else None),
    }
    _atomic_torch_save(payload, ckpt_path)
    return ckpt_path


def maybe_auto_resume_version_c(out_dir: Path, model, ema_model, optimizer, scaler, logger):
    auto_resume = str(os.environ.get("AUTO_RESUME", "0")).strip().lower() in ("1", "true", "yes", "y", "on")
    if not auto_resume:
        return 0, None
    ckpt_path = Path(out_dir) / "checkpoints" / "last.pth"
    if not ckpt_path.exists():
        return 0, None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(model_state, dict):
            unwrap_model(model).load_state_dict(model_state, strict=True)
        if isinstance(ckpt, dict) and ckpt.get("optimizer", None) is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and isinstance(ckpt, dict) and ckpt.get("scaler", None) is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass
        if ema_model is not None and isinstance(ckpt, dict) and ckpt.get("ema", None) is not None:
            try:
                ema_model.ema.load_state_dict(ckpt["ema"], strict=True)
            except Exception:
                pass
        last_epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
        start_outer = max(0, last_epoch + 1)
        best_acc1 = float(ckpt.get("best_acc1", 0.0)) if isinstance(ckpt, dict) else None
        logger.info(f"[AUTO_RESUME] loaded {ckpt_path} (start_outer={start_outer}, best_acc1={best_acc1})")
        return start_outer, best_acc1
    except Exception as e:
        logger.warning(f"[AUTO_RESUME] failed to load {ckpt_path}: {e}. Start from scratch.")
        return 0, None


def maybe_init_from_checkpoint_version_c(
    init_ckpt_path: Optional[str],
    model,
    ema_model,
    logger,
    use_resume_epoch: bool = True,
    load_ema: bool = True,
) -> Tuple[int, Optional[float]]:
    path = str(init_ckpt_path or os.environ.get("INIT_CKPT_PATH", "") or "").strip()
    if not path:
        return 0, None
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        logger.warning("[INIT_CKPT] not found: %s", str(ckpt_path))
        return 0, None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = None
        if isinstance(ckpt, dict):
            if isinstance(ckpt.get("model", None), dict):
                state_dict = ckpt.get("model")
            elif isinstance(ckpt.get("state_dict", None), dict):
                state_dict = ckpt.get("state_dict")
        if state_dict is None:
            state_dict = ckpt

        missing, unexpected = unwrap_model(model).load_state_dict(state_dict, strict=False)
        logger.info(
            "[INIT_CKPT] loaded %s (missing=%d unexpected=%d)",
            str(ckpt_path),
            int(len(missing) if isinstance(missing, list) else 0),
            int(len(unexpected) if isinstance(unexpected, list) else 0),
        )

        if ema_model is not None:
            ema_state = ckpt.get("ema", None) if isinstance(ckpt, dict) else None
            if bool(load_ema) and isinstance(ema_state, dict):
                try:
                    ema_model.ema.load_state_dict(ema_state, strict=False)
                    logger.info("[INIT_CKPT] EMA loaded from checkpoint.")
                except Exception as exc:
                    logger.warning("[INIT_CKPT] failed to load EMA from checkpoint: %s", str(exc))
                    ema_model.ema.load_state_dict(unwrap_model(model).state_dict(), strict=False)
            else:
                ema_model.ema.load_state_dict(unwrap_model(model).state_dict(), strict=False)

        start_outer = 0
        if bool(use_resume_epoch) and isinstance(ckpt, dict) and ckpt.get("epoch", None) is not None:
            start_outer = max(0, int(ckpt.get("epoch", -1)) + 1)

        best_acc1 = None
        if isinstance(ckpt, dict) and ckpt.get("best_acc1", None) is not None:
            try:
                best_acc1 = float(ckpt.get("best_acc1"))
            except Exception:
                best_acc1 = None

        logger.info(
            "[INIT_CKPT] start_outer=%d (use_resume_epoch=%s) best_acc1=%s",
            int(start_outer),
            bool(use_resume_epoch),
            str(best_acc1),
        )
        return int(start_outer), best_acc1
    except Exception as e:
        logger.warning("[INIT_CKPT] failed to load %s: %s", str(ckpt_path), str(e))
        return 0, None

from utils.trace_contract_v54 import (
    REQUIRED_GATING_KEYS,
    REQUIRED_PROXY_SANITIZE_KEYS,
)
from utils.trace_guard import (
    init_trace_dir_v54,
    append_trace_event_v54,
    finalize_trace_dir,
    update_trace_summary,
    build_baseline_trace_summary,
    build_trace_header_payload_v54,
)
from utils.trace_signature_v54 import build_signature_v54, REQUIRED_SIGNATURE_FIELDS
from utils.trace_payload_v54 import make_gating_payload_v54
from utils.config import AttrDict
from utils.config_utils import get_nested
from utils.contract_seal import assert_cfg_sealed_or_violate
from utils.pretrain_loader import maybe_load_pretrained
from utils.stable_hw import (
    apply_accuracy_guard,
    get_accuracy_metric_key,
    init_locked_acc_ref,
    init_hw_refs_from_baseline_stats,
    stable_hw_log_fields,
    stable_hw_schedule,
    update_train_acc1_ema,
    update_hw_refs_from_stats,
)

# v5 StableHW: allow_discrete_updates gates ALL discrete signature changes.
# Discrete updates include: partition updates, device mapping updates, layout optimize updates,
# channel rewires, and any track_live/refine steps that alter assignments/signatures.
# In RECOVERY we only allow continuous model training and cached evaluations.


def _get_objective_cfg(cfg) -> dict:
    """
    Unify objective config access across training/export/layout scripts.
    Preferred: cfg.objective (used by layout_agent / heuragenix configs)
    Fallback : cfg.layout (legacy)
    """
    obj = {}
    try:
        if hasattr(cfg, "objective") and cfg.objective is not None:
            obj = OmegaConf.to_container(cfg.objective, resolve=True) or {}
    except Exception:
        obj = {}
    if not obj:
        try:
            if hasattr(cfg, "layout") and cfg.layout is not None:
                obj = OmegaConf.to_container(cfg.layout, resolve=True) or {}
        except Exception:
            obj = {}
    return obj or {}


def _as_float(val, name: str) -> float:
    """Convert config values that might be strings into floats with a clear error."""
    try:
        return float(val)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected {name} to be numeric, but got {val!r}.") from exc


def _seed_worker(worker_id: int, base_seed: int) -> None:
    seed = base_seed + worker_id
    seed_everything(seed)


def _get_iso_cfg_value(iso_cfg, key: str, default=None):
    if iso_cfg is None:
        return default
    if isinstance(iso_cfg, dict):
        return iso_cfg.get(key, default)
    return getattr(iso_cfg, key, default)


def _cfg_get(obj, key: str, default=None):
    if obj is None:
        return default

    # Fast path: non-string key keeps old behavior.
    if not isinstance(key, str):
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    # Preserve old behavior first for exact flat keys / attrs.
    if isinstance(obj, dict):
        if key in obj:
            return obj.get(key, default)
    else:
        if hasattr(obj, key):
            return getattr(obj, key, default)

    # Support dotted path lookup like "a.b.c".
    if "." not in key:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    cur = obj
    for part in key.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            if part not in cur:
                return default
            cur = cur.get(part)
        else:
            if not hasattr(cur, part):
                return default
            cur = getattr(cur, part)

    return default if cur is None else cur


@torch.no_grad()
def _repair_nonfinite_params_(module: torch.nn.Module, *, max_abs: float = 1.0e4) -> bool:
    repaired = False
    for p in module.parameters(recurse=True):
        if p is None:
            continue
        if not torch.isfinite(p).all():
            p.data = torch.nan_to_num(p.data, nan=0.0, posinf=float(max_abs), neginf=-float(max_abs))
            p.data.clamp_(-float(max_abs), float(max_abs))
            repaired = True
    return repaired


def build_dataloader(cfg):
    ds = UCF101Dataset(cfg, split="train")
    batch_size = int(getattr(cfg.data, "batch_size", cfg.train.batch_size))
    base_seed = int(getattr(cfg.training, "seed", getattr(cfg.train, "seed", 0)))
    generator = torch.Generator()
    generator.manual_seed(base_seed)
    worker_init = partial(_seed_worker, base_seed=base_seed)
    pin_memory = bool(getattr(cfg.data, "pin_memory", True))
    persistent_workers = bool(getattr(cfg.data, "persistent_workers", True)) and int(cfg.data.num_workers) > 0
    prefetch_factor = int(getattr(cfg.data, "prefetch_factor", 2))
    kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        worker_init_fn=worker_init,
        generator=generator,
        pin_memory=pin_memory,
    )
    if int(cfg.data.num_workers) > 0:
        kwargs.update(dict(persistent_workers=persistent_workers, prefetch_factor=prefetch_factor))
    return DataLoader(ds, **kwargs)


def build_val_loader(cfg) -> DataLoader:
    ds = UCF101Dataset(cfg, split="val")
    batch_size = int(getattr(cfg.data, "batch_size", cfg.train.batch_size))
    base_seed = int(getattr(cfg.training, "seed", getattr(cfg.train, "seed", 0)) or 0)

    generator = torch.Generator()
    generator.manual_seed(base_seed)
    worker_init = partial(_seed_worker, base_seed=base_seed)

    pin_memory = bool(getattr(cfg.data, "pin_memory", True))
    persistent_workers = bool(getattr(cfg.data, "persistent_workers", True)) and int(cfg.data.num_workers) > 0
    prefetch_factor = int(getattr(cfg.data, "prefetch_factor", 2))

    val_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        worker_init_fn=worker_init,
        generator=generator,
        pin_memory=pin_memory,
    )
    if int(cfg.data.num_workers) > 0:
        val_kwargs.update(dict(persistent_workers=persistent_workers, prefetch_factor=prefetch_factor))

    return DataLoader(ds, **val_kwargs)


def validate_one_epoch(model: torch.nn.Module, val_loader: DataLoader, device: torch.device, amp: bool,
                       max_batches: int = 0, model_type: str = "video", amp_dtype: torch.dtype | None = None) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if max_batches and idx >= max_batches:
                break
            x = batch["video"].to(device)
            y = batch["label"].to(device)
            _dtype = amp_dtype if amp_dtype is not None else torch.float16
            with autocast(device.type, enabled=amp, dtype=_dtype):
                if model_type == "video_audio":
                    logits = model(x, batch["audio"].to(device))
                else:
                    logits = model(x)
            pred = logits.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    return float(correct) / max(1, total)


def _traffic_aware_seed(sites_xy: np.ndarray, traffic: np.ndarray, S: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy placement of hot traffic pairs onto nearest site pairs (SPEC §7.1)."""

    Ns = sites_xy.shape[0]
    assign = np.full(S, -1, dtype=int)
    used_sites: set[int] = set()
    assigned_slots: set[int] = set()

    t_sym = traffic + traffic.T
    hot_pairs: list[tuple[int, int, float]] = []
    for i in range(S):
        for j in range(i + 1, S):
            hot_pairs.append((i, j, float(t_sym[i, j])))
    hot_pairs.sort(key=lambda x: x[2], reverse=True)
    hot_pairs = [p for p in hot_pairs if p[2] > 0][: max(4, S)]

    # precompute site distances
    site_pairs: list[tuple[int, int, float]] = []
    for a in range(Ns):
        for b in range(a + 1, Ns):
            d = float(np.linalg.norm(sites_xy[a] - sites_xy[b]))
            site_pairs.append((a, b, d))
    site_pairs.sort(key=lambda x: x[2])

    for i, j, _ in hot_pairs:
        if i in assigned_slots or j in assigned_slots:
            continue
        for a, b, _ in site_pairs:
            if a in used_sites or b in used_sites:
                continue
            assign[i] = a
            assign[j] = b
            used_sites.update([a, b])
            assigned_slots.update([i, j])
            break

    # fill remaining slots deterministically
    remaining_sites = [s for s in range(Ns) if s not in used_sites]
    for s_idx in range(S):
        if assign[s_idx] == -1:
            if not remaining_sites:
                break
            assign[s_idx] = remaining_sites.pop(0)

    if np.any(assign < 0):
        missing = np.nonzero(assign < 0)[0][:8]
        raise ValueError(
            f"[traffic_aware_seed] failed to assign all slots; missing {missing.tolist()} (Ns={Ns}, S={S})"
        )
    return assign


def _micro_place_seed(
    assign_seed: np.ndarray,
    sites_xy: np.ndarray,
    evaluator: LayoutEvaluator,
    layout_state: LayoutState,
    traffic: np.ndarray,
    steps: int = 80,
    T0: float = 1.0,
    alpha: float = 0.995,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Lightweight SA-based micro placement for training seed export (SPEC §7.2)."""

    rng = rng or np.random.default_rng()
    assign = assign_seed.copy()
    layout_state.assign = assign
    eval_cur = evaluator.evaluate(layout_state)
    best = eval_cur["total_scalar"]
    best_assign = assign.copy()
    traffic_sym = traffic + traffic.T
    accepts = 0
    T = T0

    site_dists = np.linalg.norm(sites_xy[:, None, :] - sites_xy[None, :, :], axis=-1)
    neighbor_k = min(30, sites_xy.shape[0])

    for _ in range(steps):
        action = rng.random()
        new_assign = assign.copy()
        if action < 0.6:
            # swap hot pair
            pairs = [(i, j, traffic_sym[i, j]) for i in range(layout_state.S) for j in range(i + 1, layout_state.S)]
            pairs.sort(key=lambda x: x[2], reverse=True)
            i, j, _ = pairs[rng.integers(0, max(1, len(pairs)) - 1)] if pairs else (0, 1, 0)
            new_assign[i], new_assign[j] = new_assign[j], new_assign[i]
        else:
            # relocate to nearby empty site
            empty_sites = [s for s in range(sites_xy.shape[0]) if s not in new_assign]
            if empty_sites:
                slot = int(rng.integers(0, layout_state.S))
                dists = [(sid, site_dists[new_assign[slot], sid]) for sid in empty_sites]
                dists.sort(key=lambda x: x[1])
                candidate = dists[:neighbor_k]
                if candidate:
                    site_id = candidate[rng.integers(0, len(candidate))][0]
                    new_assign[slot] = site_id

        layout_state.assign = new_assign
        eval_new = evaluator.evaluate(layout_state)
        delta = eval_new["total_scalar"] - eval_cur["total_scalar"]
        if delta < 0 or math.exp(-delta / max(T, 1e-6)) > rng.random():
            assign = new_assign
            eval_cur = eval_new
            accepts += 1
            if eval_cur["total_scalar"] < best:
                best = eval_cur["total_scalar"]
                best_assign = assign.copy()
        layout_state.assign = assign
        T *= alpha

    layout_state.assign = best_assign
    return best_assign, {"steps": steps, "accepts": accepts, "best_total": float(best)}


def _export_layout_input(
    cfg,
    export_dir: Path,
    out_dir: Path,
    chiplet_slots: ChipletSlots,
    mapping_solver: MappingSolver,
    segments: List,
    mapping: List[int],
    wafer_layout: WaferLayout,
    seed: int = 0,
):
    """Export layout_input.json following SPEC v5.4 (§10.1).

    This uses deterministic square-grid sites and the latest mapping/segments
    from training to provide a reproducible offline entry point.
    """

    export_dir.mkdir(parents=True, exist_ok=True)

    # Build discrete sites shared by train/offline
    chip_types = list(chiplet_slots.library.types.values())
    chip_max_w = max(t.width_mm for t in chip_types)
    chip_max_h = max(t.height_mm for t in chip_types)
    sites_xy = build_sites(
        wafer_radius_mm=float(cfg.hw.wafer_radius_mm),
        chip_max_width_mm=chip_max_w,
        chip_max_height_mm=chip_max_h,
        margin_mm=float(getattr(cfg.hw, "site_margin_mm", 5.0)),
        method="square_grid_in_circle",
        grid_pitch_mm=None,
        seed=int(seed),
    )
    S = int(cfg.hw.num_slots)
    Ns = sites_xy.shape[0]
    assign_grid = np.arange(S, dtype=int) % Ns

    eff_specs = chiplet_slots(hard=False)["eff_specs"]
    chip_tdp = eff_specs["tdp_w"].detach().cpu().numpy().astype(float)

    traffic = mapping_solver.build_traffic_matrix(segments, mapping, num_slots=S).cpu().numpy().astype(float)
    if traffic.shape != (S, S):
        raise ValueError(
            f"[export_layout_input] traffic must be (S,S)=({S},{S}), got {traffic.shape}"
        )
    obj = _get_objective_cfg(cfg)

    sigma_mm = float(obj.get("sigma_mm", 3.0))
    baseline = obj.get("baseline", {"L_comm_baseline": 1.0, "L_therm_baseline": 1.0})
    scalar_weights = obj.get(
        "scalar_weights",
        {"w_comm": 1.0, "w_therm": 1.0, "w_penalty": 1000.0},
    )
    w_comm = float(scalar_weights.get("w_comm", 1.0))
    w_therm = float(scalar_weights.get("w_therm", 1.0))
    w_penalty = float(scalar_weights.get("w_penalty", 1000.0))
    rng = np.random.default_rng(seed)
    baseline_eval = LayoutEvaluator(
        sigma_mm=sigma_mm,
        baseline=baseline,
        scalar_w={"w_comm": w_comm, "w_therm": w_therm, "w_penalty": w_penalty},
    )
    layout_state = LayoutState(
        S=S,
        Ns=Ns,
        wafer_radius_mm=float(cfg.hw.wafer_radius_mm),
        sites_xy_mm=sites_xy,
        assign=assign_grid,
        chip_tdp_w=chip_tdp,
        traffic_bytes=traffic,
        meta={},
    )
    base_res = baseline_eval.evaluate(layout_state)
    baseline_eval = LayoutEvaluator(
        sigma_mm=sigma_mm,
        baseline={"L_comm_baseline": base_res["L_comm"], "L_therm_baseline": base_res["L_therm"]},
        scalar_w={"w_comm": w_comm, "w_therm": w_therm, "w_penalty": w_penalty},
    )
    seed_cfg = getattr(cfg, "layout_seed", None)
    method = getattr(seed_cfg, "method", "grid") if seed_cfg is not None else "grid"
    if method == "grid":
        assign_seed = assign_grid.copy()
    elif method in {"traffic", "traffic_aware", "seed"}:
        assign_seed = _traffic_aware_seed(sites_xy, traffic, S, rng)
    else:
        raise ValueError(f"[export_layout_input] unsupported layout_seed.method={method!r}")

    bad = assign_seed[(assign_seed < 0) | (assign_seed >= Ns)]
    if bad.size > 0:
        raise ValueError(
            f"[export_layout_input] assign_seed has invalid site ids: {np.unique(bad)[:8]} Ns={Ns}"
        )

    micro_enabled = bool(getattr(seed_cfg, "enable_micro_place", False)) if seed_cfg is not None else False
    micro_steps = int(getattr(seed_cfg, "micro_place_steps", 0)) if seed_cfg is not None else 0
    if micro_enabled and micro_steps > 0:
        layout_state.assign = assign_seed
        assign_seed, micro_stats = _micro_place_seed(
            assign_seed,
            sites_xy,
            baseline_eval,
            layout_state,
            traffic,
            steps=micro_steps,
            T0=float(getattr(seed_cfg, "micro_place_T0", 1.0)),
            alpha=float(getattr(seed_cfg, "micro_place_alpha", 0.995)),
            rng=rng,
        )
    else:
        micro_stats = {
            "skipped": True,
            "enable_micro_place": micro_enabled,
            "micro_place_steps": micro_steps,
        }

    layout_state.assign = assign_seed

    baseline = {
        "assign_grid": assign_grid.tolist(),
        "L_comm": base_res["L_comm"],
        "L_therm": base_res["L_therm"],
        "comm_norm": 1.0,
        "therm_norm": 1.0,
    }

    # Serialize segments minimally
    segments_json = []
    for idx, seg in enumerate(segments):
        segments_json.append(
            {
                "id": getattr(seg, "id", idx),
                "flops": getattr(seg, "flops", 0.0),
                "bytes": getattr(seg, "bytes", 0.0),
                "seq_len": getattr(seg, "seq_len", 0),
                "embed_dim": getattr(seg, "embed_dim", 0),
                "num_heads": getattr(seg, "num_heads", 0),
                "mlp_ratio": getattr(seg, "mlp_ratio", 0.0),
                "traffic_out_bytes": getattr(seg, "traffic_out_bytes", 0.0),
            }
        )

    layout_input = {
        "layout_version": "v5.4",
        "wafer": {"radius_mm": float(cfg.hw.wafer_radius_mm), "margin_mm": float(getattr(cfg.hw, "site_margin_mm", 5.0))},
        "sites": {
            "method": "square_grid_in_circle",
            "pitch_mm": None,
            "sites_xy": sites_xy.tolist(),
        },
        "slots": {"S": S, "tdp": chip_tdp.tolist()},
        "mapping": {
            "mapping_id": f"train_step_final",
            "segments": segments_json,
            "traffic_matrix": traffic.tolist(),
            "mapping": mapping,
            "used_slots": sorted(set(mapping)),
            "traffic_mode": "full_SxS_slot_order_0_to_S-1",
        },
        "baseline": baseline,
        "seed": {"assign_seed": assign_seed.tolist(), "micro_place_stats": micro_stats},
        "objective_cfg": {
            "sigma_mm": sigma_mm,
            "scalar_weights": {"w_comm": w_comm, "w_therm": w_therm, "w_penalty": w_penalty},
        },
    }

    out_path = export_dir / "layout_input.json"
    with out_path.open("w", encoding="utf-8") as f:
        safe_dump(layout_input, f, indent=2)

    # === v5.4 contract: ALWAYS materialize canonical copies (no silent failure) ===
    project_root = Path(__file__).resolve().parents[1]  # proj_ast2_ucf101_full/
    canonical_a3_dir = project_root / "outputs" / "P3" / "A3"
    canonical_a3_path = canonical_a3_dir / "layout_input.json"

    # run_root is the training out_dir
    run_root_dir = Path(out_dir).resolve()
    run_root_path = run_root_dir / "layout_input.json"

    # Ensure parents exist
    canonical_a3_dir.mkdir(parents=True, exist_ok=True)
    run_root_dir.mkdir(parents=True, exist_ok=True)

    # Write canonical copies deterministically
    canonical_a3_path.write_text(safe_dumps(layout_input, indent=2), encoding="utf-8")
    run_root_path.write_text(safe_dumps(layout_input, indent=2), encoding="utf-8")

    return out_path






def _solve_mapping_for_cache(
    model: torch.nn.Module,
    chiplet_slots: ChipletSlots,
    mapping_solver: MappingSolver,
    hw_proxy: LayerHwProxy,
    wafer_layout: WaferLayout,
    partitioner: PartitionPlanner,
    hw_cfg: Any,
    model_info: Optional[Dict[str, Any]] = None,
    mapping_strategy: Optional[str] = None,
) -> Dict[str, Any]:
    slot_out = chiplet_slots(hard=False)
    eff_specs = slot_out["eff_specs"]
    part_res = partitioner.plan(
        model,
        eff_specs,
        alpha=slot_out["alpha"],
        model_info=model_info,
        use_fine_split=getattr(hw_cfg, "use_fine_split", True),
    )
    segments = part_res["segments"]
    mapping_result = mapping_solver.solve_mapping(
        segments,
        eff_specs,
        hw_proxy,
        layout_positions=wafer_layout.current_pos_continuous(),
        strategy=(str(mapping_strategy) if mapping_strategy is not None else getattr(hw_cfg, "mapping_strategy", "greedy_local")),
        distance_scale_ms=getattr(hw_cfg, "distance_scale_ms", 0.0),
        alpha=slot_out["alpha"],
    )
    mapping = mapping_result.get("mapping", [])
    S = int(eff_specs["peak_flops"].shape[0]) if "peak_flops" in eff_specs else int(len(mapping))
    seg_sig_payload = [
        {
            "k": int(i),
            "flops": float(getattr(seg, "flops", 0.0)),
            "bytes": float(getattr(seg, "bytes", 0.0)),
            "traffic": float(getattr(seg, "traffic_out_bytes", 0.0)),
            "mem": float(getattr(seg, "mem_mb", 0.0)),
        }
        for i, seg in enumerate(segments)
    ]
    mapping_sig = stable_hash(
        {
            "S": int(S),
            "mapping": [int(x) for x in mapping],
            "segments": seg_sig_payload,
        }
    )
    signature = f"seg{len(segments)}_{mapping_sig}"
    mapping_result["segments"] = segments
    mapping_result["signature"] = signature
    mapping_result["mapping_sig"] = mapping_sig
    return mapping_result


def _solve_layout_for_cache(
    chiplet_slots: ChipletSlots,
    wafer_layout: WaferLayout,
    hw_cfg: Any,
    mapping_result: Dict[str, Any],
) -> Dict[str, Any]:
    slot_out = chiplet_slots(hard=False)
    eff_specs = slot_out["eff_specs"]
    segments = mapping_result.get("segments", [])
    mapping = mapping_result.get("mapping", [])
    if not segments or not mapping:
        return {"loss": 0.0, "stats": {}, "signature": None}
    with torch.no_grad():
        L_layout, layout_stats = wafer_layout(
            mapping,
            segments,
            eff_specs,
            lambda_boundary=hw_cfg.lambda_boundary,
            lambda_overlap=hw_cfg.lambda_overlap,
            lambda_comm=hw_cfg.lambda_comm_extra,
            lambda_thermal=hw_cfg.lambda_thermal,
            distance_scale=float(getattr(hw_cfg, "distance_scale_ms", 1.0) or 1.0),
        )
    # v5.4: layout signature must be assign-only (SPEC_B). pos is auxiliary and must NOT affect signature.
    assign = getattr(wafer_layout, "assign", None)
    signature = signature_for_assign(assign)
    return {"loss": float(L_layout.item()), "stats": layout_stats, "signature": signature}




#
# ROI-Commit (v5.4+): discrete mapping/layout safe commit
#
def _roi_get_cfg(iso_cfg) -> dict:
    roi = _get_iso_cfg_value(iso_cfg, "roi_commit", None)
    if roi is None:
        return {}
    if isinstance(roi, dict):
        return dict(roi)
    try:
        return {k: roi[k] for k in roi}
    except Exception:
        try:
            return dict(vars(roi))
        except Exception:
            return {}


def _roi_rel_improve(old: float, new: float) -> float:
    denom = max(abs(float(old)), 1e-6)
    return (float(old) - float(new)) / denom


@torch.no_grad()
def _roi_eval_hw_metric(
    *,
    cfg,
    hw_proxy,
    mapping_solver,
    wafer_layout,
    stable_hw_cfg,
    stable_hw_state,
    slot_out: dict,
    mapping_res: dict,
    metric_key: str,
):
    """Compute a cheap, deterministic HW metric for ROI-Commit.

    Note: mapping/layout do NOT affect the forward accuracy directly; they affect training only via HW loss.
    Here we evaluate the *proxy hardware metric* under the candidate discrete bundle.
    """
    segments = mapping_res.get("segments", []) if isinstance(mapping_res, dict) else []
    mapping = mapping_res.get("mapping", []) if isinstance(mapping_res, dict) else []
    if not segments or not mapping:
        return 0.0, 0.0, {}

    eff_specs = slot_out.get("eff_specs", None)
    alpha = slot_out.get("alpha", None)
    if eff_specs is None:
        return 0.0, 0.0, {}

    # signatures (for cache keys inside compute_hw_loss)
    mapping_sig = mapping_res.get("mapping_sig") or mapping_res.get("signature")
    try:
        segments_sig = stable_hash([getattr(s, "signature", None) or repr(s) for s in (segments or [])])
    except Exception:
        segments_sig = None

    layout_pos = wafer_layout.current_pos_continuous()
    L_hw, hw_stats = compute_hw_loss(
        cfg=cfg,
        hw_proxy=hw_proxy,
        model_info={},
        stable_hw_cfg=stable_hw_cfg,
        stable_hw_state=stable_hw_state,
        segments=segments,
        mapping=mapping,
        mapping_sig=str(mapping_sig) if mapping_sig is not None else None,
        segments_sig=str(segments_sig) if segments_sig is not None else None,
        eff_specs=eff_specs,
        layout_positions=layout_pos,
        mapping_solver=mapping_solver,
        wafer_layout=wafer_layout,
        alpha=alpha,
    )
    mk = str(metric_key or "proxy_raw_latency_ms")
    if mk in ("L_hw_total", "hw_total", "L_hw"):
        metric = float(L_hw.detach().cpu().item())
    else:
        metric = float(hw_stats.get(mk, hw_stats.get("proxy_raw_latency_ms", 0.0)))
    return float(metric), float(L_hw.detach().cpu().item()), hw_stats


def _roi_should_commit(
    *,
    old_metric: float,
    new_metric: float,
    min_rel_improve: float,
    min_abs_improve: float = 0.0,
) -> bool:
    abs_impr = float(old_metric) - float(new_metric)
    rel = _roi_rel_improve(old_metric, new_metric)
    return (abs_impr >= float(min_abs_improve)) and (rel >= float(min_rel_improve))


def _roi_mapping_change_frac(old_mapping_res: Optional[Dict[str, Any]], new_mapping_res: Optional[Dict[str, Any]]) -> float:
    """Cheap, candidate-specific risk proxy: fraction of mapping assignments that change.

    This is intentionally simple and deterministic: it forces ROI decisions to depend on the candidate mapping,
    enabling ACHO (via lambda_hw_effective) to influence commit/reject in a measurable way.
    """
    try:
        if not old_mapping_res or not new_mapping_res:
            return 0.0
        old_map = old_mapping_res.get("mapping", None)
        new_map = new_mapping_res.get("mapping", None)
        if not isinstance(old_map, (list, tuple)) or not isinstance(new_map, (list, tuple)):
            return 0.0
        n = max(len(old_map), len(new_map))
        if n <= 0:
            return 0.0
        m = min(len(old_map), len(new_map))
        diff = 0
        for i in range(m):
            if int(old_map[i]) != int(new_map[i]):
                diff += 1
        # Count length mismatch as changed assignments.
        diff += (n - m)
        return float(diff) / float(n)
    except Exception:
        return 0.0


def _roi_extract_keep_factors(model_info: Optional[Dict[str, Any]], depth: int) -> Dict[str, List[float]]:
    """Extract per-layer keep signals as lists to stabilize discrete planning."""
    if not model_info:
        return {"token_keep": [1.0] * depth, "head_keep": [1.0] * depth, "ch_keep": [1.0] * depth, "block_keep": [1.0] * depth}

    mi = model_info
    if isinstance(mi, dict) and ("model_info" in mi) and isinstance(mi.get("model_info"), dict):
        if not any(k in mi for k in ("keep_factors", "token_keep", "head_keep", "ch_keep", "block_keep")):
            mi = mi["model_info"]

    kf = None
    if isinstance(mi, dict) and mi.get("keep_factors") is not None:
        kf = mi.get("keep_factors")
    elif isinstance(mi, dict) and mi.get("keep_factors_t") is not None:
        kf = mi.get("keep_factors_t")
    elif isinstance(mi, dict):
        kf = mi

    def _as_list(x, n: int, default: float = 1.0) -> List[float]:
        if x is None:
            return [default] * n
        if torch.is_tensor(x):
            xx = x.detach().float().reshape(-1)
            if int(xx.numel()) == 1:
                return [float(xx.item())] * n
            if int(xx.numel()) == n:
                return [float(v) for v in xx.tolist()]
            if int(xx.numel()) > 0:
                return [float(xx[0].item())] * n
            return [default] * n
        if isinstance(x, (int, float)):
            return [float(x)] * n
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return [default] * n
            if len(x) == n:
                return [float(v) for v in x]
            return [float(x[0])] * n
        return [default] * n

    if isinstance(kf, dict):
        return {
            "token_keep": _as_list(kf.get("token_keep", 1.0), depth, 1.0),
            "head_keep": _as_list(kf.get("head_keep", 1.0), depth, 1.0),
            "ch_keep": _as_list(kf.get("ch_keep", 1.0), depth, 1.0),
            "block_keep": _as_list(kf.get("block_keep", 1.0), depth, 1.0),
        }
    return {"token_keep": [1.0] * depth, "head_keep": [1.0] * depth, "ch_keep": [1.0] * depth, "block_keep": [1.0] * depth}


def _roi_update_keep_ema(stable_hw_state: Dict[str, Any], model_info: Optional[Dict[str, Any]], *, ema_alpha: float = 0.2) -> None:
    """Update EMA of keep_factors; used for stable discrete planning + ROI evaluation."""
    if not isinstance(stable_hw_state, dict) or (not isinstance(model_info, dict)):
        return
    depth = int(stable_hw_state.get("arch_depth", 0) or 0)
    if depth <= 0:
        mi = model_info.get("model_info", model_info)
        if isinstance(mi, dict):
            kft = mi.get("keep_factors_t", None)
            if isinstance(kft, dict):
                tk = kft.get("token_keep", None)
                if torch.is_tensor(tk) and tk.ndim == 1:
                    depth = int(tk.numel())
    if depth <= 0:
        return

    cur = _roi_extract_keep_factors(model_info, depth)
    prev = stable_hw_state.get("roi_keep_ema", None)
    a = float(max(0.0, min(1.0, float(ema_alpha))))
    if not isinstance(prev, dict):
        stable_hw_state["roi_keep_ema"] = cur
        return

    out: Dict[str, List[float]] = {}
    for k in ("token_keep", "head_keep", "ch_keep", "block_keep"):
        c = cur.get(k, [1.0] * depth)
        p = prev.get(k, [1.0] * depth) if isinstance(prev.get(k), list) else [1.0] * depth
        if len(p) != depth:
            p = [float(p[0])] * depth if p else [1.0] * depth
        out[k] = [a * float(c[i]) + (1.0 - a) * float(p[i]) for i in range(depth)]
    stable_hw_state["roi_keep_ema"] = out


def _get_ast_pruner(model: torch.nn.Module):
    core = unwrap_model(model)
    pruner = getattr(core, "ast_pruner", None)
    if pruner is None:
        return None
    if not hasattr(pruner, "g_ch"):
        return None
    return pruner


def _get_layer_ch_keep_now(model: torch.nn.Module) -> Optional[torch.Tensor]:
    pruner = _get_ast_pruner(model)
    if pruner is None:
        return None
    if hasattr(pruner, "get_layer_ch_keep"):
        try:
            return pruner.get_layer_ch_keep().detach().float().cpu()
        except Exception:
            pass
    with torch.no_grad():
        return torch.sigmoid(pruner.g_ch.detach()).mean(dim=1).float().cpu()


def _summarize_layer_ch_keep(
    keep_now: Optional[torch.Tensor],
    freeze_prefix_ratio: float = 0.0,
) -> Dict[str, Any]:
    if keep_now is None:
        return {
            "ch_keep_real_mean": 1.0,
            "ch_keep_real_min": 1.0,
            "ch_keep_real_max": 1.0,
            "ch_keep_prunable_mean": 1.0,
            "ch_keep_layerwise": [],
        }

    keep_cpu = keep_now.detach().float().cpu()
    depth = int(keep_cpu.numel())
    k_freeze = int(round(float(depth) * float(max(0.0, min(1.0, freeze_prefix_ratio)))))
    if k_freeze < depth:
        prunable = keep_cpu[k_freeze:]
    else:
        prunable = keep_cpu

    return {
        "ch_keep_real_mean": float(keep_cpu.mean().item()),
        "ch_keep_real_min": float(keep_cpu.min().item()),
        "ch_keep_real_max": float(keep_cpu.max().item()),
        "ch_keep_prunable_mean": float(prunable.mean().item()),
        "ch_keep_layerwise": keep_cpu.tolist(),
    }


def _update_alloc_layer_sens_ema(
    model: torch.nn.Module,
    run_state: Dict[str, Any],
    ema_alpha: float = 0.8,
) -> Optional[torch.Tensor]:
    pruner = _get_ast_pruner(model)
    if pruner is None:
        return None
    grad = getattr(pruner.g_ch, "grad", None)
    if grad is None:
        return None
    sens = grad.detach().abs().mean(dim=1).float().cpu()
    prev = run_state.get("alloc_layer_sens_ema", None)
    if isinstance(prev, torch.Tensor) and prev.shape == sens.shape:
        sens = float(ema_alpha) * prev + (1.0 - float(ema_alpha)) * sens
    run_state["alloc_layer_sens_ema"] = sens.clone()
    return sens


def _build_eval_model_info_with_ch_override(
    last_info: Optional[Dict[str, Any]],
    depth: int,
    ch_keep_override: List[float],
) -> Dict[str, Any]:
    keep = _roi_extract_keep_factors(last_info, depth)
    keep["ch_keep"] = [float(x) for x in ch_keep_override]
    return {"keep_factors": keep}


def _safe_logit_scalar(p: float, eps: float = 1e-4) -> float:
    p = float(max(eps, min(1.0 - eps, p)))
    return float(math.log(p / (1.0 - p)))


def _structural_gate_open_logit(cfg) -> tuple[float, float, float]:
    ast_cfg = getattr(cfg, "ast", None)
    pruner_cfg = getattr(ast_cfg, "pruner", None) if ast_cfg is not None else None
    gate_init_open = float(getattr(pruner_cfg, "gate_init_open", 0.99) if pruner_cfg is not None else 0.99)
    head_init_open = float(getattr(pruner_cfg, "head_init_open", gate_init_open) if pruner_cfg is not None else gate_init_open)
    ch_init_open = float(getattr(pruner_cfg, "ch_init_open", gate_init_open) if pruner_cfg is not None else gate_init_open)
    block_init_open = float(getattr(pruner_cfg, "block_init_open", gate_init_open) if pruner_cfg is not None else gate_init_open)
    return (_safe_logit_scalar(head_init_open), _safe_logit_scalar(ch_init_open), _safe_logit_scalar(block_init_open))


def _force_open_structural_gates_(model, cfg) -> None:
    pruner = _get_ast_pruner(model)
    if pruner is None:
        return
    head_l, ch_l, block_l = _structural_gate_open_logit(cfg)
    with torch.no_grad():
        if hasattr(pruner, "g_head"):
            pruner.g_head.data.fill_(float(head_l))
        if hasattr(pruner, "g_ch"):
            pruner.g_ch.data.fill_(float(ch_l))
        if hasattr(pruner, "g_block"):
            pruner.g_block.data.fill_(float(block_l))


def _zero_structural_gate_grads_(model) -> None:
    pruner = _get_ast_pruner(model)
    if pruner is None:
        return
    for name in ("g_head", "g_ch", "g_block"):
        if hasattr(pruner, name):
            grad = getattr(getattr(pruner, name), "grad", None)
            if grad is not None:
                grad.zero_()


def _apply_layerwise_keep_candidate_to_gates(
    model: torch.nn.Module,
    keep_target: torch.Tensor,
    *,
    gate_apply_blend: float = 1.0,
) -> None:
    pruner = _get_ast_pruner(model)
    if pruner is None:
        return
    with torch.no_grad():
        keep_now = torch.sigmoid(pruner.g_ch).mean(dim=1)
        depth = int(keep_now.numel())

        try:
            pr_cfg = pruner.cfg.get("channel_prune", pruner.cfg)
            front_ratio = float(pr_cfg.get("freeze_prefix_ratio", pruner.cfg.get("ch_freeze_prefix_ratio", 0.0)) or 0.0)
            front_ratio = float(max(0.0, min(1.0, front_ratio)))
        except Exception:
            front_ratio = 0.0
        k_freeze = int(round(float(depth) * float(front_ratio)))

        tgt = keep_target.to(keep_now.device, dtype=keep_now.dtype).clone()
        if k_freeze > 0:
            tgt[:k_freeze] = 1.0

        blend = float(max(0.0, min(1.0, gate_apply_blend)))
        for i in range(k_freeze, depth):
            cur_i = float(keep_now[i].item())
            tgt_i = float(tgt[i].item())
            delta = (_safe_logit_scalar(tgt_i) - _safe_logit_scalar(cur_i)) * blend
            pruner.g_ch.data[i].add_(float(delta))


def _build_alloc_candidates_from_remaining_budget(
    keep_now: torch.Tensor,
    sens: torch.Tensor,
    *,
    freeze_prefix_ratio: float,
    remain_budget: float,
    keep_unit: float,
    min_keep_floor: float,
    pool_size: int,
    max_candidates: int,
) -> List[torch.Tensor]:
    depth = int(keep_now.numel())
    k_freeze = int(round(float(depth) * float(max(0.0, min(1.0, freeze_prefix_ratio)))))
    prunable = list(range(k_freeze, depth))
    if not prunable:
        return []

    sens = sens.clone().float()
    sens = sens / (float(sens.mean().item()) + 1e-6)

    prunable_sorted = sorted(prunable, key=lambda i: float(sens[i].item()))
    pool = prunable_sorted[: min(int(pool_size), len(prunable_sorted))]
    if not pool:
        return []

    budget = float(max(0.0, remain_budget))
    if budget <= 1e-8:
        return []

    u = float(max(1e-4, keep_unit))
    floor = float(max(0.05, min(0.95, min_keep_floor)))

    cands: List[torch.Tensor] = []

    for i in pool:
        cand = keep_now.clone()
        di = min(float(budget), float(max(0.0, cand[i].item() - floor)))
        cand[i] = cand[i] - di
        cands.append(cand)

    pair_ratios = [(0.75, 0.25), (0.5, 0.5), (0.25, 0.75)]
    for i, j in itertools.combinations(pool, 2):
        for r1, r2 in pair_ratios:
            cand = keep_now.clone()
            di = min(float(budget * r1), float(max(0.0, cand[i].item() - floor)))
            dj = min(float(budget * r2), float(max(0.0, cand[j].item() - floor)))
            total = di + dj
            if total <= 1e-8:
                continue
            cand[i] = cand[i] - di
            cand[j] = cand[j] - dj
            cands.append(cand)

    tri_ratios = [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.34, 0.33, 0.33)]
    for combo in itertools.combinations(pool, 3):
        for rs in tri_ratios:
            cand = keep_now.clone()
            used = 0.0
            for idx, rr in zip(combo, rs):
                d = min(float(budget * rr), float(max(0.0, cand[idx].item() - floor)))
                cand[idx] = cand[idx] - d
                used += d
            if used > 1e-8:
                cands.append(cand)

    uniq = []
    seen = set()
    for cand in cands:
        q = cand.clone()
        for i in range(k_freeze, depth):
            q[i] = round(float(q[i].item()) / u) * u
            q[i] = float(max(floor, min(1.0, q[i].item())))
        key = tuple(round(float(x), 4) for x in q.tolist())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(q)
        if len(uniq) >= int(max_candidates):
            break
    return uniq


def _to_cpu_tensor_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu()
        else:
            out[k] = v
    return out


def _quantize_keep_vec(q: torch.Tensor, *, k_freeze: int, u: float, floor: float) -> torch.Tensor:
    """Quantize prunable keep values to keep_unit grid and clamp to [floor, 1]."""
    depth = int(q.numel())
    out = q.clone()
    for i in range(int(k_freeze), depth):
        out[i] = round(float(out[i].item()) / float(u)) * float(u)
        out[i] = float(max(float(floor), min(1.0, float(out[i].item()))))
    return out


def _apply_sparse_direction(
    keep_now: torch.Tensor,
    *,
    items: List[Tuple[int, float]],
    delta_mag: float,
    k_freeze: int,
    u: float,
    floor: float,
) -> torch.Tensor:
    """Apply a sparse direction (layer_idx, weight) with total magnitude delta_mag to produce a probe keep vector."""
    cand = keep_now.clone()
    dm = float(max(0.0, delta_mag))
    if dm <= 0.0:
        return _quantize_keep_vec(cand, k_freeze=int(k_freeze), u=float(u), floor=float(floor))
    for li, w in items:
        ww = float(max(0.0, w))
        if ww <= 0.0:
            continue
        d = dm * ww
        maxd = float(max(0.0, float(cand[li].item()) - float(floor)))
        if d > maxd:
            d = maxd
        if d > 0.0:
            cand[li] = cand[li] - float(d)
    return _quantize_keep_vec(cand, k_freeze=int(k_freeze), u=float(u), floor=float(floor))


def _interp_keep_scaled(
    keep_now: torch.Tensor,
    *,
    keep_probe: torch.Tensor,
    scale: float,
    k_freeze: int,
    u: float,
    floor: float,
) -> torch.Tensor:
    """Interpolate from keep_now toward keep_probe by scale in [0,1], then quantize."""
    s = float(max(0.0, min(1.0, scale)))
    cand = keep_now + s * (keep_probe - keep_now)
    return _quantize_keep_vec(cand, k_freeze=int(k_freeze), u=float(u), floor=float(floor))


def _build_alloc_direction_pool(
    *,
    keep_now: torch.Tensor,
    sens: torch.Tensor,
    freeze_prefix_ratio: float,
    pool_size: int,
    max_directions: int,
    candidate_strategy: str,
    direction_orders: Optional[List[str]] = None,
    rng_seed: Optional[int] = None,
) -> List[List[Tuple[int, float]]]:
    depth = int(keep_now.numel())
    k_freeze = int(round(float(depth) * float(max(0.0, min(1.0, freeze_prefix_ratio)))))
    prunable = list(range(k_freeze, depth))
    if not prunable:
        return []

    sens_norm = sens.clone().float()
    sens_norm = sens_norm / (float(sens_norm.mean().item()) + 1e-6)
    strategy = str(candidate_strategy or "sens_pool").lower().strip()
    if strategy == "uniform_all":
        pool = list(prunable)
        random.Random(int(rng_seed) if rng_seed is not None else 0).shuffle(pool)
    else:
        prunable_sorted = sorted(prunable, key=lambda i: float(sens_norm[i].item()))
        p = int(pool_size)
        if p <= 0:
            p = len(prunable_sorted)
        pool = prunable_sorted[: min(int(p), len(prunable_sorted))]
    if not pool:
        return []

    dirs_total = int(max(1, int(max_directions)))
    orders = set(str(x).lower().strip() for x in (direction_orders or ["single", "pair", "tri"]))
    dirs: List[List[Tuple[int, float]]] = []
    seen_dir = set()

    def _push_dir(items: List[Tuple[int, float]]):
        items2 = [(int(i), float(w)) for i, w in items if float(w) > 0.0]
        s = sum(w for _, w in items2)
        if s <= 1e-12:
            return
        items2 = [(i, float(w) / float(s)) for i, w in items2]
        key = tuple((i, round(w, 4)) for i, w in sorted(items2))
        if key in seen_dir:
            return
        seen_dir.add(key)
        dirs.append(items2)

    if "single" in orders:
        for i in pool:
            _push_dir([(int(i), 1.0)])
            if len(dirs) >= dirs_total:
                return dirs

    if "pair" in orders and len(dirs) < dirs_total:
        pair_ratios = [(0.75, 0.25), (0.5, 0.5), (0.25, 0.75)]
        for i, j in itertools.combinations(pool, 2):
            for r1, r2 in pair_ratios:
                _push_dir([(int(i), float(r1)), (int(j), float(r2))])
                if len(dirs) >= dirs_total:
                    return dirs

    if "tri" in orders and len(dirs) < dirs_total:
        tri_ratios = [(0.5, 0.3, 0.2), (0.4, 0.4, 0.2), (0.34, 0.33, 0.33)]
        for combo in itertools.combinations(pool, 3):
            for rs in tri_ratios:
                _push_dir([(int(combo[0]), float(rs[0])), (int(combo[1]), float(rs[1])), (int(combo[2]), float(rs[2]))])
                if len(dirs) >= dirs_total:
                    return dirs
    return dirs


def _alloc_candidate_sens_score(
    *,
    keep_now: torch.Tensor,
    keep_apply: torch.Tensor,
    sens_norm: torch.Tensor,
) -> float:
    delta = (keep_now - keep_apply).abs().float()
    delta_sum = float(delta.sum().item())
    if delta_sum <= 1.0e-12:
        return 1.0e18
    return float((sens_norm * delta).sum().item() / max(1.0e-12, delta_sum))


def _alloc_select_candidates_tiebreak(
    candidates: List[Dict[str, Any]],
    *,
    decision_rule: str,
    tie_apply_abs: float,
    tie_apply_rel: float,
    tie_risk_abs: float,
    tie_risk_rel: float,
    tie_sens_rel: float,
) -> Optional[Dict[str, Any]]:
    valid = [c for c in candidates if isinstance(c, dict) and c.get("rel_hw_gain", None) is not None]
    if not valid:
        return None

    rule = str(decision_rule or "").lower().strip()
    if rule == "apply_only":
        return max(
            valid,
            key=lambda x: (
                float(x.get("rel_hw_gain", -1.0e18)),
                -float(x.get("acc_risk", 1.0e18)),
                -float(x.get("cand_sens", 1.0e18)),
                float(x.get("probe_rel_hw_gain", -1.0e18)),
            ),
        )

    if rule != "apply_risk_sens_then_long":
        rule = "apply_risk_sens_then_long"

    best_apply = max(float(x.get("rel_hw_gain", -1.0e18)) for x in valid)
    apply_tol = max(float(tie_apply_abs), abs(float(best_apply)) * float(tie_apply_rel))
    pool_apply = [x for x in valid if float(x.get("rel_hw_gain", -1.0e18)) >= float(best_apply) - float(apply_tol)]
    if not pool_apply:
        return None

    best_risk = min(float(x.get("acc_risk", 1.0e18)) for x in pool_apply)
    risk_tol = max(float(tie_risk_abs), abs(float(best_risk)) * float(tie_risk_rel))
    pool_risk = [x for x in pool_apply if float(x.get("acc_risk", 1.0e18)) <= float(best_risk) + float(risk_tol)]
    if not pool_risk:
        return None

    best_sens = min(float(x.get("cand_sens", 1.0e18)) for x in pool_risk)
    sens_tol = max(1.0e-8, abs(float(best_sens)) * float(tie_sens_rel))
    pool_sens = [x for x in pool_risk if float(x.get("cand_sens", 1.0e18)) <= float(best_sens) + float(sens_tol)]
    if not pool_sens:
        return None

    return max(
        pool_sens,
        key=lambda x: (
            float(x.get("probe_rel_hw_gain", -1.0e18)),
            -float(x.get("acc_risk", 1.0e18)),
            -float(x.get("cand_sens", 1.0e18)),
            float(x.get("rel_hw_gain", -1.0e18)),
        ),
    )


def _eval_alloc_direction_probe(
    *,
    direction_items: List[Tuple[int, float]],
    keep_now: torch.Tensor,
    base_obj: float,
    usable_budget: float,
    lookahead_budget: float,
    probe_points: int,
    probe_alphas: List[float],
    probe_use_final_range: bool,
    keep_unit: float,
    min_keep_floor: float,
    freeze_prefix_ratio: float,
    sens: torch.Tensor,
    model: torch.nn.Module,
    cfg,
    hw_proxy,
    wafer_layout,
    eff_specs_cpu,
    alpha_cpu,
    last_info: Optional[Dict[str, Any]],
    fine_split_threads: int,
    search_threads: int,
    max_acc_risk: float,
    pick_policy: str,
    decision_rule: str = "legacy_long_gain",
    tie_apply_abs: float = 0.0,
    tie_apply_rel: float = 0.0,
    tie_risk_abs: float = 0.0,
    tie_risk_rel: float = 0.0,
    tie_sens_rel: float = 0.0,
) -> Optional[Dict[str, Any]]:
    del probe_use_final_range  # included for logging/call compatibility
    depth = int(keep_now.numel())
    k_freeze = int(round(float(depth) * float(max(0.0, min(1.0, freeze_prefix_ratio)))))
    sens_norm = sens.clone().float()
    sens_norm = sens_norm / (float(sens_norm.mean().item()) + 1e-6)
    u = float(max(1e-4, keep_unit))
    floor = float(max(0.05, min(0.95, min_keep_floor)))

    def _acc_risk_for_apply(keep_apply: torch.Tensor) -> float:
        return float((sens_norm * (keep_now - keep_apply).abs()).sum().item() / max(1.0e-6, float(usable_budget)))

    alphas = [float(x) for x in list(probe_alphas)[: max(1, int(probe_points))]]
    probes = []
    apply_keeps = []
    apply_risks = []
    for a in alphas:
        delta_mag = float(a) * float(lookahead_budget)
        keep_probe = _apply_sparse_direction(
            keep_now,
            items=direction_items,
            delta_mag=float(delta_mag),
            k_freeze=int(k_freeze),
            u=float(u),
            floor=float(floor),
        )
        scale = 1.0 if delta_mag <= 1e-12 else min(1.0, float(usable_budget) / float(delta_mag))
        keep_apply = _interp_keep_scaled(
            keep_now,
            keep_probe=keep_probe,
            scale=float(scale),
            k_freeze=int(k_freeze),
            u=float(u),
            floor=float(floor),
        )
        probes.append(keep_probe)
        apply_keeps.append(keep_apply)
        apply_risks.append(float(_acc_risk_for_apply(keep_apply)))

    with ThreadPoolExecutor(max_workers=max(1, int(search_threads))) as ex:
        futs = [
            ex.submit(
                _eval_single_alloc_candidate,
                kp,
                model=model,
                last_info=last_info,
                cfg=cfg,
                hw_proxy=hw_proxy,
                wafer_layout=wafer_layout,
                eff_specs=eff_specs_cpu,
                alpha=alpha_cpu,
                fine_split_threads=int(max(1, int(fine_split_threads or 1))),
            )
            for kp in probes
        ]
        outs = [f.result() for f in futs]

    items_full = []
    for idx, (a, out, keep_apply, r) in enumerate(zip(alphas, outs, apply_keeps, apply_risks)):
        if max_acc_risk > 0.0 and math.isfinite(max_acc_risk) and float(r) > float(max_acc_risk):
            continue
        probe_obj = float(out.get("objective", 1.0e18))
        probe_rel_hw_gain = (float(base_obj) - float(probe_obj)) / max(1.0e-6, abs(float(base_obj)))
        cand_sens = _alloc_candidate_sens_score(
            keep_now=keep_now,
            keep_apply=keep_apply,
            sens_norm=sens_norm,
        )
        apply_out = _eval_single_alloc_candidate(
            keep_apply,
            model=model,
            last_info=last_info,
            cfg=cfg,
            hw_proxy=hw_proxy,
            wafer_layout=wafer_layout,
            eff_specs=eff_specs_cpu,
            alpha=alpha_cpu,
            fine_split_threads=int(max(1, int(fine_split_threads or 1))),
        )
        apply_obj = float(apply_out.get("objective", 1.0e18))
        rel_hw_gain_apply = (float(base_obj) - float(apply_obj)) / max(1.0e-6, abs(float(base_obj)))
        item = {
            "direction_items": [(int(i), float(w)) for i, w in direction_items],
            "keep_probe": probes[int(idx)],
            "keep_apply": keep_apply,
            "keep_cand": keep_apply,
            "probe_alpha": float(a),
            "probe_objective": float(probe_obj),
            "probe_rel_hw_gain": float(probe_rel_hw_gain),
            "acc_risk": float(r),
            "cand_sens": float(cand_sens),
            "objective": float(apply_obj),
            "base_objective": float(base_obj),
            "rel_hw_gain": float(rel_hw_gain_apply),
            "total_score": float(rel_hw_gain_apply),
            "probe_raw": out,
            "raw": apply_out,
        }
        items_full.append(item)
    if not items_full:
        return None

    best_local = None
    if str(decision_rule or "legacy_long_gain").lower().strip() == "legacy_long_gain":
        for item in items_full:
            if best_local is None or float(item.get("probe_rel_hw_gain", -1.0e18)) > float(best_local.get("probe_rel_hw_gain", -1.0e18)) + 1e-12:
                best_local = item
            elif best_local is not None and abs(float(item.get("probe_rel_hw_gain", -1.0e18)) - float(best_local.get("probe_rel_hw_gain", -1.0e18))) <= 1e-12:
                if pick_policy not in ("hw_only",) and float(item.get("acc_risk", 1.0e18)) < float(best_local.get("acc_risk", 1.0e18)) - 1e-12:
                    best_local = item
    else:
        best_local = _alloc_select_candidates_tiebreak(
            items_full,
            decision_rule=str(decision_rule),
            tie_apply_abs=float(tie_apply_abs),
            tie_apply_rel=float(tie_apply_rel),
            tie_risk_abs=float(tie_risk_abs),
            tie_risk_rel=float(tie_risk_rel),
            tie_sens_rel=float(tie_sens_rel),
        )
    return dict(best_local) if isinstance(best_local, dict) else None


def _run_alloc_candidate_search_probe(
    *,
    model: torch.nn.Module,
    cfg,
    hw_proxy,
    wafer_layout,
    eff_specs,
    alpha,
    last_info: Optional[Dict[str, Any]],
    keep_now: torch.Tensor,
    sens: torch.Tensor,
    base_obj: float,
    usable_budget: float,
    search_threads: int,
    fine_split_threads: int,
    freeze_prefix_ratio: float,
    keep_unit: float,
    min_keep_floor: float,
    pool_size: int,
    max_candidates: int,
    acc_risk_weight: float,
    pick_policy: str,
    max_acc_risk: float,
    rng_seed: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Directional long-range probing (N points on final range) + stage-limited apply.
    Shared probe logic for OURS and CEM; only direction generation differs.
    """
    eff_specs_cpu = _to_cpu_tensor_dict(eff_specs)
    alpha_cpu = alpha.detach().cpu() if torch.is_tensor(alpha) else alpha

    alloc_cfg = getattr(cfg, "alloc_search", None)
    probe_points = int(getattr(alloc_cfg, "probe_points", 0) or 0) if alloc_cfg is not None else 0
    probe_alphas = getattr(alloc_cfg, "probe_alphas", None) if alloc_cfg is not None else None
    if probe_alphas is None:
        probe_alphas = [0.3333333333, 0.6666666667, 1.0]
    probe_alphas = [float(x) for x in list(probe_alphas)[: max(1, probe_points)]]

    probe_apply_check = bool(getattr(alloc_cfg, "probe_apply_check", True)) if alloc_cfg is not None else True
    candidate_strategy = str(getattr(alloc_cfg, "candidate_strategy", "sens_pool") or "sens_pool") if alloc_cfg is not None else "sens_pool"
    direction_orders = getattr(alloc_cfg, "direction_orders", None) if alloc_cfg is not None else None

    depth = int(keep_now.numel())
    k_freeze = int(round(float(depth) * float(max(0.0, min(1.0, freeze_prefix_ratio)))))
    prunable = list(range(k_freeze, depth))
    if not prunable:
        return None

    keep_mean_prunable = float(keep_now[k_freeze:].mean().item())

    keep_end = 1.0
    try:
        sched = getattr(getattr(cfg, "ast", None), "schedule", None)
        keep_end = float(getattr(sched, "ch_keep_end", 1.0) or 1.0)
    except Exception:
        keep_end = 1.0

    exec_budget = float(max(0.0, usable_budget))
    # final remaining range: keep_mean_prunable - keep_end
    lookahead_budget = float(max(0.0, keep_mean_prunable - float(keep_end)))

    # Allow LOCAL-range probing when probe_use_final_range is False.
    # This keeps probe_points (e.g., 3-point) for budget comparability,
    # but removes the long-horizon "final remaining range" look-ahead.
    alloc_cfg = getattr(cfg, "alloc_search", None)
    probe_use_final = bool(getattr(alloc_cfg, "probe_use_final_range", True)) if alloc_cfg is not None else True
    probe_local_mult = float(getattr(alloc_cfg, "probe_local_scale_mult", 2.0) or 2.0) if alloc_cfg is not None else 2.0
    if not probe_use_final:
        # local probe range is bounded by O(exec_budget)
        lookahead_budget = float(min(lookahead_budget, max(0.0, probe_local_mult * exec_budget)))
    if lookahead_budget <= 1e-8 or exec_budget <= 1e-8:
        return None

    sens_norm = sens.clone().float()
    sens_norm = sens_norm / (float(sens_norm.mean().item()) + 1e-6)

    u = float(max(1e-4, keep_unit))
    floor = float(max(0.05, min(0.95, min_keep_floor)))

    k_total = int(max(1, int(max_candidates)))
    # Use the full probe budget for directional probing. Any optional apply-check
    # is treated as an extra constant-cost evaluation outside the probe budget.
    dirs_total = int(max(1, k_total // int(max(1, probe_points))))
    used_probe = 0

    def _acc_risk_for_apply(keep_apply: torch.Tensor) -> float:
        return float((sens_norm * (keep_now - keep_apply).abs()).sum().item() / max(1.0e-6, float(exec_budget)))

    def _dir_score(obj_probe: float) -> float:
        # Rank feasible candidates by hardware gain only; accuracy risk is a hard
        # feasibility constraint enforced before scoring.
        rel_hw_gain_probe = (float(base_obj) - float(obj_probe)) / max(1.0e-6, abs(float(base_obj)))
        return float(rel_hw_gain_probe)

    def _probe_direction(items: List[Tuple[int, float]]) -> Optional[Dict[str, Any]]:
        dir_rng_seed = int(rng_seed) if rng_seed is not None else 0
        return _eval_alloc_direction_probe(
            direction_items=items,
            keep_now=keep_now,
            base_obj=float(base_obj),
            usable_budget=float(exec_budget),
            lookahead_budget=float(lookahead_budget),
            probe_points=int(probe_points),
            probe_alphas=list(probe_alphas),
            probe_use_final_range=bool(probe_use_final),
            keep_unit=float(keep_unit),
            min_keep_floor=float(min_keep_floor),
            freeze_prefix_ratio=float(freeze_prefix_ratio),
            sens=sens,
            model=model,
            cfg=cfg,
            hw_proxy=hw_proxy,
            wafer_layout=wafer_layout,
            eff_specs_cpu=eff_specs_cpu,
            alpha_cpu=alpha_cpu,
            last_info=last_info,
            fine_split_threads=int(max(1, int(fine_split_threads or 1))),
            search_threads=int(max(1, int(search_threads))),
            max_acc_risk=float(max_acc_risk),
            pick_policy=str(pick_policy),
            rng_seed=int(dir_rng_seed),
        )

    if pick_policy in ("cem", "es"):
        pool = list(prunable)
        p_size = int(len(pool))
        if p_size <= 0:
            return None

        cem_iters = int(getattr(alloc_cfg, "cem_iters", 2) if alloc_cfg is not None else 2)
        cem_iters = max(1, cem_iters)
        elite_frac = float(getattr(alloc_cfg, "cem_elite_frac", 0.25) if alloc_cfg is not None else 0.25)
        elite_frac = float(max(0.05, min(0.8, elite_frac)))
        smooth = float(getattr(alloc_cfg, "cem_smooth", 0.7) if alloc_cfg is not None else 0.7)
        smooth = float(max(0.0, min(0.99, smooth)))

        cem_budget_mult = float(getattr(alloc_cfg, "cem_budget_mult", 1.0) if alloc_cfg is not None else 1.0)
        cem_budget_mult = float(max(0.1, cem_budget_mult))
        total_samples = int(math.ceil(float(max_candidates) * float(cem_budget_mult)))
        total_samples = int(max(1, total_samples))

        # Same budgeting rule for CEM: total_samples counts probe evaluations only.
        dirs_total2 = int(max(1, int(total_samples) // int(max(1, probe_points))))
        per_iter = [dirs_total2 // cem_iters] * cem_iters
        for r in range(dirs_total2 - sum(per_iter)):
            per_iter[r] += 1

        units_dir = int(getattr(alloc_cfg, "cem_dir_units", 16) if alloc_cfg is not None else 16)
        units_dir = int(max(1, units_dir))

        p = torch.ones(p_size, dtype=torch.float32) / float(max(1, p_size))
        best = None
        uniq_prop = 0

        for it in range(cem_iters):
            n_dir = int(per_iter[it])
            if n_dir <= 0:
                continue

            try:
                seed_hex = stable_hash([float(base_obj), float(exec_budget), float(lookahead_budget), float(it), float(total_samples)])
                seed_int = int(seed_hex[:8], 16)
            except Exception:
                seed_int = 0
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed_int))

            counts_list: List[torch.Tensor] = []
            items_list: List[List[Tuple[int, float]]] = []
            seen = set()
            attempts = 0
            max_attempts = int(max(50, n_dir * 20))
            while len(items_list) < n_dir and attempts < max_attempts:
                attempts += 1
                draws = torch.multinomial(p, num_samples=int(units_dir), replacement=True, generator=gen)
                counts = torch.bincount(draws, minlength=p_size).to(dtype=torch.int32)
                key = tuple(counts.tolist())
                if key in seen:
                    continue
                seen.add(key)
                items = []
                for jj, li in enumerate(pool):
                    c = int(counts[jj].item())
                    if c <= 0:
                        continue
                    items.append((int(li), float(c) / float(max(1, units_dir))))
                if not items:
                    continue
                counts_list.append(counts)
                items_list.append(items)
            uniq_prop += len(seen)

            while len(items_list) < n_dir:
                draws = torch.multinomial(p, num_samples=int(units_dir), replacement=True, generator=gen)
                counts = torch.bincount(draws, minlength=p_size).to(dtype=torch.int32)
                items = []
                for jj, li in enumerate(pool):
                    c = int(counts[jj].item())
                    if c <= 0:
                        continue
                    items.append((int(li), float(c) / float(max(1, units_dir))))
                if not items:
                    continue
                counts_list.append(counts)
                items_list.append(items)

            scored = []
            for items, counts in zip(items_list, counts_list):
                res = _probe_direction(items)
                used_probe += int(max(1, probe_points))
                if res is None:
                    continue
                res["counts"] = counts.detach().cpu()
                scored.append(res)
                if best is None or float(res["total_score"]) > float(best["total_score"]) + 1e-12:
                    best = dict(res)

            if scored:
                scored.sort(key=lambda x: float(x["total_score"]), reverse=True)
                n_elite = max(1, int(round(len(scored) * elite_frac)))
                elite_counts = torch.stack([scored[i]["counts"] for i in range(n_elite)], dim=0).float()
                elite_mean = elite_counts.mean(dim=0)
                elite_p = (elite_mean / float(max(1, units_dir))).clamp(min=1e-6)
                elite_p = elite_p / elite_p.sum()
                p = smooth * p + (1.0 - smooth) * elite_p
                p = p.clamp(min=1e-4)
                p = p / p.sum()

        if best is None:
            return None

        keep_apply = best["keep_apply"]
        apply_out = _eval_single_alloc_candidate(
            keep_apply,
            model=model,
            last_info=last_info,
            cfg=cfg,
            hw_proxy=hw_proxy,
            wafer_layout=wafer_layout,
            eff_specs=eff_specs_cpu,
            alpha=alpha_cpu,
            fine_split_threads=int(max(1, int(fine_split_threads or 1))),
        )
        apply_obj = float(apply_out.get("objective", 1.0e18))
        rel_hw_gain_apply = (float(base_obj) - float(apply_obj)) / max(1.0e-6, abs(float(base_obj)))
        total_score_apply = float(rel_hw_gain_apply)

        best.update(
            {
                "keep_cand": keep_apply,
                "objective": float(apply_obj),
                "base_objective": float(base_obj),
                "rel_hw_gain": float(rel_hw_gain_apply),
                "total_score": float(total_score_apply),
                "raw": apply_out,
                "policy": str(pick_policy),
                "eval_budget_total": int(1 + used_probe + (1 if probe_apply_check else 0)),
                "eval_budget_candidates": int(used_probe + (1 if probe_apply_check else 0)),
                "cem_iters": int(cem_iters),
                "cem_total_samples": int(total_samples),
                "cem_pool_size": int(len(pool)),
                "cem_dir_units": int(units_dir),
                "cem_units": int(units_dir),
                "cem_budget_mult": float(cem_budget_mult),
                "cem_unique_proposals": int(uniq_prop),
                "probe_points": int(probe_points),
                "probe_dirs": int(dirs_total2),
            }
        )
        return best

    dirs = _build_alloc_direction_pool(
        keep_now=keep_now,
        sens=sens,
        freeze_prefix_ratio=float(freeze_prefix_ratio),
        pool_size=int(pool_size),
        max_directions=int(dirs_total),
        candidate_strategy=str(candidate_strategy),
        direction_orders=list(direction_orders) if direction_orders is not None else None,
        rng_seed=int(rng_seed) if rng_seed is not None else 0,
    )

    if not dirs:
        return None

    best = None
    best_probe_obj = None

    for items in dirs:
        res = _probe_direction(items)
        used_probe += int(max(1, probe_points))
        if res is None:
            continue
        if best is None:
            best = dict(res)
            best_probe_obj = float(res["probe_objective"])
            continue

        if pick_policy in ("hw", "hw_then_risk", "rel_hw_gain", "hw_only"):
            rel_best = (float(base_obj) - float(best_probe_obj)) / max(1.0e-6, abs(float(base_obj)))
            rel_cur = (float(base_obj) - float(res["probe_objective"])) / max(1.0e-6, abs(float(base_obj)))
            if float(rel_cur) > float(rel_best) + 1e-12:
                best = dict(res)
                best_probe_obj = float(res["probe_objective"])
            elif abs(float(rel_cur) - float(rel_best)) <= 1e-12:
                # Tie-break on risk only when policy explicitly uses risk.
                if pick_policy not in ("hw_only",):
                    if float(res["acc_risk"]) < float(best["acc_risk"]) - 1e-12:
                        best = dict(res)
                        best_probe_obj = float(res["probe_objective"])
        else:
            if float(res["total_score"]) > float(best["total_score"]) + 1e-12:
                best = dict(res)
                best_probe_obj = float(res["probe_objective"])

    if best is None:
        return None

    best.update(
        {
            "policy": "heuristic_probe",
            "eval_budget_total": int(1 + used_probe + (1 if probe_apply_check else 0)),
            "eval_budget_candidates": int(used_probe + (1 if probe_apply_check else 0)),
            "probe_points": int(probe_points),
            "probe_dirs": int(len(dirs)),
        }
    )
    return best


def _ast_warm_eff_from_cfg(cfg) -> int:
    """Effective warmup epochs used by AST schedule (dense warmup length)."""
    ast = getattr(cfg, "ast", None)
    sched = getattr(ast, "schedule", None) if ast is not None else None
    if sched is None:
        return 0
    warm = int(getattr(sched, "warmup_epochs", 0) or 0)
    force_dense = int(getattr(sched, "force_dense_epochs", warm) or warm)
    return int(max(0, max(warm, force_dense)))


def _eval_single_alloc_candidate(
    keep_cand: torch.Tensor,
    *,
    model: torch.nn.Module,
    last_info: Optional[Dict[str, Any]],
    cfg,
    hw_proxy,
    wafer_layout,
    eff_specs,
    alpha,
    fine_split_threads: int = 1,
) -> Dict[str, Any]:
    try:
        depth = int(keep_cand.numel())
        eval_info = _build_eval_model_info_with_ch_override(
            last_info=last_info,
            depth=depth,
            ch_keep_override=[float(x) for x in keep_cand.tolist()],
        )

        mapping_solver_local = MappingSolver(cfg.mapping.strategy, cfg.mapping.mem_limit_factor)
        partitioner_local = PartitionPlanner(mapping_solver_local, wafer_layout, hw_proxy, cfg.partition)

        part_res = partitioner_local.plan(
            model,
            eff_specs,
            alpha=alpha,
            model_info=eval_info,
            use_fine_split=bool(getattr(cfg.hw, "use_fine_split", True)),
            fine_split_threads=int(max(1, int(fine_split_threads or 1))),
        )
        obj = float(part_res.get("objective", 1.0e18))
        mapping_sig = part_res.get("mapping_sig", None)
        return {
            "ok": True,
            "objective": obj,
            "mapping_sig": mapping_sig,
            "part_res": part_res,
        }
    except Exception as exc:
        return {
            "ok": False,
            "objective": 1.0e18,
            "error": str(exc),
        }


def _compute_alloc_phase_state(
    *,
    outer: int,
    ast_sched: Dict[str, Any],
    warmup_epochs: int,
    start_after_prune_epochs: int,
    budget_frac_start: float,
    budget_frac_max: float,
    budget_frac_ramp_epochs: int,
) -> Dict[str, Any]:
    # ast_sched itself does not carry warmup_epochs; pass the effective warmup explicitly.
    warmup_epochs = int(max(0, int(warmup_epochs)))
    prune_epoch = int(outer) - int(warmup_epochs)
    if prune_epoch < int(start_after_prune_epochs):
        return {
            "warmup_epochs": int(warmup_epochs),
            "prune_epoch": int(prune_epoch),
            "alloc_epoch": -1,
            "alloc_enabled_this_outer": False,
            "alloc_phase_progress": 0.0,
            "alloc_phase_budget_frac": 0.0,
            "alloc_budget_frac": 0.0,
        }

    alloc_epoch = int(prune_epoch) - int(start_after_prune_epochs)
    progress = float(alloc_epoch) / float(max(1, int(budget_frac_ramp_epochs) - 1))
    progress = max(0.0, min(1.0, progress))
    alloc_budget_frac = float(budget_frac_start) + progress * (float(budget_frac_max) - float(budget_frac_start))
    alloc_budget_frac = max(float(budget_frac_start), min(float(budget_frac_max), alloc_budget_frac))
    return {
        "warmup_epochs": int(warmup_epochs),
        "prune_epoch": int(prune_epoch),
        "alloc_epoch": int(alloc_epoch),
        "alloc_enabled_this_outer": True,
        "alloc_phase_progress": float(progress),
        "alloc_phase_budget_frac": float(alloc_budget_frac),
        "alloc_budget_frac": float(alloc_budget_frac),
    }


def _run_alloc_candidate_search(
    *,
    model: torch.nn.Module,
    cfg,
    hw_proxy,
    wafer_layout,
    eff_specs,
    alpha,
    last_info: Optional[Dict[str, Any]],
    keep_now: torch.Tensor,
    sens: torch.Tensor,
    usable_budget: float,
    search_threads: int,
    fine_split_threads: int,
    freeze_prefix_ratio: float,
    keep_unit: float,
    min_keep_floor: float,
    pool_size: int,
    max_candidates: int,
    acc_risk_weight: float,
    pick_policy: str = "total_score",
    max_acc_risk: float = float("inf"),
    rng_seed: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    eff_specs_cpu = _to_cpu_tensor_dict(eff_specs)
    alpha_cpu = alpha.detach().cpu() if torch.is_tensor(alpha) else alpha

    base_eval = _eval_single_alloc_candidate(
        keep_now,
        model=model,
        last_info=last_info,
        cfg=cfg,
        hw_proxy=hw_proxy,
        wafer_layout=wafer_layout,
        eff_specs=eff_specs_cpu,
        alpha=alpha_cpu,
        fine_split_threads=int(max(1, int(fine_split_threads or 1))),
    )
    base_obj = float(base_eval.get("objective", 1.0e18))

    sens_norm = sens.clone().float()
    sens_norm = sens_norm / (float(sens_norm.mean().item()) + 1e-6)

    pick_policy = str(pick_policy or "total_score").lower().strip()
    try:
        max_acc_risk = float(max_acc_risk)
    except Exception:
        max_acc_risk = float("inf")

    alloc_cfg = getattr(cfg, "alloc_search", None)
    probe_points = int(getattr(alloc_cfg, "probe_points", 0) or 0) if alloc_cfg is not None else 0
    # Probe search supports both:
    #  - final-range probing (probe_use_final_range=True): lookahead_budget = keep_mean - keep_end
    #  - local-range probing (probe_use_final_range=False): lookahead_budget ~= O(usable_budget)
    if probe_points >= 2:
        return _run_alloc_candidate_search_probe(
            model=model,
            cfg=cfg,
            hw_proxy=hw_proxy,
            wafer_layout=wafer_layout,
            eff_specs=eff_specs,
            alpha=alpha,
            last_info=last_info,
            keep_now=keep_now,
            sens=sens,
            base_obj=float(base_obj),
            usable_budget=float(usable_budget),
            search_threads=int(search_threads),
            fine_split_threads=int(fine_split_threads),
            freeze_prefix_ratio=float(freeze_prefix_ratio),
            keep_unit=float(keep_unit),
            min_keep_floor=float(min_keep_floor),
            pool_size=int(pool_size),
            max_candidates=int(max_candidates),
            acc_risk_weight=float(acc_risk_weight),
            pick_policy=str(pick_policy),
            max_acc_risk=float(max_acc_risk),
            rng_seed=rng_seed,
        )

    # ------------------------------------------------------------
    # CEM/ES baseline (standard, discrete-unit version)
    # - does NOT use sensitivity to pick pool
    # - does NOT use sens-EMA risk proxy
    # - uses discrete unit allocation (Multinomial) so small budgets still change keep
    # - evaluation budget: total_samples = ceil(max_candidates * cem_budget_mult)
    #   (so x1 / x1.5 MUST differ and cannot be clipped by len(cands)/dedup)
    # ------------------------------------------------------------
    if pick_policy in ("cem", "es"):
        depth = int(keep_now.numel())
        k_freeze = int(round(float(depth) * float(max(0.0, min(1.0, freeze_prefix_ratio)))))
        prunable = list(range(k_freeze, depth))
        if not prunable:
            return None

        # Pool: do NOT use sensitivity; use all prunable by default.
        pool_size_eff = int(pool_size)
        if pool_size_eff <= 0 or pool_size_eff >= len(prunable):
            pool = list(prunable)
        else:
            # DP/seed-safe local RNG (does not touch global random state)
            try:
                seed_hex = stable_hash([float(base_obj), float(usable_budget), float(keep_now.float().mean().item())])
                seed_int = int(seed_hex[:8], 16)
            except Exception:
                seed_int = 0
            rng = random.Random(int(seed_int))
            pool = rng.sample(prunable, pool_size_eff)
            pool = sorted(pool)
        if not pool:
            return None

        budget = float(max(0.0, usable_budget))
        if budget <= 1e-8:
            return None

        u = float(max(1e-4, keep_unit))
        floor = float(max(0.05, min(0.95, min_keep_floor)))

        cem_cfg = getattr(cfg, "alloc_search", None)
        cem_iters = int(getattr(cem_cfg, "cem_iters", 2) if cem_cfg is not None else 2)
        cem_iters = max(1, cem_iters)
        elite_frac = float(getattr(cem_cfg, "cem_elite_frac", 0.25) if cem_cfg is not None else 0.25)
        elite_frac = float(max(0.05, min(0.8, elite_frac)))
        smooth = float(getattr(cem_cfg, "cem_smooth", 0.7) if cem_cfg is not None else 0.7)
        smooth = float(max(0.0, min(0.99, smooth)))
        cem_budget_mult = float(getattr(cem_cfg, "cem_budget_mult", 1.0) if cem_cfg is not None else 1.0)
        cem_budget_mult = float(max(0.1, cem_budget_mult))
        total_samples = int(math.ceil(float(max_candidates) * float(cem_budget_mult)))
        total_samples = int(max(1, total_samples))

        # optional "task" proxy: penalize concentrated pruning (no sensitivity)
        cem_task_beta = float(getattr(cem_cfg, "cem_task_beta", 0.0) if cem_cfg is not None else 0.0)
        cem_task_beta = float(max(0.0, cem_task_beta))

        per_iter = [total_samples // cem_iters] * cem_iters
        for r in range(total_samples - sum(per_iter)):
            per_iter[r] += 1

        P = int(len(pool))
        # CEM maintains a categorical prob vector p over pool dims
        p = torch.ones(P, dtype=torch.float32) / float(max(1, P))

        best = None
        used_evals = 0
        uniq_prop = 0
        units = int(round(float(budget) / float(u)))
        units = int(max(1, units))

        def _make_cand_from_counts(counts_i: torch.Tensor) -> torch.Tensor:
            cand = keep_now.clone()
            for jj, li in enumerate(pool):
                du = int(counts_i[jj].item())
                if du <= 0:
                    continue
                d = float(du) * float(u)
                maxd = float(max(0.0, float(cand[li].item()) - floor))
                if d > maxd:
                    d = maxd
                if d > 0.0:
                    cand[li] = cand[li] - d
            q = cand.clone()
            for ii in range(k_freeze, depth):
                q[ii] = round(float(q[ii].item()) / u) * u
                q[ii] = float(max(floor, min(1.0, q[ii].item())))
            return q

        for it in range(cem_iters):
            n_samp = int(per_iter[it])
            if n_samp <= 0:
                continue
            # local torch generator (does not touch global RNG state)
            try:
                seed_hex = stable_hash([float(base_obj), float(budget), float(it), float(total_samples)])
                seed_int = int(seed_hex[:8], 16)
            except Exception:
                seed_int = 0
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(seed_int))

            # propose unique count-vectors via multinomial unit sampling
            counts_list: List[torch.Tensor] = []
            cand_list: List[torch.Tensor] = []
            seen = set()
            attempts = 0
            max_attempts = int(max(20, n_samp * 10))
            while len(cand_list) < n_samp and attempts < max_attempts:
                attempts += 1
                draws = torch.multinomial(p, num_samples=int(units), replacement=True, generator=gen)
                counts = torch.bincount(draws, minlength=P).to(dtype=torch.int32)
                key = tuple(counts.tolist())
                if key in seen:
                    continue
                seen.add(key)
                cand = _make_cand_from_counts(counts)
                counts_list.append(counts)
                cand_list.append(cand)
            uniq_prop += len(seen)

            # If not enough unique proposals, fill remaining with duplicates (still consumes budget).
            while len(cand_list) < n_samp:
                draws = torch.multinomial(p, num_samples=int(units), replacement=True, generator=gen)
                counts = torch.bincount(draws, minlength=P).to(dtype=torch.int32)
                cand = _make_cand_from_counts(counts)
                counts_list.append(counts)
                cand_list.append(cand)

            with ThreadPoolExecutor(max_workers=max(1, int(search_threads))) as ex:
                futs = [
                    ex.submit(
                        _eval_single_alloc_candidate,
                        cand,
                        model=model,
                        last_info=last_info,
                        cfg=cfg,
                        hw_proxy=hw_proxy,
                        wafer_layout=wafer_layout,
                        eff_specs=eff_specs_cpu,
                        alpha=alpha_cpu,
                        fine_split_threads=int(max(1, int(fine_split_threads or 1))),
                    )
                    for cand in cand_list
                ]

                used_evals += len(futs)

                scored = []
                for cand, counts, fut in zip(cand_list, counts_list, futs):
                    out = fut.result()
                    obj = float(out.get("objective", 1.0e18))
                    rel_hw_gain = (float(base_obj) - float(obj)) / max(1.0e-6, abs(float(base_obj)))
                    # generic "task proxy" (no sensitivity): penalize concentrated pruning via L2(delta)
                    delta = (keep_now - cand).clamp(min=0.0)
                    delta_pool = delta[pool].float()
                    task_proxy = float(torch.sqrt((delta_pool * delta_pool).sum()).item() / max(1.0e-6, float(budget)))
                    if math.isfinite(max_acc_risk) and float(task_proxy) > float(max_acc_risk):
                        continue
                    total_score = float(rel_hw_gain) - float(cem_task_beta) * float(task_proxy)
                    item = {
                        "keep_cand": cand,
                        "objective": obj,
                        "base_objective": float(base_obj),
                        "rel_hw_gain": float(rel_hw_gain),
                        "acc_risk": float(task_proxy),
                        "total_score": float(total_score),
                        "raw": out,
                        "counts": counts.detach().cpu(),
                    }
                    scored.append(item)
                    if best is None or float(item["total_score"]) > float(best["total_score"]) + 1e-12:
                        best = item

            # CEM update: p <- smooth*p + (1-smooth)*mean(elite_counts)/units
            if scored:
                scored.sort(key=lambda x: float(x["total_score"]), reverse=True)
                n_elite = max(1, int(round(len(scored) * elite_frac)))
                elite_counts = torch.stack([scored[i]["counts"] for i in range(n_elite)], dim=0).float()
                elite_mean = elite_counts.mean(dim=0)
                elite_p = (elite_mean / max(1.0, float(units))).clamp(min=1e-6)
                elite_p = elite_p / elite_p.sum()
                p = smooth * p + (1.0 - smooth) * elite_p
                p = p.clamp(min=1e-4)
                p = p / p.sum()

        if best is None:
            return None

        best["eval_budget_total"] = int(1 + used_evals)
        best["eval_budget_candidates"] = int(used_evals)
        best["policy"] = str(pick_policy)
        best["cem_iters"] = int(cem_iters)
        best["cem_total_samples"] = int(total_samples)
        best["cem_pool_size"] = int(len(pool))
        best["cem_units"] = int(units)
        best["cem_budget_mult"] = float(cem_budget_mult)
        best["cem_unique_proposals"] = int(uniq_prop)
        best["cem_task_beta"] = float(cem_task_beta)
        return best

    # --------- default heuristic candidates (ours / hw_then_risk / total_score / risk_only) ----------
    cands = _build_alloc_candidates_from_remaining_budget(
        keep_now=keep_now,
        sens=sens,
        freeze_prefix_ratio=freeze_prefix_ratio,
        remain_budget=usable_budget,
        keep_unit=keep_unit,
        min_keep_floor=min_keep_floor,
        pool_size=pool_size,
        max_candidates=max_candidates,
    )
    if not cands:
        return None

    if pick_policy in ("risk_only", "sens_only", "acc_risk_only"):
        best_cand = None
        best_risk = None
        for cand in cands:
            acc_risk = float((sens_norm * (keep_now - cand).abs()).sum().item() / max(1.0e-6, usable_budget))
            if math.isfinite(max_acc_risk) and acc_risk > float(max_acc_risk):
                continue
            if best_risk is None or acc_risk < best_risk:
                best_risk = float(acc_risk)
                best_cand = cand

        if best_cand is None:
            return None

        out = _eval_single_alloc_candidate(
            best_cand,
            model=model,
            last_info=last_info,
            cfg=cfg,
            hw_proxy=hw_proxy,
            wafer_layout=wafer_layout,
            eff_specs=eff_specs_cpu,
            alpha=alpha_cpu,
            fine_split_threads=int(max(1, int(fine_split_threads or 1))),
        )
        obj = float(out.get("objective", 1.0e18))
        rel_hw_gain = (float(base_obj) - float(obj)) / max(1.0e-6, abs(float(base_obj)))
        total_score = float(rel_hw_gain) - float(acc_risk_weight) * float(best_risk if best_risk is not None else 0.0)
        return {
            "keep_cand": best_cand,
            "objective": obj,
            "base_objective": float(base_obj),
            "rel_hw_gain": float(rel_hw_gain),
            "acc_risk": float(best_risk if best_risk is not None else 0.0),
            "total_score": float(total_score),
            "raw": out,
        }

    best = None
    with ThreadPoolExecutor(max_workers=max(1, int(search_threads))) as ex:
        futs = [
            ex.submit(
                _eval_single_alloc_candidate,
                cand,
                model=model,
                last_info=last_info,
                cfg=cfg,
                hw_proxy=hw_proxy,
                wafer_layout=wafer_layout,
                eff_specs=eff_specs_cpu,
                alpha=alpha_cpu,
                fine_split_threads=int(max(1, int(fine_split_threads or 1))),
            )
            for cand in cands
        ]

        for cand, fut in zip(cands, futs):
            out = fut.result()
            obj = float(out.get("objective", 1.0e18))
            rel_hw_gain = (float(base_obj) - float(obj)) / max(1.0e-6, abs(float(base_obj)))
            acc_risk = float((sens_norm * (keep_now - cand).abs()).sum().item() / max(1.0e-6, usable_budget))
            total_score = float(rel_hw_gain) - float(acc_risk_weight) * float(acc_risk)

            if max_acc_risk > 0.0 and math.isfinite(max_acc_risk):
                if float(acc_risk) > float(max_acc_risk):
                    continue

            item = {
                "keep_cand": cand,
                "objective": obj,
                "base_objective": base_obj,
                "rel_hw_gain": rel_hw_gain,
                "acc_risk": acc_risk,
                "total_score": total_score,
                "raw": out,
            }
            if best is None:
                best = item
                continue

            if pick_policy in ("hw", "hw_then_risk", "rel_hw_gain", "hw_only"):
                if float(item["rel_hw_gain"]) > float(best["rel_hw_gain"]) + 1e-12:
                    best = item
                elif abs(float(item["rel_hw_gain"]) - float(best["rel_hw_gain"])) <= 1e-12:
                    # Tie-break on risk only when policy explicitly uses risk.
                    if pick_policy not in ("hw_only",):
                        if float(item["acc_risk"]) < float(best["acc_risk"]) - 1e-12:
                            best = item
            else:
                if float(item["total_score"]) > float(best["total_score"]) + 1e-12:
                    best = item
    return best


def _maybe_run_alloc_search_and_apply(
    *,
    model: torch.nn.Module,
    cfg,
    run_state: Dict[str, Any],
    outer: int,
    seed: int,
    ast_sched: Dict[str, Any],
    hw_proxy,
    wafer_layout,
    chiplet_slots,
    logger,
) -> None:
    alloc_cfg = getattr(cfg, "alloc_search", None)
    if alloc_cfg is None or not bool(getattr(alloc_cfg, "enabled", False)):
        return

    if bool(run_state.get("user_recovery_active", False)):
        run_state["alloc_last_search"] = {
            "outer": int(outer),
            "enabled": False,
            "reason": "user_recovery",
            "alloc_enabled_this_outer": False,
        }
        return

    start_after_prune_epochs = int(getattr(alloc_cfg, "start_after_prune_epochs", 5))
    budget_frac_start = float(getattr(alloc_cfg, "budget_frac_start", 0.5))
    budget_frac_max = float(getattr(alloc_cfg, "budget_frac_max", 0.8))
    budget_frac_ramp_epochs = int(getattr(alloc_cfg, "budget_frac_ramp_epochs", 10))

    last_info = run_state.get("last_model_info", None)
    sens = run_state.get("alloc_layer_sens_ema", None)
    if last_info is None or sens is None:
        return

    pruner = _get_ast_pruner(model)
    if pruner is None:
        return

    keep_now = _get_layer_ch_keep_now(model)
    if keep_now is None:
        return

    try:
        pr_cfg = pruner.cfg.get("channel_prune", pruner.cfg)
        freeze_prefix_ratio = float(pr_cfg.get("freeze_prefix_ratio", pruner.cfg.get("ch_freeze_prefix_ratio", 0.0)) or 0.0)
        freeze_prefix_ratio = float(max(0.0, min(1.0, freeze_prefix_ratio)))
    except Exception:
        freeze_prefix_ratio = 0.0

    depth = int(keep_now.numel())
    k_freeze = int(round(float(depth) * float(freeze_prefix_ratio)))
    if k_freeze >= depth:
        return

    keep_mean_prunable = float(keep_now[k_freeze:].mean().item())
    target_global_keep = float(ast_sched.get("ch_keep_target", keep_mean_prunable))
    remain_budget = max(0.0, keep_mean_prunable - target_global_keep)
    warm_eff = _ast_warm_eff_from_cfg(cfg)
    phase_state = _compute_alloc_phase_state(
        outer=int(outer),
        ast_sched=ast_sched,
        warmup_epochs=int(warm_eff),
        start_after_prune_epochs=int(start_after_prune_epochs),
        budget_frac_start=float(budget_frac_start),
        budget_frac_max=float(budget_frac_max),
        budget_frac_ramp_epochs=int(budget_frac_ramp_epochs),
    )
    if not bool(phase_state.get("alloc_enabled_this_outer", False)):
        run_state["alloc_last_search"] = {
            "outer": int(outer),
            "enabled": False,
            "reason": "before_alloc_phase",
            "alloc_enabled_this_outer": False,
            "alloc_start_after_prune_epochs": int(start_after_prune_epochs),
            "alloc_phase_progress": float(phase_state.get("alloc_phase_progress", 0.0)),
            "alloc_phase_budget_frac": float(phase_state.get("alloc_phase_budget_frac", 0.0)),
            "alloc_budget_frac": 0.0,
            "alloc_applied": False,
            "alloc_remain_budget": float(remain_budget),
            "alloc_usable_budget": 0.0,
            "warmup_epochs": int(phase_state.get("warmup_epochs", warm_eff)),
            "prune_epoch": int(phase_state.get("prune_epoch", -1)),
            "alloc_epoch": int(phase_state.get("alloc_epoch", -1)),
            "target_global_keep": float(target_global_keep),
            "keep_mean_prunable": float(keep_mean_prunable),
        }
        return

    alloc_phase_budget_frac = float(phase_state.get("alloc_phase_budget_frac", phase_state.get("alloc_budget_frac", 0.0)))
    usable_budget = float(alloc_phase_budget_frac) * float(remain_budget)
    usable_budget = max(0.0, min(float(remain_budget), float(usable_budget)))

    min_remain_budget = float(getattr(alloc_cfg, "min_remain_budget", 0.005))
    if usable_budget <= min_remain_budget:
        run_state["alloc_last_search"] = {
            "outer": int(outer),
            "enabled": True,
            "reason": "usable_budget_too_small",
            "alloc_enabled_this_outer": True,
            "alloc_start_after_prune_epochs": int(start_after_prune_epochs),
            "alloc_phase_progress": float(phase_state.get("alloc_phase_progress", 0.0)),
            "alloc_phase_budget_frac": float(alloc_phase_budget_frac),
            "alloc_budget_frac": 0.0,
            "alloc_applied": False,
            "alloc_remain_budget": float(remain_budget),
            "alloc_usable_budget": float(usable_budget),
            "warmup_epochs": int(phase_state.get("warmup_epochs", warm_eff)),
            "prune_epoch": int(phase_state.get("prune_epoch", -1)),
            "alloc_epoch": int(phase_state.get("alloc_epoch", -1)),
            "target_global_keep": float(target_global_keep),
            "keep_mean_prunable": float(keep_mean_prunable),
        }
        return

    slot_out = chiplet_slots(hard=False)
    eff_specs = slot_out["eff_specs"]
    alpha = slot_out["alpha"]
    outer_threads = int(
        os.environ.get(
            "ALLOC_SEARCH_OUTER_THREADS",
            str(getattr(alloc_cfg, "search_threads_outer", 6) or 6),
        )
    )
    inner_threads = int(
        os.environ.get(
            "ALLOC_SEARCH_INNER_THREADS",
            str(getattr(alloc_cfg, "search_threads_inner", 5) or 5),
        )
    )

    max_acc_risk = float(getattr(alloc_cfg, "max_acc_risk", float("inf")) or float("inf"))
    pick_policy = str(getattr(alloc_cfg, "pick_policy", "hw_then_risk") or "hw_then_risk")
    controller = str(getattr(alloc_cfg, "controller", "legacy") or "legacy").lower().strip()
    candidate_strategy = str(getattr(alloc_cfg, "candidate_strategy", "sens_pool") or "sens_pool").lower().strip()
    if controller == "ds_fixed":
        candidate_strategy = "uniform_all"
    if controller == "new_ours" and candidate_strategy not in ("sens_pool", "uniform_all"):
        candidate_strategy = "sens_pool"
    if controller == "new_ours_tiebreak" and candidate_strategy not in ("sens_pool", "uniform_all"):
        candidate_strategy = "uniform_all"
    tie_mode = str(getattr(alloc_cfg, "tiebreak_mode", "legacy_long_gain") or "legacy_long_gain").lower().strip()
    tie_apply_abs = float(getattr(alloc_cfg, "tie_apply_abs", 0.0) or 0.0)
    tie_apply_rel = float(getattr(alloc_cfg, "tie_apply_rel", 0.0) or 0.0)
    tie_risk_abs = float(getattr(alloc_cfg, "tie_risk_abs", 0.0) or 0.0)
    tie_risk_rel = float(getattr(alloc_cfg, "tie_risk_rel", 0.0) or 0.0)
    tie_sens_rel = float(getattr(alloc_cfg, "tie_sens_rel", 0.0) or 0.0)
    direction_orders = getattr(alloc_cfg, "direction_orders", None)

    common_meta = {
        "alloc_controller": str(controller),
        "alloc_selected_source": "",
        "alloc_selected_long_gain": 0.0,
        "alloc_selected_apply_gain": 0.0,
        "alloc_inc_long_gain": 0.0,
        "alloc_best_ch_long_gain": 0.0,
        "alloc_switched": False,
        "alloc_inc_age": 0,
        "alloc_candidate_strategy": str(candidate_strategy),
        "alloc_decision_basis": "",
    }

    best = None
    if controller == "legacy":
        alloc_rng_seed = int(outer) + 100003 * int(seed)
        best = _run_alloc_candidate_search(
            model=model,
            cfg=cfg,
            hw_proxy=hw_proxy,
            wafer_layout=wafer_layout,
            eff_specs=eff_specs,
            alpha=alpha,
            last_info=last_info,
            keep_now=keep_now,
            sens=sens,
            usable_budget=usable_budget,
            search_threads=max(1, int(outer_threads)),
            fine_split_threads=max(1, int(inner_threads)),
            freeze_prefix_ratio=freeze_prefix_ratio,
            keep_unit=float(getattr(alloc_cfg, "keep_unit", 0.01)),
            min_keep_floor=float(getattr(alloc_cfg, "min_keep_floor", 0.25)),
            pool_size=int(getattr(alloc_cfg, "pool_size", 6)),
            max_candidates=int(getattr(alloc_cfg, "max_candidates", 60)),
            acc_risk_weight=float(getattr(alloc_cfg, "acc_risk_weight", 0.2)),
            max_acc_risk=float(max_acc_risk),
            pick_policy=str(pick_policy),
            rng_seed=int(alloc_rng_seed),
        )
    else:
        eff_specs_cpu = _to_cpu_tensor_dict(eff_specs)
        alpha_cpu = alpha.detach().cpu() if torch.is_tensor(alpha) else alpha
        base_eval = _eval_single_alloc_candidate(
            keep_now,
            model=model,
            last_info=last_info,
            cfg=cfg,
            hw_proxy=hw_proxy,
            wafer_layout=wafer_layout,
            eff_specs=eff_specs_cpu,
            alpha=alpha_cpu,
            fine_split_threads=max(1, int(inner_threads)),
        )
        base_obj = float(base_eval.get("objective", 1.0e18))
        alloc_probe_points = int(getattr(alloc_cfg, "probe_points", 1) or 1)
        alloc_probe_alphas = [float(x) for x in list(getattr(alloc_cfg, "probe_alphas", [1.0]) or [1.0])[: max(1, alloc_probe_points)]]
        alloc_probe_use_final = bool(getattr(alloc_cfg, "probe_use_final_range", True))
        alloc_probe_local_mult = float(getattr(alloc_cfg, "probe_local_scale_mult", 2.0) or 2.0)
        keep_end = 1.0
        try:
            sched = getattr(getattr(cfg, "ast", None), "schedule", None)
            keep_end = float(getattr(sched, "ch_keep_end", 1.0) or 1.0)
        except Exception:
            keep_end = 1.0
        lookahead_budget = float(max(0.0, keep_mean_prunable - float(keep_end)))
        if not alloc_probe_use_final:
            lookahead_budget = float(min(lookahead_budget, max(0.0, alloc_probe_local_mult * float(usable_budget))))
        if controller == "ds_fixed":
            alloc_probe_points = 1
            alloc_probe_alphas = [1.0]
            alloc_probe_use_final = False
            lookahead_budget = float(max(0.0, usable_budget))

        dirs_total = int(max(1, int(getattr(alloc_cfg, "max_candidates", 60)) // int(max(1, alloc_probe_points))))
        alloc_rng_seed = int(outer) + 100003 * int(seed)
        dirs = _build_alloc_direction_pool(
            keep_now=keep_now,
            sens=sens,
            freeze_prefix_ratio=float(freeze_prefix_ratio),
            pool_size=int(getattr(alloc_cfg, "pool_size", 6)),
            max_directions=int(dirs_total),
            candidate_strategy=str(candidate_strategy),
            direction_orders=list(direction_orders) if direction_orders is not None else None,
            rng_seed=int(alloc_rng_seed),
        )

        if lookahead_budget > 1e-8 and usable_budget > 1e-8 and dirs:
            if controller == "new_ours":
                inc_eval = None
                incumbent = run_state.get("alloc_incumbent_direction", None)
                keep_incumbent = bool(getattr(alloc_cfg, "keep_incumbent", True))
                if keep_incumbent and isinstance(incumbent, dict) and incumbent.get("direction_items", None):
                    inc_eval = _eval_alloc_direction_probe(
                        direction_items=incumbent["direction_items"],
                        keep_now=keep_now,
                        base_obj=float(base_obj),
                        usable_budget=float(usable_budget),
                        lookahead_budget=float(lookahead_budget),
                        probe_points=int(alloc_probe_points),
                        probe_alphas=list(alloc_probe_alphas),
                        probe_use_final_range=bool(alloc_probe_use_final),
                        keep_unit=float(getattr(alloc_cfg, "keep_unit", 0.01)),
                        min_keep_floor=float(getattr(alloc_cfg, "min_keep_floor", 0.25)),
                        freeze_prefix_ratio=float(freeze_prefix_ratio),
                        sens=sens,
                        model=model,
                        cfg=cfg,
                        hw_proxy=hw_proxy,
                        wafer_layout=wafer_layout,
                        eff_specs_cpu=eff_specs_cpu,
                        alpha_cpu=alpha_cpu,
                        last_info=last_info,
                        fine_split_threads=max(1, int(inner_threads)),
                        search_threads=max(1, int(outer_threads)),
                        max_acc_risk=float(max_acc_risk),
                        pick_policy=str(pick_policy),
                        decision_rule="legacy_long_gain",
                    )
                best_ch = None
                for items in dirs:
                    res = _eval_alloc_direction_probe(
                        direction_items=items,
                        keep_now=keep_now,
                        base_obj=float(base_obj),
                        usable_budget=float(usable_budget),
                        lookahead_budget=float(lookahead_budget),
                        probe_points=int(alloc_probe_points),
                        probe_alphas=list(alloc_probe_alphas),
                        probe_use_final_range=bool(alloc_probe_use_final),
                        keep_unit=float(getattr(alloc_cfg, "keep_unit", 0.01)),
                        min_keep_floor=float(getattr(alloc_cfg, "min_keep_floor", 0.25)),
                        freeze_prefix_ratio=float(freeze_prefix_ratio),
                        sens=sens,
                        model=model,
                        cfg=cfg,
                        hw_proxy=hw_proxy,
                        wafer_layout=wafer_layout,
                        eff_specs_cpu=eff_specs_cpu,
                        alpha_cpu=alpha_cpu,
                        last_info=last_info,
                        fine_split_threads=max(1, int(inner_threads)),
                        search_threads=max(1, int(outer_threads)),
                        max_acc_risk=float(max_acc_risk),
                        pick_policy=str(pick_policy),
                        decision_rule="legacy_long_gain",
                    )
                    if res is None:
                        continue
                    if best_ch is None or float(res.get("probe_rel_hw_gain", -1.0e18)) > float(best_ch.get("probe_rel_hw_gain", -1.0e18)) + 1e-12:
                        best_ch = res

                inc_gain = float(inc_eval.get("probe_rel_hw_gain", 0.0)) if isinstance(inc_eval, dict) else 0.0
                ch_gain = float(best_ch.get("probe_rel_hw_gain", 0.0)) if isinstance(best_ch, dict) else 0.0
                common_meta["alloc_inc_long_gain"] = float(inc_gain)
                common_meta["alloc_best_ch_long_gain"] = float(ch_gain)
                common_meta["alloc_decision_basis"] = "long_gain_only"

                selected = None
                selected_source = "none"
                if max(inc_gain, ch_gain) > 0.0:
                    if float(ch_gain) > float(inc_gain):
                        selected = best_ch
                        selected_source = "challenger"
                    else:
                        selected = inc_eval
                        selected_source = "incumbent"

                if selected is not None:
                    best = dict(selected)
                    common_meta["alloc_selected_source"] = str(selected_source)
                    common_meta["alloc_selected_long_gain"] = float(selected.get("probe_rel_hw_gain", 0.0) or 0.0)
                    common_meta["alloc_selected_apply_gain"] = float(selected.get("rel_hw_gain", 0.0) or 0.0)
                    if selected_source == "challenger":
                        if keep_incumbent:
                            run_state["alloc_incumbent_direction"] = {
                                "direction_items": [(int(i), float(w)) for i, w in selected.get("direction_items", [])],
                                "age": 1,
                                "last_source": "challenger",
                            }
                        else:
                            run_state["alloc_incumbent_direction"] = None
                        common_meta["alloc_switched"] = True
                    else:
                        prev_age = 0
                        if isinstance(incumbent, dict):
                            prev_age = int(incumbent.get("age", 0) or 0)
                        if keep_incumbent:
                            run_state["alloc_incumbent_direction"] = {
                                "direction_items": [(int(i), float(w)) for i, w in selected.get("direction_items", [])],
                                "age": int(prev_age) + 1,
                                "last_source": "incumbent",
                            }
                        else:
                            run_state["alloc_incumbent_direction"] = None
                    common_meta["alloc_inc_age"] = int((run_state.get("alloc_incumbent_direction") or {}).get("age", 0))
                else:
                    run_state["alloc_incumbent_direction"] = None
            elif controller == "new_ours_tiebreak":
                incumbent = run_state.get("alloc_incumbent_direction", None)
                keep_incumbent = bool(getattr(alloc_cfg, "keep_incumbent", True))
                all_candidates = []
                inc_eval = None
                if keep_incumbent and isinstance(incumbent, dict) and incumbent.get("direction_items", None):
                    inc_eval = _eval_alloc_direction_probe(
                        direction_items=incumbent["direction_items"],
                        keep_now=keep_now,
                        base_obj=float(base_obj),
                        usable_budget=float(usable_budget),
                        lookahead_budget=float(lookahead_budget),
                        probe_points=int(alloc_probe_points),
                        probe_alphas=list(alloc_probe_alphas),
                        probe_use_final_range=bool(alloc_probe_use_final),
                        keep_unit=float(getattr(alloc_cfg, "keep_unit", 0.01)),
                        min_keep_floor=float(getattr(alloc_cfg, "min_keep_floor", 0.25)),
                        freeze_prefix_ratio=float(freeze_prefix_ratio),
                        sens=sens,
                        model=model,
                        cfg=cfg,
                        hw_proxy=hw_proxy,
                        wafer_layout=wafer_layout,
                        eff_specs_cpu=eff_specs_cpu,
                        alpha_cpu=alpha_cpu,
                        last_info=last_info,
                        fine_split_threads=max(1, int(inner_threads)),
                        search_threads=max(1, int(outer_threads)),
                        max_acc_risk=float(max_acc_risk),
                        pick_policy=str(pick_policy),
                        decision_rule=str(tie_mode),
                        tie_apply_abs=float(tie_apply_abs),
                        tie_apply_rel=float(tie_apply_rel),
                        tie_risk_abs=float(tie_risk_abs),
                        tie_risk_rel=float(tie_risk_rel),
                        tie_sens_rel=float(tie_sens_rel),
                    )
                    if isinstance(inc_eval, dict):
                        inc_item = dict(inc_eval)
                        inc_item["_candidate_source"] = "incumbent"
                        all_candidates.append(inc_item)
                challenger_candidates = []
                for items in dirs:
                    res = _eval_alloc_direction_probe(
                        direction_items=items,
                        keep_now=keep_now,
                        base_obj=float(base_obj),
                        usable_budget=float(usable_budget),
                        lookahead_budget=float(lookahead_budget),
                        probe_points=int(alloc_probe_points),
                        probe_alphas=list(alloc_probe_alphas),
                        probe_use_final_range=bool(alloc_probe_use_final),
                        keep_unit=float(getattr(alloc_cfg, "keep_unit", 0.01)),
                        min_keep_floor=float(getattr(alloc_cfg, "min_keep_floor", 0.25)),
                        freeze_prefix_ratio=float(freeze_prefix_ratio),
                        sens=sens,
                        model=model,
                        cfg=cfg,
                        hw_proxy=hw_proxy,
                        wafer_layout=wafer_layout,
                        eff_specs_cpu=eff_specs_cpu,
                        alpha_cpu=alpha_cpu,
                        last_info=last_info,
                        fine_split_threads=max(1, int(inner_threads)),
                        search_threads=max(1, int(outer_threads)),
                        max_acc_risk=float(max_acc_risk),
                        pick_policy=str(pick_policy),
                        decision_rule=str(tie_mode),
                        tie_apply_abs=float(tie_apply_abs),
                        tie_apply_rel=float(tie_apply_rel),
                        tie_risk_abs=float(tie_risk_abs),
                        tie_risk_rel=float(tie_risk_rel),
                        tie_sens_rel=float(tie_sens_rel),
                    )
                    if not isinstance(res, dict):
                        continue
                    ch_item = dict(res)
                    ch_item["_candidate_source"] = "challenger"
                    challenger_candidates.append(ch_item)
                    all_candidates.append(ch_item)

                common_meta["alloc_inc_long_gain"] = float(inc_eval.get("probe_rel_hw_gain", 0.0)) if isinstance(inc_eval, dict) else 0.0
                common_meta["alloc_best_ch_long_gain"] = max(float(x.get("probe_rel_hw_gain", 0.0)) for x in challenger_candidates) if challenger_candidates else 0.0
                common_meta["alloc_decision_basis"] = str(tie_mode)

                selected = _alloc_select_candidates_tiebreak(
                    all_candidates,
                    decision_rule=str(tie_mode),
                    tie_apply_abs=float(tie_apply_abs),
                    tie_apply_rel=float(tie_apply_rel),
                    tie_risk_abs=float(tie_risk_abs),
                    tie_risk_rel=float(tie_risk_rel),
                    tie_sens_rel=float(tie_sens_rel),
                )
                selected_source = str(selected.get("_candidate_source", "none")) if isinstance(selected, dict) else "none"
                if not isinstance(selected, dict):
                    run_state["alloc_incumbent_direction"] = None
                elif float(selected.get("rel_hw_gain", 0.0) or 0.0) <= 0.0:
                    run_state["alloc_incumbent_direction"] = None
                else:
                    best = dict(selected)
                    if selected_source == "challenger":
                        if keep_incumbent:
                            run_state["alloc_incumbent_direction"] = {
                                "direction_items": [(int(i), float(w)) for i, w in selected.get("direction_items", [])],
                                "age": 1,
                                "last_source": "challenger",
                            }
                        else:
                            run_state["alloc_incumbent_direction"] = None
                        common_meta["alloc_switched"] = True
                    elif selected_source == "incumbent":
                        prev_age = 0
                        if isinstance(incumbent, dict):
                            prev_age = int(incumbent.get("age", 0) or 0)
                        if keep_incumbent:
                            run_state["alloc_incumbent_direction"] = {
                                "direction_items": [(int(i), float(w)) for i, w in selected.get("direction_items", [])],
                                "age": int(prev_age) + 1,
                                "last_source": "incumbent",
                            }
                        else:
                            run_state["alloc_incumbent_direction"] = None
                    else:
                        run_state["alloc_incumbent_direction"] = None
                    common_meta["alloc_selected_source"] = str(selected_source)
                    common_meta["alloc_selected_long_gain"] = float(selected.get("probe_rel_hw_gain", 0.0) or 0.0)
                    common_meta["alloc_selected_apply_gain"] = float(selected.get("rel_hw_gain", 0.0) or 0.0)
                common_meta["alloc_inc_age"] = int((run_state.get("alloc_incumbent_direction") or {}).get("age", 0))
            elif controller == "ds_fixed":
                best_ch = None
                for items in dirs:
                    res = _eval_alloc_direction_probe(
                        direction_items=items,
                        keep_now=keep_now,
                        base_obj=float(base_obj),
                        usable_budget=float(usable_budget),
                        lookahead_budget=float(lookahead_budget),
                        probe_points=1,
                        probe_alphas=[1.0],
                        probe_use_final_range=False,
                        keep_unit=float(getattr(alloc_cfg, "keep_unit", 0.01)),
                        min_keep_floor=float(getattr(alloc_cfg, "min_keep_floor", 0.25)),
                        freeze_prefix_ratio=float(freeze_prefix_ratio),
                        sens=sens,
                        model=model,
                        cfg=cfg,
                        hw_proxy=hw_proxy,
                        wafer_layout=wafer_layout,
                        eff_specs_cpu=eff_specs_cpu,
                        alpha_cpu=alpha_cpu,
                        last_info=last_info,
                        fine_split_threads=max(1, int(inner_threads)),
                        search_threads=max(1, int(outer_threads)),
                        max_acc_risk=float(max_acc_risk),
                        pick_policy=str(pick_policy),
                        decision_rule="legacy_long_gain",
                    )
                    if res is None:
                        continue
                    if best_ch is None or float(res.get("probe_rel_hw_gain", -1.0e18)) > float(best_ch.get("probe_rel_hw_gain", -1.0e18)) + 1e-12:
                        best_ch = res
                if isinstance(best_ch, dict) and float(best_ch.get("probe_rel_hw_gain", 0.0)) > 0.0:
                    best = dict(best_ch)
                    common_meta["alloc_selected_source"] = "challenger"
                    common_meta["alloc_selected_long_gain"] = float(best.get("probe_rel_hw_gain", 0.0))
                    common_meta["alloc_selected_apply_gain"] = float(best.get("rel_hw_gain", 0.0))
                common_meta["alloc_best_ch_long_gain"] = float(best_ch.get("probe_rel_hw_gain", 0.0)) if isinstance(best_ch, dict) else 0.0
                common_meta["alloc_decision_basis"] = "long_gain_only"
                run_state["alloc_incumbent_direction"] = None

    if best is None:
        run_state["alloc_last_search"] = {
            "outer": int(outer),
            "enabled": True,
            "reason": "no_candidate",
            "alloc_enabled_this_outer": True,
            "alloc_start_after_prune_epochs": int(start_after_prune_epochs),
            "alloc_phase_progress": float(phase_state.get("alloc_phase_progress", 0.0)),
            "alloc_phase_budget_frac": float(alloc_phase_budget_frac),
            "alloc_budget_frac": 0.0,
            "alloc_applied": False,
            "alloc_remain_budget": float(remain_budget),
            "alloc_usable_budget": float(usable_budget),
            "alloc_outer_threads": int(outer_threads),
            "alloc_inner_threads": int(inner_threads),
            "warmup_epochs": int(phase_state.get("warmup_epochs", warm_eff)),
            "prune_epoch": int(phase_state.get("prune_epoch", -1)),
            "alloc_epoch": int(phase_state.get("alloc_epoch", -1)),
            "target_global_keep": float(target_global_keep),
            "keep_mean_prunable": float(keep_mean_prunable),
            **common_meta,
        }
        return

    min_rel_hw_improve = float(getattr(alloc_cfg, "min_rel_hw_improve", 0.0))
    if controller == "legacy" and float(best["rel_hw_gain"]) < float(min_rel_hw_improve):
        run_state["alloc_last_search"] = {
            "outer": int(outer),
            "enabled": True,
            "reason": "below_min_rel_hw_improve",
            "alloc_enabled_this_outer": True,
            "alloc_start_after_prune_epochs": int(start_after_prune_epochs),
            "alloc_phase_progress": float(phase_state.get("alloc_phase_progress", 0.0)),
            "alloc_phase_budget_frac": float(alloc_phase_budget_frac),
            "alloc_budget_frac": 0.0,
            "alloc_applied": False,
            "alloc_remain_budget": float(remain_budget),
            "alloc_usable_budget": float(usable_budget),
            "alloc_outer_threads": int(outer_threads),
            "alloc_inner_threads": int(inner_threads),
            "warmup_epochs": int(phase_state.get("warmup_epochs", warm_eff)),
            "prune_epoch": int(phase_state.get("prune_epoch", -1)),
            "alloc_epoch": int(phase_state.get("alloc_epoch", -1)),
            "base_objective": float(best["base_objective"]),
            "best_objective": float(best["objective"]),
            "rel_hw_gain": float(best["rel_hw_gain"]),
            "acc_risk": float(best["acc_risk"]),
            "total_score": float(best["total_score"]),
            "target_global_keep": float(target_global_keep),
            "keep_mean_prunable": float(keep_mean_prunable),
            **common_meta,
        }
        logger.info(
            "[AllocSearch] outer=%d skipped apply: rel_hw_gain=%.6g < min_rel_hw_improve=%.6g",
            int(outer), float(best["rel_hw_gain"]), float(min_rel_hw_improve),
        )
        return

    # NOTE: Thesis gating rule uses (risk constraint) + (rel_hw_gain >= tau_hw) only.
    # The legacy min_total_score gate is disabled to avoid code–paper drift.

    _apply_layerwise_keep_candidate_to_gates(
        model,
        best["keep_cand"],
        gate_apply_blend=float(getattr(alloc_cfg, "gate_apply_blend", 1.0)),
    )

    run_state["alloc_last_keep_applied"] = best["keep_cand"].clone()
    run_state["alloc_last_search"] = {
        "outer": int(outer),
        "enabled": True,
        "reason": "applied",
        "alloc_enabled_this_outer": True,
        "alloc_start_after_prune_epochs": int(start_after_prune_epochs),
        "alloc_phase_progress": float(phase_state.get("alloc_phase_progress", 0.0)),
        "alloc_phase_budget_frac": float(alloc_phase_budget_frac),
        "alloc_budget_frac": float(alloc_phase_budget_frac),
        "alloc_applied": True,
        "alloc_remain_budget": float(remain_budget),
        "alloc_usable_budget": float(usable_budget),
        "alloc_outer_threads": int(outer_threads),
        "alloc_inner_threads": int(inner_threads),
        "warmup_epochs": int(phase_state.get("warmup_epochs", warm_eff)),
        "prune_epoch": int(phase_state.get("prune_epoch", -1)),
        "alloc_epoch": int(phase_state.get("alloc_epoch", -1)),
        "base_objective": float(best["base_objective"]),
        "best_objective": float(best["objective"]),
        "rel_hw_gain": float(best["rel_hw_gain"]),
        "acc_risk": float(best["acc_risk"]),
        "total_score": float(best["total_score"]),
        "alloc_policy": str(best.get("policy", "heuristic")),
        "alloc_eval_budget_total": int(best.get("eval_budget_total", 0) or 0),
        "alloc_eval_budget_candidates": int(best.get("eval_budget_candidates", 0) or 0),
        "cem_total_samples": int(best.get("cem_total_samples", 0) or 0),
        "cem_pool_size": int(best.get("cem_pool_size", 0) or 0),
        "cem_units": int(best.get("cem_units", 0) or 0),
        "cem_budget_mult": float(best.get("cem_budget_mult", 0.0) or 0.0),
        "cem_unique_proposals": int(best.get("cem_unique_proposals", 0) or 0),
        "cem_task_beta": float(best.get("cem_task_beta", 0.0) or 0.0),
        "cem_dir_units": int(best.get("cem_dir_units", 0) or 0),
        "probe_points": int(best.get("probe_points", 0) or 0),
        "probe_dirs": int(best.get("probe_dirs", 0) or 0),
        "probe_alpha": float(best.get("probe_alpha", 0.0) or 0.0),
        "probe_objective": float(best.get("probe_objective", 0.0) or 0.0),
        "target_global_keep": float(target_global_keep),
        "keep_mean_prunable": float(keep_mean_prunable),
        **common_meta,
    }

    logger.info(
        "[AllocSearch] outer=%d applied remain_budget=%.6g usable_budget=%.6g alloc_frac=%.3f prune_epoch=%d alloc_epoch=%d keep_mean_prunable=%.6g target_global_keep=%.6g base_obj=%.6g best_obj=%.6g rel_hw_gain=%.6g acc_risk=%.6g total=%.6g probeN=%d dirs=%d alpha*=%.2f cem_total=%d threads=%d/%d",
        int(outer),
        float(remain_budget),
        float(usable_budget),
        float(alloc_phase_budget_frac),
        int(phase_state.get("prune_epoch", -1)),
        int(phase_state.get("alloc_epoch", -1)),
        float(keep_mean_prunable),
        float(target_global_keep),
        float(best["base_objective"]),
        float(best["objective"]),
        float(best["rel_hw_gain"]),
        float(best["acc_risk"]),
        float(best["total_score"]),
        int(best.get("probe_points", 0) or 0),
        int(best.get("probe_dirs", 0) or 0),
        float(best.get("probe_alpha", 0.0) or 0.0),
        int(best.get("cem_total_samples", 0) or 0),
        int(outer_threads),
        int(inner_threads),
    )


def _maybe_light_project_ch_keep(
    *,
    model: torch.nn.Module,
    cfg,
    ast_sched: Dict[str, Any],
    outer: int,
    run_state: Dict[str, Any],
    logger,
) -> None:
    """Light projection: make mean(prunable keep) actually track ch_keep_target."""
    if bool(run_state.get("user_recovery_active", False)):
        return
    proj_cfg = getattr(getattr(cfg, "ast", None), "projection", None)
    if proj_cfg is None or not bool(getattr(proj_cfg, "enabled", False)):
        return
    if bool(ast_sched.get("force_dense", False)):
        return
    target = ast_sched.get("ch_keep_target", None)
    if target is None:
        return
    target = float(max(0.0, min(1.0, float(target))))
    pruner = _get_ast_pruner(model)
    if pruner is None or (not hasattr(pruner, "g_ch")):
        return
    try:
        pr_cfg = pruner.cfg.get("channel_prune", pruner.cfg)
        front_ratio = float(pr_cfg.get("freeze_prefix_ratio", pruner.cfg.get("ch_freeze_prefix_ratio", 0.0)) or 0.0)
        front_ratio = float(max(0.0, min(1.0, front_ratio)))
    except Exception:
        front_ratio = 0.0
    tol = float(getattr(proj_cfg, "tol", 0.002) or 0.002)
    blend = float(getattr(proj_cfg, "blend", 0.5) or 0.5)
    max_offset = float(getattr(proj_cfg, "max_offset", 8.0) or 8.0)
    iters = int(getattr(proj_cfg, "iters", 25) or 25)
    with torch.no_grad():
        ch_keep_layer = torch.sigmoid(pruner.g_ch).mean(dim=1)
        depth = int(ch_keep_layer.numel())
        k_freeze = int(round(float(depth) * float(front_ratio)))
        if k_freeze >= depth:
            return
        cur = float(ch_keep_layer[k_freeze:].mean().item())
        if cur <= target + tol:
            run_state["proj_last"] = {"outer": int(outer), "applied": False, "cur": cur, "target": target}
            return
        g = pruner.g_ch[k_freeze:].detach()
        lo, hi = 0.0, float(max_offset)
        for _ in range(max(8, iters)):
            mid = 0.5 * (lo + hi)
            m = float(torch.sigmoid(g - mid).mean().item())
            if m > target:
                lo = mid
            else:
                hi = mid
        off = float(hi)
        step = float(max(0.0, min(1.0, blend))) * off
        if step <= 0.0:
            return
        pruner.g_ch.data[k_freeze:].sub_(step)
        new_keep = float(torch.sigmoid(pruner.g_ch).mean(dim=1)[k_freeze:].mean().item())
        run_state["proj_last"] = {
            "outer": int(outer),
            "applied": True,
            "cur": float(cur),
            "target": float(target),
            "offset": float(off),
            "step": float(step),
            "new": float(new_keep),
        }
        logger.info(
            "[ProjKeep] outer=%d prunable_keep %.6f -> %.6f (target=%.6f, off=%.3f, step=%.3f)",
            int(outer), float(cur), float(new_keep), float(target), float(off), float(step)
        )


def _set_structural_gates_trainable(model: torch.nn.Module, trainable: bool) -> Dict[str, Any]:
    """Freeze/unfreeze structural gates so recovery epochs do NOT change hardware."""
    pruner = _get_ast_pruner(model)
    if pruner is None:
        return {"ok": False, "trainable": bool(trainable)}
    changed = []
    for name in ("g_head", "g_ch", "g_block"):
        if hasattr(pruner, name):
            p = getattr(pruner, name)
            try:
                p.requires_grad_(bool(trainable))
                changed.append(name)
            except Exception:
                pass
    return {"ok": True, "trainable": bool(trainable), "params": changed}


def _maybe_start_user_recovery_after_val(
    *,
    cfg,
    run_state: Dict[str, Any],
    ast_sched: Dict[str, Any],
    outer: int,
    prunable_keep: float,
    logger,
) -> None:
    rec_cfg = getattr(getattr(cfg, "ast", None), "recovery", None)
    if rec_cfg is None or not bool(getattr(rec_cfg, "enabled", False)):
        return
    if bool(run_state.get("user_recovery_started", False)):
        return
    sched = getattr(getattr(cfg, "ast", None), "schedule", None)
    keep_end = float(getattr(sched, "ch_keep_end", 1.0) or 1.0) if sched is not None else 1.0
    tol = float(getattr(rec_cfg, "tol", 0.003) or 0.003)
    epochs = max(0, int(getattr(rec_cfg, "epochs", 2) or 2))
    if epochs <= 0:
        return
    tgt = float(ast_sched.get("ch_keep_target", keep_end) or keep_end)
    if float(tgt) > float(keep_end) + 1e-6:
        return
    if float(prunable_keep) > float(keep_end) + float(tol):
        return
    run_state["user_recovery_started"] = True
    run_state["user_recovery_start_outer"] = int(outer) + 1
    run_state["user_recovery_remaining"] = int(epochs)
    logger.info(
        "[UserRecovery] scheduled: start_outer=%d epochs=%d (keep_end=%.4f tol=%.4f prunable_keep=%.4f)",
        int(run_state["user_recovery_start_outer"]), int(epochs), float(keep_end), float(tol), float(prunable_keep)
    )


def _eval_epoch_end_hw_snapshot(
    *,
    model: torch.nn.Module,
    cfg,
    run_state: Dict[str, Any],
    hw_proxy,
    wafer_layout,
    chiplet_slots,
) -> Dict[str, Any]:
    try:
        keep_now = _get_layer_ch_keep_now(model)
        if keep_now is None:
            return {
                "ok": False,
                "error": "keep_now is None",
            }

        last_info = run_state.get("last_model_info", None)
        if last_info is None:
            return {
                "ok": False,
                "error": "last_model_info is None",
            }

        eval_info = _build_eval_model_info_with_ch_override(
            last_info=last_info,
            depth=int(keep_now.numel()),
            ch_keep_override=[float(x) for x in keep_now.tolist()],
        )

        slot_out = chiplet_slots(hard=False)
        eff_specs = slot_out["eff_specs"]
        alpha = slot_out["alpha"]

        mapping_solver_local = MappingSolver(cfg.mapping.strategy, cfg.mapping.mem_limit_factor)
        partitioner_local = PartitionPlanner(mapping_solver_local, wafer_layout, hw_proxy, cfg.partition)
        fine_threads = int(os.environ.get("ALLOC_SEARCH_INNER_THREADS", "5"))

        part_res = partitioner_local.plan(
            model,
            eff_specs,
            alpha=alpha,
            model_info=eval_info,
            use_fine_split=bool(getattr(cfg.hw, "use_fine_split", True)),
            fine_split_threads=int(max(1, fine_threads)),
        )
        return {
            "ok": True,
            "objective": float(part_res.get("objective", 1.0e18)),
            "mapping_sig": part_res.get("mapping_sig", None),
            "latency_ms": float(part_res.get("latency_ms", part_res.get("raw_latency_ms", 0.0)) or 0.0),
            "energy_mj": float(part_res.get("energy_mj", part_res.get("raw_energy_mj", 0.0)) or 0.0),
            "mem_mb": float(part_res.get("mem_mb", part_res.get("raw_mem_mb", 0.0)) or 0.0),
            "comm_ms": float(part_res.get("comm_ms", part_res.get("raw_comm_ms", 0.0)) or 0.0),
            "raw_latency_ms": float(part_res.get("raw_latency_ms", 0.0) or 0.0),
            "raw_energy_mj": float(part_res.get("raw_energy_mj", 0.0) or 0.0),
            "raw_mem_mb": float(part_res.get("raw_mem_mb", 0.0) or 0.0),
            "raw_comm_ms": float(part_res.get("raw_comm_ms", 0.0) or 0.0),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc),
        }


def train_version_c(
    cfg,
    out_dir: Optional[str] = None,
    export_layout_input: bool = False,
    layout_export_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    ctr = getattr(cfg, "_contract", None)
    if ctr is None or not bool(getattr(ctr, "stamped_v54", False)):
        raise RuntimeError(
            "v5.4 CONTRACT: cfg not validated/stamped. "
            "Call validate_and_fill_defaults(...) via SPEC_D OneCommand entrypoint."
        )
    if not getattr(cfg, "contract", None) or not getattr(cfg.contract, "seal_digest", None):
        raise RuntimeError("v5.4 CONTRACT: missing seal_digest; boot not completed.")
    # --- [v5.4 HARD GATE D] trainer must refuse non-bootstrapped cfg ---
    c = getattr(cfg, "contract", None)
    if c is None or getattr(c, "validated", False) is not True or str(getattr(c, "version", "")) != "v5.4":
        raise RuntimeError(
            "v5.4 P0: cfg is not contract-validated (missing contract.validated/version). "
            "Use scripts/run_version_c.py (OneCommand) to start runs."
        )
    vb = str(getattr(c, "validated_by", "") or "")
    if vb not in ("validate_and_fill_defaults", "contract_bootstrap_v5.4"):
        raise RuntimeError(
            f"v5.4 P0: cfg.validated_by={vb!r} is not an approved bootstrap marker."
        )
    seal_digest = getattr(getattr(cfg, "contract", None), "seal_digest", None)
    if not seal_digest:
        raise RuntimeError("v5.4 P0: missing cfg.contract.seal_digest (contract evidence not sealed)")

    device = get_device(cfg.train.device)
    device_type = device.type
    logger = setup_logger()
    amp_enabled, amp_dtype, use_scaler, amp_dtype_str = _resolve_amp_settings(cfg, device_type, logger=logger)
    if out_dir is not None:
        expected_out_dir = str(getattr(getattr(cfg, "train", None), "out_dir", "") or "")
        if expected_out_dir and str(out_dir) != expected_out_dir:
            raise RuntimeError(
                f"v5.4 P0: out_dir mismatch after seal (cfg.train.out_dir={expected_out_dir}, cli={out_dir})"
            )
    if seed is not None:
        cfg_seed = int(getattr(getattr(cfg, "train", None), "seed", 0) or getattr(cfg.training, "seed", 0) or 0)
        if int(seed) != cfg_seed:
            raise RuntimeError(
                f"v5.4 P0: seed mismatch after seal (cfg.train.seed={cfg_seed}, cli={seed})"
            )
    # ---- v5.4: allow config-driven export (OneCommand) ----
    if not export_layout_input:
        export_layout_input = bool(getattr(cfg, "export_layout_input", False))

    layout_export_dir_source = "cli"
    if layout_export_dir is None:
        layout_export_dir = getattr(cfg, "export_dir", None)
        layout_export_dir_source = "cfg"

    cfg_export_dir = str(layout_export_dir or "").strip()
    if not cfg_export_dir:
        # legacy fallback (kept) — but must be auditable
        layout_export_dir = str(Path(cfg.out_dir) / "exports" / "layout_input")
        layout_export_dir_source = "default_out_dir_exports"
    # out_dir: training outputs root
    out_dir = Path(getattr(cfg.train, "out_dir", "") or "outputs/version_c")
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(getattr(cfg.train, "seed", 0) or getattr(cfg.training, "seed", 0) or 0)
    run_id = stable_hash(
        {
            "mode": "version_c_train",
            "seal_digest": str(seal_digest),
            "seed": int(seed),
        }
    )
    cfg_hash = str(seal_digest)
    cfg_path = str(getattr(cfg, "cfg_path", "") or getattr(getattr(cfg, "train", None), "cfg_path", "") or "")
    signature = build_signature_v54(cfg, method_name="ours_version_c")
    signature_v54 = signature
    # ---- v5.4 trace dir contract: out_dir/trace/<run_id>/... ----
    trace_base = out_dir / "trace"
    trace_meta = init_trace_dir_v54(
        base_dir=trace_base,
        run_id=str(run_id),
        cfg=cfg,
        signature=signature,
        signature_v54=signature_v54,
        required_signature_fields=REQUIRED_SIGNATURE_FIELDS,
        run_meta={"mode": "version_c_train", "seed_id": int(seed), "run_id": str(run_id)},
        extra_manifest={
            "task": "version_c",
            "out_dir": str(out_dir),
            "layout_export_dir_resolved": str(layout_export_dir),
            "layout_export_dir_source": str(layout_export_dir_source),
        },
    )
    trace_dir = Path(trace_meta["trace_dir"])
    trace_events_path = Path(trace_meta["trace_events"])
    trace_header_written = False
    steps_done_for_finalize = 0
    try:
        assert_cfg_sealed_or_violate(cfg, seal_digest, trace_events_path, step=0)
        # layout_export_dir: ONLY for exporting layout_input.json (optional)
        layout_export_dir = Path(layout_export_dir) if layout_export_dir else None
        if layout_export_dir is not None:
            layout_export_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "logs" / "version_c_stats.jsonl"
        epoch_audit_path = out_dir / "logs" / "epoch_end_audit.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_slim = bool(int(os.environ.get("LOG_SLIM", "1")))
        log_interval_steps = int(os.environ.get("LOG_INTERVAL_STEPS", default=(200 if log_slim else 10)))
        log_interval_steps = max(1, int(log_interval_steps))
        suite_cleanup_enabled = bool(_oc_select(cfg, "suite_cleanup.enabled", False))
        disable_stable_hw_guard_controls = bool(
            _oc_select(cfg, "suite_cleanup.disable_stable_hw_guard_controls", False)
        )
        suppress_stable_hw_epoch_summary = bool(
            _oc_select(cfg, "suite_cleanup.suppress_stable_hw_epoch_summary", False)
        )
        suppress_stable_hw_debug_lines = bool(
            _oc_select(cfg, "suite_cleanup.suppress_stable_hw_debug_lines", False)
        )
        suppress_alloc_fields_when_disabled = bool(
            _oc_select(cfg, "suite_cleanup.suppress_alloc_fields_when_disabled", False)
        )
        suppress_hw_fields_when_zero = bool(_oc_select(cfg, "suite_cleanup.suppress_hw_fields_when_zero", False))
        suppress_head_block_keep_console = bool(
            _oc_select(cfg, "suite_cleanup.suppress_head_block_keep_console", False)
        )
        slim_console_log = bool(_oc_select(cfg, "suite_cleanup.slim_console_log", False))
        force_dense_freeze_structural_gates = bool(
            _oc_select(cfg, "training.force_dense_freeze_structural_gates", False)
        )
        # Reduce log bloat from repeated per-step warnings in long Version-C runs.
        # - train.warn_every_steps (or WARN_EVERY_STEPS env) controls how often repeated WARN lines are emitted.
        # - We still count all events and print a per-outer summary (so issues remain auditable).
        warn_every_steps = int(getattr(getattr(cfg, "train", object()), "warn_every_steps", 0) or 0)
        if warn_every_steps <= 0:
            warn_every_steps = int(os.environ.get("WARN_EVERY_STEPS", str(log_interval_steps)))
        warn_every_steps = max(1, int(warn_every_steps))

        nan_guard_warn_every_steps = int(getattr(getattr(cfg, "train", object()), "nan_guard_warn_every_steps", 0) or 0)
        if nan_guard_warn_every_steps <= 0:
            nan_guard_warn_every_steps = warn_every_steps
        nan_guard_warn_every_steps = max(1, int(nan_guard_warn_every_steps))

        def _warn_throttled(key: str, step_i: int, msg: str, *args, every_steps: int, first_n: int = 1) -> None:
            total_key = f"warn_{key}_total"
            logged_key = f"warn_{key}_logged"
            last_key = f"warn_{key}_last_step"
            total = int(run_state.get(total_key, 0)) + 1
            run_state[total_key] = total
            last = run_state.get(last_key, None)
            should = False
            if total <= int(first_n):
                should = True
            elif last is None:
                should = True
            else:
                try:
                    should = (int(step_i) - int(last)) >= int(every_steps)
                except Exception:
                    should = True
            if should:
                run_state[last_key] = int(step_i)
                run_state[logged_key] = int(run_state.get(logged_key, 0)) + 1
                logger.warning(msg, *args)
        # ---- v5.4 contract log (avoid missing-key crash) ----
        def _boolish(x):
            if x is None:
                return None
            if isinstance(x, bool):
                return x
            if isinstance(x, (int, float)):
                return bool(x)
            if isinstance(x, dict):
                return bool(x.get("enabled", False))
            try:
                return bool(getattr(x, "enabled"))
            except Exception:
                return None

        nds_en = _boolish(getattr(getattr(cfg, "stable_hw", None), "no_double_scale", None))
        logger.info(
            f"[v5.4 contract] train.mode={getattr(cfg.train, 'mode', None)} "
            f"strict={getattr(getattr(cfg, 'contract', None), 'strict', None)} "
            f"seal_digest={seal_digest} "
            f"stable_hw.enabled={getattr(cfg.stable_hw, 'enabled', None)} "
            f"stable_hw.locked_acc_ref.enabled={getattr(getattr(cfg.stable_hw, 'locked_acc_ref', None), 'enabled', None)} "
            f"stable_hw.no_drift.enabled={getattr(getattr(cfg.stable_hw, 'no_drift', None), 'enabled', None)} "
            f"stable_hw.no_double_scale.enabled={nds_en}"
        )

        def _update_manifest_gating_summary(trace_dir: Path, cfg_obj, stable_state: Dict[str, Any]) -> None:
            manifest_path = trace_dir / "manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                manifest = {}
            stable_cfg = getattr(cfg_obj, "stable_hw", None)
            no_drift_cfg = _cfg_get(stable_cfg, "no_drift", None)
            guard_cfg = _cfg_get(stable_cfg, "accuracy_guard", None)
            gating_summary = {
                "requested": {
                    "stable_hw_enabled": bool(_cfg_get(stable_cfg, "enabled", False)),
                    "no_drift_enabled": bool(_cfg_get(no_drift_cfg, "enabled", False)),
                    "hard_gating": bool(_cfg_get(guard_cfg, "hard_gating", False)),
                },
                "effective": {
                    "stable_hw_enabled": bool(stable_state.get("stable_hw_enabled", _cfg_get(stable_cfg, "enabled", False))),
                    "allow_discrete_updates": bool(stable_state.get("allow_discrete_updates", True)),
                    "gating_reason_code": str(stable_state.get("gating_reason_code", "") or ""),
                },
            }
            manifest["gating_policy_summary"] = gating_summary
            manifest_path.write_text(safe_dumps(manifest, indent=2), encoding="utf-8")

        loader = build_dataloader(cfg)
        # ---- build val/test loader for stable_hw accuracy_guard ----
        val_ds = UCF101Dataset(cfg, split="val")
        base_seed = int(getattr(cfg.training, "seed", getattr(cfg.train, "seed", 0)))
        generator = torch.Generator()
        generator.manual_seed(base_seed)

        def _seed_worker(worker_id: int):
            s = base_seed + worker_id
            seed_everything(s)

        pin_memory = bool(getattr(cfg.data, "pin_memory", True))
        persistent_workers = bool(getattr(cfg.data, "persistent_workers", True)) and int(getattr(cfg.data, "num_workers", 4)) > 0
        prefetch_factor = int(getattr(cfg.data, "prefetch_factor", 2))
        val_kwargs = dict(
            dataset=val_ds,
            batch_size=int(getattr(cfg.data, "batch_size", cfg.train.batch_size)),
            shuffle=False,
            num_workers=int(getattr(cfg.data, "num_workers", 4)),
            worker_init_fn=_seed_worker,
            generator=generator,
            pin_memory=pin_memory,
        )
        if int(getattr(cfg.data, "num_workers", 4)) > 0:
            val_kwargs.update(dict(persistent_workers=persistent_workers, prefetch_factor=prefetch_factor))
        val_loader = DataLoader(**val_kwargs)

        # training.stable_hw_eval_max_batches:
        #   >0  : evaluate at most N val batches (fast eval)
        #   <=0 : FULL eval (all val batches). NOTE: eval_acc1 treats max_batches=0 as "run 0 batches",
        #         so we convert <=0 -> None here.
        max_eval_batches_cfg = int(getattr(cfg.training, "stable_hw_eval_max_batches", 20))
        max_eval_batches_for_eval = None if max_eval_batches_cfg <= 0 else max_eval_batches_cfg
        data_iter = iter(loader)

        # NOTE(v5.4 contract): cfg is sealed after validate_and_fill_defaults().
        # Do NOT mutate cfg here (or anywhere after seal), otherwise seal_digest check will fail.
        mapping_only = bool(getattr(cfg.training, "mapping_only", False))
        layout_only = bool(getattr(cfg.training, "layout_only", False))
        twostage = bool(getattr(cfg.training, "twostage", False))

        base_update_alpha = not layout_only
        # v5.4: layout updates must be discrete and auditable; continuous pos optimization in trainer is forbidden
        stable_hw_cfg = getattr(cfg, "stable_hw", None)
        iso_cfg_global = getattr(stable_hw_cfg, "discrete_isolation", None) if stable_hw_cfg else None
        update_layout = bool(_get_iso_cfg_value(iso_cfg_global, "optimize_layout", False))
        if update_layout:
            raise RuntimeError(
                "[P0][v5.4] optimize_layout in Version-C trainer is forbidden. "
                "Use layout_agent/HeurAgenix to generate cached layouts (assign-only) and run with cache-only."
            )

        model_type = getattr(cfg.training, "model_type", "video")
        num_frames = int(getattr(cfg.data, "num_frames", cfg.model.num_frames))
        audio_feat_dim = int(getattr(cfg.data, "audio_feat_dim", cfg.audio.feat_dim))
        if model_type == "video_audio":
            model = VideoAudioAST(
                img_size=cfg.model.img_size,
                num_frames=num_frames,
                num_classes=cfg.model.num_classes,
                embed_dim=cfg.model.embed_dim,
                depth=cfg.model.depth,
                num_heads=cfg.model.num_heads,
                mlp_ratio=cfg.model.mlp_ratio,
                patch_size=cfg.model.patch_size,
                audio_feat_dim=audio_feat_dim,
                in_chans=cfg.model.in_chans,
                drop_rate=cfg.model.drop_rate,
                attn_drop=cfg.model.attn_drop,
                drop_path_rate=cfg.model.drop_path_rate,
                use_ast_prune=cfg.ast.use_ast_prune,
                ast_cfg=cfg.ast,
            ).to(device)
        else:
            model = VideoViT(
                img_size=cfg.model.img_size,
                num_frames=num_frames,
                num_classes=cfg.model.num_classes,
                embed_dim=cfg.model.embed_dim,
                depth=cfg.model.depth,
                num_heads=cfg.model.num_heads,
                mlp_ratio=cfg.model.mlp_ratio,
                patch_size=cfg.model.patch_size,
                in_chans=cfg.model.in_chans,
                drop_rate=cfg.model.drop_rate,
                attn_drop=cfg.model.attn_drop,
                drop_path_rate=cfg.model.drop_path_rate,
                use_ast_prune=cfg.ast.use_ast_prune,
                ast_cfg=cfg.ast,
            ).to(device)

        # Optional pretrained initialization (local weights).
        # If pretrain.enabled=true and weights are missing/incompatible, we fail loudly
        # to avoid accidental "training from scratch".
        maybe_load_pretrained(cfg=cfg, model=model, logger=logger)

        # ---- Optional multi-GPU (single-process) via torch.nn.DataParallel ----
        # Enable with: USE_DP=1 and CUDA_VISIBLE_DEVICES=0,1,2 (or any N GPUs).
        use_dp = str(os.environ.get("USE_DP", "0")).strip().lower() in ("1", "true", "yes", "y", "on")
        if use_dp and device.type == "cuda":
            n = int(torch.cuda.device_count())
            if n > 1:
                # DataParallel requires the *base* module + input tensors to live on device_ids[0].
                # If cfg.train.device was set to cuda:1 (or similar), hard-pin to cuda:0 within the visible set.
                primary = torch.device("cuda:0")
                if device != primary:
                    logger.info("[DP] overriding device %s -> %s (DataParallel primary)", str(device), str(primary))
                    device = primary
                    model = model.to(device)

                try:
                    torch.cuda.set_device(0)
                except Exception:
                    pass

                logger.info(
                    "[DP] env CUDA_VISIBLE_DEVICES=%s torch.cuda.device_count()=%d primary=%s",
                    str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
                    int(n),
                    str(device),
                )
                logger.info("[DP] enabling DataParallel with %d visible GPUs", n)
                model = torch.nn.DataParallel(model, device_ids=list(range(n)), output_device=0)
            else:
                logger.info("[DP] USE_DP=1 but only 1 visible GPU; running single-GPU")

        ema_cfg = getattr(cfg.train, "ema", None)
        ema_enabled = bool(getattr(ema_cfg, "enabled", False))
        ema_decay = float(getattr(ema_cfg, "decay", 0.9999) or 0.9999)
        ema_eval = bool(getattr(ema_cfg, "eval", True))
        ema_model = ModelEMA(model, decay=ema_decay) if ema_enabled else None
        logger.info("[EMA] enabled=%s decay=%.6f eval=%s", bool(ema_enabled), float(ema_decay), bool(ema_eval))

        lr = _as_float(cfg.train.lr, "cfg.train.lr")
        weight_decay = _as_float(cfg.train.weight_decay, "cfg.train.weight_decay")
        backbone_lr_scale = float(getattr(cfg.train, "backbone_lr_scale", 1.0) or 1.0)
        freeze_backbone_epochs = int(getattr(cfg.train, "freeze_backbone_epochs", 0) or 0)
        model_param_groups, model_param_group_meta = _build_model_param_groups(
            model=model,
            lr=lr,
            backbone_lr_scale=backbone_lr_scale,
        )
        optimizer_model = torch.optim.AdamW(model_param_groups, lr=lr, weight_decay=weight_decay)
        logger.info(
            "[OPT] model_lr=%.6g backbone_lr_scale=%.4f freeze_backbone_epochs=%d backbone_tensors=%d head_tensors=%d backbone_params=%d head_params=%d",
            float(lr),
            float(backbone_lr_scale),
            int(freeze_backbone_epochs),
            int(model_param_group_meta.get("backbone_tensors", 0)),
            int(model_param_group_meta.get("head_tensors", 0)),
            int(model_param_group_meta.get("backbone_params", 0)),
            int(model_param_group_meta.get("head_params", 0)),
        )
        # ------------------------------------------------------------------
        # v5.4 hotfix:
        # Version-C trainer historically used a single variable name `optimizer`
        # for checkpointing / auto-resume, but this implementation split into
        # optimizer_model / optimizer_alpha and forgot to define `optimizer`.
        #
        # Keep checkpoint semantics unchanged: `optimizer` refers to the MODEL
        # optimizer (optimizer_model). This avoids NameError and does NOT affect
        # SMOKE vs official output directory routing.
        # ------------------------------------------------------------------
        optimizer = optimizer_model
        scaler = GradScaler(device_type, enabled=use_scaler)
        logger.info("[AMP] enabled=%s amp_dtype=%s autocast_dtype=%s scaler=%s",
                    bool(amp_enabled), str(amp_dtype_str), str(amp_dtype).replace("torch.", ""), bool(use_scaler))

        library = ChipletLibrary(cfg.hw.gpu_yaml)
        chiplet_slots = ChipletSlots(
            library,
            cfg.chiplet.candidate_types,
            cfg.hw.num_slots,
            cfg.chiplet.tau_init,
        ).to(device)

        # Chapter-3 fairness:
        # all methods must share the same fixed hardware-slot substrate.
        # chiplet_slots can be READ to produce alpha / eff_specs,
        # but must not be TRAINED by any method (including HWLOSS).
        for p in chiplet_slots.parameters():
            p.requires_grad_(False)

        optimizer_alpha = None
        logger.info("[ChipletSlots] trainable=%s policy=%s", False, "fixed_for_ch3_fairness")

        label_smoothing = float(getattr(cfg.train, "label_smoothing", 0.0) or 0.0)
        mixup_alpha = float(getattr(cfg.train, "mixup_alpha", 0.0) or 0.0)
        mixup_prob = float(getattr(cfg.train, "mixup_prob", 1.0) or 1.0)
        mixup_switch_off_epoch = int(getattr(cfg.train, "mixup_switch_off_epoch", -1) or -1)

        lr_schedule = str(getattr(cfg.train, "lr_schedule", "none") or "none").lower()
        min_lr = float(getattr(cfg.train, "min_lr", 0.0) or 0.0)
        warmup_epochs = int(getattr(cfg.train, "warmup_epochs", 0) or 0)
        outer_epochs_total = int(getattr(cfg.training, "outer_epochs", getattr(cfg.train, "epochs", 0)) or 0)
        inner_steps_ast_cfg = int(getattr(cfg.training, "inner_steps_ast", 0) or 0)
        steps_per_outer = len(loader) if (inner_steps_ast_cfg <= 0 and hasattr(loader, "__len__")) else max(1, inner_steps_ast_cfg)
        warmup_steps = int(warmup_epochs) * int(steps_per_outer)
        total_steps = int(outer_epochs_total) * int(steps_per_outer)

        def _compute_lr(global_step: int) -> float:
            if lr_schedule != "cosine":
                return float(lr)
            if total_steps <= 0:
                return float(lr)
            if warmup_steps > 0 and global_step < warmup_steps:
                return float(lr) * float(global_step + 1) / float(max(1, warmup_steps))
            denom = max(1, total_steps - warmup_steps)
            progress = float(global_step - warmup_steps) / float(denom)
            progress = max(0.0, min(1.0, progress))
            return float(min_lr) + 0.5 * (float(lr) - float(min_lr)) * (1.0 + math.cos(math.pi * progress))

        proxy_weight_dir = str(getattr(cfg.hw, "weight_dir", "") or getattr(cfg.hw, "proxy_weight_dir", ""))
        if not proxy_weight_dir:
            raise RuntimeError("[ProxyMissing] cfg.hw.weight_dir or cfg.hw.proxy_weight_dir must be set.")
        # ---- Proxy run context (IMPORTANT: keep consistent with proxy training features) ----
        # NOTE: batch_size is under cfg.train.batch_size (NOT cfg.data.batch_size)
        bs_train = int(getattr(cfg.train, "batch_size", 1) or 1)

        # For VideoViT, forward flattens (B, T, ...) -> (B*T, ...),
        # so the effective batch seen by attention/MLP is B*T.
        # This helps keep proxy inputs in the correct scale.
        nf = int(num_frames) if num_frames is not None else int(getattr(cfg.model, "num_frames", 1) or 1)
        bs_eff = int(bs_train * max(1, nf))

        run_ctx = {
            "img": int(cfg.model.img_size),
            "bs": int(bs_eff),
            "num_frames": int(nf),
            "depth": int(cfg.model.depth),
            "embed_dim": int(cfg.model.embed_dim),
            "num_heads": int(cfg.model.num_heads),
            "mlp_ratio": float(cfg.model.mlp_ratio),
            "tp_world_size": int(getattr(cfg.hw, "tp_world_size", 1) or 1),
            "runs": int(getattr(cfg.hw, "proxy_runs", 10) or 10),
            "warmup": int(getattr(cfg.hw, "proxy_warmup", 5) or 5),
            # ---- Stabilize tabular fixed-point (ms<->mem) ----
            # These priors prevent ms0 from falling far outside training distribution.
            "proxy_ms_prior_min": float(getattr(cfg.hw, "proxy_ms_prior_min", 0.05) or 0.05),
            "proxy_ms_prior_max": float(getattr(cfg.hw, "proxy_ms_prior_max", 100.0) or 100.0),
        }
        use_multi_proxy = bool(getattr(cfg.hw, "multi_device_proxy", True))
        if use_multi_proxy:
            hw_proxy = MultiDeviceHwOracle(cfg.hw.gpu_yaml, weight_dir=proxy_weight_dir, run_ctx=run_ctx)
        else:
            hw_proxy = LayerHwProxy(cfg.hw.device_name, cfg.hw.gpu_yaml, proxy_weight_dir, run_ctx=run_ctx)
        mapping_solver = MappingSolver(cfg.mapping.strategy, cfg.mapping.mem_limit_factor)
        # v5.4: build discrete sites + deterministic initial assign (SPEC_B)
        chip_max_w = max(library.get(n).width_mm for n in cfg.chiplet.candidate_types)
        chip_max_h = max(library.get(n).height_mm for n in cfg.chiplet.candidate_types)

        sites_xy_np = build_sites(
            wafer_radius_mm=float(cfg.hw.wafer_radius_mm),
            chip_max_width_mm=float(chip_max_w),
            chip_max_height_mm=float(chip_max_h),
            margin_mm=float(getattr(cfg.hw, "site_margin_mm", 0.0) or 0.0),
            method=str(getattr(cfg.hw, "site_method", "square_grid_in_circle")),
            grid_pitch_mm=(
                float(getattr(cfg.hw, "site_pitch_mm")) if getattr(cfg.hw, "site_pitch_mm", None) is not None else None
            ),
            seed=int(seed),
        )
        if int(sites_xy_np.shape[0]) < int(cfg.hw.num_slots):
            raise RuntimeError(
                f"[P0][v5.4] Not enough discrete sites: Ns={int(sites_xy_np.shape[0])} < num_slots={int(cfg.hw.num_slots)}"
            )

        sites_xy = torch.tensor(sites_xy_np, device=device, dtype=torch.float32)
        assign0 = torch.arange(int(cfg.hw.num_slots), device=device, dtype=torch.long)  # slot i -> site i

        wafer_layout = WaferLayout(
            num_slots=int(cfg.hw.num_slots),
            wafer_radius_mm=float(cfg.hw.wafer_radius_mm),
            sites_xy=sites_xy,
            assign=assign0,
        ).to(device)
        partitioner = PartitionPlanner(mapping_solver, wafer_layout, hw_proxy, cfg.partition)
        optimizer_layout = None  # v5.4: forbidden (see P0-3)

        layout_opt_steps = int(_get_iso_cfg_value(iso_cfg_global, "layout_opt_steps", 10) or 10)
        layout_opt_lr = float(_get_iso_cfg_value(iso_cfg_global, "layout_opt_lr", 5e-2) or 5e-2)
        layout_opt_grad_clip = float(_get_iso_cfg_value(iso_cfg_global, "layout_opt_grad_clip", 1.0) or 1.0)
        layout_opt = None

        last_segments: List = []
        last_mapping: List[int] = []
        stable_hw_state: Dict[str, Any] = {}
        stable_hw_state["seed"] = int(seed)
        stable_hw_state["run_signature"] = signature
        stable_hw_state["out_dir"] = str(out_dir)
        stable_hw_state["stable_hw_enabled"] = bool(getattr(stable_hw_cfg, "enabled", True)) if stable_hw_cfg else False
        try:
            mu = unwrap_model(model)
            if hasattr(mu, "cfg") and getattr(mu.cfg, "depth", None) is not None:
                stable_hw_state["arch_depth"] = int(mu.cfg.depth)
        except Exception:
            pass

        hw_ratio_cap = float(getattr(cfg.training, "hw_term_ratio_cap", 0.1) or 0.1)
        hw_norm_enabled = bool(getattr(cfg.training, "hw_loss_norm_enabled", True))
        hw_norm_alpha = float(getattr(cfg.training, "hw_loss_norm_ema_alpha", 0.98) or 0.98)
        hw_norm_min_denom = float(getattr(cfg.training, "hw_loss_norm_min_denom", 1e-3) or 1e-3)
        hw_norm_clip = float(getattr(cfg.training, "hw_loss_norm_clip", 10.0) or 10.0)
        hw_kill_epochs = int(getattr(cfg.training, "hw_kill_epochs_on_nan", 3) or 3)
        if "hw_mag_ema" not in stable_hw_state:
            stable_hw_state["hw_mag_ema"] = 1.0
        if stable_hw_cfg is None:
            locked_cfg = {}
            no_drift_cfg = {}
            guard_cfg = {}
            guard_ctrl_cfg = {}
        else:
            locked_cfg = stable_hw_cfg.get("locked_acc_ref", {}) if isinstance(stable_hw_cfg, dict) else getattr(
                stable_hw_cfg, "locked_acc_ref", {}
            )
            no_drift_cfg = stable_hw_cfg.get("no_drift", {}) if isinstance(stable_hw_cfg, dict) else getattr(
                stable_hw_cfg, "no_drift", {}
            )
            guard_cfg = stable_hw_cfg.get("accuracy_guard", {}) if isinstance(stable_hw_cfg, dict) else getattr(
                stable_hw_cfg, "accuracy_guard", {}
            )
            guard_ctrl_cfg = guard_cfg.get("controller", {}) if isinstance(guard_cfg, dict) else getattr(
                guard_cfg, "controller", {}
            )
        lar_source = (
            locked_cfg.get("source", None) if isinstance(locked_cfg, dict) else getattr(locked_cfg, "source", None)
        ) or "unknown"
        locked_acc_ref_enabled = bool(
            (locked_cfg.get("enabled", True) if isinstance(locked_cfg, dict) else getattr(locked_cfg, "enabled", True))
        )
        no_drift_enabled = bool(
            (no_drift_cfg.get("enabled", True) if isinstance(no_drift_cfg, dict) else getattr(no_drift_cfg, "enabled", True))
        )
        allow_train_ema_fallback = _oc_select(cfg, "stable_hw.allow_train_ema_fallback", None)
        if allow_train_ema_fallback is None:
            allow_train_ema_fallback = _oc_select(cfg, "stable_hw.accuracy_guard.allow_train_ema_fallback", None)
        if stable_hw_cfg is not None:
            nd_cfg = getattr(stable_hw_cfg, "no_drift", None)
            if isinstance(nd_cfg, bool):
                nd_enabled = bool(nd_cfg)
            elif isinstance(nd_cfg, dict):
                nd_enabled = bool(nd_cfg.get("enabled", False))
            elif nd_cfg is not None:
                nd_enabled = bool(getattr(nd_cfg, "enabled", False))
            else:
                nd_enabled = False
            stable_hw_state["no_drift_enabled"] = nd_enabled
            nds_cfg = getattr(stable_hw_cfg, "no_double_scale", False)
            if isinstance(nds_cfg, dict):
                nds_enabled = bool(nds_cfg.get("enabled", False))
            else:
                nds_enabled = bool(nds_cfg)
            stable_hw_state["no_double_scale_enabled"] = nds_enabled
        else:
            stable_hw_state["no_drift_enabled"] = False
            stable_hw_state["no_double_scale_enabled"] = False
        stable_hw_state.setdefault(
            "discrete_cache",
            {
                "mapping": None,
                "layout": None,
                "mapping_signature": None,
                "layout_signature": None,
                "hw_mats": {},
            },
        )
        if stable_hw_cfg and bool(getattr(stable_hw_cfg, "enabled", True)):
            # v5.4 stable_hw APIs: init_locked_acc_ref does NOT accept stable_hw_cfg/output_dir
            init_locked_acc_ref(cfg, stable_hw_state)

            # init_hw_refs_from_baseline_stats accepts stable_hw_cfg but NOT output_dir
            init_hw_refs_from_baseline_stats(cfg, stable_hw_state, stable_hw_cfg=stable_hw_cfg)

            # ---- v5.4: fail-fast preflight for LockedAccRef (baseline stdout curve) ----
            try:
                pre_decision, _ = apply_accuracy_guard(
                    epoch=0,
                    stable_hw_cfg=cfg,
                    stable_hw_state=stable_hw_state,
                    val_metric_or_none=None,
                    has_val_this_epoch=False,
                    train_ema_or_none=None,
                )
                stable_hw_state = pre_decision.state
            except Exception as e:
                raise RuntimeError(
                    "[P0][v5.4] LockedAccRef preflight failed before training. "
                    "Check BASELINE_STDOUT / stable_hw.locked_acc_ref.baseline_stdout_path and ensure stdout contains "
                    "lines like: [val] epoch=.. mode=fast/full acc_video=.."
                ) from e

            try:
                locked2 = getattr(getattr(cfg, "stable_hw", None), "locked_acc_ref", None)
                freeze_ep = int(getattr(locked2, "freeze_epoch", 0) or 0) if locked2 is not None else 0
                curve2 = stable_hw_state.get("acc_ref_curve", None)
                if isinstance(curve2, list) and len(curve2) > 0:
                    logger.info(
                        "[LockedAccRef] preflight ok: source=%s curve_len=%s freeze_epoch=%s",
                        str(stable_hw_state.get("acc_ref_source", "")),
                        int(len(curve2)),
                        int(freeze_ep),
                    )
            except Exception:
                pass
        (out_dir / "stable_hw_state.json").write_text(
            safe_dumps(stable_hw_state, indent=2),
            encoding="utf-8",
        )
        stable_hw_state.setdefault("gating_reason_code", "")
        requested_cfg = _oc_select(cfg, "_contract.requested_config_snapshot", {}) or {}
        effective_cfg = OmegaConf.to_container(cfg, resolve=True)
        contract_overrides = _oc_select(cfg, "_contract.overrides", []) or []

        def _get_req(path, default=None):
            cur = requested_cfg
            for key in path.split("."):
                if not isinstance(cur, dict) or key not in cur:
                    return default
                cur = cur[key]
            return cur

        req_fb = _get_req("stable_hw.allow_train_ema_fallback", None)
        if req_fb is None:
            req_fb = _get_req("stable_hw.accuracy_guard.allow_train_ema_fallback", None)
        eff_fb = _oc_select(cfg, "stable_hw.allow_train_ema_fallback", None)
        if eff_fb is None:
            eff_fb = _oc_select(cfg, "stable_hw.accuracy_guard.allow_train_ema_fallback", None)

        trace_header_payload = build_trace_header_payload_v54(
            signature=signature,
            requested_config=requested_cfg,
            effective_config=effective_cfg,
            contract_overrides=contract_overrides,
            requested={
                "mode": "version_c",
                "stable_hw_enabled": bool(_get_req("stable_hw.enabled", None))
                if _get_req("stable_hw.enabled", None) is not None
                else None,
                "locked_acc_ref_enabled": bool(_get_req("stable_hw.locked_acc_ref.enabled", None))
                if _get_req("stable_hw.locked_acc_ref.enabled", None) is not None
                else None,
                "no_drift_enabled": bool(_get_req("stable_hw.no_drift.enabled", None))
                if _get_req("stable_hw.no_drift.enabled", None) is not None
                else None,
                "no_double_scale_enabled": bool(_get_req("stable_hw.no_double_scale.enabled", None))
                if _get_req("stable_hw.no_double_scale.enabled", None) is not None
                else None,
                "allow_train_ema_fallback": req_fb,
            },
            effective={
                "mode": "version_c",
                "stable_hw_enabled": bool(getattr(getattr(cfg, "stable_hw", None), "enabled", False)),
                "locked_acc_ref_enabled": bool(locked_acc_ref_enabled),
                "no_drift_enabled": bool(no_drift_enabled),
                "no_double_scale_enabled": bool(
                    getattr(getattr(getattr(cfg, "stable_hw", None), "no_double_scale", None), "enabled", False)
                ),
                "allow_train_ema_fallback": bool(eff_fb) if eff_fb is not None else None,
            },
            no_drift_enabled=bool(no_drift_enabled),
            acc_ref_source=str(lar_source),
            seal_digest=str(getattr(getattr(cfg, "contract", None), "seal_digest", "")),
        )
        trace_header_payload.update(
            {
                "stable_hw_effective": {
                    "enabled": bool(stable_hw_state.get("stable_hw_enabled", False)),
                    "no_drift_effective": bool(
                        stable_hw_state.get("no_drift_effective", stable_hw_state.get("no_drift_enabled", True))
                    ),
                    "acc_ref_source": stable_hw_state.get("acc_ref_source", ""),
                    "hw_ref_source": stable_hw_state.get("hw_ref_source", ""),
                    "lambda_hw_base": float(stable_hw_state.get("lambda_hw_base", 0.0)),
                    "lambda_hw_effective": float(stable_hw_state.get("lambda_hw_effective", 0.0)),
                    "allow_discrete_updates": bool(stable_hw_state.get("allow_discrete_updates", False)),
                    "gating_reason_code": stable_hw_state.get("gating_reason_code", ""),
                },
                "stable_hw_requested": {
                    "enabled": bool(_get_req("stable_hw.enabled", None))
                    if _get_req("stable_hw.enabled", None) is not None
                    else None,
                    "no_drift_enabled": bool(_get_req("stable_hw.no_drift.enabled", None))
                    if _get_req("stable_hw.no_drift.enabled", None) is not None
                    else None,
                    "lambda_hw": float(_get_req("hw.lambda_hw", 0.0))
                    if _get_req("hw.lambda_hw", None) is not None
                    else None,
                },
            }
        )
        append_trace_event_v54(
            trace_events_path,
            "trace_header",
            payload=trace_header_payload,
            run_id=run_id,
            step=0,
        )
        trace_header_written = True
        run_state: Dict[str, Any] = {
            "last_model_info": None,
            "alloc_layer_sens_ema": None,
            "alloc_last_keep_applied": None,
            "alloc_last_search": None,
            "alloc_incumbent_direction": None,
        }
        last_acc1: Optional[float] = None
        best_acc1: Optional[float] = None
        last_hw_stats = None
        ran_epochs = 0
        early_stop_triggered = False
        reason = "done"
        steps_done = 0
        best_solution_valid = True
        gating_epochs = 0
        freeze_epochs = 0
        total_epochs = 0
        try:
            init_ckpt_cfg_path = str(getattr(getattr(cfg, "training", None), "init_ckpt_path", "") or "")
            init_use_resume_epoch = bool(getattr(getattr(cfg, "training", None), "init_ckpt_use_resume_epoch", True))
            init_load_ema = bool(getattr(getattr(cfg, "training", None), "init_ckpt_load_ema", True))
            start_outer_init, best_from_init = maybe_init_from_checkpoint_version_c(
                init_ckpt_cfg_path,
                model,
                ema_model,
                logger,
                use_resume_epoch=bool(init_use_resume_epoch),
                load_ema=bool(init_load_ema),
            )
            start_outer_resume, best_from_ckpt = maybe_auto_resume_version_c(out_dir, model, ema_model, optimizer, scaler, logger)
            start_outer = int(max(int(start_outer_init), int(start_outer_resume)))
            if best_from_init is not None:
                best_acc1 = float(best_from_init)
            if best_from_ckpt is not None:
                best_acc1 = float(best_from_ckpt)
            global_step = int(start_outer) * int(steps_per_outer)
            ch3_ref_lat = os.environ.get("HW_REF_LAT_MS", "")
            ch3_ref_mem = os.environ.get("HW_REF_MEM_MB", "")
            ch3_ref_comm = os.environ.get("HW_REF_COMM_MS", "")
            logger.info("[CH3 common refs] lat=%s mem=%s comm=%s", ch3_ref_lat, ch3_ref_mem, ch3_ref_comm)
            for outer in range(start_outer, cfg.training.outer_epochs):
                assert_cfg_sealed_or_violate(cfg, seal_digest, trace_events_path, step=outer)
                backbone_trainable_now = bool(int(outer) >= int(freeze_backbone_epochs))
                freeze_state = _set_backbone_trainable(model, trainable=backbone_trainable_now)
                if run_state.get("_backbone_trainable_last") is None or bool(run_state.get("_backbone_trainable_last")) != bool(backbone_trainable_now):
                    logger.info(
                        "[FT] outer=%d backbone_trainable=%s freeze_backbone_epochs=%d affected_tensors=%d affected_params=%d",
                        int(outer),
                        bool(backbone_trainable_now),
                        int(freeze_backbone_epochs),
                        int(freeze_state.get("tensors", 0)),
                        int(freeze_state.get("params", 0)),
                    )
                    run_state["_backbone_trainable_last"] = bool(backbone_trainable_now)
                run_state["backbone_trainable"] = bool(backbone_trainable_now)
                ran_epochs += 1
                total_epochs += 1
                # Per-outer warning deltas (keeps stdout small but still auditable).
                amp_no_grads_total_before = int(run_state.get("warn_amp_alpha_no_grads_total", 0))
                amp_no_grads_logged_before = int(run_state.get("warn_amp_alpha_no_grads_logged", 0))
                nan_repair_total_before = int(run_state.get("warn_nan_repair_total", 0))
                nan_repair_logged_before = int(run_state.get("warn_nan_repair_logged", 0))
                nan_loss_total_before = int(run_state.get("warn_nan_total_loss_total", 0))
                nan_loss_logged_before = int(run_state.get("warn_nan_total_loss_logged", 0))
                stable_hw_enabled = bool(getattr(stable_hw_cfg, "enabled", True)) if stable_hw_cfg else False
                guard_controls_enabled = bool(stable_hw_enabled and _stable_hw_guard_controls_enabled(cfg))
                if stable_hw_enabled:
                    legacy_loss_lambda = float(getattr(getattr(cfg, "loss", None), "lambda_hw", 0.0) or 0.0)
                    legacy_hw_lambda = float(getattr(getattr(cfg, "hw", None), "lambda_hw", 0.0) or 0.0)
                    if (legacy_loss_lambda != 0.0 or legacy_hw_lambda != 0.0) and not stable_hw_state.get(
                        "_legacy_lambda_warned", False
                    ):
                        logger.info(
                            "[StableHW] NOTE: legacy cfg.loss.lambda_hw/cfg.hw.lambda_hw will be ignored; "
                            "using stable_hw_state.lambda_hw_effective."
                        )
                        stable_hw_state["_legacy_lambda_warned"] = True
                    # ---- v5.4 StableHW schedule ----
                    stable_hw_schedule(outer, stable_hw_cfg, stable_hw_state)

                    # ===== StableHW: v5.4 Acc-First Hard Gating (single source of truth) =====
                    metric_key = get_accuracy_metric_key(cfg)

                    # make sure acc_ref exists if LockedAccRef is enabled (may still be None until set)
                    if stable_hw_state.get("acc_ref") is None:
                        init_locked_acc_ref(cfg, stable_hw_state)

                    # guard based on *previous* metric (v5.4)
                    prev_val = stable_hw_state.get("val_acc1_last", None)
                    prev_train_ema = stable_hw_state.get("train_acc1_ema", None)

                    mk = str(metric_key).lower().strip()
                    use_train_ema = mk in ("train_acc1_ema", "train_ema")

                    # IMPORTANT: even when prev_val is None (first epoch), apply_accuracy_guard()
                    # will set guard_mode=WARMUP and allow_discrete_updates=True by contract.
                    if guard_controls_enabled:
                        stable_decision, allow_discrete = apply_accuracy_guard(
                            epoch=outer,
                            stable_hw_cfg=cfg,
                            stable_hw_state=stable_hw_state,
                            val_metric_or_none=float(prev_val) if (not use_train_ema and prev_val is not None) else None,
                            has_val_this_epoch=bool((not use_train_ema) and (prev_val is not None)),
                            train_ema_or_none=float(prev_train_ema) if (use_train_ema and prev_train_ema is not None) else None,
                        )
                        stable_hw_state = stable_decision.state
                        stable_hw_state["allow_discrete_updates"] = bool(allow_discrete)
                        if str(stable_hw_state.get("guard_mode", "")).upper() != "HW_OPT":
                            gating_epochs += 1
                        if not bool(stable_hw_state.get("allow_discrete_updates", True)):
                            freeze_epochs += 1
                    else:
                        stable_hw_state["allow_discrete_updates"] = True
                        stable_hw_state["guard_mode"] = "HW_OPT"
                        stable_hw_state["freeze_schedule"] = False
                    # ---- invariants (v5.4) ----
                    if stable_hw_state.get("acc_ref") is not None:
                        cur = float(stable_hw_state["acc_ref"])
                        prev = stable_hw_state.get("_acc_ref_once", None)
                        locked_ep = stable_hw_state.get("acc_ref_locked_epoch", None)
                        # allow one update on the locking epoch (warmup->locked transition), or for dynamic/curve refs
                        allow_dynamic = bool(stable_hw_state.get("acc_ref_dynamic", False))
                        if prev is None:
                            stable_hw_state["_acc_ref_once"] = cur
                        else:
                            prevf = float(prev)
                            if abs(prevf - cur) > 1e-9:
                                if (locked_ep is not None and int(locked_ep) == int(outer)) or allow_dynamic:
                                    stable_hw_state["_acc_ref_once"] = cur
                                else:
                                    logger.warning(
                                        f"[StableHW] acc_ref changed (prev={prevf:.6f}, cur={cur:.6f}). "
                                        "Resetting _acc_ref_once to avoid crash. "
                                        "If this is unexpected, inspect locked_acc_ref/curve settings."
                                    )
                                    stable_hw_state["_acc_ref_once"] = cur
                    # ---- v5.4 restart window: apply lr_restart_mul once per restart epoch ----
                    if stable_hw_enabled and bool(stable_hw_state.get("request_lr_restart", False)):
                        last_applied = int(stable_hw_state.get("_lr_restart_applied_epoch", -999999))
                        if last_applied != int(outer):
                            _ctrl = getattr(getattr(stable_hw_cfg, "accuracy_guard", None), "controller", None)
                            if _ctrl is None:
                                _ctrl = getattr(stable_hw_cfg, "controller", {})  # legacy fallback
                            mul = float(getattr(_ctrl, "lr_restart_mul", 2.0) or 2.0)
                            for pg in optimizer_model.param_groups:
                                pg["lr"] = float(pg.get("lr", lr)) * mul
                            stable_hw_state["_lr_restart_applied_epoch"] = int(outer)
                        stable_hw_state["request_lr_restart"] = False
                # ---- v5.4 canonical ----
                # stable_hw enabled: MUST use lambda_hw_effective written by schedule+guard
                # stable_hw disabled: fallback to legacy cfg.hw.lambda_hw (for ablations/baselines)
                if stable_hw_enabled:
                    kill_rem = int(stable_hw_state.get("hw_kill_remaining", 0) or 0)
                    if guard_controls_enabled and kill_rem > 0:
                        stable_hw_state["hw_kill_remaining"] = kill_rem - 1
                        stable_hw_state["lambda_hw_effective"] = 0.0
                        stable_hw_state["allow_discrete_updates"] = False
                        stable_hw_state["freeze_schedule"] = True
                        stable_hw_state["guard_mode"] = "RECOVERY"
                        logger.warning("[HWKill] active: force lambda_hw_eff=0 (remaining=%s)", kill_rem)
                    lambda_hw_eff = float(stable_hw_state.get("lambda_hw_effective", 0.0))
                else:
                    lambda_hw_eff = float(getattr(getattr(cfg, "hw", None), "lambda_hw", 0.0) or 0.0)

                # ---- use effective lambda ONLY (already gated) ----
                stable_hw_state["lambda_hw_effective"] = float(lambda_hw_eff)
                stable_hw_state.setdefault("lambda_hw_base", float(stable_hw_state.get("lambda_hw_base", 0.0)))

                start_rec = int(run_state.get("user_recovery_start_outer", 10**9))
                rem_rec = int(run_state.get("user_recovery_remaining", 0) or 0)
                user_rec_active = (rem_rec > 0) and (int(outer) >= int(start_rec))
                run_state["user_recovery_active"] = bool(user_rec_active)
                if guard_controls_enabled and user_rec_active:
                    stable_hw_state["freeze_schedule"] = True
                    stable_hw_state["in_recovery"] = True
                    stable_hw_state["guard_mode"] = "USER_RECOVERY"
                    _set_structural_gates_trainable(model, trainable=False)
                else:
                    if str(stable_hw_state.get("guard_mode", "")) == "USER_RECOVERY":
                        _set_structural_gates_trainable(model, trainable=True)
                        stable_hw_state.pop("in_recovery", None)
                        stable_hw_state.pop("freeze_schedule", None)
                        stable_hw_state.pop("guard_mode", None)
                # -------------------------
                # AST schedule (dense warmup -> ramp -> stabilize)
                # This affects token gating (rho/temperature) and lambda_AST multiplier only.
                # v5.4 fix: when StableHW enters VIOLATE/RECOVERY (freeze_schedule=True),
                # pause the AST schedule so token_keep does NOT keep dropping (otherwise recovery becomes impossible).
                # -------------------------
                # Ensure the virtual epoch is aligned to resume start point.
                stable_hw_state.setdefault("ast_sched_virtual_epoch", int(outer))
                ast_sched, ast_epoch_used = compute_ast_schedule_effective_with_stable_hw_freeze(cfg, stable_hw_state, int(outer))
                freeze_now = bool(_stablehw_freeze_ast_now(stable_hw_state)) if guard_controls_enabled else False
                stable_hw_state["freeze_ast_schedule"] = bool(freeze_now)
                stable_hw_state["ast_sched_epoch_used"] = int(ast_epoch_used)

                lambda_ast_eff = float(getattr(getattr(cfg, "loss", None), "lambda_AST", 1.0) or 1.0)
                freeze_structural_gates_now = False
                if isinstance(ast_sched, dict) and ast_sched.get("phase") != "disabled":
                    lambda_ast_eff = float(ast_sched.get("lambda_ast", lambda_ast_eff))
                    _apply_ast_runtime_overrides_to_model(model, cfg, ast_sched)
                    if ema_model is not None:
                        _apply_ast_runtime_overrides_to_model(ema_model.ema, cfg, ast_sched)

                    freeze_structural_gates_now = bool(ast_sched.get("force_dense", False)) and bool(force_dense_freeze_structural_gates)
                    if freeze_structural_gates_now:
                        _force_open_structural_gates_(model, cfg)
                        if ema_model is not None:
                            _force_open_structural_gates_(ema_model.ema, cfg)

                    try:
                        _maybe_run_alloc_search_and_apply(
                            model=model,
                            cfg=cfg,
                            run_state=run_state,
                            outer=int(outer),
                            seed=int(seed),
                            ast_sched=ast_sched,
                            hw_proxy=hw_proxy,
                            wafer_layout=wafer_layout,
                            chiplet_slots=chiplet_slots,
                            logger=logger,
                        )
                    except Exception as _alloc_exc:
                        logger.warning("[AllocSearch] outer=%d failed with error: %s", int(outer), str(_alloc_exc))

                    try:
                        _maybe_light_project_ch_keep(
                            model=model,
                            cfg=cfg,
                            ast_sched=ast_sched,
                            outer=int(outer),
                            run_state=run_state,
                            logger=logger,
                        )
                    except Exception as _proj_exc:
                        logger.warning("[ProjKeep] outer=%d failed with error: %s", int(outer), str(_proj_exc))

                    # Log at key transition points and whenever we freeze/unfreeze (prevents blind 3-day runs).
                    sched0 = getattr(getattr(cfg, "ast", None), "schedule", None)
                    warm_e = int(getattr(sched0, "warmup_epochs", 0) or 0) if sched0 is not None else 0
                    if outer == start_outer or outer == warm_e or outer == (warm_e + 1) or freeze_now or (outer % 5 == 0):
                        logger.info(
                            "[ASTSchedule] outer=%s ast_epoch=%s phase=%s force_dense=%s rho_token=%.4f temp=%.4f lambda_ast=%.4f freeze=%s guard_mode=%s",
                            int(outer),
                            int(ast_epoch_used),
                            str(ast_sched.get("phase")),
                            bool(ast_sched.get("force_dense", False)),
                            float(ast_sched.get("rho_token", 1.0)),
                            float(ast_sched.get("token_temperature", 0.1)),
                            float(ast_sched.get("lambda_ast", lambda_ast_eff)),
                            bool(freeze_now),
                            str(stable_hw_state.get("guard_mode", "disabled")) if stable_hw_enabled else "disabled",
                        )
                    if outer == start_outer:
                        logger.info(
                            "[ASTSchedule] enabled: warmup=%s, ramp=%s, curve=%s",
                            getattr(sched0, "warmup_epochs", None),
                            getattr(sched0, "ramp_epochs", None),
                            getattr(sched0, "curve", None),
                        )
                allow_discrete_updates = (
                    bool(stable_hw_state.get("allow_discrete_updates", True)) if guard_controls_enabled else True
                )
                # v5 discrete update gating (allow_discrete_updates=False in RECOVERY):
                #   - partition/mapping updates
                #   - device mapping updates
                #   - layout optimization updates (layout_opt.step/track_live/refine)
                #   - any step that changes discrete assignment/signature
                iso = getattr(stable_hw_cfg, "discrete_isolation", None) if stable_hw_cfg else None
                map_every = int(getattr(iso, "mapping_update_every_epochs", 1) if iso else 1)
                lay_every = int(getattr(iso, "layout_update_every_epochs", 1) if iso else 1)

                cache = stable_hw_state.setdefault(
                    "discrete_cache",
                    {"mapping": None, "layout": None, "mapping_signature": None, "layout_signature": None, "hw_mats": {}},
                )
                stable_hw_state["discrete_frozen_init_mapping"] = False

                if guard_controls_enabled and (not allow_discrete_updates) and (cache.get("mapping") is None or cache.get("layout") is None):
                    gm = str(stable_hw_state.get("guard_mode", "")).upper()

                    # Only allow cache init during WARMUP (Acc-First, no HW influence anyway)
                    if gm == "WARMUP":
                        stable_hw_state["gating_reason_code"] = "warmup_cache_init"
                        allow_discrete_updates = True
                        stable_hw_state["allow_discrete_updates"] = True
                    else:
                        # v5.4: RECOVERY/VIOLATE must be read-only; cache-empty means run is not auditable => fail-fast
                        append_trace_event_v54(
                            trace_events_path,
                            "boundary",
                            payload={
                                "candidate_id": int(outer),
                                "boundary_type": "stable_hw_cache_empty",
                                "violated_fields": [
                                    "stable_hw.accuracy_guard.controller.freeze_discrete_updates",
                                    "stable_hw.discrete_isolation.cache_mapping_layout",
                                ],
                                "severity": "P0",
                                "action": "fail_fast",
                            },
                            run_id=run_id,
                            step=int(outer),
                        )
                        raise RuntimeError(
                            "[v5.4 P0] stable_hw guard_mode is not WARMUP but discrete cache is empty "
                            "(mapping/layout missing). This would silently bypass RECOVERY read-only semantics. "
                            "Fix: run warmup to populate cache, or resume from a run directory with valid cache."
                        )

                allow_discrete = allow_discrete_updates
                update_alpha = base_update_alpha and allow_discrete

                # ---- HW stabilize window: prevent discrete updates / alpha step right after HW enables ----
                hw_stabilize_epochs = int(getattr(cfg.training, "hw_stabilize_epochs", 2) or 2)
                hw_enabled_now = (not twostage) and (float(lambda_hw_eff) > 0.0)

                # record first outer when HW becomes active
                if hw_enabled_now and stable_hw_state.get("hw_first_outer", None) is None:
                    stable_hw_state["hw_first_outer"] = int(outer)

                hw_first_outer = stable_hw_state.get("hw_first_outer", None)
                in_hw_stabilize = (
                    hw_enabled_now
                    and hw_first_outer is not None
                    and int(outer) < int(hw_first_outer) + int(hw_stabilize_epochs)
                )

                if guard_controls_enabled and in_hw_stabilize:
                    # disable discrete updates and alpha step; reuse cached mapping/layout only
                    if (allow_discrete_updates or update_alpha) and (not suppress_stable_hw_debug_lines):
                        logger.warning(
                            "[HWGuard] stabilize window active (outer=%s first=%s len=%s): disable discrete updates + alpha",
                            int(outer),
                            int(hw_first_outer),
                            int(hw_stabilize_epochs),
                        )
                    allow_discrete_updates = False
                    allow_discrete = False
                    update_alpha = False
                    stable_hw_state["allow_discrete_updates"] = False
                    # IMPORTANT (v5.4):
                    # HW stabilize window is NOT a recovery state.
                    # It should ONLY pause discrete updates / alpha steps to avoid a spike right after HW enables.
                    # Do NOT force guard_mode=RECOVERY or freeze_schedule here; otherwise we incorrectly freeze
                    # the AST pruning schedule and cut lambda_hw_eff to 0 on the next outer (run becomes "locked").
                    stable_hw_state["hw_stabilizing"] = True
                    stable_hw_state["hw_stabilize_first_outer"] = int(hw_first_outer)
                    stable_hw_state["hw_stabilize_len"] = int(hw_stabilize_epochs)
                    stable_hw_state["gating_reason_code"] = "hw_stabilize_window"
                else:
                    stable_hw_state["hw_stabilizing"] = False

                mapping_updated = False
                layout_updated = False

                need_update_mapping = ((outer % map_every) == 0) or (cache["mapping"] is None)
                need_update_layout = ((outer % lay_every) == 0) or (cache["layout"] is None)

                if not allow_discrete_updates:
                    need_update_mapping = False
                    need_update_layout = False

                if stable_hw_enabled:
                    if stable_hw_state.get("lambda_hw_effective", 0.0) <= 0.0:
                        need_update_mapping = need_update_mapping and (cache["mapping"] is None)
                        need_update_layout = need_update_layout and (cache["layout"] is None)

                    if guard_controls_enabled and (need_update_mapping or need_update_layout) and (not allow_discrete_updates):
                        stable_hw_state["gating_reason_code"] = "discrete_updates_blocked"
                        if not suppress_stable_hw_debug_lines:
                            print("[StableHW] Discrete updates frozen; reuse cached mapping/layout this step.")
                        need_update_mapping = False
                        need_update_layout = False

                # ROI-Commit pre-gating: cooldown + accuracy margin gate (optional)
                roi_cfg = _roi_get_cfg(iso)
                roi_enabled = bool(_cfg_get(roi_cfg, "enabled", False)) and bool(stable_hw_enabled)
                if roi_enabled:
                    cd_until = int(stable_hw_state.get("roi_cooldown_until", -1) or -1)
                    if cd_until >= 0 and int(outer) < cd_until:
                        stable_hw_state["roi_cooldown_active"] = True
                        stable_hw_state["gating_reason_code"] = "roi_cooldown"
                        allow_discrete_updates = False
                        allow_discrete = False
                        update_alpha = False
                        stable_hw_state["allow_discrete_updates"] = False
                    else:
                        stable_hw_state["roi_cooldown_active"] = False

                    margin_last = float(stable_hw_state.get("acc_margin_last", 0.0) or 0.0)
                    min_margin = float(_cfg_get(roi_cfg, "min_margin", 0.0) or 0.0)
                    stable_hw_state["roi_margin_last"] = float(margin_last)
                    stable_hw_state["roi_min_margin"] = float(min_margin)

                    require_hw_active = bool(_cfg_get(roi_cfg, "require_hw_active", True))
                    if require_hw_active and float(stable_hw_state.get("lambda_hw_effective", 0.0) or 0.0) <= 0.0:
                        need_update_mapping = need_update_mapping and (cache["mapping"] is None)
                        need_update_layout = need_update_layout and (cache["layout"] is None)
                    elif margin_last < min_margin:
                        stable_hw_state["gating_reason_code"] = "roi_margin_low"
                        need_update_mapping = False
                        need_update_layout = False

                # NOTE:
                # ROI pre-gating (e.g., cooldown) can flip allow_discrete_updates AFTER we computed
                # need_update_mapping/need_update_layout. Re-apply the gate here so we never enter
                # mapping/layout discrete-update paths while the StableHW gate is closed.
                if not allow_discrete_updates:
                    need_update_mapping = False
                    need_update_layout = False

                if (not allow_discrete_updates) and cache["mapping"] is None:
                    stable_hw_state["discrete_frozen_init_mapping"] = True

                if need_update_mapping:
                    assert allow_discrete_updates, (
                        "StableHW gate closed (guard_mode=%s reason=%s): discrete updates must not run"
                        % (
                            str(stable_hw_state.get("guard_mode", "")),
                            str(stable_hw_state.get("gating_reason_code", "")),
                        )
                    )
                    roi_cfg_local = _roi_get_cfg(iso)
                    roi_enabled_local = bool(_cfg_get(roi_cfg_local, "enabled", False)) and bool(stable_hw_enabled)
                    old_mapping_res = cache.get("mapping")
                    old_sig = cache.get("mapping_signature")

                    # Prepare a stable model_info snapshot for discrete planning.
                    # When ROI is enabled, use EMA keep_factors to reduce noisy churn.
                    last_info = run_state.get("last_model_info")
                    model_info_for_discrete = last_info
                    if roi_enabled_local and isinstance(stable_hw_state.get("roi_keep_ema", None), dict):
                        model_info_for_discrete = {}
                        if isinstance(last_info, dict):
                            arch = last_info.get("arch")
                            if arch is None and isinstance(last_info.get("model_info"), dict):
                                arch = last_info["model_info"].get("arch")
                            if isinstance(arch, dict):
                                model_info_for_discrete["arch"] = arch
                        model_info_for_discrete["keep_factors"] = stable_hw_state.get("roi_keep_ema")

                    # Candidate generation: optionally try multiple mapping strategies and pick the best by ROI metric.
                    strategies = _cfg_get(roi_cfg_local, "candidate_strategies", None)
                    if not strategies:
                        strategies = [None]

                    cand_mapping_res = None
                    cand_sig = None
                    cand_strategy = None
                    cand_metric = None
                    cand_rank_score = None

                    # Pre-compute the old metric once (if ROI is active and we have a previous mapping).
                    old_metric = None
                    metric_key = str(_cfg_get(roi_cfg_local, "metric", "proxy_raw_latency_ms") or "proxy_raw_latency_ms")
                    if roi_enabled_local and (old_mapping_res is not None):
                        gm = str(stable_hw_state.get("guard_mode", "")).upper()
                        if (gm == "OK") and (not bool(stable_hw_state.get("hw_stabilizing", False))):
                            try:
                                old_metric, _, _ = _roi_eval_hw_metric(
                                    cfg=cfg, hw_proxy=hw_proxy, mapping_solver=mapping_solver, wafer_layout=wafer_layout,
                                    stable_hw_cfg=stable_hw_cfg, stable_hw_state=stable_hw_state, slot_out=slot_out,
                                    mapping_res=old_mapping_res, metric_key=metric_key,
                                )
                            except Exception:
                                old_metric = None

                    rank_use_score = False
                    rank_lambda_scale = 1.0
                    rank_lambda_min = 0.0
                    rank_lambda_max = 1.0e9
                    rank_risk_weight = 1.5
                    rank_margin0 = 0.01
                    rank_margin_scale = 6.0

                    if roi_enabled_local and (old_metric is not None):
                        try:
                            lag_cfg_rank = _cfg_get(roi_cfg_local, "lagrangian", {}) or {}
                            if not isinstance(lag_cfg_rank, dict):
                                try:
                                    lag_cfg_rank = dict(lag_cfg_rank)
                                except Exception:
                                    lag_cfg_rank = {}
                            rank_use_score = bool(
                                _cfg_get(lag_cfg_rank, "enabled", None)
                                if _cfg_get(lag_cfg_rank, "enabled", None) is not None
                                else _cfg_get(roi_cfg_local, "lagrangian_enabled", False)
                            )
                            rank_lambda_scale = float(_cfg_get(lag_cfg_rank, "rank_lambda_scale", _cfg_get(lag_cfg_rank, "lambda_scale", 30.0)) or 30.0)
                            rank_lambda_min = float(_cfg_get(lag_cfg_rank, "rank_lambda_min", _cfg_get(lag_cfg_rank, "lambda_min", 0.0)) or 0.0)
                            rank_lambda_max = float(_cfg_get(lag_cfg_rank, "rank_lambda_max", _cfg_get(lag_cfg_rank, "lambda_max", 1e9)) or 1e9)
                            rank_risk_weight = float(_cfg_get(lag_cfg_rank, "rank_risk_weight", _cfg_get(lag_cfg_rank, "risk_weight", 1.5)) or 1.5)
                            rank_margin0 = float(_cfg_get(lag_cfg_rank, "rank_margin0", _cfg_get(lag_cfg_rank, "margin0", 0.01)) or 0.01)
                            rank_margin_scale = float(_cfg_get(lag_cfg_rank, "rank_margin_scale", _cfg_get(lag_cfg_rank, "margin_scale", 6.0)) or 6.0)
                        except Exception:
                            rank_use_score = False

                    for stg in strategies:
                        cand = _solve_mapping_for_cache(
                            model=model,
                            chiplet_slots=chiplet_slots,
                            mapping_solver=mapping_solver,
                            hw_proxy=hw_proxy,
                            wafer_layout=wafer_layout,
                            partitioner=partitioner,
                            hw_cfg=cfg.hw,
                            model_info=model_info_for_discrete,
                            mapping_strategy=(str(stg) if stg is not None else None),
                        )
                        mval = None
                        if roi_enabled_local and (old_metric is not None):
                            try:
                                mval, _, _ = _roi_eval_hw_metric(
                                    cfg=cfg, hw_proxy=hw_proxy, mapping_solver=mapping_solver, wafer_layout=wafer_layout,
                                    stable_hw_cfg=stable_hw_cfg, stable_hw_state=stable_hw_state, slot_out=slot_out,
                                    mapping_res=cand, metric_key=metric_key,
                                )
                            except Exception:
                                mval = None

                        cand_rank_score_this = None
                        if rank_use_score and (mval is not None) and (old_mapping_res is not None):
                            try:
                                lam_eff_rank = float(stable_hw_state.get("lambda_hw_effective", 0.0) or 0.0)
                                lam_rank_use = float(lam_eff_rank) * float(rank_lambda_scale)
                                lam_rank_use = max(float(rank_lambda_min), min(float(lam_rank_use), float(rank_lambda_max)))

                                acc_drop_now_rank = float(stable_hw_state.get("acc_drop", 0.0) or 0.0)
                                eps_drop_rank = float(stable_hw_state.get("epsilon_drop", 0.0) or 0.0)
                                margin_rank = float(eps_drop_rank) - float(acc_drop_now_rank)
                                margin_risk_rank = 0.0
                                if float(rank_margin0) > 0.0:
                                    margin_risk_rank = max(0.0, (float(rank_margin0) - float(margin_rank)) / float(rank_margin0))

                                chg_rank = float(_roi_mapping_change_frac(old_mapping_res, cand))
                                base_unit_rank = float(old_metric) if (old_metric is not None and float(old_metric) > 0.0) else 1.0
                                penalty_rank = (
                                    float(rank_risk_weight)
                                    * float(base_unit_rank)
                                    * float(chg_rank)
                                    * (1.0 + float(rank_margin_scale) * float(margin_risk_rank))
                                )
                                cand_rank_score_this = float(mval) + float(lam_rank_use) * float(penalty_rank)

                                logger.info(
                                    "[ROICommit][rank] outer=%d strategy=%s metric=%s mval=%.6g rank_score=%.6g "
                                    "lam_eff=%.6g lam_rank_use=%.6g chg=%.4f margin=%.6g pen=%.6g",
                                    int(outer),
                                    str(stg) if stg is not None else str(getattr(cfg.hw, "mapping_strategy", "greedy_local")),
                                    str(metric_key),
                                    float(mval),
                                    float(cand_rank_score_this),
                                    float(lam_eff_rank),
                                    float(lam_rank_use),
                                    float(chg_rank),
                                    float(margin_rank),
                                    float(penalty_rank),
                                )
                            except Exception as _exc_rank:
                                cand_rank_score_this = None

                        if cand_mapping_res is None:
                            cand_mapping_res = cand
                            cand_metric = mval
                            cand_rank_score = cand_rank_score_this
                            cand_strategy = (str(stg) if stg is not None else str(getattr(cfg.hw, "mapping_strategy", "greedy_local")))
                        else:
                            use_replace = False
                            if rank_use_score and (cand_rank_score_this is not None) and (cand_rank_score is not None):
                                use_replace = (float(cand_rank_score_this) < float(cand_rank_score))
                            elif rank_use_score and (cand_rank_score_this is not None) and (cand_rank_score is None):
                                use_replace = True
                            elif (mval is not None) and (cand_metric is not None):
                                use_replace = (float(mval) < float(cand_metric))

                            if use_replace:
                                cand_mapping_res = cand
                                cand_metric = mval
                                cand_rank_score = cand_rank_score_this
                                cand_strategy = (str(stg) if stg is not None else str(getattr(cfg.hw, "mapping_strategy", "greedy_local")))

                        # If ROI isn't active yet, don't waste time scoring more candidates.
                        if (not roi_enabled_local) or (old_metric is None):
                            break

                    if cand_mapping_res is None:
                        cand_mapping_res = _solve_mapping_for_cache(
                            model=model,
                            chiplet_slots=chiplet_slots,
                            mapping_solver=mapping_solver,
                            hw_proxy=hw_proxy,
                            wafer_layout=wafer_layout,
                            partitioner=partitioner,
                            hw_cfg=cfg.hw,
                            model_info=model_info_for_discrete,
                        )
                        cand_strategy = str(getattr(cfg.hw, "mapping_strategy", "greedy_local"))

                    cand_sig = cand_mapping_res.get("mapping_sig") or cand_mapping_res.get("signature")

                    commit = True
                    roi_reason = "roi_disabled"
                    if roi_enabled_local and (old_mapping_res is not None):
                        gm = str(stable_hw_state.get("guard_mode", "")).upper()
                        if (gm == "OK") and (not bool(stable_hw_state.get("hw_stabilizing", False))):
                            metric_key = str(_cfg_get(roi_cfg_local, "metric", "proxy_raw_latency_ms") or "proxy_raw_latency_ms")
                            min_rel = float(_cfg_get(roi_cfg_local, "min_rel_improve", 0.01) or 0.01)
                            min_abs = float(_cfg_get(roi_cfg_local, "min_abs_improve", 0.0) or 0.0)
                            try:
                                om = old_metric
                                if om is None:
                                    om, _, _ = _roi_eval_hw_metric(
                                        cfg=cfg, hw_proxy=hw_proxy, mapping_solver=mapping_solver, wafer_layout=wafer_layout,
                                        stable_hw_cfg=stable_hw_cfg, stable_hw_state=stable_hw_state, slot_out=slot_out,
                                        mapping_res=old_mapping_res, metric_key=metric_key,
                                    )
                                nm = cand_metric
                                if nm is None:
                                    nm, _, _ = _roi_eval_hw_metric(
                                        cfg=cfg, hw_proxy=hw_proxy, mapping_solver=mapping_solver, wafer_layout=wafer_layout,
                                        stable_hw_cfg=stable_hw_cfg, stable_hw_state=stable_hw_state, slot_out=slot_out,
                                        mapping_res=cand_mapping_res, metric_key=metric_key,
                                    )
                                # ROI commit criterion: optionally use Lagrangian net score
                                # (hardware metric + lambda * candidate risk proxy).
                                # Read nested lagrangian config explicitly to avoid silent misses.
                                lag_cfg = _cfg_get(roi_cfg_local, "lagrangian", {}) or {}
                                if not isinstance(lag_cfg, dict):
                                    try:
                                        lag_cfg = dict(lag_cfg)
                                    except Exception:
                                        lag_cfg = {}

                                lag_enabled = bool(
                                    _cfg_get(lag_cfg, "enabled", None)
                                    if _cfg_get(lag_cfg, "enabled", None) is not None
                                    else _cfg_get(roi_cfg_local, "lagrangian_enabled", False)
                                )

                                # Make lambda influence intentionally strong to avoid identical trajectories.
                                lambda_scale = float(_cfg_get(lag_cfg, "lambda_scale", 30.0) or 30.0)
                                lambda_min = float(_cfg_get(lag_cfg, "lambda_min", 0.0) or 0.0)
                                lambda_max = float(_cfg_get(lag_cfg, "lambda_max", 1e9) or 1e9)
                                risk_weight = float(_cfg_get(lag_cfg, "risk_weight", 1.5) or 1.5)
                                margin0 = float(_cfg_get(lag_cfg, "margin0", 0.01) or 0.01)
                                margin_scale = float(_cfg_get(lag_cfg, "margin_scale", 6.0) or 6.0)

                                if lag_enabled and (not bool(stable_hw_state.get("_roi_lagrangian_cfg_logged", False))):
                                    logger.info(
                                        "[ROICommit] lagrangian enabled metric=%s lambda_scale=%.6g lambda_min=%.6g "
                                        "lambda_max=%.6g risk_weight=%.6g margin0=%.6g margin_scale=%.6g",
                                        str(metric_key),
                                        float(lambda_scale),
                                        float(lambda_min),
                                        float(lambda_max),
                                        float(risk_weight),
                                        float(margin0),
                                        float(margin_scale),
                                    )
                                    stable_hw_state["_roi_lagrangian_cfg_logged"] = True

                                # Baseline: compare pure hardware metric.
                                rel = float(_roi_rel_improve(om, nm))
                                commit = _roi_should_commit(
                                    old_metric=om,
                                    new_metric=nm,
                                    min_rel_improve=min_rel,
                                    min_abs_improve=min_abs,
                                )
                                abs_impr = float(om) - float(nm)

                                # Lagrangian mode: score = metric + lambda_use * penalty(candidate)
                                # penalty uses a cheap candidate-specific proxy (mapping change fraction) amplified near the accuracy boundary.
                                if lag_enabled:
                                    try:
                                        lam_eff = float(stable_hw_state.get("lambda_hw_effective", 0.0) or 0.0)
                                        lam_use = float(lam_eff) * float(lambda_scale)
                                        lam_use = max(float(lambda_min), min(float(lam_use), float(lambda_max)))

                                        acc_drop_now = float(stable_hw_state.get("acc_drop", 0.0) or 0.0)
                                        eps_drop = float(stable_hw_state.get("epsilon_drop", 0.0) or 0.0)
                                        margin = float(eps_drop) - float(acc_drop_now)
                                        margin_risk = 0.0
                                        if float(margin0) > 0.0:
                                            margin_risk = max(0.0, (float(margin0) - float(margin)) / float(margin0))

                                        chg = float(_roi_mapping_change_frac(old_mapping_res, cand_mapping_res))
                                        # Scale penalty to the same unit as the metric (hardware loss) to make lambda effect visible.
                                        base_unit = float(om) if (om is not None and float(om) > 0.0) else 1.0
                                        penalty = float(risk_weight) * float(base_unit) * float(chg) * (1.0 + float(margin_scale) * float(margin_risk))

                                        old_score = float(om)  # old mapping vs itself has 0 change-penalty
                                        new_score = float(nm) + float(lam_use) * float(penalty)

                                        abs_impr = float(old_score) - float(new_score)
                                        rel = float(_roi_rel_improve(old_score, new_score))
                                        commit = _roi_should_commit(
                                            old_metric=float(old_score),
                                            new_metric=float(new_score),
                                            min_rel_improve=min_rel,
                                            min_abs_improve=min_abs,
                                        )
                                        roi_reason = (
                                            f"metric={metric_key} old={om:.6g} new={nm:.6g} "
                                            f"score_old={old_score:.6g} score_new={new_score:.6g} "
                                            f"abs={abs_impr:.6g} rel={rel:.4f} "
                                            f"lam_eff={lam_eff:.6g} lam_use={lam_use:.6g} chg={chg:.4f} "
                                            f"margin={margin:.6g} pen={penalty:.6g} "
                                            f"min_abs={min_abs:.6g} min_rel={min_rel:.4f} strategy={cand_strategy} "
                                            f"rank_score={(cand_rank_score if cand_rank_score is not None else float('nan')):.6g}"
                                        )
                                        stable_hw_state["roi_last_eval"] = {
                                            "outer": int(outer),
                                            "metric": metric_key,
                                            "old": float(om),
                                            "new": float(nm),
                                            "old_score": float(old_score),
                                            "new_score": float(new_score),
                                            "penalty": float(penalty),
                                            "chg_frac": float(chg),
                                            "margin": float(margin),
                                            "lam_eff": float(lam_eff),
                                            "lam_use": float(lam_use),
                                            "abs_improve": float(abs_impr),
                                            "min_abs": float(min_abs),
                                            "rel_improve": float(rel),
                                            "min_rel": float(min_rel),
                                            "commit": bool(commit),
                                        }
                                    except Exception as _exc_lag:
                                        # Fallback to metric-only decision if anything goes wrong.
                                        roi_reason = (
                                            f"metric={metric_key} old={om:.6g} new={nm:.6g} "
                                            f"abs={abs_impr:.6g} rel={rel:.4f} "
                                            f"min_abs={min_abs:.6g} min_rel={min_rel:.4f} "
                                            f"strategy={cand_strategy} lag_err={_exc_lag}"
                                        )
                                        stable_hw_state["roi_last_eval"] = {
                                            "outer": int(outer),
                                            "metric": metric_key,
                                            "old": float(om),
                                            "new": float(nm),
                                            "abs_improve": float(abs_impr),
                                            "min_abs": float(min_abs),
                                            "rel_improve": float(rel),
                                            "min_rel": float(min_rel),
                                            "commit": bool(commit),
                                        }
                                else:
                                    roi_reason = (
                                        f"metric={metric_key} old={om:.6g} new={nm:.6g} "
                                        f"abs={abs_impr:.6g} rel={rel:.4f} min_abs={min_abs:.6g} min_rel={min_rel:.4f} strategy={cand_strategy}"
                                    )
                                    stable_hw_state["roi_last_eval"] = {
                                        "outer": int(outer),
                                        "metric": metric_key,
                                        "old": float(om),
                                        "new": float(nm),
                                        "abs_improve": float(abs_impr),
                                        "min_abs": float(min_abs),
                                        "rel_improve": float(rel),
                                        "min_rel": float(min_rel),
                                        "commit": bool(commit),
                                    }
                                # Record a short decision window for ACHO feedback.
                                w = int(_cfg_get(roi_cfg_local, "window_size", 10) or 10)
                                win = stable_hw_state.get("roi_decisions_window", [])
                                if not isinstance(win, list):
                                    win = []
                                win.append(bool(commit))
                                if w > 0 and len(win) > w:
                                    win = win[-w:]
                                stable_hw_state["roi_decisions_window"] = win
                                try:
                                    stable_hw_state["roi_reject_rate"] = float(1.0 - (sum(1 for x in win if x) / max(1.0, float(len(win)))))
                                except Exception:
                                    pass
                            except Exception as exc:
                                commit = True
                                roi_reason = f"roi_eval_error:{exc}"
                        else:
                            commit = False
                            roi_reason = f"guard_mode_or_stabilize_blocked:{gm}"

                    if old_mapping_res is None:
                        commit = True
                        roi_reason = "cache_init"

                    if commit:
                        cache["mapping"] = cand_mapping_res
                        cache["mapping_signature"] = cand_sig
                        mapping_res = cand_mapping_res
                        mapping_updated = bool(old_sig is None or (str(cand_sig) != str(old_sig)))
                        if roi_enabled_local and mapping_updated:
                            base_cd = int(_cfg_get(roi_cfg_local, "cooldown_epochs", 2) or 2)
                            scale = int(_cfg_get(roi_cfg_local, "cooldown_scale", 0) or 0)
                            max_extra = int(_cfg_get(roi_cfg_local, "cooldown_max_extra", 0) or 0)
                            rel = 0.0
                            try:
                                rel = float(stable_hw_state.get("roi_last_eval", {}).get("rel_improve", 0.0))
                            except Exception:
                                rel = 0.0
                            extra = 0
                            if scale > 0:
                                extra = int(math.ceil(max(0.0, rel) * float(scale)))
                                if max_extra > 0:
                                    extra = min(extra, int(max_extra))
                            cd = int(base_cd) + int(extra)
                            stable_hw_state["roi_cooldown_until"] = int(outer) + int(cd)
                            stable_hw_state["roi_last_commit_outer"] = int(outer)
                            stable_hw_state["roi_last_commit_reason"] = str(roi_reason)
                            stable_hw_state["roi_last_commit_rel_improve"] = float(rel)
                            stable_hw_state["roi_last_commit_cooldown"] = int(cd)
                            logger.info("[ROICommit] mapping COMMIT outer=%s %s", int(outer), str(roi_reason))
                    else:
                        mapping_res = old_mapping_res
                        mapping_updated = False
                        stable_hw_state["roi_last_reject_outer"] = int(outer)
                        stable_hw_state["roi_last_reject_reason"] = str(roi_reason)
                        if roi_enabled_local:
                            logger.info("[ROICommit] mapping REJECT outer=%s %s", int(outer), str(roi_reason))
                else:
                    mapping_res = cache["mapping"]
                    mapping_updated = False

                if mapping_res is None:
                    raise RuntimeError("Mapping cache is empty after mapping step (mapping_res is None).")

                if need_update_layout:
                    assert allow_discrete_updates, (
                        "StableHW gate closed (guard_mode=%s reason=%s): discrete updates must not run"
                        % (
                            str(stable_hw_state.get("guard_mode", "")),
                            str(stable_hw_state.get("gating_reason_code", "")),
                        )
                    )
                    old_layout_sig = cache.get("layout_signature")
                    layout_res = _solve_layout_for_cache(
                        chiplet_slots=chiplet_slots,
                        wafer_layout=wafer_layout,
                        hw_cfg=cfg.hw,
                        mapping_result=mapping_res,
                    )
                    cache["layout"] = layout_res
                    cache["layout_signature"] = layout_res.get("signature")
                    layout_updated = bool(old_layout_sig is None or (str(cache["layout_signature"]) != str(old_layout_sig)))
                else:
                    layout_res = cache["layout"]
                    layout_updated = False

                if layout_res is None:
                    raise RuntimeError("Layout cache is empty after layout step (layout_res is None).")
                tau = max(cfg.chiplet.tau_min, cfg.chiplet.tau_init * (cfg.chiplet.tau_decay ** outer))
                chiplet_slots.set_tau(tau)
                last_hw_stats = None
                sum_latency = 0.0
                sum_energy = 0.0
                sum_mem = 0.0
                sum_comm = 0.0
                hw_stats_count = 0
                grad_clip_norm = float(getattr(cfg.train, "grad_clip_norm", 0.0) or 0.0)
                inner_steps_ast = int(getattr(cfg.training, "inner_steps_ast", 0) or 0)

                def _next_batch():
                    nonlocal data_iter
                    try:
                        return next(data_iter)
                    except StopIteration:
                        data_iter = iter(loader)
                        return next(data_iter)

                if inner_steps_ast <= 0:
                    step_iter = enumerate(loader)
                else:
                    step_iter = ((i, _next_batch()) for i in range(inner_steps_ast))

                # Ensure training mode (eval_acc1 sets model.eval(); must switch back)
                model.train()
                chiplet_slots.train()

                for step, batch in step_iter:
                    x = batch["video"].to(device)
                    y = batch["label"].to(device)
                    mixup_enabled = (
                        mixup_alpha > 0.0
                        and (random.random() < mixup_prob)
                        and (mixup_switch_off_epoch < 0 or outer < mixup_switch_off_epoch)
                    )
                    y_a = y
                    y_b = y
                    mixup_lam = 1.0
                    audio = batch.get("audio", None)
                    if mixup_enabled:
                        mixup_lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                        perm = torch.randperm(x.size(0), device=x.device)
                        x = x * mixup_lam + x[perm] * (1.0 - mixup_lam)
                        if audio is not None:
                            audio = audio.to(device)
                            audio = audio * mixup_lam + audio[perm] * (1.0 - mixup_lam)
                        y_b = y[perm]
                    if lr_schedule == "cosine":
                        lr_cur = _compute_lr(global_step)
                        for pg in optimizer_model.param_groups:
                            pg["lr"] = lr_cur * float(pg.get("lr_scale", 1.0) or 1.0)
                        if optimizer_alpha is not None:
                            for pg in optimizer_alpha.param_groups:
                                pg["lr"] = lr_cur
                    optimizer_model.zero_grad()
                    if optimizer_alpha is not None:
                        optimizer_alpha.zero_grad()
                    with autocast(device_type, enabled=amp_enabled, dtype=amp_dtype):
                        if model_type == "video_audio":
                            if audio is None:
                                audio = batch["audio"]
                            audio = audio.to(device)
                            logits, info = model(x, audio, return_intermediate=True)
                        else:
                            logits, info = model(x, return_intermediate=True)
                        # DP gather: reduce stacked per-GPU scalars/vectors in info/model_info back to expected shapes.
                        if isinstance(info, dict) and torch.cuda.is_available() and int(torch.cuda.device_count()) > 1:
                            try:
                                ng = int(torch.cuda.device_count())

                                def _reduce_dp(v):
                                    if torch.is_tensor(v) and v.numel() > 1:
                                        if v.ndim >= 1 and int(v.shape[0]) == ng:
                                            return v.mean(dim=0)
                                        return v.float().mean()
                                    return v

                                if torch.is_tensor(info.get("L_AST", None)):
                                    info["L_AST"] = _reduce_dp(info["L_AST"])

                                mi = info.get("model_info", None)
                                if isinstance(mi, dict):
                                    for k in (
                                        "token_keep",
                                        "head_keep",
                                        "ch_keep",
                                        "block_keep",
                                        "seq_len_total",
                                        "seq_len_effective",
                                        "est_attn_flops_ratio",
                                        "est_token_linear_flops_ratio",
                                    ):
                                        if k in mi:
                                            mi[k] = _reduce_dp(mi[k])

                                    for kk in ("keep_factors_t", "keep_factors"):
                                        kfd = mi.get(kk, None)
                                        if isinstance(kfd, dict):
                                            for name in ("token_keep", "head_keep", "ch_keep", "block_keep"):
                                                if name in kfd:
                                                    kfd[name] = _reduce_dp(kfd[name])
                            except Exception:
                                pass
                        run_state["last_model_info"] = info
                        if stable_hw_enabled:
                            # Stabilize discrete planning with EMA keep signals.
                            ema_a = 0.2
                            try:
                                iso_cfg = getattr(getattr(cfg, "stable_hw", None), "discrete_isolation", None)
                                roi_cfg = getattr(iso_cfg, "roi_commit", None) if iso_cfg is not None else None
                                if roi_cfg is not None:
                                    ema_a = float(getattr(roi_cfg, "keep_ema_alpha", 0.2) or 0.2)
                            except Exception:
                                ema_a = 0.2
                            _roi_update_keep_ema(stable_hw_state, info, ema_alpha=ema_a)
                        # Hard guard: if model produced non-finite logits, skip the step.
                        if not torch.isfinite(logits).all():
                            run_state["nan_guard_skipped_steps"] = int(run_state.get("nan_guard_skipped_steps", 0)) + 1
                            logger.warning("[NaNGuard] non-finite logits detected (outer=%s step=%s); skipping step.", outer, step)
                            optimizer_model.zero_grad(set_to_none=True)
                            if optimizer_alpha is not None:
                                optimizer_alpha.zero_grad(set_to_none=True)
                            continue
                        if mixup_enabled:
                            L_task = (
                                mixup_lam * F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
                                + (1.0 - mixup_lam) * F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)
                            )
                        else:
                            L_task = F.cross_entropy(logits, y, label_smoothing=label_smoothing)
                        model_info = info.get("model_info", {}) if isinstance(info, dict) else {}
                        slot_out = chiplet_slots(hard=False)
                        alpha = slot_out["alpha"]
                        eff_specs = slot_out["eff_specs"]

                        segments_cached = mapping_res.get("segments", []) if mapping_res else []
                        mapping_cached = mapping_res.get("mapping", []) if mapping_res else []

                        track_live = False
                        use_cached_mapping = True
                        use_cached_layout = True
                        iso_cfg = None
                        if stable_hw_cfg:
                            iso_cfg = getattr(stable_hw_cfg, "discrete_isolation", None)
                            if iso_cfg is not None:
                                track_live = bool(getattr(iso_cfg, "track_live_segments", False))
                                use_cached_mapping = bool(getattr(iso_cfg, "use_cached_mapping_for_inner_steps", True))
                                use_cached_layout = bool(getattr(iso_cfg, "use_cached_layout_for_inner_steps", True))
                        if not allow_discrete:
                            track_live = False
                            use_cached_mapping = True
                            use_cached_layout = True

                        segments_for_hw = segments_cached
                        mapping_for_hw = mapping_cached
                        step_mapping_updated = mapping_updated
                        step_layout_updated = layout_updated
                        if allow_discrete_updates and track_live and iso_cfg is not None:
                            track_every = int(getattr(iso_cfg, "track_live_every_steps", 1) or 1)
                            if (step % track_every) == 0:
                                if not use_cached_mapping:
                                    assert bool(stable_hw_state.get("allow_discrete_updates", True)), (
                                        "StableHW gate closed: discrete updates must not run in RECOVERY/WARMUP"
                                    )
                                    part_res = partitioner.plan(
                                        model,
                                        eff_specs,
                                        alpha=chiplet_slots()["alpha"],
                                        model_info=run_state.get("last_model_info"),
                                        use_fine_split=bool(getattr(cfg.hw, "use_fine_split", True)),
                                    )
                                    segments_for_hw = part_res["segments"]

                                    mapping_res_live = mapping_solver.solve_mapping(
                                        segments_for_hw,
                                        eff_specs,
                                        hw_proxy,
                                        layout_positions=wafer_layout.current_pos_continuous(),
                                        strategy=str(getattr(cfg.hw, "mapping_strategy", "greedy_local")),
                                        distance_scale_ms=float(getattr(cfg.hw, "distance_scale_ms", 0.0) or 0.0),
                                        alpha=chiplet_slots()["alpha"],
                                    )
                                    mapping_for_hw = mapping_res_live.get("mapping", [])
                                    stable_hw_state["discrete_cache"]["mapping_signature"] = str(
                                        stable_hash(
                                            {
                                                "mapping": [int(x) for x in mapping_for_hw],
                                                "segments": [
                                                    {
                                                        "k": int(i),
                                                        "flops": float(getattr(seg, "flops", 0.0)),
                                                        "bytes": float(getattr(seg, "bytes", 0.0)),
                                                        "traffic": float(getattr(seg, "traffic_out_bytes", 0.0)),
                                                        "mem": float(getattr(seg, "mem_mb", 0.0)),
                                                    }
                                                    for i, seg in enumerate(segments_for_hw)
                                                ],
                                            }
                                        )
                                    )
                                    stable_hw_state["discrete_cache"]["layout_signature"] = str(
                                        getattr(wafer_layout, "signature", lambda: "unknown")()
                                    )
                                    last_segments = segments_for_hw
                                    last_mapping = mapping_for_hw
                                    step_mapping_updated = True
                                if not use_cached_layout:
                                    assert bool(stable_hw_state.get("allow_discrete_updates", True)), (
                                        "StableHW gate closed: discrete updates must not run in RECOVERY/WARMUP"
                                    )
                                    layout_res = _solve_layout_for_cache(
                                        chiplet_slots=chiplet_slots,
                                        wafer_layout=wafer_layout,
                                        hw_cfg=cfg.hw,
                                        mapping_result={
                                            "segments": segments_for_hw,
                                            "mapping": mapping_for_hw,
                                        },
                                    )
                                    cache["layout"] = layout_res
                                    cache["layout_signature"] = layout_res.get("signature")
                                    step_layout_updated = True

                        if (
                            allow_discrete_updates
                            and layout_opt is not None
                            and bool(_get_iso_cfg_value(iso_cfg, "optimize_layout", False))
                        ):
                            if step_layout_updated and segments_for_hw and mapping_for_hw:
                                prev_requires = {}
                                for n, p in model.named_parameters():
                                    prev_requires[n] = p.requires_grad
                                    p.requires_grad_(False)

                                wafer_layout.train()
                                for _k in range(layout_opt_steps):
                                    layout_opt.zero_grad(set_to_none=True)
                                    out = wafer_layout.forward(
                                        mapping=mapping_for_hw,
                                        segments=segments_for_hw,
                                        eff_specs=eff_specs,
                                        lambda_boundary=cfg.hw.lambda_boundary,
                                        lambda_overlap=cfg.hw.lambda_overlap,
                                        lambda_comm=cfg.hw.lambda_comm_extra,
                                        lambda_thermal=cfg.hw.lambda_thermal,
                                        distance_scale=float(getattr(cfg.hw, "distance_scale_ms", 0.0)),
                                    )
                                    loss_layout = out[0] if isinstance(out, (tuple, list)) else out["total"]
                                    loss_layout.backward()
                                    torch.nn.utils.clip_grad_norm_([wafer_layout.pos], max_norm=layout_opt_grad_clip)
                                    layout_opt.step()

                                for n, p in model.named_parameters():
                                    p.requires_grad_(prev_requires[n])

                                cache["layout_signature"] = signature_for_assign(getattr(wafer_layout, "assign", None))
                                cache["layout"] = {"signature": cache["layout_signature"]}

                        layout_positions = wafer_layout.current_pos_continuous()

                        segments_sig = None
                        try:
                            segments_sig = stable_hash(
                                [getattr(s, "signature", None) or repr(s) for s in (segments_for_hw or [])]
                            )
                        except Exception:
                            segments_sig = None

                        mapping_sig_for_hw = None
                        if (segments_for_hw is segments_cached) or (segments_for_hw == segments_cached):
                            mapping_sig_for_hw = (
                                (mapping_res.get("mapping_sig") or mapping_res.get("signature")) if mapping_res else None
                            )
                        else:
                            mapping_sig_for_hw = None

                        compute_hw_when_lambda0 = bool(getattr(cfg.training, "compute_hw_when_lambda0", True))
                        if (float(lambda_hw_eff) <= 0.0) and (not compute_hw_when_lambda0):
                            # HW term is disabled => skipping HW proxy has NO effect on gradients.
                            L_hw = torch.zeros_like(L_task)
                            hw_stats = {
                                "proxy_raw_latency_ms": 0.0,
                                "raw_latency_ms": 0.0,
                                "energy_mj": 0.0,
                                "mem_mb": 0.0,
                                "comm_ms": 0.0,
                                "L_hw_total": 0.0,
                                "proxy_raw": {},
                                "proxy_used": {},
                                "proxy_had_invalid": False,
                            }
                        else:
                            L_hw, hw_stats = compute_hw_loss(
                                cfg,
                                hw_proxy,
                                model_info=model_info,
                                stable_hw_cfg=stable_hw_cfg,
                                stable_hw_state=stable_hw_state,
                                segments=segments_for_hw,
                                mapping=mapping_for_hw,
                                mapping_sig=mapping_sig_for_hw,
                                segments_sig=segments_sig,
                                eff_specs=eff_specs,
                                layout_positions=layout_positions,
                                mapping_solver=mapping_solver,
                                wafer_layout=wafer_layout,
                                alpha=alpha,
                            )
                        last_hw_stats = hw_stats
                        sum_latency += float(
                            hw_stats.get(
                                "raw_latency_ms",
                                hw_stats.get("proxy_raw_latency_ms", hw_stats.get("latency_ms", 0.0)),
                            )
                        )
                        sum_energy += float(hw_stats.get("energy_mj", 0.0))
                        sum_mem += float(hw_stats.get("mem_mb", 0.0))
                        sum_comm += float(hw_stats.get("comm_ms", 0.0))
                        hw_stats_count += 1
                        # v5.4 trace: proxy sanitize events (SPEC_E)
                        # 合同要求：每个被 sanitize 的 metric 都必须单独记录 raw_value / used_value / penalty_added
                        hw_proxy_raw = hw_stats.get("proxy_raw", {}) or {}
                        hw_proxy_used = hw_stats.get("proxy_used", {}) or {}
                        changed = bool(hw_stats.get("proxy_had_invalid", False))
                        if not changed:
                            for metric_name in hw_proxy_used:
                                raw_v = float(hw_proxy_raw.get(metric_name, hw_proxy_used[metric_name]))
                                used_v = float(hw_proxy_used[metric_name])
                                if abs(raw_v - used_v) > 1e-12:
                                    changed = True
                                    break
                        if changed:
                            hw_metrics = hw_proxy_raw
                            hw_used = hw_proxy_used
                            # ---- v5.4: trace proxy sanitize evidence (SPEC_E) ----
                            for metric in sorted(hw_metrics.keys()):
                                raw = float(hw_metrics[metric])
                                used = float(hw_used.get(metric, raw))
                                # IMPORTANT: penalty_added must reflect the actual penalty used in loss,
                                # not a derived (used-raw) heuristic.
                                pen_key = f"proxy_penalty_{metric}"
                                penalty_added = float(hw_stats.get(pen_key, 0.0) or 0.0)

                                append_trace_event_v54(
                                    trace_events_path,
                                    "proxy_sanitize",
                                    payload={
                                        "candidate_id": int(outer),  # v5.4: candidate id for audit (epoch-level candidate)
                                        "outer_iter": int(outer),
                                        "inner_step": int(step),
                                        "metric": str(metric),
                                        "raw_value": raw,
                                        "used_value": used,
                                        "penalty_added": penalty_added,
                                    },
                                    run_id=run_id,
                                    step=int(step),
                                )
                        # ---- NaN-safe HW term (avoid 0*NaN propagation) ----
                        hw_nonfinite = not torch.isfinite(L_hw).all()

                        acc_part_t = L_task + float(lambda_ast_eff) * info["L_AST"]

                        # ---- normalize L_hw to reduce scale mismatch / spikes ----
                        L_hw_clean = torch.nan_to_num(L_hw, nan=0.0, posinf=0.0, neginf=0.0)

                        hw_mag = 0.0
                        hw_mag_ema = float(stable_hw_state.get("hw_mag_ema", 1.0) or 1.0)
                        if hw_norm_enabled:
                            try:
                                hw_mag = float(L_hw_clean.detach().abs().mean().item())
                                hw_mag_ema = float(hw_norm_alpha) * float(hw_mag_ema) + (1.0 - float(hw_norm_alpha)) * float(hw_mag)
                                hw_mag_ema = max(float(hw_mag_ema), float(hw_norm_min_denom))
                                stable_hw_state["hw_mag_ema"] = float(hw_mag_ema)
                            except Exception:
                                hw_mag = 0.0
                                hw_mag_ema = max(float(hw_mag_ema), float(hw_norm_min_denom))
                                stable_hw_state["hw_mag_ema"] = float(hw_mag_ema)

                        if hw_norm_enabled:
                            L_hw_normed = L_hw_clean / float(hw_mag_ema)
                            if hw_norm_clip > 0:
                                L_hw_normed = L_hw_normed.clamp(min=-float(hw_norm_clip), max=float(hw_norm_clip))
                        else:
                            L_hw_normed = L_hw_clean

                        if twostage or float(lambda_hw_eff) <= 0.0:
                            hw_term = torch.zeros_like(L_hw_normed)
                            hw_scale = 1.0
                            hw_ratio = 0.0
                        else:
                            hw_term_raw = float(lambda_hw_eff) * L_hw_normed

                            # ratio cap: |hw_term| <= hw_ratio_cap * |acc_part|
                            denom = (acc_part_t.detach().abs() + 1.0e-6)
                            hw_ratio = float((hw_term_raw.detach().abs() / denom).clamp(min=0.0, max=1.0e6).item())
                            hw_scale = 1.0
                            if hw_ratio_cap > 0.0 and hw_ratio > hw_ratio_cap:
                                hw_scale = float(hw_ratio_cap / max(1e-9, hw_ratio))
                                hw_term_raw = hw_term_raw * hw_scale
                            hw_term = hw_term_raw

                        # keep for logging/audit
                        if isinstance(hw_stats, dict):
                            hw_stats["hw_mag"] = float(hw_mag)
                            hw_stats["hw_mag_ema"] = float(hw_mag_ema)
                            hw_stats["hw_norm_enabled"] = bool(hw_norm_enabled)
                            hw_stats["hw_norm_clip"] = float(hw_norm_clip)
                            hw_stats["hw_ratio"] = float(hw_ratio)
                            hw_stats["hw_ratio_cap"] = float(hw_ratio_cap)
                            hw_stats["hw_ratio_scale"] = float(hw_scale)

                        # If HW loss went non-finite, skip this step entirely (keeps model/alpha stable).
                        if hw_nonfinite:
                            run_state["nan_guard_skipped_steps"] = int(run_state.get("nan_guard_skipped_steps", 0)) + 1
                            logger.warning("[NaNGuard] non-finite L_hw detected (outer=%s step=%s); skipping step.", outer, step)
                            optimizer_model.zero_grad(set_to_none=True)
                            if optimizer_alpha is not None:
                                optimizer_alpha.zero_grad(set_to_none=True)
                            continue
                        loss = acc_part_t + hw_term
                        # DP: if any term was gathered as a vector (e.g., shape [ngpu]), loss becomes non-scalar.
                        if torch.is_tensor(loss) and loss.numel() > 1:
                            loss = loss.mean()
                        # ---- v5.4 audit: capture the exact loss components used ----
                        try:
                            ast_loss_val = info["L_AST"]
                        except Exception:
                            ast_loss_val = 0.0

                        def _to_f(x):
                            if torch.is_tensor(x):
                                # DataParallel gather may turn scalar tensors into vectors
                                # (e.g. shape [num_gpus]); reduce before item().
                                xd = x.detach()
                                if xd.numel() > 1:
                                    xd = xd.float().mean()
                                return float(xd.cpu().item())
                            if hasattr(x, "detach"):
                                return float(x.detach().cpu().item())
                            return float(x)

                        ast_term = _to_f(ast_loss_val) * float(lambda_ast_eff)
                        acc_part = _to_f(L_task) + float(ast_term)
                        run_state["last_loss_components"] = {
                            "acc_loss": _to_f(L_task),
                            "ast_loss": _to_f(ast_loss_val),
                            "acc_part": float(acc_part),
                            "ast_term": float(ast_term),
                            "hw_loss_raw": _to_f(L_hw),
                            "hw_loss_used": _to_f(hw_term),
                            "total_loss": _to_f(loss),
                        }
                        assert "hw_loss_weighted" not in (hw_stats or {}), (
                            "NoDoubleScale violated: hw_loss should not be weighted inside hw_loss module."
                        )
                    # v5.4 contract: NoDoubleScale (lambda_hw only applied once via stable_hw lambda_hw_eff)
                    assert "lambda_hw" not in str(type(L_hw)).lower()  # cheap guard (won't catch all, but prevents accidental wrapping)
                    assert float(lambda_hw_eff) >= 0.0

                    # Hard guard: skip if total loss becomes non-finite (e.g., rare NaN from HW/AST).
                    if not torch.isfinite(loss).all():
                        run_state["nan_guard_skipped_steps"] = int(run_state.get("nan_guard_skipped_steps", 0)) + 1
                        _warn_throttled(
                            "nan_total_loss",
                            step,
                            "[NaNGuard] non-finite total loss (outer=%s step=%s); skipping step.",
                            outer,
                            step,
                            every_steps=nan_guard_warn_every_steps,
                            first_n=3,
                        )
                        optimizer_model.zero_grad(set_to_none=True)
                        if optimizer_alpha is not None:
                            optimizer_alpha.zero_grad(set_to_none=True)
                        continue
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer_model)
                    try:
                        alloc_cfg = getattr(cfg, "alloc_search", None)
                        if alloc_cfg is not None and bool(getattr(alloc_cfg, "enabled", False)):
                            _update_alloc_layer_sens_ema(
                                model,
                                run_state,
                                ema_alpha=float(getattr(alloc_cfg, "sens_ema_alpha", 0.8)),
                            )
                    except Exception:
                        pass
                    bad_grad = False
                    for p in model.parameters():
                        if p.grad is not None and (not torch.isfinite(p.grad).all()):
                            bad_grad = True
                            break

                    if (not twostage) and (optimizer_alpha is not None) and update_alpha and (float(lambda_hw_eff) > 0.0) and _optimizer_has_any_grad(optimizer_alpha):
                        scaler.unscale_(optimizer_alpha)
                        for p in chiplet_slots.parameters():
                            if p.grad is not None and (not torch.isfinite(p.grad).all()):
                                bad_grad = True
                                break

                    if bad_grad:
                        run_state["nan_guard_skipped_steps"] = int(run_state.get("nan_guard_skipped_steps", 0)) + 1
                        stable_hw_state["hw_kill_remaining"] = max(
                            int(stable_hw_state.get("hw_kill_remaining", 0) or 0),
                            hw_kill_epochs,
                        )
                        _warn_throttled(
                            "nan_grad",
                            step,
                            "[HWKill] non-finite grads (outer=%s step=%s); skip step and kill HW for %s outers.",
                            outer,
                            step,
                            hw_kill_epochs,
                            every_steps=nan_guard_warn_every_steps,
                            first_n=3,
                        )
                        optimizer_model.zero_grad(set_to_none=True)
                        if optimizer_alpha is not None:
                            optimizer_alpha.zero_grad(set_to_none=True)
                        # IMPORTANT: unscale_ was called; must call scaler.update() before continuing
                        # to reset GradScaler per-optimizer state and (if inf detected) reduce the scale.
                        try:
                            scaler.update()
                        except Exception:
                            pass
                        continue

                    if freeze_structural_gates_now:
                        _zero_structural_gate_grads_(model)

                    if grad_clip_norm > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    scaler.step(optimizer_model)
                    if (not twostage) and (optimizer_alpha is not None) and update_alpha and (float(lambda_hw_eff) > 0.0):
                        # Chapter-3 fairness: optimizer_alpha is disabled by default.
                        if _optimizer_has_any_grad(optimizer_alpha):
                            if grad_clip_norm > 0.0:
                                torch.nn.utils.clip_grad_norm_(chiplet_slots.parameters(), grad_clip_norm)
                            scaler.step(optimizer_alpha)
                        # If HW enabled but still no grads, keep silent (avoid log bloat); audit is via stats/trace.
                    # v5.4: forbidden (P0-3) — layout must be updated via discrete assign-only agent + cache
                    scaler.update()
                    if freeze_structural_gates_now:
                        _force_open_structural_gates_(model, cfg)
                        if ema_model is not None:
                            _force_open_structural_gates_(ema_model.ema, cfg)
                    repaired = _repair_nonfinite_params_(model)
                    if update_alpha:
                        repaired = _repair_nonfinite_params_(chiplet_slots) or repaired
                    if repaired:
                        _warn_throttled(
                            "nan_repair",
                            step,
                            "[NaNGuard] repaired non-finite parameters (outer=%s step=%s).",
                            outer,
                            step,
                            every_steps=nan_guard_warn_every_steps,
                            first_n=3,
                        )
                        stable_hw_state["hw_kill_remaining"] = max(
                            int(stable_hw_state.get("hw_kill_remaining", 0) or 0),
                            hw_kill_epochs,
                        )
                    if ema_model is not None:
                        try:
                            ema_model.update(model)
                        except Exception:
                            pass
                    if step % log_interval_steps == 0:
                        with torch.no_grad():
                            if mixup_enabled:
                                top1_a = _topk_correct_frac(logits.detach(), y_a, 1)
                                top1_b = _topk_correct_frac(logits.detach(), y_b, 1)
                                acc1 = (mixup_lam * top1_a + (1.0 - mixup_lam) * top1_b).mean()
                            else:
                                acc1 = _topk_correct_frac(logits.detach(), y, 1).mean()
                        last_acc1 = float(acc1.item())
                        best_acc1 = float(acc1.item()) if best_acc1 is None else max(best_acc1, float(acc1.item()))
                        if stable_hw_enabled:
                            metric = get_accuracy_metric_key(stable_hw_cfg)
                            if metric in ("train_acc1_ema", "train_ema"):
                                update_train_acc1_ema(stable_hw_cfg, stable_hw_state, float(acc1))
                        model_info = info.get("model_info", {}) if isinstance(info, dict) else {}
                        pruner = _get_ast_pruner(model)
                        if pruner is not None:
                            try:
                                pr_cfg = pruner.cfg.get("channel_prune", pruner.cfg)
                                freeze_prefix_ratio = float(pr_cfg.get("freeze_prefix_ratio", pruner.cfg.get("ch_freeze_prefix_ratio", 0.0)) or 0.0)
                                freeze_prefix_ratio = float(max(0.0, min(1.0, freeze_prefix_ratio)))
                            except Exception:
                                freeze_prefix_ratio = 0.0
                        else:
                            freeze_prefix_ratio = 0.0

                        keep_now_real = _get_layer_ch_keep_now(model)
                        keep_summary = _summarize_layer_ch_keep(keep_now_real, freeze_prefix_ratio=freeze_prefix_ratio)
                        alloc_last = run_state.get("alloc_last_search", None)
                        if not isinstance(alloc_last, dict):
                            alloc_last = {}

                        stats_full = {
                            "outer": outer,
                            "step": step,
                            "loss": loss.item(),
                            "acc1": acc1.item(),
                            "token_keep": _to_pyfloat(model_info.get("token_keep", 1.0), 1.0) if isinstance(model_info, dict) else 1.0,
                            "head_keep_real_mean": _to_pyfloat(model_info.get("head_keep", 1.0), 1.0) if isinstance(model_info, dict) else 1.0,
                            "ch_keep_real_mean": float(keep_summary.get("ch_keep_real_mean", 1.0)),
                            "ch_keep_real_min": float(keep_summary.get("ch_keep_real_min", 1.0)),
                            "ch_keep_real_max": float(keep_summary.get("ch_keep_real_max", 1.0)),
                            "ch_keep_prunable_mean": float(keep_summary.get("ch_keep_prunable_mean", 1.0)),
                            "ch_prune_ratio_real": 1.0 - float(keep_summary.get("ch_keep_prunable_mean", 1.0)),
                            "block_keep_real_mean": _to_pyfloat(model_info.get("block_keep", 1.0), 1.0) if isinstance(model_info, dict) else 1.0,
                            "ch_keep_target": float(ast_sched.get("ch_keep_target", 1.0)) if isinstance(ast_sched, dict) else 1.0,
                            "force_dense": bool(ast_sched.get("force_dense", False)) if isinstance(ast_sched, dict) else False,
                            "lambda_hw": float(lambda_hw_eff),
                            "guard_mode": str(stable_hw_state.get("guard_mode", "")) if stable_hw_enabled else "",
                            "freeze_schedule": bool(stable_hw_state.get("freeze_schedule", False)) if stable_hw_enabled else False,
                            "allow_discrete_updates": bool(allow_discrete),
                            "alloc_enabled_this_outer": bool(alloc_last.get("alloc_enabled_this_outer", False)),
                            "alloc_applied": bool(alloc_last.get("alloc_applied", False)),
                            "mapping_updated": step_mapping_updated,
                            "layout_updated": step_layout_updated,
                            "mapping_cache_hit": not step_mapping_updated,
                            "layout_cache_hit": not step_layout_updated,
                            "mapping_signature": cache["mapping_signature"],
                            "layout_signature": cache["layout_signature"],
                        }
                        if stats_full["alloc_enabled_this_outer"]:
                            stats_full.update(
                                {
                                    "alloc_controller": str(alloc_last.get("alloc_controller", "")),
                                    "alloc_selected_source": str(alloc_last.get("alloc_selected_source", "")),
                                    "alloc_selected_long_gain": float(alloc_last.get("alloc_selected_long_gain", 0.0)),
                                    "alloc_selected_apply_gain": float(alloc_last.get("alloc_selected_apply_gain", 0.0)),
                                    "alloc_switched": bool(alloc_last.get("alloc_switched", False)),
                                    "alloc_inc_age": int(alloc_last.get("alloc_inc_age", 0) or 0),
                                }
                            )
                        if hw_stats:
                            stats_full.update(hw_stats)

                        if suite_cleanup_enabled and slim_console_log:
                            stats_console = {
                                "outer": stats_full["outer"],
                                "step": stats_full["step"],
                                "loss": stats_full["loss"],
                                "acc1": stats_full["acc1"],
                                "token_keep": stats_full["token_keep"],
                                "ch_keep_real_mean": stats_full["ch_keep_real_mean"],
                                "ch_keep_real_min": stats_full["ch_keep_real_min"],
                                "ch_keep_real_max": stats_full["ch_keep_real_max"],
                                "ch_keep_prunable_mean": stats_full["ch_keep_prunable_mean"],
                                "ch_prune_ratio_real": stats_full["ch_prune_ratio_real"],
                                "ch_keep_target": stats_full["ch_keep_target"],
                                "lambda_hw": stats_full["lambda_hw"],
                            }
                            if not suppress_head_block_keep_console:
                                stats_console["head_keep_real_mean"] = stats_full["head_keep_real_mean"]
                                stats_console["block_keep_real_mean"] = stats_full["block_keep_real_mean"]
                            alloc_is_active = bool(stats_full["alloc_enabled_this_outer"])
                            if alloc_is_active:
                                stats_console["alloc_applied"] = stats_full["alloc_applied"]
                                for _k in (
                                    "alloc_controller",
                                    "alloc_selected_source",
                                    "alloc_selected_long_gain",
                                    "alloc_selected_apply_gain",
                                    "alloc_switched",
                                    "alloc_inc_age",
                                ):
                                    if _k in stats_full:
                                        stats_console[_k] = stats_full[_k]
                            elif not suppress_alloc_fields_when_disabled:
                                stats_console["alloc_enabled_this_outer"] = stats_full["alloc_enabled_this_outer"]
                                stats_console["alloc_applied"] = stats_full["alloc_applied"]
                        else:
                            stats_console = {
                                "outer": stats_full["outer"],
                                "step": stats_full["step"],
                                "loss": stats_full["loss"],
                                "acc1": stats_full["acc1"],
                                "token_keep": stats_full["token_keep"],
                                "head_keep_real_mean": stats_full["head_keep_real_mean"],
                                "ch_keep_real_mean": stats_full["ch_keep_real_mean"],
                                "ch_keep_real_min": stats_full["ch_keep_real_min"],
                                "ch_keep_real_max": stats_full["ch_keep_real_max"],
                                "ch_keep_prunable_mean": stats_full["ch_keep_prunable_mean"],
                                "ch_prune_ratio_real": stats_full["ch_prune_ratio_real"],
                                "block_keep_real_mean": stats_full["block_keep_real_mean"],
                                "ch_keep_target": stats_full["ch_keep_target"],
                                "force_dense": stats_full["force_dense"],
                                "lambda_hw": stats_full["lambda_hw"],
                                "guard_mode": stats_full["guard_mode"],
                                "freeze_schedule": stats_full["freeze_schedule"],
                                "allow_discrete_updates": stats_full["allow_discrete_updates"],
                                "alloc_enabled_this_outer": stats_full["alloc_enabled_this_outer"],
                                "alloc_applied": stats_full["alloc_applied"],
                            }
                        if stats_full["alloc_enabled_this_outer"]:
                            for _k in (
                                "alloc_controller",
                                "alloc_selected_source",
                                "alloc_selected_long_gain",
                                "alloc_selected_apply_gain",
                                "alloc_switched",
                                "alloc_inc_age",
                            ):
                                if _k in stats_full:
                                    stats_console[_k] = stats_full[_k]

                        hw_console_enabled = bool(hw_stats) and (
                            float(lambda_hw_eff) > 0.0 or bool(_oc_select(cfg, "stable_hw.enabled", False))
                        )
                        if suite_cleanup_enabled and suppress_hw_fields_when_zero:
                            hw_console_enabled = bool(hw_stats) and (float(lambda_hw_eff) > 0.0) and any(
                                float(hw_stats.get(_k, hw_stats.get(_fallback, 0.0)) or 0.0) != 0.0
                                for _k, _fallback in (
                                    ("L_hw_total", "L_hw"),
                                    ("raw_latency_ms", "latency_ms"),
                                    ("mem_mb", "raw_mem_mb"),
                                    ("comm_ms", "raw_comm_ms"),
                                )
                            )
                        if hw_console_enabled:
                            stats_console["L_hw_total"] = float(hw_stats.get("L_hw_total", 0.0))
                            stats_console["raw_latency_ms"] = float(hw_stats.get("raw_latency_ms", hw_stats.get("latency_ms", 0.0)))
                            stats_console["mem_mb"] = float(hw_stats.get("mem_mb", hw_stats.get("raw_mem_mb", 0.0)))
                            stats_console["comm_ms"] = float(hw_stats.get("comm_ms", hw_stats.get("raw_comm_ms", 0.0)))

                        log_stats(logger, stats_console)
                        with log_path.open("a", encoding="utf-8") as f:
                            if log_slim:
                                payload = dict(stats_console)
                                payload["step"] = int(global_step)
                                payload["outer"] = int(outer)
                                f.write(safe_dumps(payload) + "\n")
                            else:
                                payload = dict(stats_full)
                                payload["step"] = int(global_step)
                                payload["outer"] = int(outer)
                                payload["stable_hw"] = stable_hw_log_fields(stable_hw_state, cfg)
                                f.write(safe_dumps(payload) + "\n")
                    global_step += 1

                # Step B: alpha refinement (only meaningful when HW term enabled)
                if (optimizer_alpha is not None) and update_alpha and (float(lambda_hw_eff) > 0.0):
                    for _ in range(cfg.training.inner_steps_alpha):
                        model_info = {}
                        last_info = run_state.get("last_model_info")
                        if isinstance(last_info, dict):
                            model_info = last_info.get("model_info", {})
                        slot_out = chiplet_slots(hard=False)
                        alpha = slot_out["alpha"]
                        eff_specs = slot_out["eff_specs"]
                        layout_positions = wafer_layout.current_pos_continuous()

                        segments_cached = mapping_res.get("segments", []) if mapping_res else []
                        mapping_cached = mapping_res.get("mapping", []) if mapping_res else []

                        segments_sig = None
                        try:
                            segments_sig = stable_hash(
                                [getattr(s, "signature", None) or repr(s) for s in (segments_cached or [])]
                            )
                        except Exception:
                            segments_sig = None

                        L_hw, _ = compute_hw_loss(
                            cfg,
                            hw_proxy,
                            model_info=model_info,
                            stable_hw_cfg=stable_hw_cfg,
                            stable_hw_state=stable_hw_state,
                            segments=segments_cached,
                            mapping=mapping_cached,
                            mapping_sig=(mapping_res.get("mapping_sig") or mapping_res.get("signature")) if mapping_res else None,
                            segments_sig=segments_sig,
                            eff_specs=eff_specs,
                            layout_positions=layout_positions,
                            mapping_solver=mapping_solver,
                            wafer_layout=wafer_layout,
                            alpha=alpha,
                        )

                        optimizer_alpha.zero_grad(set_to_none=True)
                        loss_alpha = float(lambda_hw_eff) * L_hw
                        if (not twostage) and torch.isfinite(loss_alpha).all():
                            loss_alpha.backward()
                            optimizer_alpha.step()
                        else:
                            # Skip alpha update if HW term is disabled or non-finite
                            run_state["nan_guard_skipped_alpha"] = int(run_state.get("nan_guard_skipped_alpha", 0)) + 1
                            if not torch.isfinite(loss_alpha).all():
                                _warn_throttled(
                                    "nan_alpha_loss",
                                    int(global_step),
                                    "[NaNGuard] non-finite alpha loss; skipping alpha step.",
                                    every_steps=nan_guard_warn_every_steps,
                                    first_n=3,
                                )

                # Step D: layout refinement
                if update_layout and allow_discrete:
                    for _ in range(cfg.training.inner_steps_layout):
                        slot_out = chiplet_slots(hard=False)
                        eff_specs = slot_out["eff_specs"]
                        segments = mapping_res.get("segments", []) if mapping_res else []
                        mapping = mapping_res.get("mapping", []) if mapping_res else []
                        if not segments or not mapping:
                            part_res = partitioner.plan(
                                model,
                                eff_specs,
                                alpha=slot_out["alpha"],
                                model_info=run_state.get("last_model_info"),
                                use_fine_split=getattr(cfg.hw, "use_fine_split", True),
                            )
                            segments = part_res["segments"]
                            mapping = part_res["mapping"]
                        L_layout, layout_stats = wafer_layout(
                            mapping,
                            segments,
                            eff_specs,
                            lambda_boundary=cfg.hw.lambda_boundary,
                            lambda_overlap=cfg.hw.lambda_overlap,
                            lambda_comm=cfg.hw.lambda_comm_extra,
                            lambda_thermal=cfg.hw.lambda_thermal,
                            distance_scale=float(getattr(cfg.hw, "distance_scale_ms", 1.0) or 1.0),
                        )
                        if not torch.isfinite(L_layout).all():
                            run_state["nan_guard_skipped_layout"] = int(run_state.get("nan_guard_skipped_layout", 0)) + 1
                            _warn_throttled(
                                "nan_layout_loss",
                                int(global_step),
                                "[NaNGuard] non-finite layout loss; disabling layout refinement for this run.",
                                every_steps=nan_guard_warn_every_steps,
                                first_n=1,
                            )
                            _repair_nonfinite_params_(wafer_layout)
                            update_layout = False
                            break
                        optimizer_layout.zero_grad(set_to_none=True)
                        L_layout.backward()
                        optimizer_layout.step()

                # ---- per-outer warning summary (keeps stdout small) ----
                amp_no_grads_total_after = int(run_state.get("warn_amp_alpha_no_grads_total", 0))
                amp_no_grads_logged_after = int(run_state.get("warn_amp_alpha_no_grads_logged", 0))
                amp_d_total = amp_no_grads_total_after - amp_no_grads_total_before
                amp_d_logged = amp_no_grads_logged_after - amp_no_grads_logged_before
                if amp_d_total > 0:
                    amp_d_supp = max(0, amp_d_total - amp_d_logged)
                    if amp_d_supp > 0:
                        logger.info(
                            "[LOG] AMP alpha-no-grads warnings throttled: outer=%s total=%s logged=%s suppressed=%s (warn_every_steps=%s)",
                            outer,
                            amp_d_total,
                            amp_d_logged,
                            amp_d_supp,
                            warn_every_steps,
                        )

                nan_repair_total_after = int(run_state.get("warn_nan_repair_total", 0))
                nan_repair_logged_after = int(run_state.get("warn_nan_repair_logged", 0))
                nan_d_total = nan_repair_total_after - nan_repair_total_before
                nan_d_logged = nan_repair_logged_after - nan_repair_logged_before
                if nan_d_total > 0:
                    nan_d_supp = max(0, nan_d_total - nan_d_logged)
                    if nan_d_supp > 0:
                        logger.info(
                            "[LOG] NaNGuard repair warnings throttled: outer=%s total=%s logged=%s suppressed=%s (nan_guard_warn_every_steps=%s)",
                            outer,
                            nan_d_total,
                            nan_d_logged,
                            nan_d_supp,
                            nan_guard_warn_every_steps,
                        )

                nan_loss_total_after = int(run_state.get("warn_nan_total_loss_total", 0))
                nan_loss_logged_after = int(run_state.get("warn_nan_total_loss_logged", 0))
                nloss_d_total = nan_loss_total_after - nan_loss_total_before
                nloss_d_logged = nan_loss_logged_after - nan_loss_logged_before
                if nloss_d_total > 0:
                    nloss_d_supp = max(0, nloss_d_total - nloss_d_logged)
                    if nloss_d_supp > 0:
                        logger.info(
                            "[LOG] NaNGuard non-finite-loss warnings throttled: outer=%s total=%s logged=%s suppressed=%s (nan_guard_warn_every_steps=%s)",
                            outer,
                            nloss_d_total,
                            nloss_d_logged,
                            nloss_d_supp,
                            nan_guard_warn_every_steps,
                        )

                val_agg = str(getattr(getattr(cfg, "data", object()), "eval_aggregate", "clip"))
                acc_eval_model = ema_model.ema if (ema_model is not None and ema_eval) else model
                hw_audit_model = model

                val_acc1 = eval_acc1(
                    acc_eval_model,
                    val_loader,
                    device,
                    model_type=str(getattr(cfg.training, "model_type", "video")),
                    max_batches=max_eval_batches_for_eval,
                    aggregate=val_agg,
                )

                pruner_eval = _get_ast_pruner(hw_audit_model)
                if pruner_eval is not None:
                    try:
                        pr_cfg_eval = pruner_eval.cfg.get("channel_prune", pruner_eval.cfg)
                        freeze_prefix_ratio_eval = float(
                            pr_cfg_eval.get("freeze_prefix_ratio", pruner_eval.cfg.get("ch_freeze_prefix_ratio", 0.0)) or 0.0
                        )
                        freeze_prefix_ratio_eval = float(max(0.0, min(1.0, freeze_prefix_ratio_eval)))
                    except Exception:
                        freeze_prefix_ratio_eval = 0.0
                else:
                    freeze_prefix_ratio_eval = 0.0

                # IMPORTANT:
                # - accuracy is evaluated on EMA (if enabled)
                # - pruning-state summary and hardware audit must reflect the current raw model
                keep_now_eval = _get_layer_ch_keep_now(hw_audit_model)
                keep_summary_eval = _summarize_layer_ch_keep(
                    keep_now_eval,
                    freeze_prefix_ratio=freeze_prefix_ratio_eval,
                )

                epoch_end_hw = _eval_epoch_end_hw_snapshot(
                    model=hw_audit_model,
                    cfg=cfg,
                    run_state=run_state,
                    hw_proxy=hw_proxy,
                    wafer_layout=wafer_layout,
                    chiplet_slots=chiplet_slots,
                )

                alloc_last_eval = run_state.get("alloc_last_search", None)
                if not isinstance(alloc_last_eval, dict):
                    alloc_last_eval = {}

                epoch_audit_record = {
                    "outer": int(outer),
                    "eval_mode": "ema" if (ema_model is not None and ema_eval) else "raw",
                    "acc_eval_mode": "ema" if (ema_model is not None and ema_eval) else "raw",
                    "hw_eval_mode": "raw",
                    "val_acc1": float(val_acc1) if val_acc1 is not None else None,
                    "ch_keep_real_mean": float(keep_summary_eval.get("ch_keep_real_mean", 1.0)),
                    "ch_keep_real_min": float(keep_summary_eval.get("ch_keep_real_min", 1.0)),
                    "ch_keep_real_max": float(keep_summary_eval.get("ch_keep_real_max", 1.0)),
                    "ch_keep_prunable_mean": float(keep_summary_eval.get("ch_keep_prunable_mean", 1.0)),
                    "ch_keep_layerwise": keep_summary_eval.get("ch_keep_layerwise", []),
                    "epoch_end_hw_ok": bool(epoch_end_hw.get("ok", False)),
                    "epoch_end_hw_objective": float(epoch_end_hw.get("objective", 0.0)) if epoch_end_hw.get("ok", False) else None,
                    "epoch_end_hw_mapping_sig": epoch_end_hw.get("mapping_sig", None),
                    "epoch_end_hw_latency_ms": float(epoch_end_hw.get("latency_ms", 0.0)) if epoch_end_hw.get("ok", False) else None,
                    "epoch_end_hw_energy_mj": float(epoch_end_hw.get("energy_mj", 0.0)) if epoch_end_hw.get("ok", False) else None,
                    "epoch_end_hw_mem_mb": float(epoch_end_hw.get("mem_mb", 0.0)) if epoch_end_hw.get("ok", False) else None,
                    "epoch_end_hw_comm_ms": float(epoch_end_hw.get("comm_ms", 0.0)) if epoch_end_hw.get("ok", False) else None,
                    "epoch_end_hw_error": epoch_end_hw.get("error", None),
                    "alloc_enabled_this_outer": bool(alloc_last_eval.get("alloc_enabled_this_outer", False)),
                    "alloc_start_after_prune_epochs": int(alloc_last_eval.get("alloc_start_after_prune_epochs", -1)),
                    "alloc_phase_progress": float(alloc_last_eval.get("alloc_phase_progress", 0.0)),
                    "alloc_phase_budget_frac": float(alloc_last_eval.get("alloc_phase_budget_frac", 0.0)),
                    "alloc_budget_frac": float(alloc_last_eval.get("alloc_budget_frac", 0.0)),
                    "alloc_applied": bool(alloc_last_eval.get("alloc_applied", False)),
                    "alloc_remain_budget": float(alloc_last_eval.get("alloc_remain_budget", 0.0)),
                    "alloc_usable_budget": float(alloc_last_eval.get("alloc_usable_budget", 0.0)),
                    "alloc_base_objective": float(alloc_last_eval.get("base_objective", 0.0)),
                    "alloc_best_objective": float(alloc_last_eval.get("best_objective", 0.0)),
                    "alloc_rel_hw_gain": float(alloc_last_eval.get("rel_hw_gain", 0.0)),
                    "alloc_acc_risk": float(alloc_last_eval.get("acc_risk", 0.0)),
                    "alloc_total_score": float(alloc_last_eval.get("total_score", 0.0)),
                    "alloc_controller": str(alloc_last_eval.get("alloc_controller", "")),
                    "alloc_selected_source": str(alloc_last_eval.get("alloc_selected_source", "")),
                    "alloc_selected_long_gain": float(alloc_last_eval.get("alloc_selected_long_gain", 0.0)),
                    "alloc_selected_apply_gain": float(alloc_last_eval.get("alloc_selected_apply_gain", 0.0)),
                    "alloc_inc_long_gain": float(alloc_last_eval.get("alloc_inc_long_gain", 0.0)),
                    "alloc_best_ch_long_gain": float(alloc_last_eval.get("alloc_best_ch_long_gain", 0.0)),
                    "alloc_switched": bool(alloc_last_eval.get("alloc_switched", False)),
                    "alloc_inc_age": int(alloc_last_eval.get("alloc_inc_age", 0) or 0),
                    "alloc_candidate_strategy": str(alloc_last_eval.get("alloc_candidate_strategy", "")),
                    "alloc_decision_basis": str(alloc_last_eval.get("alloc_decision_basis", "")),
                }
                with epoch_audit_path.open("a", encoding="utf-8") as f:
                    f.write(safe_dumps(epoch_audit_record) + "\n")
                # ---- audit-friendly [val] line (used by LockedAccRef curve parser) ----
                try:
                    n_val = len(val_loader)
                except Exception:
                    n_val = -1
                if int(max_eval_batches_cfg) <= 0 or (n_val > 0 and int(max_eval_batches_cfg) >= int(n_val)):
                    val_mode = "full"
                else:
                    val_mode = "fast"
                if val_acc1 is not None:
                    logger.info(
                        "[val] epoch=%s mode=%s acc_clip=%.4f acc_video=%.4f real_ch_keep=%.4f prunable_ch_keep=%.4f end_hw_obj=%.6g end_lat=%.4f end_mem=%.4f end_comm=%.4f alloc_frac=%.3f (pref=%s acc_model=%s hw_model=%s)",
                        int(outer),
                        str(val_mode),
                        float(val_acc1),
                        float(val_acc1),
                        float(keep_summary_eval.get("ch_keep_real_mean", 1.0)),
                        float(keep_summary_eval.get("ch_keep_prunable_mean", 1.0)),
                        float(epoch_end_hw.get("objective", 0.0)) if epoch_end_hw.get("ok", False) else 0.0,
                        float(epoch_end_hw.get("latency_ms", 0.0)) if epoch_end_hw.get("ok", False) else 0.0,
                        float(epoch_end_hw.get("mem_mb", 0.0)) if epoch_end_hw.get("ok", False) else 0.0,
                        float(epoch_end_hw.get("comm_ms", 0.0)) if epoch_end_hw.get("ok", False) else 0.0,
                        float(alloc_last_eval.get("alloc_budget_frac", 0.0)),
                        str(val_agg),
                        "ema" if (ema_model is not None and ema_eval) else "raw",
                        "raw",
                    )

                try:
                    _maybe_start_user_recovery_after_val(
                        cfg=cfg,
                        run_state=run_state,
                        ast_sched=ast_sched if isinstance(ast_sched, dict) else {},
                        outer=int(outer),
                        prunable_keep=float(keep_summary_eval.get("ch_keep_prunable_mean", 1.0)),
                        logger=logger,
                    )
                except Exception as _rec_exc:
                    logger.warning("[UserRecovery] failed to schedule: %s", str(_rec_exc))

                if bool(run_state.get("user_recovery_active", False)):
                    rem = int(run_state.get("user_recovery_remaining", 0) or 0)
                    if rem > 0:
                        run_state["user_recovery_remaining"] = int(rem - 1)
                        if int(rem - 1) == 0:
                            logger.info("[UserRecovery] finished at outer=%d", int(outer))
                if stable_hw_enabled and guard_controls_enabled:
                    stable_decision, _ = apply_accuracy_guard(
                        epoch=outer,
                        stable_hw_cfg=cfg,
                        stable_hw_state=stable_hw_state,
                        val_metric_or_none=float(val_acc1) if val_acc1 is not None else None,
                        has_val_this_epoch=True,
                        train_ema_or_none=float(stable_hw_state.get("train_acc1_ema", 0.0))
                        if stable_hw_state.get("train_acc1_ema") is not None
                        else None,
                    )
                    stable_hw_state = stable_decision.state
                    epoch = int(outer)
                    outer_iter = int(outer)
                    last_loss = run_state.get("last_loss_components", {}) or {}
                    acc_ref = stable_hw_state.get("acc_ref", None)
                    acc_now = stable_hw_state.get("acc_now", None)
                    acc_drop = stable_hw_state.get("acc_drop", None)
                    acc_ref_val = float(acc_ref) if acc_ref is not None else 0.0
                    acc_now = float(acc_now) if acc_now is not None else 0.0
                    acc_drop = float(acc_drop) if acc_drop is not None else 0.0
                    acc_loss = last_loss.get("acc_loss", 0.0)
                    acc_part = float(last_loss.get("acc_part", acc_loss or 0.0))
                    total_loss = last_loss.get("total_loss", 0.0)
                    hw_loss_norm = last_loss.get("hw_loss_raw", 0.0)
                    hw_loss_used = last_loss.get("hw_loss_used", 0.0)
                    # v5.4 trace: gating event (spec_e)
                    # 合同要求：每个 outer step 都必须产出 gating 证据（allow_hw / reject_hw），不能只在触发时记录
                    decision = "accept_hw"
                    if float(getattr(stable_decision, "lambda_hw_effective", 0.0) or 0.0) <= 0.0:
                        decision = "reject_hw"
                    if float(hw_loss_used or 0.0) <= 0.0:
                        decision = "reject_hw"
                    if str(getattr(stable_decision, "guard_mode", "")) in ("VIOLATE", "RECOVERY", "WARMUP"):
                        decision = "reject_hw"
                    reason_code = str(stable_hw_state.get("gating_reason_code", "") or "")
                    if not reason_code:
                        reason_code = "hw_enabled" if decision == "accept_hw" else "hw_cut_or_warmup_or_recovery"
                    gate = "reject_hw" if decision == "reject_hw" else "allow_hw"
                    acc_drop_max = float(
                        stable_hw_state.get(
                            "acc_drop_max",
                            stable_hw_state.get(
                                "guard_eps_used",
                                stable_hw_state.get("epsilon_drop", 0.0),
                            ),
                        )
                    )
                    lambda_hw_eff = float(getattr(stable_decision, "lambda_hw_effective", 0.0) or 0.0)
                    hw_part = 0.0
                    if float(lambda_hw_eff) != 0.0:
                        hw_part = float(lambda_hw_eff) * float(hw_loss_used)

                    acc_used_source = str(stable_hw_state.get("acc_used_source", "") or "")
                    acc_used_value = stable_hw_state.get("acc_used_value", acc_now)
                    eps = 1e-12
                    hw_ref = {
                        "latency_ms": float(hw_stats.get("ref_latency_ms", 0.0)),
                        "energy_mj": float(hw_stats.get("ref_energy_mj", 0.0)),
                        "mem_mb": float(hw_stats.get("ref_mem_mb", 0.0)),
                        "comm_ms": float(hw_stats.get("ref_comm_ms", 0.0)),
                    }
                    hw_raw = {
                        "latency_ms": float(hw_stats.get("proxy_raw_latency_ms", 0.0)),
                        "energy_mj": float(hw_stats.get("proxy_raw_energy_mj", 0.0)),
                        "mem_mb": float(hw_stats.get("proxy_raw_mem_mb", 0.0)),
                        "comm_ms": float(hw_stats.get("raw_comm_ms", hw_stats.get("proxy_raw_comm_norm", 0.0))),
                    }
                    hw_used = {
                        "latency_ms": float(hw_stats.get("proxy_used_latency_ms", hw_stats.get("latency_ms", 0.0))),
                        "energy_mj": float(hw_stats.get("proxy_used_energy_mj", hw_stats.get("energy_mj", 0.0))),
                        "mem_mb": float(hw_stats.get("proxy_used_mem_mb", hw_stats.get("mem_mb", 0.0))),
                        "comm_ms": float(hw_stats.get("comm_ms", 0.0)),
                    }
                    hw_normed = {k: (hw_used[k] / max(eps, hw_ref[k])) for k in hw_ref.keys()}

                    total_loss_scalar = float(total_loss.item()) if hasattr(total_loss, "item") else float(total_loss or 0.0)
                    payload = make_gating_payload_v54(
                        cfg=cfg,
                        stable_state=stable_hw_state,
                        epoch=epoch,
                        step=global_step,
                        loss_scalar=total_loss_scalar,
                        gate_ok=gate == "allow_hw",
                        gate_reason=reason_code,
                        overrides={
                            "candidate_id": int(outer),
                            "gate": gate,
                            "acc_ref": float(acc_ref_val),
                            "acc_now": float(acc_now),
                            "acc_used": float(acc_used_value if acc_used_value is not None else acc_now),
                            "acc_drop": float(stable_hw_state.get("acc_drop", 0.0)),
                            "acc_drop_max": float(acc_drop_max),
                            "acc_used_source": acc_used_source,
                            "acc_used_value": float(acc_used_value if acc_used_value is not None else acc_now),
                            "lambda_hw_effective": float(lambda_hw_eff),
                            "guard_mode": str(getattr(stable_decision, "guard_mode", "")),
                            "lambda_hw_base": float(getattr(stable_decision, "lambda_hw_base", 0.0) or 0.0),
                            "hw_loss_raw": float(hw_loss_norm or 0.0),
                            "hw_loss_used": float(hw_loss_used or 0.0),
                            "total_loss_scalar": total_loss_scalar,
                            "total_loss_acc_part": float(acc_part),
                            "total_loss_hw_part": float(hw_part),
                            "acc_loss": float(acc_loss.item()) if hasattr(acc_loss, "item") else float(acc_loss or 0.0),
                            "total_loss": total_loss_scalar,
                            "reason": dict(getattr(stable_decision, "reason", {}) or {}),
                            "allow_discrete_updates": bool(stable_hw_state.get("allow_discrete_updates", True)),
                            "hw_scale_schema_version": "v5.4_ratio_v1",
                            "hw_metric_ref": hw_ref,
                            "hw_metric_raw": hw_raw,
                            "hw_metric_normed": hw_normed,
                            "hw_metric_used": hw_used,
                            "hw_metric_used_sanitized": hw_used,
                            "hw_metric_key_order": list(hw_ref.keys()),
                            "violate_streak": int(stable_hw_state.get("violate_streak", 0) or 0),
                            "epoch": int(epoch),
                            "outer_iter": int(outer_iter),
                            "global_step": int(global_step),
                            "notes": "version_c",
                            "ast_phase": str(ast_sched.get("phase", "disabled")) if isinstance(ast_sched, dict) else "disabled",
                            "ast_force_dense": bool(ast_sched.get("force_dense", False)) if isinstance(ast_sched, dict) else False,
                            "ast_rho_token": float(ast_sched.get("rho_token", 1.0)) if isinstance(ast_sched, dict) else 1.0,
                            "ast_token_temperature": float(ast_sched.get("token_temperature", 0.1)) if isinstance(ast_sched, dict) else 0.1,
                            "ast_lambda_ast": float(ast_sched.get("lambda_ast", 0.0)) if isinstance(ast_sched, dict) else 0.0,
                            "ast_freeze_now": bool(_stablehw_freeze_ast_now(stable_hw_state)) if stable_hw_enabled else False,
                            "ast_sched_virtual_epoch": int(stable_hw_state.get("ast_sched_virtual_epoch", int(outer) + 1)),
                            "ast_sched_epoch_used": int(stable_hw_state.get("ast_sched_epoch_used", int(outer))),
                        },
                    )
                    missing_keys = [k for k in REQUIRED_GATING_KEYS if k not in payload]
                    if missing_keys:
                        raise KeyError(f"gating payload missing keys: {missing_keys}")
                    append_trace_event_v54(
                        trace_events_path,
                        "gating",
                        payload=payload,
                        run_id=run_id,
                        step=int(outer),
                    )
                    stable_hw_state["gating_reason_code"] = ""
                    # ===== v5.4 Acc-First Hard Gating: stop_on_violation 必须真的停止 =====
                    if bool(stable_decision.stop_training):
                        val_acc1_str = f"{val_acc1:.6f}" if val_acc1 is not None else "None"
                        logger.warning(
                            f"[StableHW] stop_on_violation triggered at epoch={outer}: "
                            f"val_acc1={val_acc1_str}, acc_ref={stable_hw_state.get('acc_ref')}, "
                            f"acc_floor={stable_hw_state.get('acc_floor')}. Stop training now."
                        )
                        early_stop_triggered = True
                        break
                    # ---- invariants (v5.4) ----
                    if stable_hw_state.get("acc_ref") is not None:
                        cur = float(stable_hw_state["acc_ref"])
                        prev = stable_hw_state.get("_acc_ref_once", None)
                        locked_ep = stable_hw_state.get("acc_ref_locked_epoch", None)
                        # allow one update on the locking epoch (warmup->locked transition), or for dynamic/curve refs
                        allow_dynamic = bool(stable_hw_state.get("acc_ref_dynamic", False))
                        if prev is None:
                            stable_hw_state["_acc_ref_once"] = cur
                        else:
                            prevf = float(prev)
                            if abs(prevf - cur) > 1e-9:
                                if (locked_ep is not None and int(locked_ep) == int(epoch)) or allow_dynamic:
                                    stable_hw_state["_acc_ref_once"] = cur
                                else:
                                    logger.warning(
                                        f"[StableHW] acc_ref changed (prev={prevf:.6f}, cur={cur:.6f}). "
                                        "Resetting _acc_ref_once to avoid crash. "
                                        "If this is unexpected, inspect locked_acc_ref/curve settings."
                                    )
                                    stable_hw_state["_acc_ref_once"] = cur
                elif stable_hw_enabled:
                    class _StableFallbackDecision:
                        lambda_hw_effective = float(stable_hw_state.get("lambda_hw_effective", 0.0) or 0.0)
                        guard_mode = "HW_OPT"
                        lambda_hw_base = float(stable_hw_state.get("lambda_hw_base", 0.0) or 0.0)
                        reason = {}
                        stop_training = False

                    stable_decision = _StableFallbackDecision()

                if stable_hw_enabled:
                    # v5.4: always call; stable_hw decides freeze vs ema-fallback internally
                    before = {k: stable_hw_state.get(k) for k in ["ref_T", "ref_E", "ref_M", "ref_C"]}
                    update_hw_refs_from_stats(
                        cfg,
                        stable_hw_state,
                        latest_hw_stats=last_hw_stats or {},
                        stable_hw_cfg=stable_hw_cfg,
                    )
                    after = {k: stable_hw_state.get(k) for k in ["ref_T", "ref_E", "ref_M", "ref_C"]}
                    # v5.4 contract: NoDrift requested => never allow ref_update
                    if before != after:
                        allow_self_hw_lock_once = bool(stable_hw_state.get("hw_ref_just_locked", False))
                        if stable_hw_state.get("no_drift_enabled", False) and (not allow_self_hw_lock_once):
                            raise RuntimeError(
                                "[SPEC v5.4] NoDrift violation: ref changed while no_drift.enabled=True. "
                                "This must not happen (no silent fallback)."
                            )

                        def _maybe_float(val):
                            if val is None:
                                return None
                            return float(val)

                        requested_mode = str(getattr(getattr(stable_hw_cfg, "normalize", None), "ref_update", "ema"))
                        effective_mode = str(stable_hw_state.get("_force_ref_update_mode", requested_mode))
                        no_drift_enabled = bool(stable_hw_state.get("no_drift_enabled", False))

                        for ref_name in ("ref_T", "ref_E", "ref_M", "ref_C"):
                            if before.get(ref_name) != after.get(ref_name):
                                append_trace_event_v54(
                                    trace_events_path,
                                    "ref_update",
                                    payload={
                                        "outer_iter": int(outer),
                                        "ref_name": str(ref_name),
                                        "old_value": _maybe_float(before.get(ref_name)),
                                        "new_value": _maybe_float(after.get(ref_name)),
                                        "update_type": "hw_ref_update",
                                        "allowed_by_no_drift": ((not no_drift_enabled) or allow_self_hw_lock_once),
                                        "requested_mode": requested_mode,
                                        "effective_mode": effective_mode,
                                        "reason": ("self_hw_lock" if allow_self_hw_lock_once else "hw_ref_update"),
                                    },
                                    run_id=run_id,
                                    step=int(outer),
                                )
                        if stable_hw_state.get("hw_ref_just_locked", False):
                            stable_hw_state["hw_ref_just_locked"] = False
                guard_mode = str(stable_hw_state.get("guard_mode", "HW_OPT")) if stable_hw_enabled else "disabled"
                allow_discrete = (
                    bool(stable_hw_state.get("allow_discrete_updates", True)) if stable_hw_enabled else True
                )
                if not suppress_stable_hw_epoch_summary:
                    print(
                        f"[StableHW] epoch={outer} mode={guard_mode} "
                        f"lambda_hw_eff={lambda_hw_eff:.6g} allow_discrete={allow_discrete}"
                    )
                    logger.info(
                        f"[StableHW][epoch={outer}] "
                        f"mode={stable_hw_state.get('guard_mode')} "
                        f"acc_used_raw={stable_hw_state.get('acc_used_raw')} "
                        f"acc_used_ema={stable_hw_state.get('acc_used_enter')} "
                        f"acc_ref={stable_hw_state.get('acc_ref_value', stable_hw_state.get('acc_ref'))} "
                        f"eps={stable_hw_state.get('eps_enter')} "
                        f"drop={stable_hw_state.get('acc_drop_enter')} "
                        f"below={stable_hw_state.get('below_cnt')}/{stable_hw_state.get('k_enter')} "
                        f"lambda_base={stable_hw_state.get('lambda_hw_base')} "
                        f"lambda_eff={stable_hw_state.get('lambda_hw_effective')} "
                        f"allow_discrete={stable_hw_state.get('allow_discrete_updates')} "
                        f"freeze_schedule={stable_hw_state.get('freeze_schedule')}"
                    )
                # ---- v5.4: auditable acc source for gating / locked ref (SPEC_C) ----
                acc_used_source = stable_hw_state.get("acc_used_source", None)
                acc_used_value = stable_hw_state.get(
                    "acc_used_value",
                    stable_hw_state.get("acc_used_last", None),
                )

                if acc_used_source and acc_used_value is not None:
                    acc_used_source = str(acc_used_source)
                    acc_used_value = float(acc_used_value)
                else:
                    if val_acc1 is not None:
                        acc_used_source = "val"
                        acc_used_value = float(val_acc1)
                    else:
                        # 1) try last known val
                        prev_last = last_acc1 if last_acc1 is not None else stable_hw_state.get("val_acc1_last", None)
                        if prev_last is not None:
                            acc_used_source = "val_last"
                            acc_used_value = float(prev_last)
                        else:
                            # 2) fall back to EMA train acc (SPEC_C)
                            ema = stable_hw_state.get("train_acc1_ema", None)
                            allow_ema_fb = bool(getattr(stable_hw_cfg, "allow_train_ema_fallback", False))
                            if allow_ema_fb and (ema is not None):
                                acc_used_source = "train_ema"
                                acc_used_value = float(ema)
                            else:
                                # contract: cannot decide gating without any auditable accuracy signal
                                raise RuntimeError(
                                    "[V5.4 CONTRACT] val_acc1 missing and train_ema fallback is disabled or unavailable. "
                                    "Set stable_hw.allow_train_ema_fallback=True explicitly if you want this fallback."
                                )

                    if acc_used_source is not None:
                        stable_hw_state.setdefault("acc_used_source", str(acc_used_source))
                    if acc_used_value is not None:
                        stable_hw_state.setdefault("acc_used_value", float(acc_used_value))

                last_acc1 = float(acc_used_value)

                prev_best_acc1 = best_acc1
                if best_acc1 is None:
                    best_acc1 = float(last_acc1)
                else:
                    best_acc1 = max(float(best_acc1), float(last_acc1))

                improved_best = (prev_best_acc1 is None) or (
                    best_acc1 is not None and float(best_acc1) > float(prev_best_acc1)
                )
                try:
                    save_checkpoint_version_c(
                        out_dir,
                        "last",
                        model=model,
                        ema_model=ema_model,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=outer,
                        best_acc1=best_acc1,
                        seal_digest=seal_digest,
                        run_id=run_id,
                    )
                    if improved_best:
                        save_checkpoint_version_c(
                            out_dir,
                            "best",
                            model=model,
                            ema_model=ema_model,
                            optimizer=optimizer,
                            scaler=scaler,
                            epoch=outer,
                            best_acc1=best_acc1,
                            seal_digest=seal_digest,
                            run_id=run_id,
                        )
                except Exception as _ckpt_e:
                    logger.warning(f"[CKPT] save failed: {_ckpt_e}")

                stable_hw_state["val_acc1_best_seen"] = float(best_acc1) if best_acc1 is not None else None

                # v5.4 strict: single source of truth
                if getattr(cfg, "no_drift", None) not in (None, {}, ""):
                    raise ValueError(
                        "P0(v5.4): legacy root-level cfg.no_drift is forbidden at runtime. "
                        "Use cfg.stable_hw.no_drift only."
                    )

                no_drift_cfg = getattr(stable_hw_cfg, "no_drift", {}) or {}
                stable_hw_state["_contract_no_drift"] = bool(getattr(no_drift_cfg, "enabled", False)) if no_drift_cfg else False
                norm = getattr(stable_hw_cfg, "normalize", None)
                stable_hw_state["_contract_ref_update"] = (
                    "frozen" if norm is None else str(getattr(norm, "ref_update", "frozen") or "frozen")
                )

                stable_state_path = log_path.parent / "stable_hw_state.json"
                stable_state_path.write_text(
                    safe_dumps(stable_hw_state, indent=2),
                    encoding="utf-8",
                )
                if early_stop_triggered or ran_epochs == 0:
                    reason = "early_stop_or_zero_step"
            steps_done = int(ran_epochs)
            steps_done_for_finalize = int(steps_done)
        except Exception:
            reason = "error"
            best_solution_valid = False
            raise
        finally:
            _update_manifest_gating_summary(
                trace_dir,
                cfg,
                stable_hw_state if "stable_hw_state" in locals() else {},
            )
            summary_payload = {
                "ok": (str(reason) != "error"),
                "reason": str(reason),
                "steps_done": int(steps_done),
                "best_solution_valid": bool(best_solution_valid),
            }
            summary_payload.update(
                build_baseline_trace_summary(
                    cfg,
                    stable_hw_state if "stable_hw_state" in locals() else {},
                )
            )
            update_trace_summary(trace_dir, summary_payload)
            finalize_trace_dir(
                trace_events_path,
                reason=str(reason),
                steps_done=int(steps_done),
                best_solution_valid=bool(best_solution_valid),
            )

        # write run_manifest.json (auditable LockedAccRef)  -- v5.4 compliant
        from utils.run_manifest import write_run_manifest

        git_sha = None

        _telemetry = {
            "gating_on_ratio": float(gating_epochs) / max(1, int(total_epochs)),
            "freeze_epoch_ratio": float(freeze_epochs) / max(1, int(total_epochs)),
        }
        _metrics_summary = {
            "status": str(reason),
            "ran_epochs": int(ran_epochs),
            "early_stop": bool(early_stop_triggered),
            "best_acc1": float(best_acc1) if ("best_acc1" in locals() and best_acc1 is not None) else None,
            "last_acc1": float(last_acc1) if ("last_acc1" in locals() and last_acc1 is not None) else None,
            "telemetry": _telemetry,
        }

        write_run_manifest(
            out_dir=str(out_dir),
            cfg_path=cfg_path,
            cfg_hash=cfg_hash,
            run_id=run_id,
            seed=int(seed),
            git_sha=git_sha,
            code_root=str(Path(__file__).resolve().parents[1]),
            stable_hw_state=stable_hw_state if "stable_hw_state" in locals() else {},
            cfg=cfg,
            metrics_summary=_metrics_summary,
            extra={
                "task": "version_c",
                "out_dir": str(out_dir),
            },
        )

        stable_fields = stable_hw_log_fields(stable_hw_state, cfg)
        metrics = {
            "run_id": run_id,
            "stable_hw_disabled": False if stable_hw_cfg and bool(getattr(stable_hw_cfg, "enabled", True)) else True,
            "best_acc1": float(best_acc1) if best_acc1 is not None else 0.0,
            "last_acc1": float(last_acc1) if last_acc1 is not None else 0.0,
            "val_acc1": float(last_acc1) if last_acc1 is not None else 0.0,
            "last_hw_stats": last_hw_stats if last_hw_stats is not None else {},
            "mapping_signature": stable_hw_state.get("discrete_cache", {}).get("mapping_signature"),
            "layout_signature": stable_hw_state.get("discrete_cache", {}).get("layout_signature"),
            "stable_hw": stable_fields,
            "acc_used_source": stable_hw_state.get("acc_used_source", None),
            "acc_used_value": stable_hw_state.get("acc_used_value", None),
            "telemetry": {
                "gating_on_ratio": float(gating_epochs) / max(1, int(total_epochs)),
                "freeze_epoch_ratio": float(freeze_epochs) / max(1, int(total_epochs)),
            },
        }
        for k, v in stable_fields.items():
            metrics[k] = v
        with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
            safe_dump(metrics, f, indent=2)

        # Optional: export baseline stats for downstream StableHW runs.
        # This mirrors trainer_single_device behavior (BASELINE_STATS_EXPORT) so users can
        # generate outputs/dense_baseline/metrics.json from a Version-C dense baseline.
        baseline_export_path = os.environ.get("BASELINE_STATS_EXPORT", "").strip()
        if baseline_export_path:
            export_path = Path(baseline_export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = export_path.with_suffix(export_path.suffix + f".tmp.{os.getpid()}")
            try:
                with tmp_path.open("w", encoding="utf-8") as tmp_f:
                    safe_dump(metrics, tmp_f, indent=2)
                os.replace(tmp_path, export_path)
            finally:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
        hw_stats_out = dict(last_hw_stats or {})
        hw_stats_out.update(
            {
                "cfg_hash": cfg_hash,
                "seed": seed,
            }
        )
        with (out_dir / "hw_stats.json").open("w", encoding="utf-8") as f:
            safe_dump(hw_stats_out, f, indent=2)
        if export_layout_input:
            export_dir_path = Path(layout_export_dir or "outputs/P3")
            slot_out = chiplet_slots(hard=False)
            eff_specs = slot_out["eff_specs"]
            part_res = partitioner.plan(
                model,
                eff_specs,
                alpha=slot_out["alpha"],
                model_info=run_state.get("last_model_info"),
                use_fine_split=getattr(cfg.hw, "use_fine_split", True),
            )
            canonical_segments = part_res["segments"]
            mapping_result = mapping_solver.solve_mapping(
                canonical_segments,
                eff_specs,
                hw_proxy,
                layout_positions=wafer_layout.current_pos_continuous(),
                strategy=getattr(cfg.hw, "mapping_strategy", "greedy_local"),
                distance_scale_ms=getattr(cfg.hw, "distance_scale_ms", 0.0),
                alpha=slot_out["alpha"],
            )
            mapping_canonical = mapping_result["mapping"]
            _export_layout_input(
                cfg=cfg,
                export_dir=export_dir_path,
                out_dir=out_dir,
                chiplet_slots=chiplet_slots,
                mapping_solver=mapping_solver,
                segments=canonical_segments,
                mapping=mapping_canonical,
                wafer_layout=wafer_layout,
                seed=int(getattr(cfg.train, "seed", 0)),
            )

        try:
            (out_dir / "DONE").write_text("done\n", encoding="utf-8")
        except Exception:
            pass
    except Exception as e:
        from utils.trace_guard import write_exception_json
        write_exception_json(trace_dir, e, stage="trainer_version_c")

        if not trace_header_written:
            # Ensure trace_events_path is defined even if failure happened early
            try:
                _tep = trace_events_path
            except NameError:
                _tep = trace_dir / "trace_events.jsonl"
                _tep.parent.mkdir(parents=True, exist_ok=True)
                if not _tep.exists():
                    _tep.write_text("", encoding="utf-8")

            requested_cfg = _oc_select(cfg, "_contract.requested_config_snapshot", {}) or {}
            effective_cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
            contract_overrides = _oc_select(cfg, "_contract.overrides", []) or []
            signature = build_signature_v54(cfg, method_name="ours_version_c")

            trace_header_payload = build_trace_header_payload_v54(
                signature=signature,
                requested_config=requested_cfg,
                effective_config=effective_cfg,
                contract_overrides=contract_overrides,
                requested={"mode": "version_c"},
                effective={"mode": "version_c"},
                no_drift_enabled=bool(
                    getattr(getattr(getattr(cfg, "stable_hw", None), "no_drift", None), "enabled", False)
                ),
                acc_ref_source=str(
                    getattr(getattr(getattr(cfg, "stable_hw", None), "locked_acc_ref", None), "source", "none")
                ),
                seal_digest=str(getattr(getattr(cfg, "contract", None), "seal_digest", "")),
            )
            trace_header_payload["stable_hw_state"] = {"init_status": "failed_before_full_init"}

            append_trace_event_v54(
                _tep,
                "trace_header",
                payload=trace_header_payload,
                run_id=str(getattr(locals().get("run_id", ""), "strip", lambda: "")() or "unknown"),
                step=0,
            )
            trace_header_written = True

        finalize_flag = trace_dir / "finalized.flag"
        if not finalize_flag.exists():
            finalize_trace_dir(
                trace_events_path,
                reason="exception",
                steps_done=int(steps_done_for_finalize),
                best_solution_valid=False,
            )
        raise
    finally:
        if not trace_header_written:
            pass
