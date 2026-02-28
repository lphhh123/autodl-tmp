"""Single-device AST2.0-lite trainer (SPEC §12.1)."""
from __future__ import annotations

import json
import math
import os
import numbers
import random
import sys
from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from omegaconf import OmegaConf

from hw_proxy.layer_hw_proxy import LayerHwProxy
from hw_proxy.hw_loss import compute_hw_loss
from mapping.segments import build_segments_from_model
from models.video_vit import VideoViT, VideoAudioAST
from utils.data_ucf101 import UCF101Dataset
from utils.logging_utils import setup_logger, log_stats
from utils.distributed_utils import get_device
from utils.eval_utils import eval_acc1
from utils.seed import seed_everything
from utils.trace_guard import (
    init_trace_dir_v54,
    append_trace_event_v54,
    finalize_trace_dir,
    update_trace_summary,
    build_baseline_trace_summary,
    build_trace_header_payload_v54,
    build_trace_signature_v54,
)
from utils.trace_signature_v54 import REQUIRED_SIGNATURE_FIELDS
from utils.stable_hash import stable_hash
from utils.config import AttrDict
from utils.config_utils import get_nested
from utils.contract_seal import assert_cfg_sealed_or_violate
from utils.trace_contract_v54 import assert_trace_header_v54
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


def _ast_interp(a: float, b: float, t: float, curve: str = "linear") -> float:
    t = float(max(0.0, min(1.0, t)))
    curve = str(curve or "linear").lower()
    if curve == "cosine":
        # cosine ease-in-out
        tt = 0.5 - 0.5 * math.cos(math.pi * t)
    else:
        tt = t
    return float(a + (b - a) * tt)


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
    ramp = int(getattr(sched, "ramp_epochs", 0) or 0)
    curve = str(getattr(sched, "curve", "cosine") or "cosine")

    rho_end = float(getattr(sched, "rho_end", getattr(ast, "rho_token_target", 1.0)) or getattr(ast, "rho_token_target", 1.0))
    rho_start = float(getattr(sched, "rho_start", 1.0) or 1.0)

    temp_end = float(getattr(sched, "temp_end", getattr(ast, "token_temperature", 0.1)) or getattr(ast, "token_temperature", 0.1))
    temp_start = float(getattr(sched, "temp_start", 1.0) or 1.0)

    loss_cfg = getattr(cfg, "loss", None)
    lam_end = float(getattr(sched, "lambda_ast_end", getattr(loss_cfg, "lambda_AST", 1.0) if loss_cfg is not None else 1.0) or (getattr(loss_cfg, "lambda_AST", 1.0) if loss_cfg is not None else 1.0))
    lam_start = float(getattr(sched, "lambda_ast_start", 0.0) or 0.0)

    if epoch < warm:
        return {
            "phase": "warmup",
            "t": 0.0,
            "force_dense": True,
            "rho_token": 1.0,
            "token_temperature": temp_start,
            "lambda_ast": lam_start,
        }

    if ramp <= 0:
        return {
            "phase": "stabilize",
            "t": 1.0,
            "force_dense": False,
            "rho_token": rho_end,
            "token_temperature": temp_end,
            "lambda_ast": lam_end,
        }

    # Ramp progress for this epoch.
    # Use t=0 at the first ramp epoch (epoch == warmup_epochs) so
    # rho/temp/lambda start from *_start without an immediate jump.
    t = float(epoch - warm) / float(ramp)
    t = float(max(0.0, min(1.0, t)))
    phase = "ramp" if t < 1.0 else "stabilize"
    return {
        "phase": phase,
        "t": t,
        "force_dense": False,
        "rho_token": _ast_interp(rho_start, rho_end, t, curve=curve),
        "token_temperature": _ast_interp(temp_start, temp_end, t, curve=curve),
        "lambda_ast": _ast_interp(lam_start, lam_end, t, curve=curve),
    }


# -------------------------
# AST schedule runtime helpers
# -------------------------

def _apply_ast_runtime_overrides_to_model(model: nn.Module, cfg, ast_sched: dict) -> Optional[dict]:
    """Apply AST schedule to a model's pruner (if present) without mutating cfg."""
    pruner = getattr(model, "ast_pruner", None)
    if pruner is None or not hasattr(pruner, "set_runtime_overrides"):
        return None

    ast_cfg = getattr(cfg, "ast", None)
    force_dense = bool(ast_sched.get("force_dense", False))
    rho_token = float(
        ast_sched.get("rho_token", getattr(ast_cfg, "rho_token_target", 1.0) if ast_cfg is not None else 1.0)
    )
    token_temperature = float(
        ast_sched.get(
            "token_temperature",
            getattr(ast_cfg, "token_temperature", 0.1) if ast_cfg is not None else 0.1,
        )
    )

    pruner.set_runtime_overrides(
        force_dense=force_dense,
        rho_token=rho_token,
        token_temperature=token_temperature,
    )
    return {
        "force_dense": force_dense,
        "rho_token": float(rho_token),
        "token_temperature": float(token_temperature),
    }


def _maybe_freeze_ast_schedule(ast_sched: dict, stable_state: Dict[str, Any], epoch: int) -> dict:
    """Freeze AST schedule when StableHW is in recovery (prevents over-pruning)."""
    if not isinstance(stable_state, dict):
        return ast_sched

    freeze = bool(stable_state.get("freeze_schedule", False)) or bool(stable_state.get("freeze_ast_schedule", False))
    if not freeze:
        stable_state["ast_sched_last_applied"] = dict(ast_sched)
        stable_state["ast_sched_last_epoch"] = int(epoch)
        # clear frozen snapshot when leaving recovery
        stable_state.pop("ast_sched_frozen", None)
        stable_state.pop("ast_sched_frozen_epoch", None)
        return ast_sched

    frozen = stable_state.get("ast_sched_frozen") or stable_state.get("ast_sched_last_applied")
    if isinstance(frozen, dict) and frozen:
        stable_state["ast_sched_frozen"] = dict(frozen)
        stable_state["ast_sched_frozen_epoch"] = int(epoch)
        return dict(frozen)
    return ast_sched


def _extract_ast_pruner_runtime(model: nn.Module) -> Optional[dict]:
    pruner = getattr(model, "ast_pruner", None)
    if pruner is None:
        return None
    out: Dict[str, Any] = {}

    # Some pruners expose a dict; others store private runtime fields.
    ro = getattr(pruner, "runtime_overrides", None)
    if isinstance(ro, dict):
        out["runtime_overrides"] = dict(ro)

    # v5.4 AST2 pruner uses private runtime fields.
    if hasattr(pruner, "_runtime_force_dense"):
        try:
            out["force_dense"] = bool(getattr(pruner, "_runtime_force_dense"))
        except Exception:
            pass
    if hasattr(pruner, "_runtime_rho_token"):
        try:
            v = getattr(pruner, "_runtime_rho_token")
            out["rho_token"] = (None if v is None else float(v))
        except Exception:
            pass
    if hasattr(pruner, "_runtime_token_temperature"):
        try:
            v = getattr(pruner, "_runtime_token_temperature")
            out["token_temperature"] = (None if v is None else float(v))
        except Exception:
            pass

    # Best-effort: some older pruners expose public fields.
    for k in ("force_dense", "rho_token", "token_temperature"):
        if k in out:
            continue
        if hasattr(pruner, k):
            try:
                v = getattr(pruner, k)
                out[k] = bool(v) if k == "force_dense" else float(v)
            except Exception:
                pass

    return out if out else None


def _topk_correct_frac(logits: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    """Per-sample correctness for top-k. Returns float tensor in [0,1] of shape [B]."""
    k = int(k)
    if k <= 1:
        return (logits.argmax(dim=1) == targets).float()
    _, pred = logits.topk(k, dim=1)
    return pred.eq(targets.view(-1, 1)).any(dim=1).float()


def _as_float(val, name: str) -> float:
    """Convert config values that might be strings into floats with a clear error."""
    try:
        return float(val)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected {name} to be numeric, but got {val!r}.") from exc


def _split_stats_for_metrics(d: Dict[str, Any]) -> tuple[Dict[str, float], Dict[str, Any]]:
    """
    Split stats dict into:
      - scalars: values castable to float for flat metrics.json
      - extra: structured values (dict/list/tuple/str/None/others) preserved for audit

    Also converts torch tensors to python scalars when possible.
    """
    scalars: Dict[str, float] = {}
    extra: Dict[str, Any] = {}

    if not d:
        return scalars, extra

    for k, v in d.items():
        # torch tensor -> python scalar if possible
        if torch.is_tensor(v):
            try:
                v = v.detach().cpu().item()
            except Exception:
                # keep tensor as repr in extra
                extra[k] = str(v)
                continue

        # numeric -> scalar
        if isinstance(v, bool):
            scalars[k] = float(int(v))
            continue
        if isinstance(v, numbers.Number):
            try:
                scalars[k] = float(v)
                continue
            except Exception:
                extra[k] = str(v)
                continue

        # structured -> extra
        if isinstance(v, (dict, list, tuple, str)) or v is None:
            extra[k] = v
            continue

        # fallback -> extra as string
        extra[k] = str(v)

    # best-effort JSON-serializable extra
    try:
        json.dumps(extra)
    except Exception:
        def _to_jsonable(x):
            if torch.is_tensor(x):
                try:
                    return float(x.detach().cpu().item())
                except Exception:
                    return str(x)
            if isinstance(x, dict):
                return {str(kk): _to_jsonable(vv) for kk, vv in x.items()}
            if isinstance(x, (list, tuple)):
                return [_to_jsonable(xx) for xx in x]
            if isinstance(x, bool):
                return bool(x)
            if isinstance(x, numbers.Number):
                return float(x)
            if isinstance(x, (str, type(None))):
                return x
            return str(x)

        extra = _to_jsonable(extra)

    return scalars, extra


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
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.ema.state_dict().items():
                if k in msd:
                    v.copy_(v * self.decay + msd[k] * (1.0 - self.decay))

def save_checkpoint_single_device(
    out_dir: Path,
    tag: str,
    *,
    model: nn.Module,
    ema_model: Optional[ModelEMA],
    optimizer: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    epoch: int,
    best_acc1: float,
    seal_digest: str,
    run_id: str,
    ast_sched_applied: Optional[dict] = None,
) -> Path:
    ckpt_dir = Path(out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{tag}.pth"
    payload = {
        "epoch": int(epoch),
        "best_acc1": float(best_acc1),
        "seal_digest": str(seal_digest),
        "run_id": str(run_id),
        "model": model.state_dict(),
        "ema": (ema_model.ema.state_dict() if ema_model is not None else None),
        # v5.4+: persist AST schedule + runtime overrides for reproducible eval/test
        "ast_sched_applied": (dict(ast_sched_applied) if isinstance(ast_sched_applied, dict) else None),
        "ast_pruner_runtime_model": _extract_ast_pruner_runtime(model),
        "ast_pruner_runtime_ema": (_extract_ast_pruner_runtime(ema_model.ema) if ema_model is not None else None),
        "optimizer": optimizer.state_dict(),
        "scaler": (scaler.state_dict() if scaler is not None else None),
    }
    _atomic_torch_save(payload, ckpt_path)
    return ckpt_path


def _cfg_get(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _cfg_select(cfg: Any, key: str, default: Any = None) -> Any:
    """Safe OmegaConf path read.
    Works for DictConfig and plain dicts.
    Returns default when path is missing.
    Compatible with older OmegaConf that may not support `default=` kwarg.
    """
    try:
        # OmegaConf 2.2+ supports default=
        v = OmegaConf.select(cfg, key, default=default)
        return v
    except TypeError:
        # older OmegaConf: no default kwarg
        try:
            v = OmegaConf.select(cfg, key)
        except Exception:
            v = None
        return default if v is None else v
    except Exception:
        return default

def _build_run_signature(cfg) -> Dict[str, Any]:
    detailed_cfg = _cfg_get(cfg, "detailed_place", None)
    lookahead_cfg = _cfg_get(detailed_cfg, "lookahead", _cfg_get(cfg, "lookahead", {}))
    policy_switch_cfg = _cfg_get(detailed_cfg, "policy_switch", _cfg_get(cfg, "policy_switch", {}))
    action_families = _cfg_get(policy_switch_cfg, "action_families", None)
    moves_enabled = bool(action_families) if action_families is not None else bool(_cfg_get(cfg, "moves_enabled", False))
    lookahead_k = int(_cfg_get(lookahead_cfg, "topk", _cfg_get(lookahead_cfg, "k", 0) or 0))
    bandit_type = str(_cfg_get(policy_switch_cfg, "bandit_type", "eps_greedy"))
    policy_switch_enabled = bool(_cfg_get(policy_switch_cfg, "enabled", False))
    cache_size = int(_cfg_get(policy_switch_cfg, "cache_size", 0) or 0)
    cache_enabled = bool(cache_size > 0)
    cache_key_schema_version = str(_cfg_get(policy_switch_cfg, "cache_key_schema_version", "v5.4"))
    return {
        "moves_enabled": moves_enabled,
        "lookahead_k": lookahead_k,
        "bandit_type": bandit_type,
        "policy_switch": policy_switch_enabled,
        "cache_enabled": cache_enabled,
        "cache_key_schema_version": cache_key_schema_version,
    }


def _seed_worker(worker_id: int, base_seed: int) -> None:
    seed = base_seed + worker_id
    seed_everything(seed)


def build_dataloaders(cfg):
    train_ds = UCF101Dataset(cfg, split="train")
    val_ds = UCF101Dataset(cfg, split="val")
    print(f"[DEBUG] len(train_ds)={len(train_ds)}, len(val_ds)={len(val_ds)}")
    batch_size = int(getattr(cfg.data, "batch_size", cfg.train.batch_size))
    # Some single-device configs don't have `training:` (they use `train:`).
    base_seed = int(_cfg_select(cfg, "training.seed", _cfg_select(cfg, "train.seed", 0)) or 0)
    generator = torch.Generator()
    generator.manual_seed(base_seed)
    worker_init = partial(_seed_worker, base_seed=base_seed)
    pin_memory = bool(getattr(cfg.data, "pin_memory", True))
    persistent_workers = bool(getattr(cfg.data, "persistent_workers", True)) and int(cfg.data.num_workers) > 0
    prefetch_factor = int(getattr(cfg.data, "prefetch_factor", 2))
    train_kwargs = dict(
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        worker_init_fn=worker_init,
        generator=generator,
        pin_memory=pin_memory,
    )
    if int(cfg.data.num_workers) > 0:
        train_kwargs.update(dict(persistent_workers=persistent_workers, prefetch_factor=prefetch_factor))
    train_loader = DataLoader(
        train_ds,
        **train_kwargs,
    )
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
    val_loader = DataLoader(
        val_ds,
        **val_kwargs,
    )
    return train_loader, val_loader


def build_test_loader(cfg):
    test_ds = UCF101Dataset(cfg, split="test")
    batch_size = int(getattr(cfg.data, "batch_size", cfg.train.batch_size))
    base_seed = int(_cfg_select(cfg, "training.seed", _cfg_select(cfg, "train.seed", 0)) or 0)
    generator = torch.Generator()
    generator.manual_seed(base_seed)
    worker_init = partial(_seed_worker, base_seed=base_seed)
    pin_memory = bool(getattr(cfg.data, "pin_memory", True))
    persistent_workers = bool(getattr(cfg.data, "persistent_workers", True)) and int(cfg.data.num_workers) > 0
    prefetch_factor = int(getattr(cfg.data, "prefetch_factor", 2))
    test_kwargs = dict(
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        worker_init_fn=worker_init,
        generator=generator,
        pin_memory=pin_memory,
    )
    if int(cfg.data.num_workers) > 0:
        test_kwargs.update(dict(persistent_workers=persistent_workers, prefetch_factor=prefetch_factor))
    test_loader = DataLoader(
        test_ds,
        **test_kwargs,
    )
    return test_loader


def train_single_device(
    cfg,
    out_dir: str | Path | None = None,
    trace_events_path: str | Path | None = None,
    run_id: str | None = None,
    seal_digest: str | None = None,
):
    ctr = getattr(cfg, "_contract", None)
    if ctr is None or not bool(getattr(ctr, "stamped_v54", False)):
        raise RuntimeError(
            "v5.4 CONTRACT: cfg not validated/stamped. "
            "Call validate_and_fill_defaults(...) via SPEC_D OneCommand entrypoint."
        )
    if not getattr(cfg, "contract", None) or not getattr(cfg.contract, "seal_digest", None):
        raise RuntimeError("v5.4 CONTRACT: missing seal_digest; boot not completed.")
    contract = getattr(cfg, "contract", None)
    if contract is None or not bool(getattr(contract, "validated", False)):
        raise RuntimeError(
            "v5.4 contract violation: cfg must be validated via validate_and_fill_defaults() "
            "so requested/effective config is auditable."
        )
    specv = str(getattr(contract, "spec_version", getattr(contract, "version", "")))
    if specv not in ("v5.4", "v5.4-stable"):
        raise RuntimeError(f"v5.4 contract violation: unexpected spec_version={specv!r}")
    if seal_digest is None:
        seal_digest = getattr(contract, "seal_digest", None)
    if not seal_digest:
        raise RuntimeError("v5.4 P0: missing cfg.contract.seal_digest (contract evidence not sealed)")
    device = get_device(cfg.train.device)
    device_type = device.type
    logger = setup_logger()
    metrics_path = None
    trace_dir = None
    # Robust seed read (no `training:` in A1 dense baseline).
    seed = int(
        (_cfg_select(cfg, "train.seed", 0) or 0)
        or (_cfg_select(cfg, "training.seed", 0) or 0)
        or 0
    )
    if out_dir is None and hasattr(cfg, "train") and getattr(cfg.train, "out_dir", None):
        out_dir = Path(cfg.train.out_dir)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "metrics.json"
        cfg_hash = str(seal_digest)
        cfg_path = str(getattr(cfg, "cfg_path", "") or getattr(getattr(cfg, "train", None), "cfg_path", "") or "")
    baseline_export_path = os.environ.get("BASELINE_STATS_EXPORT", "").strip()

    if run_id is None:
        run_id = str(getattr(getattr(cfg, "train", None), "run_id", "") or "")
    if not run_id:
        run_id = stable_hash({"mode": "single_device_train", "seed": int(seed), "seal_digest": str(seal_digest)})
    signature = build_trace_signature_v54(cfg=cfg, run_id=run_id, seal_digest=seal_digest)

    if trace_events_path is None and out_dir is not None:
        trace_base = out_dir / "trace"
        trace_meta = init_trace_dir_v54(
            base_dir=trace_base,
            run_id=str(run_id),
            cfg=cfg,
            signature=signature,
            signature_v54=signature,
            required_signature_fields=REQUIRED_SIGNATURE_FIELDS,
            run_meta={"mode": "single_device_train", "seed_id": int(seed), "run_id": str(run_id)},
            extra_manifest={"task": "single_device", "out_dir": str(out_dir)},
        )
        trace_dir = Path(trace_meta["trace_dir"])
        trace_events_path = Path(trace_meta["trace_events"])
    elif trace_events_path is not None:
        trace_events_path = Path(trace_events_path)
        trace_dir = trace_events_path.parent
        trace_dir.mkdir(parents=True, exist_ok=True)

    if trace_events_path is None:
        raise RuntimeError("v5.4 contract violation: trace_events_path missing for single-device training")

    if trace_events_path is not None:
        def _to_plain(obj, resolve: bool):
            if obj is None:
                return None
            if OmegaConf.is_config(obj):
                return OmegaConf.to_container(obj, resolve=resolve)
            return obj

        contract_meta = getattr(cfg, "_contract", None)
        requested_cfg = getattr(contract_meta, "requested_config_snapshot", None) if contract_meta is not None else None
        if requested_cfg is None:
            requested_cfg = get_nested(cfg, "_contract.requested_config_snapshot", None)
        requested_cfg = _to_plain(requested_cfg, resolve=False)
        if requested_cfg is None:
            requested_cfg = {}

        effective_cfg = getattr(contract_meta, "effective_config_snapshot", None) if contract_meta is not None else None
        if effective_cfg is None:
            effective_cfg = OmegaConf.to_container(cfg, resolve=True)
        effective_cfg = _to_plain(effective_cfg, resolve=True)
        if effective_cfg is None:
            effective_cfg = OmegaConf.to_container(cfg, resolve=True)

        contract_overrides = getattr(contract_meta, "overrides", None) if contract_meta is not None else None
        if contract_overrides is None:
            contract_overrides = get_nested(cfg, "_contract.overrides", None)
        contract_overrides = _to_plain(contract_overrides, resolve=False)
        if contract_overrides is None:
            contract_overrides = []
        if isinstance(contract_overrides, dict):
            contract_overrides = [contract_overrides]
        elif not isinstance(contract_overrides, list):
            contract_overrides = []

        trace_header_payload = build_trace_header_payload_v54(
            cfg=cfg,
            signature=signature,
            requested_config_snapshot=requested_cfg,
            effective_config_snapshot=effective_cfg,
            contract_overrides=contract_overrides,
        )
        assert_trace_header_v54(trace_header_payload, strict=True)
        append_trace_event_v54(
            trace_events_path,
            "trace_header",
            payload=trace_header_payload,
            run_id=run_id,
            step=0,
        )
    train_loader, val_loader = build_dataloaders(cfg)
    test_loader = None
    model_type = str(_cfg_select(cfg, "training.model_type", "video") or "video")
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

    lr = _as_float(cfg.train.lr, "cfg.train.lr")
    weight_decay = _as_float(cfg.train.weight_decay, "cfg.train.weight_decay")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=cfg.train.amp)
    ema_cfg = getattr(cfg.train, "ema", None)
    ema_enabled = bool(getattr(ema_cfg, "enabled", False))
    ema_decay = float(getattr(ema_cfg, "decay", 0.9999) or 0.9999)
    ema_eval = bool(getattr(ema_cfg, "eval", True))
    ema_model = ModelEMA(model, decay=ema_decay) if ema_enabled else None

    lr_schedule = str(getattr(cfg.train, "lr_schedule", "none") or "none").lower()
    min_lr = float(getattr(cfg.train, "min_lr", 0.0) or 0.0)
    warmup_epochs = int(getattr(cfg.train, "warmup_epochs", 0) or 0)
    steps_per_epoch = len(train_loader) if hasattr(train_loader, "__len__") else 0
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = int(cfg.train.epochs) * steps_per_epoch

    label_smoothing = float(getattr(cfg.train, "label_smoothing", 0.0) or 0.0)
    mixup_alpha = float(getattr(cfg.train, "mixup_alpha", 0.0) or 0.0)
    mixup_prob = float(getattr(cfg.train, "mixup_prob", 1.0) or 1.0)
    mixup_switch_off_epoch = int(getattr(cfg.train, "mixup_switch_off_epoch", -1) or -1)

    grad_clip_norm = float(getattr(cfg.train, "grad_clip_norm", 0.0) or 0.0)
    nan_guard = bool(getattr(cfg.train, "nan_guard", True))
    skip_step_on_nonfinite = bool(getattr(cfg.train, "skip_step_on_nonfinite", True))
    proxy_weight_dir = str(getattr(cfg.hw, "weight_dir", "") or getattr(cfg.hw, "proxy_weight_dir", ""))
    if not proxy_weight_dir:
        raise RuntimeError("[ProxyMissing] cfg.hw.weight_dir or cfg.hw.proxy_weight_dir must be set.")
    hw_proxy = LayerHwProxy(
        cfg.hw.device_name,
        cfg.hw.gpu_yaml,
        proxy_weight_dir,
        run_ctx={
            "img": int(cfg.model.img_size),
            "bs": int(getattr(cfg.train, "batch_size", getattr(cfg.data, "batch_size", 1)) or 1),
            "depth": int(cfg.model.depth),
            "embed_dim": int(cfg.model.embed_dim),
            "num_heads": int(cfg.model.num_heads),
            "mlp_ratio": float(cfg.model.mlp_ratio),
            "tp_world_size": int(getattr(cfg.hw, "tp_world_size", 1) or 1),
            "runs": int(getattr(cfg.hw, "proxy_runs", 10) or 10),
            "warmup": int(getattr(cfg.hw, "proxy_warmup", 5) or 5),
        },
    )
    stable_hw_cfg = getattr(cfg, "stable_hw", None)
    stable_state: Dict[str, Any] = {}
    if out_dir is not None:
        stable_state["out_dir"] = str(out_dir)
    if stable_hw_cfg and bool(getattr(stable_hw_cfg, "enabled", True)):
        init_locked_acc_ref(cfg, stable_state)
        init_hw_refs_from_baseline_stats(cfg, stable_state, stable_hw_cfg=stable_hw_cfg)
        # ---- v5.4: always materialize stable_hw_state.json even if epochs==0 / early stop ----
        if out_dir is not None and stable_hw_cfg:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            try:
                with (out_path / "stable_hw_state.json").open("w", encoding="utf-8") as f:
                    json.dump(stable_state, f, indent=2)
            except Exception:
                pass

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

    def _to_scalar(val, default: float = 0.0) -> float:
        if hasattr(val, "detach"):
            return float(val.detach().cpu().item())
        if isinstance(val, (int, float)):
            return float(val)
        return float(default)

    def _build_gating_payload(
        *,
        epoch: int,
        inner_step: int,
        loss_value,
        hw_loss_value,
        acc_now_value: float,
        lambda_hw_eff_value: float,
    ) -> Dict[str, Any]:
        def _sanitize_hw_metric(metric: Any) -> Dict[str, Any]:
            defaults = {"latency_ms": None, "peak_mem_mb": None, "power_w": None}
            payload: Dict[str, Any] = {}
            if isinstance(metric, dict):
                payload.update(metric)
            for key, default in defaults.items():
                payload.setdefault(key, default)
            for key, value in list(payload.items()):
                if isinstance(value, float) and math.isnan(value):
                    payload[key] = None
            return payload

        hw_dbg = stable_state.get("hw_dbg", stable_state.get("hw_debug", {})) or {}
        loss_scalar = _to_scalar(loss_value, 0.0)
        hw_loss_raw = _to_scalar(hw_loss_value, 0.0)
        lambda_hw_eff_scalar = float(lambda_hw_eff_value)
        total_loss_hw_part = float(lambda_hw_eff_scalar) * float(hw_loss_raw)
        total_loss_acc_part = float(loss_scalar) - total_loss_hw_part
        inner_step = int(inner_step)
        global_step = int(epoch) * max(1, len(train_loader)) + inner_step
        guard_mode = str(stable_state.get("guard_mode", "")).upper()
        gate = "allow_hw"
        if float(lambda_hw_eff_scalar) <= 0.0 or guard_mode in ("VIOLATE", "RECOVERY", "WARMUP"):
            gate = "reject_hw"
        acc_ref = _to_scalar(stable_state.get("acc_ref"), 0.0)   # allow None during warmup
        acc_now = float(acc_now_value)
        acc_used_raw = stable_state.get("acc_used", stable_state.get("acc_used_value", None))
        acc_used = _to_scalar(acc_used_raw, acc_now)            # if None -> fall back to acc_now
        acc_drop = float(stable_state.get("acc_drop", 0.0) or 0.0)
        acc_drop_max = stable_state.get("acc_drop_max", None)
        if acc_drop_max is None:
            acc_drop_max = stable_state.get("epsilon_drop", 0.0)
        acc_drop_max = float(acc_drop_max or 0.0)
        return {
            "outer_iter": int(epoch),
            "inner_step": int(inner_step),
            "epoch": int(epoch),
            "global_step": int(global_step),
            "candidate_id": f"single_device_ep{int(epoch):04d}_st{int(inner_step):06d}",
            "gate": str(gate),
            "reason_code": str(stable_state.get("gating_reason", "n/a")),
            "gating_id": str(stable_state.get("gating_id", f"g{epoch}-{inner_step}")),
            "gating_reason": str(stable_state.get("gating_reason", "n/a")),
            "acc_ref": float(acc_ref),
            "acc_used": float(acc_used),
            "acc_now": float(acc_now),
            "acc_ref_source": str(stable_state.get("acc_ref_source", "unknown")),
            "acc_drop": float(acc_drop),
            "acc_drop_max": float(acc_drop_max),
            "acc_guard_mode": str(stable_state.get("acc_guard_mode", stable_state.get("guard_mode", "hard"))),
            "acc_guard_delta": float(stable_state.get("acc_guard_delta", stable_state.get("acc_drop", 0.0))),
            "acc_guard_emergency": bool(stable_state.get("acc_guard_emergency", False)),
            "hard_gating_active": bool(stable_state.get("hard_gating_active", False)),
            "locked_acc_ref_enabled": bool(
                getattr(getattr(getattr(cfg, "stable_hw", None), "locked_acc_ref", None), "enabled", True)
            ),
            "no_drift_enabled": bool(
                getattr(getattr(getattr(cfg, "stable_hw", None), "no_drift", None), "enabled", True)
            ),
            "hw_metric_raw": _sanitize_hw_metric(hw_dbg.get("raw_metric")),
            "hw_metric_ref": _sanitize_hw_metric(hw_dbg.get("ref_metric")),
            "hw_metric_normed": _sanitize_hw_metric(hw_dbg.get("normed_metric")),
            "hw_loss_raw": float(hw_loss_raw),
            "hw_loss_used": float(hw_loss_raw) if float(lambda_hw_eff_scalar) > 0.0 else 0.0,
            "lambda_hw_effective": float(lambda_hw_eff_scalar),
            "lambda_hw_max": float(stable_state.get("lambda_hw_max", float(getattr(cfg.hw, "lambda_hw", 0.0)))),
            "hw_scale_schema_version": str(hw_dbg.get("normalize_version") or "v5.4_ratio_v1"),
            "total_loss": float(loss_scalar),
            "total_loss_hw_part": float(total_loss_hw_part),
            "total_loss_acc_part": float(total_loss_acc_part),
            "total_loss_scalar": float(loss_scalar),
            "mapping_decision_used": False,
            "layout_decision_used": False,
            "mapping_algo": "single_device",
            "layout_algo": "single_device",
            "mapping_effective": {},
            "layout_effective": {},
            "mapping_was_cached": True,
            "layout_was_cached": True,
        }

    val_cfg = _cfg_get(cfg, "val", None)
    test_cfg = _cfg_get(cfg, "test", None)

    fast_val_max_batches = int(_cfg_get(val_cfg, "fast_max_batches", 0) or 0)
    fast_val_every_epochs = int(_cfg_get(val_cfg, "fast_every_epochs", 1) or 1)

    # full_every_epochs:
    #   0 => never run full during training (except maybe last epoch if full_on_last_epoch=True)
    #  >0 => run full every N epochs (and maybe last epoch)
    full_val_every_epochs = int(_cfg_get(val_cfg, "full_every_epochs", 0) or 0)
    full_val_max_batches = int(_cfg_get(val_cfg, "full_max_batches", 0) or 0)

    log_slim = bool(int(os.environ.get("LOG_SLIM", "0")))

    full_on_last_epoch = bool(_cfg_get(val_cfg, "full_on_last_epoch", True))
    val_log_interval = int(_cfg_get(val_cfg, "log_interval", 50) or 50)
    if log_slim:
        val_log_interval = max(int(val_log_interval), 200)

    log_interval_steps = int(os.environ.get("LOG_INTERVAL_STEPS", default=(200 if log_slim else 10)))
    log_interval_steps = max(1, int(log_interval_steps))
    val_use_tqdm = bool(_cfg_get(val_cfg, "use_tqdm", True))
    run_final_test = bool(_cfg_get(test_cfg, "run_final_test", False))

    best_acc = 0.0
    best_ckpt_path: Optional[Path] = None
    start_epoch = 0

    # ---- AUTO RESUME (crash recovery) ------------------------------------
    # Opt-in via env AUTO_RESUME=1.
    # Restores: model/optimizer/scaler + epoch + best_acc1.
    auto_resume = str(os.environ.get("AUTO_RESUME", "0")).strip().lower() in ("1", "true", "yes", "y", "on")
    if auto_resume:
        ckpt_path = Path(out_dir) / "checkpoints" / "last.pth"
        if ckpt_path.exists():
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                # Accept either full payload or bare state_dict
                model_state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
                if isinstance(model_state, dict):
                    model.load_state_dict(model_state, strict=True)
                if isinstance(ckpt, dict) and "optimizer" in ckpt and ckpt["optimizer"] is not None:
                    optimizer.load_state_dict(ckpt["optimizer"])
                if scaler is not None and isinstance(ckpt, dict) and ckpt.get("scaler", None) is not None:
                    try:
                        scaler.load_state_dict(ckpt["scaler"])
                    except Exception:
                        pass
                if ema_enabled and ema_model is not None and isinstance(ckpt, dict) and ckpt.get("ema", None) is not None:
                    try:
                        ema_model.ema.load_state_dict(ckpt["ema"], strict=True)
                    except Exception:
                        pass

                last_epoch = int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1
                start_epoch = max(0, last_epoch + 1)
                if isinstance(ckpt, dict) and "best_acc1" in ckpt and ckpt["best_acc1"] is not None:
                    best_acc = float(ckpt["best_acc1"])

                logger.info(f"[AUTO_RESUME] loaded {ckpt_path} (start_epoch={start_epoch}, best_acc1={best_acc:.6f})")
            except Exception as e:
                logger.warning(f"[AUTO_RESUME] failed to load {ckpt_path}: {e}. Start from scratch.")
                start_epoch = 0
    last_acc = 0.0
    ran_epochs = 0
    early_stop_triggered = False
    ok = False
    steps_done = 0
    gating_epochs = 0
    freeze_epochs = 0
    total_epochs = 0
    try:
        assert_cfg_sealed_or_violate(cfg, seal_digest, trace_events_path, step=0)
        for epoch in range(start_epoch, cfg.train.epochs):
            assert_cfg_sealed_or_violate(cfg, seal_digest, trace_events_path, step=epoch)
            ran_epochs += 1
            steps_done = ran_epochs
            total_epochs += 1
            stable_hw_enabled = bool(getattr(stable_hw_cfg, "enabled", True)) if stable_hw_cfg else False
            if stable_hw_enabled:
                legacy_loss_lambda = float(getattr(getattr(cfg, "loss", None), "lambda_hw", 0.0) or 0.0)
                legacy_hw_lambda = float(getattr(getattr(cfg, "hw", None), "lambda_hw", 0.0) or 0.0)
                if (legacy_loss_lambda != 0.0 or legacy_hw_lambda != 0.0) and not stable_state.get(
                    "_legacy_lambda_warned", False
                ):
                    logger.info(
                        "[StableHW] NOTE: legacy cfg.loss.lambda_hw/cfg.hw.lambda_hw will be ignored; "
                        "using stable_hw_state.lambda_hw_effective."
                    )
                    stable_state["_legacy_lambda_warned"] = True
                stable_hw_schedule(epoch, stable_hw_cfg, stable_state)
                prev_val = stable_state.get("val_acc1_last", None)
                prev_train_ema = stable_state.get("train_acc1_ema", None)
                mk = str(get_accuracy_metric_key(cfg)).lower().strip()
                use_train_ema = mk in ("train_acc1_ema", "train_ema")
                stable_decision, allow_discrete_updates = apply_accuracy_guard(
                    epoch=epoch,
                    stable_hw_cfg=cfg,
                    stable_hw_state=stable_state,
                    val_metric_or_none=float(prev_val) if (not use_train_ema and prev_val is not None) else None,
                    has_val_this_epoch=bool((not use_train_ema) and (prev_val is not None)),
                    train_ema_or_none=float(prev_train_ema) if (use_train_ema and prev_train_ema is not None) else None,
                )
                stable_state = stable_decision.state
                stable_state["allow_discrete_updates"] = bool(allow_discrete_updates)
                if str(stable_state.get("guard_mode", "")).upper() != "HW_OPT":
                    gating_epochs += 1
                if not bool(stable_state.get("allow_discrete_updates", True)):
                    freeze_epochs += 1
                if trace_dir is not None:
                    acc_now = float(last_acc) if last_acc is not None else _to_scalar(stable_state.get("acc_ref"), 0.0)
                    gating_payload = _build_gating_payload(
                        epoch=epoch,
                        inner_step=0,
                        loss_value=locals().get("loss", None),
                        hw_loss_value=locals().get("L_hw", None),
                        acc_now_value=acc_now,
                        lambda_hw_eff_value=float(stable_state.get("lambda_hw_effective", 0.0)),
                    )
                    append_trace_event_v54(
                        trace_events_path,
                        "gating",
                        payload=gating_payload,
                        run_id=run_id,
                        step=int(epoch),
                    )

                # v5.4: if stop_on_violation already triggered, do not proceed with training
                if bool(stable_decision.stop_training):
                    logger.warning(
                        f"[StableHW] stop_on_violation already triggered before epoch={epoch} training. Stop now."
                    )
                    early_stop_triggered = True
                    break
                # ---- v5.4 restart window: apply lr_restart_mul once per restart epoch ----
                if stable_hw_enabled and bool(stable_state.get("request_lr_restart", False)):
                    last_applied = int(stable_state.get("_lr_restart_applied_epoch", -999999))
                    if last_applied != int(epoch):
                        _ctrl = getattr(getattr(stable_hw_cfg, "accuracy_guard", None), "controller", None)
                        if _ctrl is None:
                            _ctrl = getattr(stable_hw_cfg, "controller", {})  # legacy fallback
                        mul = float(getattr(_ctrl, "lr_restart_mul", 2.0) or 2.0)
                        for pg in optimizer.param_groups:
                            pg["lr"] = float(pg.get("lr", lr)) * mul
                        stable_state["_lr_restart_applied_epoch"] = int(epoch)
                    stable_state["request_lr_restart"] = False
            if stable_hw_enabled:
                lambda_hw_eff = float(stable_state.get("lambda_hw_effective", 0.0))
            else:
                lambda_hw_eff = float(getattr(getattr(cfg, "hw", None), "lambda_hw", 0.0) or 0.0)

            stable_state["lambda_hw_effective"] = float(lambda_hw_eff)
            stable_state.setdefault("lambda_hw_base", float(stable_state.get("lambda_hw_base", 0.0)))

            # -------------------------
            # AST schedule (dense warmup -> ramp -> stabilize)
            # This affects token gating (rho/temperature) and lambda_AST multiplier only.
            # IMPORTANT: runtime overrides MUST be applied to BOTH model and EMA model,
            # otherwise eval/test silently uses default rho_target and drifts from training.
            # -------------------------
            ast_sched = compute_ast_schedule_effective(cfg, int(epoch))
            ast_sched = _maybe_freeze_ast_schedule(ast_sched, stable_state, epoch=int(epoch))
            stable_state["ast_sched_last_applied"] = (
                dict(ast_sched) if isinstance(ast_sched, dict) else {"phase": "disabled"}
            )
            stable_state["ast_sched_last_epoch"] = int(epoch)
            lambda_ast_eff = float(getattr(getattr(cfg, "loss", None), "lambda_AST", 1.0) or 1.0)
            if isinstance(ast_sched, dict) and ast_sched.get("phase") != "disabled":
                lambda_ast_eff = float(ast_sched.get("lambda_ast", lambda_ast_eff))
                _apply_ast_runtime_overrides_to_model(model, cfg, ast_sched)
                if ema_model is not None:
                    _apply_ast_runtime_overrides_to_model(ema_model.ema, cfg, ast_sched)
                sched0 = getattr(getattr(cfg, "ast", None), "schedule", None)
                warm_e = int(getattr(sched0, "warmup_epochs", 0) or 0) if sched0 is not None else 0
                freeze_now = bool(stable_state.get("freeze_schedule", False)) or bool(
                    stable_state.get("freeze_ast_schedule", False)
                )

                # Log at key transition points to confirm pruning actually starts.
                if epoch == start_epoch or epoch == warm_e or epoch == (warm_e + 1) or freeze_now or (epoch % 5 == 0):
                    logger.info(
                        "[ASTSchedule] epoch=%s phase=%s force_dense=%s rho_token=%.4f temp=%.4f lambda_ast=%.4f freeze=%s",
                        int(epoch),
                        str(ast_sched.get("phase")),
                        bool(ast_sched.get("force_dense", False)),
                        float(ast_sched.get("rho_token", 1.0)),
                        float(ast_sched.get("token_temperature", 0.1)),
                        float(ast_sched.get("lambda_ast", lambda_ast_eff)),
                        bool(freeze_now),
                    )
                if epoch == start_epoch:
                    logger.info(
                        "[ASTSchedule] enabled: warmup=%s, ramp=%s, curve=%s",
                        getattr(sched0, "warmup_epochs", None),
                        getattr(sched0, "ramp_epochs", None),
                        getattr(sched0, "curve", None),
                    )

            model.train()
            last_hw_stats = None
            total_steps = len(train_loader) if hasattr(train_loader, "__len__") else None
            disable_tqdm = log_slim or (not sys.stdout.isatty()) or bool(int(os.environ.get("TQDM_DISABLE", "0")))
            train_pbar = tqdm(
                enumerate(train_loader),
                total=total_steps,
                desc=f"Train e{epoch}",
                leave=True,
                disable=disable_tqdm,
            )
            for step, batch in train_pbar:
                global_step = int(epoch) * max(1, steps_per_epoch) + int(step)
                if lr_schedule == "cosine":
                    lr_cur = _compute_lr(global_step)
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr_cur
                x = batch["video"].to(device)
                y = batch["label"].to(device)
                if epoch == 0 and step == 0:
                    logger.info("[DEBUG] train batch video.shape=%s", tuple(x.shape))
                mixup_enabled = (
                    mixup_alpha > 0.0
                    and (random.random() < mixup_prob)
                    and (mixup_switch_off_epoch < 0 or epoch < mixup_switch_off_epoch)
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
                    y_a = y
                    y_b = y[perm]
                optimizer.zero_grad(set_to_none=True)
                with autocast(device_type, enabled=cfg.train.amp):
                    if model_type == "video_audio":
                        if audio is None:
                            audio = batch["audio"].to(device)
                        logits, info = model(x, audio, return_intermediate=True)
                    else:
                        logits, info = model(x, return_intermediate=True)
                    if mixup_enabled:
                        L_task = (
                            mixup_lam * F.cross_entropy(logits, y_a, label_smoothing=label_smoothing)
                            + (1.0 - mixup_lam) * F.cross_entropy(logits, y_b, label_smoothing=label_smoothing)
                        )
                    else:
                        L_task = F.cross_entropy(logits, y, label_smoothing=label_smoothing)

                    model_info = info.get("model_info", {}) if isinstance(info, dict) else {}
                    if hw_proxy is not None and "layers_cfg" not in model_info:
                        try:
                            segments = build_segments_from_model(model, cfg, model_info=model_info, precision=1)
                            bs = int(getattr(cfg.train, "batch_size", getattr(cfg.data, "batch_size", 1)) or 1)
                            nf = int(getattr(cfg.data, "num_frames", 1))
                            bs_eff = max(1, bs * nf)
                            layers_cfg = []
                            for seg in segments:
                                keep = seg.keep_factors or {}
                                layers_cfg.append(
                                    {
                                        "layer_kind": seg.kind,
                                        "flops": float(seg.flops) * bs_eff,
                                        "bytes": float(seg.bytes) * bs_eff,
                                        "embed_dim": int(seg.embed_dim),
                                        "num_heads": int(seg.num_heads),
                                        "mlp_ratio": float(seg.mlp_ratio),
                                        "seq_len": int(seg.seq_len),
                                        "precision": float(seg.precision),
                                        "keep_ratio": float(keep.get("token_keep", 1.0)),
                                    }
                                )
                            model_info = dict(model_info)
                            model_info["layers_cfg"] = layers_cfg
                            info["model_info"] = model_info
                        except Exception as exc:
                            if step == 0 and epoch == 0:
                                logger.warning(
                                    "[WARN] failed to build layers_cfg for proxy: %s: %s",
                                    type(exc).__name__,
                                    exc,
                                )
                    L_hw, hw_stats = compute_hw_loss(
                        cfg,
                        hw_proxy,
                        model_info=model_info,
                        stable_hw_cfg=stable_hw_cfg,
                        stable_hw_state=stable_state,
                    )
                    if stable_hw_cfg:
                        last_hw_stats = dict(hw_stats) if isinstance(hw_stats, dict) else {}

                    L_ast = info["L_AST"] if isinstance(info, dict) and "L_AST" in info else logits.new_tensor(0.0)

                    loss = L_task + float(lambda_ast_eff) * L_ast + lambda_hw_eff * L_hw
                    assert "hw_loss_weighted" not in (hw_stats or {}), (
                        "NoDoubleScale violated: hw_loss should not be weighted inside hw_loss module."
                    )
                if nan_guard and (not torch.isfinite(loss.detach())):
                    logger.error(
                        "[NaNGuard] non-finite loss at epoch=%s step=%s: %s",
                        epoch,
                        step,
                        float(loss.detach().cpu().item()) if torch.is_tensor(loss) else loss,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    if skip_step_on_nonfinite:
                        continue
                    raise FloatingPointError("Non-finite loss encountered in training step.")
                # v5.4 contract: NoDoubleScale (lambda_hw only applied once via stable_hw lambda_hw_eff)
                assert "lambda_hw" not in str(type(L_hw)).lower()  # cheap guard (won't catch all, but prevents accidental wrapping)
                assert float(lambda_hw_eff) >= 0.0
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
                if ema_model is not None:
                    ema_model.update(model)
                if step % log_interval_steps == 0:
                    # NOTE: Under Mixup, raw argmax-vs-y accuracy is NOT meaningful.
                    # We compute a mixup-weighted correctness so train acc can be compared to val/test.
                    with torch.no_grad():
                        if mixup_enabled:
                            top1_a = _topk_correct_frac(logits.detach(), y_a, 1)
                            top1_b = _topk_correct_frac(logits.detach(), y_b, 1)
                            top5_a = _topk_correct_frac(logits.detach(), y_a, 5)
                            top5_b = _topk_correct_frac(logits.detach(), y_b, 5)
                            acc1 = (mixup_lam * top1_a + (1.0 - mixup_lam) * top1_b).mean()
                            acc5 = (mixup_lam * top5_a + (1.0 - mixup_lam) * top5_b).mean()
                        else:
                            acc1 = _topk_correct_frac(logits.detach(), y, 1).mean()
                            acc5 = _topk_correct_frac(logits.detach(), y, 5).mean()

                    acc1_val = float(acc1.item())
                    acc5_val = float(acc5.item())
                    acc1_pct = acc1_val * 100.0
                    acc5_pct = acc5_val * 100.0

                    if stable_hw_enabled:
                        metric = get_accuracy_metric_key(stable_hw_cfg)
                        if metric in ("train_acc1_ema", "train_ema"):
                            # v5.4 contract: accuracy values are in [0,1]
                            update_train_acc1_ema(stable_hw_cfg, stable_state, float(acc1_val))
                    keep_ratio = float(model_info.get("token_keep", 1.0)) if isinstance(model_info, dict) else 1.0
                    attn_ratio = float(model_info.get("est_attn_flops_ratio", 1.0)) if isinstance(model_info, dict) else 1.0
                    train_pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "acc1": f"{acc1_val:.4f}",
                            "acc1%": f"{acc1_pct:.1f}",
                            "sparsity": f"{info['gates'].get('sparsity', {}).get('token', torch.tensor(0)).item():.4f}",
                            "keep": f"{keep_ratio:.3f}",
                            "attnR": f"{attn_ratio:.3f}",
                        }
                    )
                    stats = {
                        "epoch": epoch,
                        "step": step,
                        "loss": float(loss.item()),
                        "acc1": float(acc1_val),
                        "acc5": float(acc5_val),
                        "acc1_pct": float(acc1_pct),
                        "acc5_pct": float(acc5_pct),
                        "mixup": int(bool(mixup_enabled)),
                        "mixup_lam": float(mixup_lam) if mixup_enabled else 1.0,
                        "sparsity_token": info["gates"].get("sparsity", {}).get("token", torch.tensor(0)).item(),
                        # Masking-based pruning compute estimates (theoretical; does not claim real speedup).
                        "token_keep": float(model_info.get("token_keep", 1.0)) if isinstance(model_info, dict) else 1.0,
                        "seq_len_total": float(model_info.get("seq_len_total", 0.0)) if isinstance(model_info, dict) else 0.0,
                        "seq_len_effective": float(model_info.get("seq_len_effective", 0.0)) if isinstance(model_info, dict) else 0.0,
                        "est_attn_flops_ratio": float(model_info.get("est_attn_flops_ratio", 1.0)) if isinstance(model_info, dict) else 1.0,
                        "est_token_linear_flops_ratio": float(model_info.get("est_token_linear_flops_ratio", 1.0)) if isinstance(model_info, dict) else 1.0,
                        "lambda_hw": float(lambda_hw_eff),
                        "lambda_ast": float(lambda_ast_eff),
                        "hw_loss": float(L_hw.detach().cpu().item()),
                    }
                    stats.update(
                        {
                            "total_latency_ms": float(hw_stats.get("raw_latency_ms", hw_stats.get("latency_ms", 0.0))),
                            "peak_mem_mb": float(hw_stats.get("raw_mem_mb", hw_stats.get("mem_mb", 0.0))),
                            "comm_ms": float(hw_stats.get("comm_ms", hw_stats.get("comm_norm", 0.0))),
                        }
                    )
                    log_stats(logger, stats)
            last_acc = None
            last_epoch = epoch == cfg.train.epochs - 1

            # decide whether to run full/fast/skip
            do_full = False
            if full_on_last_epoch and last_epoch:
                do_full = True
            elif full_val_every_epochs and full_val_every_epochs > 0:
                do_full = (epoch % int(full_val_every_epochs) == 0)

            do_fast = False
            if (not do_full) and fast_val_every_epochs and fast_val_every_epochs > 0:
                do_fast = (epoch % int(fast_val_every_epochs) == 0)

            if not (do_full or do_fast):
                logger.info("[VAL] epoch=%s skipped (do_full=%s do_fast=%s)", epoch, do_full, do_fast)
            else:
                if do_full:
                    tag = "full"
                    max_batches = int(full_val_max_batches)  # 0 => ALL
                else:
                    tag = "fast"
                    max_batches = int(fast_val_max_batches)

                logger.info(
                    "[VAL] epoch=%s mode=%s max_batches=%s",
                    epoch,
                    tag,
                    "ALL" if max_batches == 0 else max_batches,
                )
                eval_model = ema_model.ema if (ema_model is not None and ema_eval) else model
                last_acc = validate(
                    eval_model,
                    val_loader,
                    device,
                    logger,
                    epoch,
                    cfg,
                    max_batches=max_batches,
                    log_interval=val_log_interval,
                    tag=tag,
                    use_tqdm=val_use_tqdm,
                    tqdm_leave=(tag in ("full", "test")),
                    ast_sched_override=stable_state.get("ast_sched_last_applied"),
                )
            if last_acc is not None and do_full:
                if last_acc > best_acc:
                    best_acc = float(last_acc)
                    best_ckpt_path = save_checkpoint_single_device(
                        out_dir,
                        "best",
                        model=model,
                        ema_model=ema_model,
                        optimizer=optimizer,
                        scaler=scaler,
                        epoch=epoch,
                        best_acc1=best_acc,
                        seal_digest=seal_digest,
                        run_id=run_id,
                        ast_sched_applied=stable_state.get("ast_sched_last_applied"),
                    )
            if stable_hw_enabled:
                stable_decision, _ = apply_accuracy_guard(
                    epoch=epoch,
                    stable_hw_cfg=cfg,
                    stable_hw_state=stable_state,
                    val_metric_or_none=float(last_acc) if last_acc is not None else None,
                    has_val_this_epoch=bool(last_acc is not None),
                    train_ema_or_none=float(stable_state.get("train_acc1_ema", 0.0))
                    if stable_state.get("train_acc1_ema") is not None
                    else None,
                )
                stable_state = stable_decision.state
                if trace_dir is not None:
                    inner_step = int(step) if "step" in locals() else 0
                    acc_now = float(last_acc) if last_acc is not None else _to_scalar(stable_state.get("acc_ref"), 0.0)
                    gating_payload = _build_gating_payload(
                        epoch=epoch,
                        inner_step=inner_step,
                        loss_value=locals().get("loss", None),
                        hw_loss_value=locals().get("L_hw", None),
                        acc_now_value=acc_now,
                        lambda_hw_eff_value=float(stable_state.get("lambda_hw_effective", 0.0)),
                    )
                    append_trace_event_v54(
                        trace_events_path,
                        "gating",
                        payload=gating_payload,
                        run_id=run_id,
                        step=int(epoch),
                    )

                # ===== v5.4 Acc-First Hard Gating: stop_on_violation 必须真的停止 =====
                if bool(stable_decision.stop_training):
                    val_acc1_str = f"{last_acc:.6f}" if last_acc is not None else "None"
                    logger.warning(
                        f"[StableHW] stop_on_violation triggered at epoch={epoch}: "
                        f"val_acc1={val_acc1_str}, acc_ref={stable_state.get('acc_ref')}, "
                        f"acc_floor={stable_state.get('acc_floor')}. Stop training now."
                    )
                    early_stop_triggered = True
                    break

                # v5.4: always call; stable_hw decides freeze vs ema-fallback internally
                update_hw_refs_from_stats(
                    cfg,
                    stable_state,
                    latest_hw_stats=last_hw_stats or {},
                    stable_hw_cfg=stable_hw_cfg,
                )
            guard_mode = str(stable_state.get("guard_mode", "HW_OPT")) if stable_hw_enabled else "disabled"
            allow_discrete = bool(stable_state.get("allow_discrete_updates", True)) if stable_hw_enabled else True
            print(
                f"[StableHW] epoch={epoch} mode={guard_mode} "
                f"lambda_hw_eff={lambda_hw_eff:.6g} allow_discrete={allow_discrete}"
            )
            logger.info(
                f"[StableHW][epoch={epoch}] "
                f"lambda_base={stable_state.get('lambda_hw_base')}, "
                f"lambda_eff={stable_state.get('lambda_hw_effective')}, "
                f"acc_ref={stable_state.get('acc_ref')}, "
                f"acc_floor={stable_state.get('acc_floor')}, "
                f"locked={stable_state.get('locked_acc_ref', stable_state.get('acc_ref_locked'))}, "
                f"allow_discrete={stable_state.get('allow_discrete_updates')}"
            )
            if metrics_path:
                stable_fields = stable_hw_log_fields(stable_state, cfg)
                metrics = {
                    "epoch": int(epoch),
                    "acc1": float(last_acc) if last_acc is not None else None,
                    "best_acc1": float(best_acc),
                    "loss": float(loss.item()),
                    "sparsity_token": float(info["gates"].get("sparsity", {}).get("token", torch.tensor(0)).item()),
                    "rho_target": float(getattr(cfg.ast, "rho_target", 0.0)),
                    "lambda_hw": float(lambda_hw_eff),
                    "hw_loss": float(L_hw.detach().cpu().item()),
                    "stable_hw_disabled": not bool(getattr(cfg.stable_hw, "enabled", False))
                    if getattr(cfg, "stable_hw", None)
                    else True,
                    "telemetry": {
                        "gating_on_ratio": float(gating_epochs) / max(1, int(total_epochs)),
                        "freeze_epoch_ratio": float(freeze_epochs) / max(1, int(total_epochs)),
                    },
                }
                metrics["last_hw_stats"] = {
                    "latency_ms": float((last_hw_stats or {}).get("latency_ms", 0.0)),
                    "energy_mj": float((last_hw_stats or {}).get("energy_mj", 0.0)),
                    "mem_mb": float((last_hw_stats or {}).get("mem_mb", 0.0)),
                    "comm_ms": float((last_hw_stats or {}).get("comm_ms", 0.0)),
                    "mem_peak_mb": float((last_hw_stats or {}).get("mem_mb", 0.0)),
                }
                metrics["stable_hw"] = stable_fields
                for k, v in stable_fields.items():
                    metrics[k] = v

                # ---- v5.4: hw_stats may contain structured fields (dict/list), do NOT float() everything ----
                _hw_scalars, _hw_extra = _split_stats_for_metrics(hw_stats or {})
                metrics.update(_hw_scalars)
                if _hw_extra:
                    metrics["hw_stats_extra"] = _hw_extra

                with metrics_path.open("w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                if baseline_export_path:
                    export_path = Path(baseline_export_path)
                    export_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_path = export_path.with_suffix(export_path.suffix + ".tmp")
                    with tmp_path.open("w", encoding="utf-8") as tmp_f:
                        json.dump(metrics, tmp_f, indent=2)
                    os.replace(tmp_path, export_path)

                # ---- v5.4: persist FULL hw_stats for audit (not only last_hw_stats) ----
                hw_stats_out = dict(hw_stats or {})
                hw_stats_out["last_hw_stats"] = dict(last_hw_stats or {})
                hw_stats_out.update(
                    {
                        "cfg_hash": cfg_hash,
                        "seed": int(getattr(cfg.train, "seed", 0)),
                    }
                )
                with (out_dir / "hw_stats.json").open("w", encoding="utf-8") as f:
                    json.dump(hw_stats_out, f, indent=2, ensure_ascii=False)
            if out_dir is not None and stable_hw_cfg:
                with (out_dir / "stable_hw_state.json").open("w", encoding="utf-8") as f:
                    json.dump(stable_state, f, indent=2)
        ok = True
    finally:
        if trace_dir is not None:
            # Must happen BEFORE finalize event
            early_stop = bool(early_stop_triggered) if "early_stop_triggered" in locals() else False
            epochs_ran = int(ran_epochs) if "ran_epochs" in locals() else 0
            summary_payload = {
                "ok": bool(ok),
                "reason": "done" if ok else "error",
                "steps_done": int(steps_done),
                "best_solution_valid": bool(ok and not early_stop_triggered),
            }
            summary_payload.update(
                build_baseline_trace_summary(
                    cfg,
                    stable_state if "stable_state" in locals() else {},
                )
            )
            update_trace_summary(trace_dir, summary_payload)
            finalize_trace_dir(
                trace_events_path,
                reason="done" if ok else "error",
                steps_done=int(steps_done),
                best_solution_valid=bool(ok and not early_stop_triggered),
            )

    if out_dir is not None:
        from utils.run_manifest import write_run_manifest
        git_sha = None
        seed = int(_cfg_select(cfg, "training.seed", _cfg_select(cfg, "train.seed", 0)) or 0)

        _metrics_summary = {
            "best_acc1": float(best_acc) if ("best_acc" in locals() and best_acc is not None) else None,
            "last_acc1": float(last_acc) if ("last_acc" in locals() and last_acc is not None) else None,
            "early_stop": bool(early_stop_triggered),
            "epochs_ran": int(ran_epochs),
            "telemetry": {
                "gating_on_ratio": float(gating_epochs) / max(1, int(total_epochs)),
                "freeze_epoch_ratio": float(freeze_epochs) / max(1, int(total_epochs)),
            },
        }

        write_run_manifest(
            out_dir=str(out_dir),
            cfg_path=cfg_path,
            cfg_hash=cfg_hash,
            run_id=run_id,
            seed=seed,
            git_sha=git_sha,
            code_root=str(Path(__file__).resolve().parents[1]),
            stable_hw_state=stable_state if "stable_state" in locals() else {},
            cfg=cfg,
            metrics_summary=_metrics_summary,
            extra={
                "task": "single_device",
                "out_dir": str(out_dir),
            },
        )

    if out_dir is not None and ran_epochs > 0:
        last_epoch_idx = max(0, int(ran_epochs) - 1)
        save_checkpoint_single_device(
            out_dir,
            "last",
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=last_epoch_idx,
            best_acc1=best_acc,
            seal_digest=seal_digest,
            run_id=run_id,
            ast_sched_applied=stable_state.get("ast_sched_last_applied")
            if isinstance(locals().get("stable_state"), dict)
            else None,
        )

    if run_final_test:
        if test_loader is None:
            test_loader = build_test_loader(cfg)
        best_path = best_ckpt_path
        if best_path is None and out_dir is not None:
            best_path = Path(out_dir) / "checkpoints" / "best.pth"
        ckpt = None
        test_ast_sched = None
        if best_path is not None and best_path.exists():
            ckpt = torch.load(best_path, map_location="cpu")
            model.load_state_dict(ckpt.get("model", ckpt))
            if ema_enabled and ema_model is not None and ckpt.get("ema", None) is not None:
                try:
                    ema_model.ema.load_state_dict(ckpt.get("ema"), strict=True)
                except Exception:
                    pass
            logger.info("[CKPT] Loaded best checkpoint from %s", best_path)
            # Restore the exact AST schedule used when this checkpoint was saved.
            if isinstance(ckpt, dict) and isinstance(ckpt.get("ast_sched_applied"), dict):
                test_ast_sched = dict(ckpt.get("ast_sched_applied"))
            else:
                ckpt_epoch = (
                    int(ckpt.get("epoch", int(cfg.train.epochs))) if isinstance(ckpt, dict) else int(cfg.train.epochs)
                )
                test_ast_sched = compute_ast_schedule_effective(cfg, ckpt_epoch)
        else:
            logger.info("[FINAL TEST] No best checkpoint captured; using current model state.")
            test_ast_sched = compute_ast_schedule_effective(cfg, int(cfg.train.epochs))

        # Apply schedule to both model and EMA model to avoid silent drift.
        if isinstance(test_ast_sched, dict) and test_ast_sched.get("phase") != "disabled":
            _apply_ast_runtime_overrides_to_model(model, cfg, test_ast_sched)
            if ema_model is not None:
                _apply_ast_runtime_overrides_to_model(ema_model.ema, cfg, test_ast_sched)
        test_epoch = int(cfg.train.epochs)
        eval_model = ema_model.ema if (ema_model is not None and ema_eval) else model
        validate(
            eval_model,
            test_loader,
            device,
            logger,
            test_epoch,
            cfg,
            max_batches=0,
            log_interval=val_log_interval,
            tag="test",
            use_tqdm=val_use_tqdm,
            tqdm_leave=True,
            ast_sched_override=test_ast_sched,
        )


def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    logger,
    epoch: int,
    cfg,
    max_batches: int | None = None,
    log_interval: int = 0,
    tag: str = "full",
    use_tqdm: bool = True,
    tqdm_leave: bool = False,
    ast_sched_override: Optional[dict] = None,
) -> float:
    model.eval()
    # Ensure eval/test uses the SAME AST runtime overrides as training.
    # (Critical for EMA eval and for final-test after loading best checkpoint.)
    try:
        ast_sched = (
            ast_sched_override if isinstance(ast_sched_override, dict) else compute_ast_schedule_effective(cfg, int(epoch))
        )
        if isinstance(ast_sched, dict) and ast_sched.get("phase") != "disabled":
            _apply_ast_runtime_overrides_to_model(model, cfg, ast_sched)
    except Exception:
        # Never let schedule plumbing crash eval.
        pass
    # Clip-level accuracy: each sampled clip/window counts as one sample.
    total = 0
    correct = 0

    # Video-level accuracy: aggregate logits over clips sharing video_id.
    video_logits_sum: dict[str, torch.Tensor] = {}
    video_counts: dict[str, int] = {}
    video_labels: dict[str, int] = {}
    saw_video_id = False
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    if max_batches is not None and max_batches > 0 and total_batches is not None:
        total_batches = min(total_batches, max_batches)
    logger.info("Starting validation epoch=%s mode=%s batches=%s", epoch, tag, total_batches)

    with torch.no_grad():
        iterable = enumerate(loader, start=1)
        pbar = None
        disable_tqdm = bool(int(os.environ.get("LOG_SLIM", "0"))) or (not sys.stdout.isatty()) or bool(int(os.environ.get("TQDM_DISABLE", "0")))
        if use_tqdm:
            pbar = tqdm(
                iterable,
                total=total_batches,
                desc=f"Val e{epoch} ({tag})",
                leave=bool(tqdm_leave),
                disable=disable_tqdm,
            )
            iterable = pbar

        # if user didn't set log_interval, still refresh postfix occasionally
        refresh_every = int(log_interval) if int(log_interval) > 0 else 20

        for batch_idx, batch in iterable:
            x = batch["video"].to(device)
            y = batch["label"].to(device)
            vids = batch.get("video_id", None)
            if vids is not None:
                saw_video_id = True
            if str(_cfg_select(cfg, "training.model_type", "video") or "video") == "video_audio":
                logits = model(x, batch["audio"].to(device))
            else:
                logits = model(x)

            pred = logits.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()

            if vids is not None:
                logits_cpu = logits.detach().float().cpu()
                y_cpu = y.detach().cpu()
                for j, vid in enumerate(vids):
                    k = str(vid)
                    if k not in video_logits_sum:
                        video_logits_sum[k] = logits_cpu[j].clone()
                        video_counts[k] = 1
                        video_labels[k] = int(y_cpu[j].item())
                    else:
                        video_logits_sum[k] += logits_cpu[j]
                        video_counts[k] += 1

            if pbar is not None and (batch_idx == 1 or batch_idx % refresh_every == 0):
                acc_now = correct / max(1, total)
                pbar.set_postfix({"acc1": f"{acc_now:.4f}", "seen": total})

            if log_interval and batch_idx % log_interval == 0:
                logger.info(
                    "[VAL PROGRESS] epoch=%s mode=%s batch=%s/%s acc1=%.4f",
                    epoch,
                    tag,
                    batch_idx,
                    total_batches if total_batches is not None else "?",
                    correct / max(1, total),
                )

            if max_batches is not None and max_batches > 0 and batch_idx >= max_batches:
                break

    acc_clip = correct / max(1, total)
    acc_video = acc_clip
    if saw_video_id and len(video_logits_sum) > 0:
        vids_all = list(video_logits_sum.keys())
        logits_video = torch.stack(
            [video_logits_sum[k] / float(max(1, video_counts[k])) for k in vids_all],
            dim=0,
        )
        labels_video = torch.tensor([video_labels[k] for k in vids_all], dtype=torch.long)
        pred_video = logits_video.argmax(dim=1)
        acc_video = float((pred_video == labels_video).float().mean().item())

    pref = str(getattr(getattr(cfg, "data", object()), "eval_aggregate", "clip")).lower()
    logger.info(
        "[val] epoch=%s mode=%s acc_clip=%.4f acc_video=%.4f n_video=%s (pref=%s)",
        epoch,
        tag,
        acc_clip,
        acc_video,
        len(video_logits_sum),
        pref,
    )
    logger.info("Finished validation epoch=%s mode=%s", epoch, tag)
    if pref in {"video", "video_avg", "video_mean"}:
        return float(acc_video)
    return float(acc_clip)
