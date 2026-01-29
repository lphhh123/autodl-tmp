"""Single-device AST2.0-lite trainer (SPEC §12.1)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from functools import partial
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from omegaconf import OmegaConf

from hw_proxy.layer_hw_proxy import LayerHwProxy
from hw_proxy.hw_loss import compute_hw_loss
from models.video_vit import VideoViT, VideoAudioAST
from utils.data_ucf101 import UCF101Dataset
from utils.logging_utils import setup_logger, log_stats
from utils.metrics import topk_accuracy
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


def _as_float(val, name: str) -> float:
    """Convert config values that might be strings into floats with a clear error."""
    try:
        return float(val)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected {name} to be numeric, but got {val!r}.") from exc


def _cfg_get(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


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
    base_seed = int(getattr(cfg.training, "seed", getattr(cfg.train, "seed", 0)))
    generator = torch.Generator()
    generator.manual_seed(base_seed)
    worker_init = partial(_seed_worker, base_seed=base_seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        worker_init_fn=worker_init,
        generator=generator,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        worker_init_fn=worker_init,
        generator=generator,
    )
    return train_loader, val_loader


def build_test_loader(cfg):
    test_ds = UCF101Dataset(cfg, split="test")
    batch_size = int(getattr(cfg.data, "batch_size", cfg.train.batch_size))
    base_seed = int(getattr(cfg.training, "seed", getattr(cfg.train, "seed", 0)))
    generator = torch.Generator()
    generator.manual_seed(base_seed)
    worker_init = partial(_seed_worker, base_seed=base_seed)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        worker_init_fn=worker_init,
        generator=generator,
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
    seed = int(getattr(cfg.train, "seed", 0) or getattr(cfg.training, "seed", 0) or 0)
    if out_dir is None and hasattr(cfg, "train") and getattr(cfg.train, "out_dir", None):
        out_dir = Path(cfg.train.out_dir)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "metrics.json"
        cfg_hash = str(seal_digest)
        cfg_path = str(getattr(cfg, "cfg_path", "") or getattr(getattr(cfg, "train", None), "cfg_path", "") or "")

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

    lr = _as_float(cfg.train.lr, "cfg.train.lr")
    weight_decay = _as_float(cfg.train.weight_decay, "cfg.train.weight_decay")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=cfg.train.amp)
    proxy_weight_dir = str(getattr(cfg.hw, "weight_dir", "") or getattr(cfg.hw, "proxy_weight_dir", ""))
    if not proxy_weight_dir:
        raise RuntimeError("[ProxyMissing] cfg.hw.weight_dir or cfg.hw.proxy_weight_dir must be set.")
    hw_proxy = LayerHwProxy(
        cfg.hw.device_name,
        cfg.hw.gpu_yaml,
        proxy_weight_dir,
        run_ctx={
            "img": int(cfg.model.img_size),
            "bs": int(getattr(cfg.data, "batch_size", 1) or 1),
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
    full_val_every_epochs = int(_cfg_get(val_cfg, "full_every_epochs", 1) or 1)
    val_log_interval = int(_cfg_get(val_cfg, "log_interval", 50) or 50)
    val_use_tqdm = bool(_cfg_get(val_cfg, "use_tqdm", True))
    run_final_test = bool(_cfg_get(test_cfg, "run_final_test", False))

    best_acc = 0.0
    best_state_dict = None
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
        for epoch in range(cfg.train.epochs):
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
                        for pg in opt.param_groups:
                            pg["lr"] = float(pg.get("lr", lr)) * mul
                        stable_state["_lr_restart_applied_epoch"] = int(epoch)
                    stable_state["request_lr_restart"] = False
            if stable_hw_enabled:
                lambda_hw_eff = float(stable_state.get("lambda_hw_effective", 0.0))
            else:
                lambda_hw_eff = float(getattr(getattr(cfg, "hw", None), "lambda_hw", 0.0) or 0.0)

            stable_state["lambda_hw_effective"] = float(lambda_hw_eff)
            stable_state.setdefault("lambda_hw_base", float(stable_state.get("lambda_hw_base", 0.0)))
            model.train()
            last_hw_stats = None
            total_steps = len(train_loader) if hasattr(train_loader, "__len__") else None
            train_pbar = tqdm(
                enumerate(train_loader),
                total=total_steps,
                desc=f"Train e{epoch}",
                leave=True,
            )
            for step, batch in train_pbar:
                x = batch["video"].to(device)
                y = batch["label"].to(device)
                if epoch == 0 and step == 0:
                    logger.info("[DEBUG] train batch video.shape=%s", tuple(x.shape))
                opt.zero_grad()
                with autocast(device_type, enabled=cfg.train.amp):
                    if model_type == "video_audio":
                        logits, info = model(x, batch["audio"].to(device), return_intermediate=True)
                    else:
                        logits, info = model(x, return_intermediate=True)
                    L_task = F.cross_entropy(logits, y)

                    model_info = info.get("model_info", {}) if isinstance(info, dict) else {}
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

                    loss = L_task + float(getattr(cfg.loss, "lambda_AST", 1.0)) * L_ast + lambda_hw_eff * L_hw
                    assert "hw_loss_weighted" not in (hw_stats or {}), (
                        "NoDoubleScale violated: hw_loss should not be weighted inside hw_loss module."
                    )
                # v5.4 contract: NoDoubleScale (lambda_hw only applied once via stable_hw lambda_hw_eff)
                assert "lambda_hw" not in str(type(L_hw)).lower()  # cheap guard (won't catch all, but prevents accidental wrapping)
                assert float(lambda_hw_eff) >= 0.0
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                if step % 10 == 0:
                    acc1, acc5 = topk_accuracy(logits.detach(), y, topk=(1, 5))
                    if stable_hw_enabled:
                        metric = get_accuracy_metric_key(stable_hw_cfg)
                        if metric in ("train_acc1_ema", "train_ema"):
                            update_train_acc1_ema(stable_hw_cfg, stable_state, float(acc1))
                    train_pbar.set_postfix(
                        {
                            "loss": f"{loss.item():.4f}",
                            "acc1": f"{acc1.item():.4f}",
                            "sparsity": f"{info['gates'].get('sparsity', {}).get('token', torch.tensor(0)).item():.4f}",
                        }
                    )
                    stats = {
                        "epoch": epoch,
                        "step": step,
                        "loss": loss.item(),
                        "acc1": acc1.item(),
                        "acc5": acc5.item(),
                        "sparsity_token": info["gates"].get("sparsity", {}).get("token", torch.tensor(0)).item(),
                        "lambda_hw": float(lambda_hw_eff),
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
            full_every = max(1, int(full_val_every_epochs))
            do_full = last_epoch or (epoch % full_every == 0)
            if do_full:
                tag = "full"
                max_batches = 0
            else:
                tag = "fast"
                max_batches = int(fast_val_max_batches)
            logger.info(
                "[VAL] epoch=%s mode=%s max_batches=%s",
                epoch,
                tag,
                "ALL" if max_batches == 0 else max_batches,
            )
            last_acc = validate(
                model,
                val_loader,
                device,
                logger,
                epoch,
                cfg,
                max_batches=max_batches,
                log_interval=val_log_interval,
                tag=tag,
                use_tqdm=val_use_tqdm,
            )
            if last_acc is not None and do_full:
                if last_acc > best_acc:
                    best_acc = float(last_acc)
                    best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
                    "mem_peak_mb": float((last_hw_stats or {}).get("mem_mb", 0.0)),
                }
                metrics["stable_hw"] = stable_fields
                for k, v in stable_fields.items():
                    metrics[k] = v
                metrics.update({k: float(v) for k, v in hw_stats.items()})
                with metrics_path.open("w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                hw_stats_out = dict(last_hw_stats or {})
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
        seed = int(getattr(cfg.training, "seed", getattr(cfg.train, "seed", 0)) or 0)

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

    if run_final_test:
        if test_loader is None:
            test_loader = build_test_loader(cfg)
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)
            logger.info("[FINAL TEST] Loaded best checkpoint from full validation acc=%.6f", best_acc)
        else:
            logger.info("[FINAL TEST] No best checkpoint captured; using current model state.")
        test_epoch = int(cfg.train.epochs)
        validate(
            model,
            test_loader,
            device,
            logger,
            test_epoch,
            cfg,
            max_batches=0,
            log_interval=val_log_interval,
            tag="test",
            use_tqdm=val_use_tqdm,
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
) -> float:
    model.eval()
    total = 0
    correct = 0
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    if max_batches is not None and max_batches > 0 and total_batches is not None:
        total_batches = min(total_batches, max_batches)
    logger.info("Starting validation epoch=%s mode=%s batches=%s", epoch, tag, total_batches)
    with torch.no_grad():
        iterable = enumerate(loader, start=1)
        if use_tqdm:
            iterable = tqdm(iterable, total=total_batches, desc=f"Val e{epoch} ({tag})", leave=False)
        for batch_idx, batch in iterable:
            x = batch["video"].to(device)
            y = batch["label"].to(device)
            if cfg.training.model_type == "video_audio":
                logits = model(x, batch["audio"].to(device))
            else:
                logits = model(x)
            pred = logits.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
            if log_interval and batch_idx % log_interval == 0:
                logger.info(
                    "[VAL PROGRESS] epoch=%s mode=%s batch=%s/%s",
                    epoch,
                    tag,
                    batch_idx,
                    total_batches if total_batches is not None else "?",
                )
            if max_batches is not None and max_batches > 0 and batch_idx >= max_batches:
                break
    acc = correct / max(1, total)
    logger.info(f"[val] epoch {epoch} acc={acc:.4f}")
    logger.info("Finished validation epoch=%s", epoch)
    return float(acc)
