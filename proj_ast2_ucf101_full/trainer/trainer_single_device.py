"""Single-device AST2.0-lite trainer (SPEC §12.1)."""
from __future__ import annotations

import json
from pathlib import Path
from functools import partial
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from hw_proxy.layer_hw_proxy import LayerHwProxy
from hw_proxy.hw_loss import compute_hw_loss
from models.video_vit import VideoViT, VideoAudioAST
from utils.data_ucf101 import UCF101Dataset
from utils.logging_utils import setup_logger, log_stats
from utils.metrics import topk_accuracy
from utils.distributed_utils import get_device
from utils.eval_utils import eval_acc1
from utils.seed import seed_everything
from utils.trace_guard import init_trace_dir, append_trace_event_v54, finalize_trace_dir
from utils.trace_signature_v54 import build_signature_v54, REQUIRED_SIGNATURE_FIELDS
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


def train_single_device(cfg, out_dir: str | Path | None = None):
    device = get_device(cfg.train.device)
    device_type = device.type
    logger = setup_logger()
    metrics_path = None
    trace_dir = None
    run_id = str(getattr(cfg.train, "run_id", "")) or "single_device"
    if out_dir is None and hasattr(cfg, "train") and getattr(cfg.train, "out_dir", None):
        out_dir = Path(cfg.train.out_dir)
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = out_dir / "metrics.json"
        trace_dir = out_dir / "trace"
        sig = build_signature_v54(cfg, method_name="train_single_device")
        init_trace_dir(
            trace_dir,
            signature=sig,
            run_meta={"mode": "train_single_device", "run_id": run_id},
            required_signature_keys=REQUIRED_SIGNATURE_FIELDS,
        )
        trace_events_path = trace_dir / "trace_events.jsonl"
    train_loader, val_loader = build_dataloaders(cfg)
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
    hw_proxy = LayerHwProxy(cfg.hw.device_name, cfg.hw.gpu_yaml, cfg.hw.proxy_weight_dir)
    stable_hw_cfg = getattr(cfg, "stable_hw", None)
    stable_state: Dict[str, Any] = {}
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

    best_acc = 0.0
    last_acc = 0.0
    ran_epochs = 0
    early_stop_triggered = False
    ok = False
    steps_done = 0
    try:
        for epoch in range(cfg.train.epochs):
            ran_epochs += 1
            steps_done = ran_epochs
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
                if trace_dir is not None:
                    append_trace_event_v54(
                        trace_events_path,
                        "gating_decision",
                        payload={
                            "epoch": int(epoch),
                            "metric": str(stable_decision.reason.get("metric")),
                            "acc_ref": float(stable_decision.reason.get("acc_ref"))
                            if stable_decision.reason.get("acc_ref") is not None
                            else None,
                            "acc_current": (
                                float(stable_decision.reason.get("acc_current"))
                                if stable_decision.reason.get("acc_current") is not None
                                else None
                            ),
                            "guard_mode": str(stable_decision.guard_mode),
                            "lambda_hw_effective": float(stable_decision.lambda_hw_effective),
                            "gated": bool(stable_decision.guard_mode in ("VIOLATE", "RECOVERY")),
                            "reason": str(stable_decision.reason.get("violate", "")),
                        },
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
            for step, batch in enumerate(train_loader):
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
                            "total_latency_ms": float(hw_stats.get("latency_ms", 0.0)),
                            "peak_mem_mb": float(hw_stats.get("mem_mb", 0.0)),
                            "comm_ms": float(hw_stats.get("comm_ms", 0.0)),
                        }
                    )
                    log_stats(logger, stats)
            last_acc = validate(model, val_loader, device, logger, epoch, cfg)
            best_acc = max(best_acc, last_acc)
            if stable_hw_enabled:
                stable_decision, _ = apply_accuracy_guard(
                    epoch=epoch,
                    stable_hw_cfg=cfg,
                    stable_hw_state=stable_state,
                    val_metric_or_none=float(last_acc) if last_acc is not None else None,
                    has_val_this_epoch=True,
                    train_ema_or_none=float(stable_state.get("train_acc1_ema", 0.0))
                    if stable_state.get("train_acc1_ema") is not None
                    else None,
                )
                stable_state = stable_decision.state
                if trace_dir is not None:
                    append_trace_event_v54(
                        trace_events_path,
                        "gating_decision",
                        payload={
                            "epoch": int(epoch),
                            "metric": str(stable_decision.reason.get("metric")),
                            "acc_ref": float(stable_decision.reason.get("acc_ref"))
                            if stable_decision.reason.get("acc_ref") is not None
                            else None,
                            "acc_current": (
                                float(stable_decision.reason.get("acc_current"))
                                if stable_decision.reason.get("acc_current") is not None
                                else None
                            ),
                            "guard_mode": str(stable_decision.guard_mode),
                            "lambda_hw_effective": float(stable_decision.lambda_hw_effective),
                            "gated": bool(stable_decision.guard_mode in ("VIOLATE", "RECOVERY")),
                            "reason": str(stable_decision.reason.get("violate", "")),
                        },
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
                    latest_stats=last_hw_stats or {},
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
            if trace_dir is not None and last_hw_stats is not None:
                append_trace_event_v54(
                    trace_events_path,
                    "proxy_sanitize_summary",
                    payload={
                        "epoch": int(epoch),
                        "had_negative_latency": bool(last_hw_stats.get("sanitize", {}).get("had_negative", False))
                        if isinstance(last_hw_stats.get("sanitize", None), dict)
                        else False,
                        "latency_penalty": float(last_hw_stats.get("sanitize", {}).get("penalty", 0.0))
                        if isinstance(last_hw_stats.get("sanitize", None), dict)
                        else 0.0,
                    },
                    run_id=run_id,
                    step=int(epoch),
                )
            if metrics_path:
                metrics = {
                    "epoch": int(epoch),
                    "acc1": float(last_acc),
                    "best_acc1": float(best_acc),
                    "loss": float(loss.item()),
                    "sparsity_token": float(info["gates"].get("sparsity", {}).get("token", torch.tensor(0)).item()),
                    "rho_target": float(getattr(cfg.ast, "rho_target", 0.0)),
                    "lambda_hw": float(lambda_hw_eff),
                    "hw_loss": float(L_hw.detach().cpu().item()),
                    "stable_hw_disabled": not bool(getattr(cfg.stable_hw, "enabled", False))
                    if getattr(cfg, "stable_hw", None)
                    else True,
                }
                metrics["last_hw_stats"] = {
                    "latency_ms": float((last_hw_stats or {}).get("latency_ms", 0.0)),
                    "energy_mj": float((last_hw_stats or {}).get("energy_mj", 0.0)),
                    "mem_peak_mb": float((last_hw_stats or {}).get("mem_mb", 0.0)),
                }
                metrics["stable_hw"] = stable_hw_log_fields(stable_state)
                metrics.update({k: float(v) for k, v in hw_stats.items()})
                with metrics_path.open("w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                hw_stats_out = dict(last_hw_stats or {})
                hw_stats_out.update(
                    {
                        "cfg_hash": str(getattr(cfg.train, "cfg_hash", "")),
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
            finalize_trace_dir(
                trace_dir,
                summary_extra={
                    "reason": "done" if ok else "error",
                    "steps_done": int(steps_done),
                    "best_solution_valid": bool(ok and not early_stop_triggered),
                },
                run_id=run_id,
                step=int(steps_done),
            )

    if out_dir is not None:
        try:
            from utils.run_manifest import write_run_manifest

            write_run_manifest(
                out_dir=str(out_dir),
                cfg_path=str(getattr(cfg.train, "cfg_path", "")),
                cfg_hash=str(getattr(cfg.train, "cfg_hash", "")),
                seed=int(getattr(cfg.train, "seed", 0) or getattr(cfg.training, "seed", 0) or 0),
                stable_hw_state=stable_state,
                extra={
                    "budget_main_axis": "wall_time_s",
                    "dataset_id": getattr(getattr(cfg, "dataset", None), "dataset_id", None)
                    or getattr(getattr(cfg, "dataset", None), "name", "unknown"),
                },
                run_id=str(getattr(cfg.train, "run_id", "")) or "single_device",
                spec_version="v5.4",
                command=None,
            )
        except Exception:
            pass
    if trace_dir is not None:
        append_trace_event_v54(
            trace_events_path,
            "training_complete",
            payload={"early_stop": bool(early_stop_triggered), "epochs_ran": int(ran_epochs)},
            run_id=run_id,
            step=int(steps_done),
        )


def validate(model: nn.Module, loader: DataLoader, device: torch.device, logger, epoch: int, cfg) -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["video"].to(device)
            y = batch["label"].to(device)
            if cfg.training.model_type == "video_audio":
                logits = model(x, batch["audio"].to(device))
            else:
                logits = model(x)
            pred = logits.argmax(dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    acc = correct / max(1, total)
    logger.info(f"[val] epoch {epoch} acc={acc:.4f}")
    return float(acc)
