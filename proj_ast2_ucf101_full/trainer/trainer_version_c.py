"""Version-C full trainer (SPEC §12.2)."""
from __future__ import annotations

import json
import math
import random
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
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


def save_checkpoint_version_c(
    out_dir: Path,
    tag: str,
    *,
    model: torch.nn.Module,
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
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": (scaler.state_dict() if scaler is not None else None),
    }
    _atomic_torch_save(payload, ckpt_path)
    return ckpt_path


def maybe_auto_resume_version_c(out_dir: Path, model, optimizer, scaler, logger):
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
            model.load_state_dict(model_state, strict=True)
        if isinstance(ckpt, dict) and ckpt.get("optimizer", None) is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and isinstance(ckpt, dict) and ckpt.get("scaler", None) is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
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
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


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
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        worker_init_fn=worker_init,
        generator=generator,
    )


def build_val_loader(cfg) -> DataLoader:
    ds = UCF101Dataset(cfg, split="val")
    batch_size = int(getattr(cfg.data, "batch_size", cfg.train.batch_size))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )


def validate_one_epoch(model: torch.nn.Module, val_loader: DataLoader, device: torch.device, amp: bool,
                       max_batches: int = 0, model_type: str = "video") -> float:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if max_batches and idx >= max_batches:
                break
            x = batch["video"].to(device)
            y = batch["label"].to(device)
            with autocast(device.type, enabled=amp):
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
        strategy=getattr(hw_cfg, "mapping_strategy", "greedy_local"),
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
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_slim = bool(int(os.environ.get("LOG_SLIM", "1")))
        log_interval_steps = int(os.environ.get("LOG_INTERVAL_STEPS", default=(200 if log_slim else 10)))
        log_interval_steps = max(1, int(log_interval_steps))
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

        val_loader = DataLoader(
            val_ds,
            batch_size=int(getattr(cfg.data, "batch_size", cfg.train.batch_size)),
            shuffle=False,
            num_workers=int(getattr(cfg.data, "num_workers", 4)),
            worker_init_fn=_seed_worker,
            generator=generator,
        )

        max_eval_batches = int(getattr(cfg.training, "stable_hw_eval_max_batches", 20))
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

        lr = _as_float(cfg.train.lr, "cfg.train.lr")
        weight_decay = _as_float(cfg.train.weight_decay, "cfg.train.weight_decay")
        optimizer_model = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
        scaler = GradScaler(device_type, enabled=cfg.train.amp)

        library = ChipletLibrary(cfg.hw.gpu_yaml)
        chiplet_slots = ChipletSlots(library, cfg.chiplet.candidate_types, cfg.hw.num_slots, cfg.chiplet.tau_init).to(device)
        optimizer_alpha = torch.optim.Adam(chiplet_slots.parameters(), lr=lr)

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

        global_step = 0
        last_segments: List = []
        last_mapping: List[int] = []
        stable_hw_state: Dict[str, Any] = {}
        stable_hw_state["run_signature"] = signature
        stable_hw_state["out_dir"] = str(out_dir)
        stable_hw_state["stable_hw_enabled"] = bool(getattr(stable_hw_cfg, "enabled", True)) if stable_hw_cfg else False
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
        allow_train_ema_fallback = get_nested(cfg, "stable_hw.allow_train_ema_fallback", None)
        if allow_train_ema_fallback is None:
            allow_train_ema_fallback = get_nested(cfg, "stable_hw.accuracy_guard.allow_train_ema_fallback", None)
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
        (out_dir / "stable_hw_state.json").write_text(
            safe_dumps(stable_hw_state, indent=2),
            encoding="utf-8",
        )
        stable_hw_state.setdefault("gating_reason_code", "")
        requested_cfg = get_nested(cfg, "_contract.requested_config_snapshot", {}) or {}
        effective_cfg = OmegaConf.to_container(cfg, resolve=True)
        contract_overrides = get_nested(cfg, "_contract.overrides", []) or []

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
        eff_fb = get_nested(cfg, "stable_hw.allow_train_ema_fallback", None)
        if eff_fb is None:
            eff_fb = get_nested(cfg, "stable_hw.accuracy_guard.allow_train_ema_fallback", None)

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
        run_state: Dict[str, Any] = {"last_model_info": None}
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
            start_outer, best_from_ckpt = maybe_auto_resume_version_c(out_dir, model, optimizer, scaler, logger)
            if best_from_ckpt is not None:
                best_acc1 = float(best_from_ckpt)
            for outer in range(start_outer, cfg.training.outer_epochs):
                assert_cfg_sealed_or_violate(cfg, seal_digest, trace_events_path, step=outer)
                ran_epochs += 1
                total_epochs += 1
                stable_hw_enabled = bool(getattr(stable_hw_cfg, "enabled", True)) if stable_hw_cfg else False
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
                    # ---- invariants (v5.4) ----
                    if stable_hw_state.get("acc_ref") is not None:
                        stable_hw_state.setdefault("_acc_ref_once", stable_hw_state["acc_ref"])
                        assert float(stable_hw_state["_acc_ref_once"]) == float(
                            stable_hw_state["acc_ref"]
                        ), "acc_ref drift detected"
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
                    lambda_hw_eff = float(stable_hw_state.get("lambda_hw_effective", 0.0))
                else:
                    lambda_hw_eff = float(getattr(getattr(cfg, "hw", None), "lambda_hw", 0.0) or 0.0)

                # ---- use effective lambda ONLY (already gated) ----
                stable_hw_state["lambda_hw_effective"] = float(lambda_hw_eff)
                stable_hw_state.setdefault("lambda_hw_base", float(stable_hw_state.get("lambda_hw_base", 0.0)))
                # -------------------------
                # AST schedule (dense warmup -> ramp -> stabilize)
                # This affects token gating (rho/temperature) and lambda_AST multiplier only.
                # -------------------------
                ast_sched = compute_ast_schedule_effective(cfg, int(outer))
                lambda_ast_eff = float(getattr(getattr(cfg, "loss", None), "lambda_AST", 1.0) or 1.0)
                if ast_sched.get("phase") != "disabled":
                    lambda_ast_eff = float(ast_sched.get("lambda_ast", lambda_ast_eff))
                    pruner = getattr(model, "ast_pruner", None)
                    if pruner is not None and hasattr(pruner, "set_runtime_overrides"):
                        pruner.set_runtime_overrides(
                            force_dense=bool(ast_sched.get("force_dense", False)),
                            rho_token=float(ast_sched.get("rho_token", getattr(getattr(cfg, "ast", None), "rho_token_target", 1.0))),
                            token_temperature=float(ast_sched.get("token_temperature", getattr(getattr(cfg, "ast", None), "token_temperature", 0.1))),
                        )
                    if outer == start_outer:
                        sched0 = getattr(getattr(cfg, "ast", None), "schedule", None)
                        logger.info(
                            "[ASTSchedule] enabled: warmup=%s, ramp=%s, curve=%s",
                            getattr(sched0, "warmup_epochs", None),
                            getattr(sched0, "ramp_epochs", None),
                            getattr(sched0, "curve", None),
                        )
                allow_discrete_updates = (
                    bool(stable_hw_state.get("allow_discrete_updates", True)) if stable_hw_enabled else True
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

                if (not allow_discrete_updates) and (cache.get("mapping") is None or cache.get("layout") is None):
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

                mapping_updated = False
                layout_updated = False

                need_update_mapping = ((outer % map_every) == 0) or (cache["mapping"] is None)
                need_update_layout = ((outer % lay_every) == 0) or (cache["layout"] is None)

                if stable_hw_enabled:
                    if stable_hw_state.get("lambda_hw_effective", 0.0) <= 0.0:
                        need_update_mapping = need_update_mapping and (cache["mapping"] is None)
                        need_update_layout = need_update_layout and (cache["layout"] is None)

                    if (need_update_mapping or need_update_layout) and (not allow_discrete_updates):
                        stable_hw_state["gating_reason_code"] = "discrete_updates_blocked"
                        print("[StableHW] Discrete updates frozen; reuse cached mapping/layout this step.")
                        need_update_mapping = False
                        need_update_layout = False

                if (not allow_discrete_updates) and cache["mapping"] is None:
                    stable_hw_state["discrete_frozen_init_mapping"] = True

                if need_update_mapping:
                    assert allow_discrete_updates, (
                        "StableHW gate closed: discrete updates must not run in RECOVERY/WARMUP"
                    )
                    mapping_res = _solve_mapping_for_cache(
                        model=model,
                        chiplet_slots=chiplet_slots,
                        mapping_solver=mapping_solver,
                        hw_proxy=hw_proxy,
                        wafer_layout=wafer_layout,
                        partitioner=partitioner,
                        hw_cfg=cfg.hw,
                        model_info=run_state.get("last_model_info"),
                    )
                    cache["mapping"] = mapping_res
                    cache["mapping_signature"] = mapping_res.get("mapping_sig") or mapping_res.get("signature")
                    mapping_updated = True
                else:
                    mapping_res = cache["mapping"]
                    mapping_updated = False

                if mapping_res is None:
                    raise RuntimeError("Mapping cache is empty after mapping step (mapping_res is None).")

                if need_update_layout:
                    assert allow_discrete_updates, (
                        "StableHW gate closed: discrete updates must not run in RECOVERY/WARMUP"
                    )
                    layout_res = _solve_layout_for_cache(
                        chiplet_slots=chiplet_slots,
                        wafer_layout=wafer_layout,
                        hw_cfg=cfg.hw,
                        mapping_result=mapping_res,
                    )
                    cache["layout"] = layout_res
                    cache["layout_signature"] = layout_res.get("signature")
                    layout_updated = True
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
                for step in range(cfg.training.inner_steps_ast):
                    try:
                        batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(loader)
                        batch = next(data_iter)
                    x = batch["video"].to(device)
                    y = batch["label"].to(device)
                    optimizer_model.zero_grad()
                    optimizer_alpha.zero_grad()
                    with autocast(device_type, enabled=cfg.train.amp):
                        if model_type == "video_audio":
                            logits, info = model(x, batch["audio"].to(device), return_intermediate=True)
                        else:
                            logits, info = model(x, return_intermediate=True)
                        run_state["last_model_info"] = info
                        # Hard guard: if model produced non-finite logits, skip the step.
                        if not torch.isfinite(logits).all():
                            run_state["nan_guard_skipped_steps"] = int(run_state.get("nan_guard_skipped_steps", 0)) + 1
                            logger.warning("[NaNGuard] non-finite logits detected (outer=%s step=%s); skipping step.", outer, step)
                            optimizer_model.zero_grad(set_to_none=True)
                            optimizer_alpha.zero_grad(set_to_none=True)
                            continue
                        L_task = F.cross_entropy(logits, y)
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
                        if twostage or float(lambda_hw_eff) <= 0.0:
                            hw_term = torch.zeros_like(L_hw)
                        else:
                            hw_term = float(lambda_hw_eff) * torch.nan_to_num(L_hw, nan=0.0, posinf=0.0, neginf=0.0)
                        # If HW loss went non-finite, skip this step entirely (keeps model/alpha stable).
                        if hw_nonfinite:
                            run_state["nan_guard_skipped_steps"] = int(run_state.get("nan_guard_skipped_steps", 0)) + 1
                            logger.warning("[NaNGuard] non-finite L_hw detected (outer=%s step=%s); skipping step.", outer, step)
                            optimizer_model.zero_grad(set_to_none=True)
                            optimizer_alpha.zero_grad(set_to_none=True)
                            continue
                        loss = L_task + float(lambda_ast_eff) * info["L_AST"] + hw_term
                        # ---- v5.4 audit: capture the exact loss components used ----
                        try:
                            ast_loss_val = info["L_AST"]
                        except Exception:
                            ast_loss_val = 0.0

                        def _to_f(x):
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
                        logger.warning("[NaNGuard] non-finite total loss (outer=%s step=%s); skipping step.", outer, step)
                        optimizer_model.zero_grad(set_to_none=True)
                        optimizer_alpha.zero_grad(set_to_none=True)
                        continue
                    scaler.scale(loss).backward()
                    scaler.step(optimizer_model)
                    if not twostage and update_alpha:
                        scaler.step(optimizer_alpha)
                    # v5.4: forbidden (P0-3) — layout must be updated via discrete assign-only agent + cache
                    scaler.update()
                    repaired = _repair_nonfinite_params_(model)
                    if update_alpha:
                        repaired = _repair_nonfinite_params_(chiplet_slots) or repaired
                    if repaired:
                        logger.warning("[NaNGuard] repaired non-finite parameters (outer=%s step=%s).", outer, step)
                    if step % log_interval_steps == 0:
                        acc1 = (logits.argmax(dim=1) == y).float().mean()
                        last_acc1 = float(acc1.item())
                        best_acc1 = float(acc1.item()) if best_acc1 is None else max(best_acc1, float(acc1.item()))
                        if stable_hw_enabled:
                            metric = get_accuracy_metric_key(stable_hw_cfg)
                            if metric in ("train_acc1_ema", "train_ema"):
                                update_train_acc1_ema(stable_hw_cfg, stable_hw_state, float(acc1))
                        model_info = info.get("model_info", {}) if isinstance(info, dict) else {}
                        stats = {
                            "outer": outer,
                            "step": step,
                            "loss": loss.item(),
                            "acc1": acc1.item(),
                            # Masking-based pruning compute estimates (theoretical; does not claim real speedup).
                            "token_keep": float(model_info.get("token_keep", 1.0)) if isinstance(model_info, dict) else 1.0,
                            "seq_len_total": float(model_info.get("seq_len_total", 0.0)) if isinstance(model_info, dict) else 0.0,
                            "seq_len_effective": float(model_info.get("seq_len_effective", 0.0)) if isinstance(model_info, dict) else 0.0,
                            "est_attn_flops_ratio": float(model_info.get("est_attn_flops_ratio", 1.0)) if isinstance(model_info, dict) else 1.0,
                            "est_token_linear_flops_ratio": float(model_info.get("est_token_linear_flops_ratio", 1.0)) if isinstance(model_info, dict) else 1.0,
                            "lambda_hw": float(lambda_hw_eff),
                            "allow_discrete_updates": bool(allow_discrete),
                            "mapping_updated": step_mapping_updated,
                            "layout_updated": step_layout_updated,
                            "mapping_cache_hit": not step_mapping_updated,
                            "layout_cache_hit": not step_layout_updated,
                            "mapping_signature": cache["mapping_signature"],
                            "layout_signature": cache["layout_signature"],
                        }
                        if (not log_slim) and hw_stats:
                            stats.update(hw_stats)
                        elif hw_stats:
                            # slim mode: keep only a tiny, stable subset
                            for _k in (
                                "L_hw_total",
                                "L_hw_norm",
                                "raw_latency_ms",
                                "raw_energy_mj",
                                "raw_mem_mb",
                                "raw_comm_ms",
                                "latency_ms",
                                "energy_mj",
                                "mem_mb",
                                "comm_ms",
                                "proxy_invalid_count",
                                "proxy_sanitize_count",
                                "hw_loss_nonfinite",
                            ):
                                if _k in hw_stats:
                                    stats[_k] = hw_stats[_k]
                        log_stats(logger, stats)
                        with log_path.open("a", encoding="utf-8") as f:
                            if log_slim:
                                # Minimal per-step record (keeps files small & easy to grep).
                                f.write(
                                    safe_dumps(
                                        {
                                            "step": int(global_step),
                                            "outer": int(outer),
                                            "loss": float(loss.item()),
                                            "acc1": float(acc1.item()),
                                            "lambda_hw": float(lambda_hw_eff),
                                            "lat_ms": float(
                                                hw_stats.get("raw_latency_ms", hw_stats.get("latency_ms", 0.0)) if hw_stats else 0.0
                                            ),
                                            "energy_mj": float(hw_stats.get("energy_mj", 0.0) if hw_stats else 0.0),
                                            "mem_mb": float(hw_stats.get("mem_mb", 0.0) if hw_stats else 0.0),
                                            "comm_ms": float(hw_stats.get("comm_ms", 0.0) if hw_stats else 0.0),
                                            "hw_loss": float(hw_stats.get("L_hw_total", 0.0) if hw_stats else 0.0),
                                            "mapping_cache_hit": (not step_mapping_updated),
                                            "layout_cache_hit": (not step_layout_updated),
                                        }
                                    )
                                    + "\n"
                                )
                            else:
                                f.write(
                                    safe_dumps(
                                        {
                                            "step": int(global_step),
                                            "outer": int(outer),
                                            "loss": float(loss.item()),
                                            "lat_ms": float(
                                                hw_stats.get(
                                                    "raw_latency_ms",
                                                    hw_stats.get("proxy_raw_latency_ms", hw_stats.get("latency_ms", 0.0)),
                                                )
                                            ),
                                            "energy_mj": float(hw_stats.get("energy_mj", 0.0)),
                                            "mem_mb": float(hw_stats.get("mem_mb", 0.0)),
                                            "lambda_hw": float(lambda_hw_eff),
                                            "stable_hw": stable_hw_log_fields(stable_hw_state, cfg),
                                            "mapping_updated": step_mapping_updated,
                                            "layout_updated": step_layout_updated,
                                            "mapping_cache_hit": (not step_mapping_updated),
                                            "layout_cache_hit": (not step_layout_updated),
                                            "mapping_signature": cache["mapping_signature"],
                                            "layout_signature": cache["layout_signature"],
                                        }
                                    )
                                    + "\n"
                                )
                    global_step += 1

                # Step B: alpha refinement
                if update_alpha:
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
                        if (not twostage) and (float(lambda_hw_eff) > 0.0) and torch.isfinite(loss_alpha).all():
                            loss_alpha.backward()
                            optimizer_alpha.step()
                        else:
                            # Skip alpha update if HW term is disabled or non-finite
                            run_state["nan_guard_skipped_alpha"] = int(run_state.get("nan_guard_skipped_alpha", 0)) + 1
                            if not torch.isfinite(loss_alpha).all():
                                logger.warning("[NaNGuard] non-finite alpha loss; skipping alpha step.")

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
                            logger.warning("[NaNGuard] non-finite layout loss; disabling layout refinement for this run.")
                            _repair_nonfinite_params_(wafer_layout)
                            update_layout = False
                            break
                        optimizer_layout.zero_grad(set_to_none=True)
                        L_layout.backward()
                        optimizer_layout.step()

                val_agg = str(getattr(getattr(cfg, "data", object()), "eval_aggregate", "clip"))
                val_acc1 = eval_acc1(
                    model,
                    val_loader,
                    device,
                    model_type=str(getattr(cfg.training, "model_type", "video")),
                    max_batches=max_eval_batches,
                    aggregate=val_agg,
                )
                if stable_hw_enabled:
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
                        stable_hw_state.setdefault("_acc_ref_once", stable_hw_state["acc_ref"])
                        assert float(stable_hw_state["_acc_ref_once"]) == float(
                            stable_hw_state["acc_ref"]
                        ), "acc_ref drift detected"

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
                        if stable_hw_state.get("no_drift_enabled", False):
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
                                        "allowed_by_no_drift": (not no_drift_enabled),
                                        "requested_mode": requested_mode,
                                        "effective_mode": effective_mode,
                                        "reason": "hw_ref_update",
                                    },
                                    run_id=run_id,
                                    step=int(outer),
                                )
                guard_mode = str(stable_hw_state.get("guard_mode", "HW_OPT")) if stable_hw_enabled else "disabled"
                allow_discrete = (
                    bool(stable_hw_state.get("allow_discrete_updates", True)) if stable_hw_enabled else True
                )
                print(
                    f"[StableHW] epoch={outer} mode={guard_mode} "
                    f"lambda_hw_eff={lambda_hw_eff:.6g} allow_discrete={allow_discrete}"
                )
                logger.info(
                    f"[StableHW][epoch={outer}] "
                    f"lambda_base={stable_hw_state.get('lambda_hw_base')}, "
                    f"lambda_eff={stable_hw_state.get('lambda_hw_effective')}, "
                    f"acc_ref={stable_hw_state.get('acc_ref')}, "
                    f"acc_floor={stable_hw_state.get('acc_floor')}, "
                    f"locked={stable_hw_state.get('locked_acc_ref', stable_hw_state.get('acc_ref_locked'))}, "
                    f"allow_discrete={stable_hw_state.get('allow_discrete_updates')}"
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

            requested_cfg = get_nested(cfg, "_contract.requested_config_snapshot", {}) or {}
            effective_cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
            contract_overrides = get_nested(cfg, "_contract.overrides", []) or []
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
