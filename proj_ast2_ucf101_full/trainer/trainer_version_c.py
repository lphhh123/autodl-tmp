"""Version-C full trainer (SPEC §12.2)."""
from __future__ import annotations

import json
import math
import random
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
from utils.trace_contract_v54 import REQUIRED_GATING_KEYS, REQUIRED_PROXY_SANITIZE_KEYS
from utils.trace_guard import (
    init_trace_dir_v54,
    append_trace_event_v54,
    finalize_trace_dir,
    update_trace_summary,
    build_baseline_trace_summary,
)
from utils.trace_signature_v54 import build_signature_v54, REQUIRED_SIGNATURE_FIELDS
from utils.config import AttrDict
from utils.config_utils import get_nested
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
        json.dump(layout_input, f, indent=2)

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
    canonical_a3_path.write_text(json.dumps(layout_input, indent=2), encoding="utf-8")
    run_root_path.write_text(json.dumps(layout_input, indent=2), encoding="utf-8")

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
    pos = wafer_layout.current_pos_continuous().detach().cpu().tolist()
    pos_round = [[round(float(x), 6) for x in row] for row in pos]
    signature = stable_hash({"pos": pos_round})
    return {"loss": float(L_layout.item()), "stats": layout_stats, "signature": signature}


def train_version_c(
    cfg,
    out_dir: Optional[str] = None,
    export_layout_input: bool = False,
    layout_export_dir: Optional[str] = None,
    seed: Optional[int] = None,
):
    # ===== v5.4 contract gate: forbid running with unvalidated cfg =====
    if (
        (not hasattr(cfg, "contract"))
        or (not bool(getattr(cfg.contract, "validated", False)))
        or (getattr(cfg.contract, "version", None) != "v5.4")
    ):
        raise RuntimeError(
            "[P0][v5.4] cfg is not validated/stamped by validate_and_fill_defaults(). "
            "Do NOT call train_version_c() with raw load_config(); use scripts/run_version_c.py "
            "or call validate_and_fill_defaults(cfg) explicitly."
        )

    device = get_device(cfg.train.device)
    device_type = device.type
    logger = setup_logger()
    if out_dir:
        cfg.out_dir = str(out_dir)
        cfg.train.out_dir = str(out_dir)
    if seed is not None:
        cfg.train.seed = int(seed)
    # ---- v5.4: allow config-driven export (OneCommand) ----
    if not export_layout_input:
        export_layout_input = bool(getattr(cfg, "export_layout_input", False))

    layout_export_dir_source = "cli"
    if layout_export_dir is None:
        layout_export_dir = getattr(cfg, "export_dir", None)
        layout_export_dir_source = "cfg"
    else:
        cfg.export_dir = layout_export_dir

    cfg_export_dir = str(layout_export_dir or "").strip()
    if not cfg_export_dir:
        # legacy fallback (kept) — but must be auditable
        layout_export_dir = str(Path(cfg.out_dir) / "exports" / "layout_input")
        layout_export_dir_source = "default_out_dir_exports"
    # out_dir: training outputs root
    out_dir = Path(getattr(cfg.train, "out_dir", "") or "outputs/version_c")
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_hash = getattr(getattr(cfg, "train", None), "cfg_hash", "") or ""
    cfg_path = getattr(getattr(cfg, "train", None), "cfg_path", "") or ""
    seed = int(getattr(cfg.train, "seed", 0) or getattr(cfg.training, "seed", 0) or 0)
    run_id = stable_hash(
        {
            "mode": "version_c_train",
            "cfg_hash": str(cfg_hash),
            "seed": int(seed),
        }
    )
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
        # layout_export_dir: ONLY for exporting layout_input.json (optional)
        layout_export_dir = Path(layout_export_dir) if layout_export_dir else None
        if layout_export_dir is not None:
            layout_export_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "logs" / "version_c_stats.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
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
            f"stable_hw.enabled={getattr(cfg.stable_hw, 'enabled', None)} "
            f"locked_acc_ref.enabled={getattr(cfg.locked_acc_ref, 'enabled', None)} "
            f"no_drift.enabled={getattr(cfg.no_drift, 'enabled', None)} "
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
            gating_summary = {
                "requested": {
                    "stable_hw_enabled": bool(_cfg_get(stable_cfg, "enabled", False)),
                    "no_drift_enabled": bool(_cfg_get(no_drift_cfg, "enabled", False)),
                    "hard_gating": bool(_cfg_get(getattr(cfg_obj, "accuracy_guard", None), "hard_gating", False)),
                },
                "effective": {
                    "stable_hw_enabled": bool(stable_state.get("stable_hw_enabled", _cfg_get(stable_cfg, "enabled", False))),
                    "allow_discrete_updates": bool(stable_state.get("allow_discrete_updates", True)),
                    "gating_reason_code": str(stable_state.get("gating_reason_code", "") or ""),
                },
            }
            manifest["gating_policy_summary"] = gating_summary
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

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

        mapping_only = bool(getattr(cfg.training, "mapping_only", False))
        layout_only = bool(getattr(cfg.training, "layout_only", False))
        twostage = bool(getattr(cfg.training, "twostage", False))
        if mapping_only:
            setattr(cfg.hw, "optimize_layout", False)
            setattr(cfg.hw, "mapping_only", True)
        if layout_only:
            setattr(cfg.hw, "layout_only", True)

        base_update_alpha = not layout_only
        update_layout = bool(getattr(cfg.hw, "optimize_layout", True))

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
        optimizer_model = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = GradScaler(device_type, enabled=cfg.train.amp)

        library = ChipletLibrary(cfg.hw.gpu_yaml)
        chiplet_slots = ChipletSlots(library, cfg.chiplet.candidate_types, cfg.hw.num_slots, cfg.chiplet.tau_init).to(device)
        optimizer_alpha = torch.optim.Adam(chiplet_slots.parameters(), lr=lr)

        hw_proxy = LayerHwProxy(cfg.hw.device_name, cfg.hw.gpu_yaml, cfg.hw.proxy_weight_dir)
        mapping_solver = MappingSolver(cfg.mapping.strategy, cfg.mapping.mem_limit_factor)
        wafer_layout = WaferLayout(cfg.hw.num_slots, cfg.hw.wafer_radius_mm).to(device)
        partitioner = PartitionPlanner(mapping_solver, wafer_layout, hw_proxy, cfg.partition)
        optimizer_layout = torch.optim.Adam(wafer_layout.parameters(), lr=1e-3)

        stable_hw_cfg = getattr(cfg, "stable_hw", None)
        iso_cfg_global = getattr(stable_hw_cfg, "discrete_isolation", None) if stable_hw_cfg else None
        layout_opt_steps = int(_get_iso_cfg_value(iso_cfg_global, "layout_opt_steps", 10) or 10)
        layout_opt_lr = float(_get_iso_cfg_value(iso_cfg_global, "layout_opt_lr", 5e-2) or 5e-2)
        layout_opt_grad_clip = float(_get_iso_cfg_value(iso_cfg_global, "layout_opt_grad_clip", 1.0) or 1.0)
        layout_opt = None
        if bool(_get_iso_cfg_value(iso_cfg_global, "optimize_layout", False)):
            layout_opt = torch.optim.Adam([wafer_layout.pos], lr=layout_opt_lr)

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
        allow_train_ema_fallback = (
            guard_ctrl_cfg.get("allow_train_ema_fallback", None)
            if isinstance(guard_ctrl_cfg, dict)
            else getattr(guard_ctrl_cfg, "allow_train_ema_fallback", None)
        )
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
            json.dumps(stable_hw_state, indent=2, ensure_ascii=False),
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

        append_trace_event_v54(
            trace_events_path,
            "trace_header",
            payload={
                "requested_config": requested_cfg,
                "effective_config": effective_cfg,
                "contract_overrides": contract_overrides,
                "requested": {
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
                    "allow_train_ema_fallback": _get_req(
                        "stable_hw.accuracy_guard.controller.allow_train_ema_fallback", None
                    ),
                },
                "effective": {
                    "mode": "version_c",
                    "stable_hw_enabled": bool(getattr(getattr(cfg, "stable_hw", None), "enabled", False)),
                    "locked_acc_ref_enabled": bool(locked_acc_ref_enabled),
                    "no_drift_enabled": bool(no_drift_enabled),
                    "no_double_scale_enabled": bool(
                        getattr(getattr(getattr(cfg, "stable_hw", None), "no_double_scale", None), "enabled", False)
                    ),
                    "allow_train_ema_fallback": bool(allow_train_ema_fallback)
                    if allow_train_ema_fallback is not None
                    else None,
                },
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
                    "hard_gating": bool(_get_req("accuracy_guard.hard_gating", None))
                    if _get_req("accuracy_guard.hard_gating", None) is not None
                    else None,
                },
                "signature": signature,
                "no_drift_enabled": bool(no_drift_enabled),
                "acc_ref_source": str(lar_source),
            },
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
            for outer in range(cfg.training.outer_epochs):
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

                # ---- P0 guard: never allow discrete gate to cause empty-cache crash ----
                if (not allow_discrete_updates) and (cache.get("mapping") is None or cache.get("layout") is None):
                    logger.info(
                        "[StableHW] Discrete gate is closed but cache is empty; initializing mapping/layout once."
                    )

                    # ---- v5.4 contract: explicit auditable fallback (requested vs effective) ----
                    cached_mapping = cache.get("mapping")
                    cached_layout = cache.get("layout")
                    prev_reason = str(stable_hw_state.get("decision_reason", "") or "")
                    fb_reason = "fallback_cache_empty_force_enable_discrete_updates"
                    stable_hw_state["decision_reason"] = (prev_reason + "|" + fb_reason) if prev_reason else fb_reason
                    stable_hw_state["discrete_frozen_init_mapping"] = True
                    stable_hw_state["gating_reason_code"] = fb_reason

                    allow_discrete_updates = True
                    stable_hw_state["allow_discrete_updates"] = True

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
                    optimizer_layout.zero_grad()
                    with autocast(device_type, enabled=cfg.train.amp):
                        if model_type == "video_audio":
                            logits, info = model(x, batch["audio"].to(device), return_intermediate=True)
                        else:
                            logits, info = model(x, return_intermediate=True)
                        run_state["last_model_info"] = info
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

                                layout_positions = wafer_layout.current_pos_continuous()
                                pos_round = [
                                    [round(float(x), 6) for x in row]
                                    for row in layout_positions.detach().cpu().tolist()
                                ]
                                cache["layout_signature"] = stable_hash({"pos": pos_round})
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
                        sum_latency += float(hw_stats.get("latency_ms", 0.0))
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
                        hw_term = float(lambda_hw_eff) * L_hw if not twostage else (L_hw * 0.0)
                        loss = L_task + cfg.loss.lambda_AST * info["L_AST"] + hw_term
                        # ---- v5.4 audit: capture the exact loss components used ----
                        try:
                            ast_loss_val = info["L_AST"]
                        except Exception:
                            ast_loss_val = 0.0

                        def _to_f(x):
                            if hasattr(x, "detach"):
                                return float(x.detach().cpu().item())
                            return float(x)

                        run_state["last_loss_components"] = {
                            "acc_loss": _to_f(L_task),
                            "ast_loss": _to_f(ast_loss_val),
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
                    scaler.scale(loss).backward()
                    scaler.step(optimizer_model)
                    if not twostage and update_alpha:
                        scaler.step(optimizer_alpha)
                    if not twostage and update_layout and allow_discrete:
                        scaler.step(optimizer_layout)
                    scaler.update()
                    if step % 10 == 0:
                        acc1 = (logits.argmax(dim=1) == y).float().mean()
                        last_acc1 = float(acc1.item())
                        best_acc1 = float(acc1.item()) if best_acc1 is None else max(best_acc1, float(acc1.item()))
                        if stable_hw_enabled:
                            metric = get_accuracy_metric_key(stable_hw_cfg)
                            if metric in ("train_acc1_ema", "train_ema"):
                                update_train_acc1_ema(stable_hw_cfg, stable_hw_state, float(acc1))
                        stats = {
                            "outer": outer,
                            "step": step,
                            "loss": loss.item(),
                            "acc1": acc1.item(),
                            "lambda_hw": float(lambda_hw_eff),
                            "allow_discrete_updates": bool(allow_discrete),
                            "mapping_updated": step_mapping_updated,
                            "layout_updated": step_layout_updated,
                            "mapping_cache_hit": not step_mapping_updated,
                            "layout_cache_hit": not step_layout_updated,
                            "mapping_signature": cache["mapping_signature"],
                            "layout_signature": cache["layout_signature"],
                        }
                        stats.update(hw_stats)
                        log_stats(logger, stats)
                        with log_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps({
                                "step": int(global_step),
                                "outer": int(outer),
                                "loss": float(loss.item()),
                                "lat_ms": float(hw_stats.get("latency_ms", 0.0)),
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
                            }) + "\n")
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

                        optimizer_alpha.zero_grad()
                        (loss_alpha := (float(lambda_hw_eff) * L_hw)).backward()
                        optimizer_alpha.step()

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
                        optimizer_layout.zero_grad()
                        L_layout.backward()
                        optimizer_layout.step()

                val_acc1 = eval_acc1(
                    model,
                    val_loader,
                    device,
                    model_type=str(getattr(cfg.training, "model_type", "video")),
                    max_batches=max_eval_batches,
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
                    payload = {
                        "candidate_id": int(outer),
                        "gate": gate,
                        "acc_ref": float(acc_ref_val),
                        "acc_now": float(acc_now),
                        "acc_used": float(acc_used_value if acc_used_value is not None else acc_now),
                        "acc_drop": float(stable_hw_state.get("acc_drop", 0.0)),
                        "acc_drop_max": float(acc_drop_max),
                        "acc_used_source": acc_used_source,
                        "acc_used_value": float(acc_used_value if acc_used_value is not None else acc_now),
                        "reason_code": reason_code,
                        "lambda_hw_effective": float(lambda_hw_eff),
                        "guard_mode": str(getattr(stable_decision, "guard_mode", "")),
                        "lambda_hw_base": float(getattr(stable_decision, "lambda_hw_base", 0.0) or 0.0),
                        "hw_loss_raw": float(hw_loss_norm or 0.0),
                        "hw_loss_used": float(hw_loss_used or 0.0),
                        "total_loss_scalar": float(total_loss.item())
                        if hasattr(total_loss, "item")
                        else float(total_loss or 0.0),
                        "total_loss_acc_part": float(acc_loss.item())
                        if hasattr(acc_loss, "item")
                        else float(acc_loss or 0.0),
                        "total_loss_hw_part": float(hw_part),
                        "acc_loss": float(acc_loss.item()) if hasattr(acc_loss, "item") else float(acc_loss or 0.0),
                        "total_loss": float(total_loss.item()) if hasattr(total_loss, "item") else float(total_loss or 0.0),
                        "reason": dict(getattr(stable_decision, "reason", {}) or {}),
                        "allow_discrete_updates": bool(stable_hw_state.get("allow_discrete_updates", True)),
                        "hw_metric_ref": hw_stats.get("proxy_refs", {}),
                        "hw_metric_raw": hw_stats.get("proxy_raw", {}),
                        "hw_metric_normed": hw_stats.get("proxy_used", {}),
                        "hw_scale_schema_version": "v5.4_hw_loss_norm",
                        "violate_streak": int(stable_hw_state.get("violate_streak", 0) or 0),
                        "epoch": int(epoch),
                        "outer_iter": int(outer_iter),
                        "global_step": int(global_step),
                    }
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

                if best_acc1 is None:
                    best_acc1 = float(last_acc1)
                else:
                    best_acc1 = max(float(best_acc1), float(last_acc1))

                stable_hw_state["val_acc1_best_seen"] = float(best_acc1) if best_acc1 is not None else None

                no_drift_cfg = getattr(cfg, "no_drift", None)
                if no_drift_cfg is None:
                    no_drift_cfg = getattr(stable_hw_cfg, "no_drift", None)
                stable_hw_state["_contract_no_drift"] = bool(getattr(no_drift_cfg, "enabled", False)) if no_drift_cfg else False
                norm = getattr(stable_hw_cfg, "normalize", None)
                stable_hw_state["_contract_ref_update"] = (
                    "frozen" if norm is None else str(getattr(norm, "ref_update", "frozen") or "frozen")
                )

                stable_state_path = log_path.parent / "stable_hw_state.json"
                stable_state_path.write_text(
                    json.dumps(stable_hw_state, indent=2, ensure_ascii=False),
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
            cfg_path=str(cfg.train.cfg_path),
            cfg_hash=str(getattr(cfg.train, "cfg_hash", "")),
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
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        hw_stats_out = dict(last_hw_stats or {})
        hw_stats_out.update(
            {
                "cfg_hash": cfg_hash,
                "seed": seed,
            }
        )
        with (out_dir / "hw_stats.json").open("w", encoding="utf-8") as f:
            json.dump(hw_stats_out, f, indent=2, ensure_ascii=False)
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
            requested_cfg = get_nested(cfg, "_contract.requested_config_snapshot", {}) or {}
            effective_cfg = cfg.to_dict() if hasattr(cfg, "to_dict") else {}
            signature = build_signature_v54(cfg, method_name="ours_version_c")

            append_trace_event_v54(
                trace_events_path=trace_events_path,
                event_type="trace_header",
                payload={
                    "schema_version": "v5.4",
                    "signature": signature,
                    "requested_config": requested_cfg,
                    "effective_config": effective_cfg,
                    "no_drift_enabled": bool(
                        getattr(getattr(getattr(cfg, "stable_hw", None), "no_drift", None), "enabled", False)
                    ),
                    "acc_ref_source": str(
                        getattr(getattr(getattr(cfg, "stable_hw", None), "locked_acc_ref", None), "source", "none")
                    ),
                    "stable_hw_state": {
                        "init_status": "failed_before_full_init",
                    },
                },
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
