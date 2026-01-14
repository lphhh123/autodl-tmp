"""Version-C full trainer (SPEC §12.2)."""
from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch
import torch.nn.functional as F
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
from mapping.segments import build_segments_from_model
from models.video_vit import VideoViT, VideoAudioAST
from utils.distributed_utils import get_device
from utils.logging_utils import setup_logger, log_stats
from utils.seed import seed_everything
from utils.stable_hw import apply_accuracy_guard, stable_hw_schedule, update_hw_refs_from_stats


def _as_float(val, name: str) -> float:
    """Convert config values that might be strings into floats with a clear error."""
    try:
        return float(val)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected {name} to be numeric, but got {val!r}.") from exc


def _seed_worker(worker_id: int, base_seed: int) -> None:
    seed = base_seed + worker_id
    seed_everything(seed)


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
    chiplet_slots: ChipletSlots,
    mapping_solver: MappingSolver,
    segments: List,
    mapping: List[int],
    wafer_layout: WaferLayout,
    seed: int = 0,
):
    """Export layout_input.json following SPEC v4.3.2 (§10.1).

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
    sigma_mm = float(getattr(getattr(cfg, "layout", None), "sigma_mm", 20.0))
    sw = getattr(getattr(cfg, "layout", None), "scalar_weights", None)
    w_comm = float(getattr(sw, "w_comm", 0.7)) if sw is not None else 0.7
    w_therm = float(getattr(sw, "w_therm", 0.3)) if sw is not None else 0.3
    w_penalty = float(getattr(sw, "w_penalty", 1000.0)) if sw is not None else 1000.0
    rng = np.random.default_rng(seed)
    baseline_eval = LayoutEvaluator(
        sigma_mm=sigma_mm,
        baseline={"L_comm_baseline": 1.0, "L_therm_baseline": 1.0},
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
        "layout_version": "v4.3.2",
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
    signature = f"seg{len(segments)}_map{hash(tuple(mapping))}"
    mapping_result["segments"] = segments
    mapping_result["signature"] = signature
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
            distance_scale=1e-9,
        )
    signature = f"{mapping_result.get('signature')}_layout"
    return {"loss": float(L_layout.item()), "stats": layout_stats, "signature": signature}


def train_version_c(cfg, export_layout_input: bool = False, export_dir: Optional[str] = None):
    device = get_device(cfg.train.device)
    device_type = device.type
    seed_everything(int(getattr(cfg.train, "seed", 0)))
    logger = setup_logger()
    out_dir = Path(export_dir) if export_dir else Path("outputs/version_c")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "logs" / "version_c_stats.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    loader = build_dataloader(cfg)
    data_iter = iter(loader)

    mapping_only = bool(getattr(cfg.training, "mapping_only", False))
    layout_only = bool(getattr(cfg.training, "layout_only", False))
    twostage = bool(getattr(cfg.training, "twostage", False))
    if mapping_only:
        setattr(cfg.hw, "optimize_layout", False)
        setattr(cfg.hw, "mapping_only", True)
    if layout_only:
        setattr(cfg.hw, "layout_only", True)

    update_alpha = not layout_only
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

    global_step = 0
    last_segments: List = []
    last_mapping: List[int] = []

    stable_hw_cfg = getattr(cfg, "stable_hw", None)
    stable_hw_state: Dict[str, Any] = {
        "lambda_hw": float(getattr(cfg.hw, "lambda_hw", 0.0)),
        "refs_inited": False,
        "ref_source": "unset",
    }
    stable_hw_state.setdefault(
        "discrete_cache",
        {
            "mapping": None,
            "layout": None,
            "mapping_signature": None,
            "layout_signature": None,
        },
    )
    run_state: Dict[str, Any] = {"last_model_info": None}
    last_acc1: Optional[float] = None
    best_acc1: Optional[float] = None
    last_hw_stats = None

    for outer in range(cfg.training.outer_epochs):
        if stable_hw_cfg:
            stable_hw_schedule(outer, stable_hw_cfg, stable_hw_state)
        iso = getattr(stable_hw_cfg, "discrete_isolation", None) if stable_hw_cfg else None
        map_every = int(getattr(iso, "mapping_update_every_epochs", 1) if iso else 1)
        lay_every = int(getattr(iso, "layout_update_every_epochs", 1) if iso else 1)

        cache = stable_hw_state["discrete_cache"]

        mapping_updated = False
        layout_updated = False

        if (outer % map_every) == 0 or cache["mapping"] is None:
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
            cache["mapping_signature"] = mapping_res.get("signature")
            mapping_updated = True
        else:
            mapping_res = cache["mapping"]

        if (outer % lay_every) == 0 or cache["layout"] is None:
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
        tau = max(cfg.chiplet.tau_min, cfg.chiplet.tau_init * (cfg.chiplet.tau_decay ** outer))
        chiplet_slots.set_tau(tau)
        last_hw_stats = None
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
                layout_positions = wafer_layout.current_pos_continuous()

                segments_cached = mapping_res.get("segments", []) if mapping_res else []
                mapping_cached = mapping_res.get("mapping", []) if mapping_res else []

                L_hw, hw_stats = compute_hw_loss(
                    cfg,
                    hw_proxy,
                    model_info=model_info,
                    stable_hw_cfg=stable_hw_cfg,
                    stable_hw_state=stable_hw_state,
                    segments=segments_cached,
                    mapping=mapping_cached,
                    eff_specs=eff_specs,
                    layout_positions=layout_positions,
                    mapping_solver=mapping_solver,
                    wafer_layout=wafer_layout,
                    alpha=alpha,
                )
                last_hw_stats = hw_stats
                lambda_hw = float(stable_hw_state.get("lambda_hw", float(getattr(cfg.hw, "lambda_hw", 0.0))))
                hw_term = float(lambda_hw) * L_hw if not twostage else (L_hw * 0.0)
                loss = L_task + cfg.loss.lambda_AST * info["L_AST"] + hw_term
            scaler.scale(loss).backward()
            scaler.step(optimizer_model)
            if not twostage and update_alpha:
                scaler.step(optimizer_alpha)
            if not twostage and update_layout:
                scaler.step(optimizer_layout)
            scaler.update()
            # P1-3(A): by default, DO NOT re-plan each inner step (wasteful + breaks discrete isolation).
            track_live = False
            if stable_hw_cfg:
                iso_cfg = getattr(stable_hw_cfg, "discrete_isolation", None)
                if iso_cfg is not None:
                    track_live = bool(getattr(iso_cfg, "track_live_segments", False))

            if track_live:
                part_res = partitioner.plan(
                    model,
                    chiplet_slots(hard=False)["eff_specs"],
                    alpha=chiplet_slots(hard=False)["alpha"],
                    model_info=run_state.get("last_model_info"),
                    use_fine_split=getattr(cfg.hw, "use_fine_split", True),
                )
                last_segments = part_res.get("segments", [])
                last_mapping = part_res.get("mapping", [])
            if step % 10 == 0:
                acc1 = (logits.argmax(dim=1) == y).float().mean()
                if stable_hw_cfg:
                    apply_accuracy_guard(float(acc1.item()), stable_hw_cfg, stable_hw_state)
                last_acc1 = float(acc1.item())
                best_acc1 = float(acc1.item()) if best_acc1 is None else max(best_acc1, float(acc1.item()))
                stats = {
                    "outer": outer,
                    "step": step,
                    "loss": loss.item(),
                    "acc1": acc1.item(),
                    "lambda_hw": float(lambda_hw),
                    "mapping_updated": mapping_updated,
                    "layout_updated": layout_updated,
                    "mapping_cache_hit": not mapping_updated,
                    "layout_cache_hit": not layout_updated,
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
                        "lambda_hw": float(lambda_hw),
                        "acc_drop": float(stable_hw_state.get("acc_drop", 0.0)),
                        "schedule_phase": stable_hw_state.get("schedule_phase"),
                        "mapping_updated": mapping_updated,
                        "layout_updated": layout_updated,
                        "mapping_cache_hit": (not mapping_updated),
                        "layout_cache_hit": (not layout_updated),
                        "mapping_signature": cache["mapping_signature"],
                        "layout_signature": cache["layout_signature"],
                    }) + "\n")
            global_step += 1

        if stable_hw_cfg and last_hw_stats is not None:
            update_hw_refs_from_stats(stable_hw_cfg, stable_hw_state, last_hw_stats)

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

                L_hw, _ = compute_hw_loss(
                    cfg,
                    hw_proxy,
                    model_info=model_info,
                    stable_hw_cfg=stable_hw_cfg,
                    stable_hw_state=stable_hw_state,
                    segments=segments_cached,
                    mapping=mapping_cached,
                    eff_specs=eff_specs,
                    layout_positions=layout_positions,
                    mapping_solver=mapping_solver,
                    wafer_layout=wafer_layout,
                    alpha=alpha,
                )

                lambda_hw = float(stable_hw_state.get("lambda_hw", float(getattr(cfg.hw, "lambda_hw", 0.0))))
                optimizer_alpha.zero_grad()
                (loss_alpha := (float(lambda_hw) * L_hw)).backward()
                optimizer_alpha.step()

        # Step D: layout refinement
        if update_layout:
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
                    distance_scale=1e-9,
                )
                optimizer_layout.zero_grad()
                L_layout.backward()
                optimizer_layout.step()

        stable_state_path = log_path.parent / "stable_hw_state.json"
        stable_state_path.write_text(
            json.dumps(stable_hw_state, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    metrics = {
        "stable_hw_disabled": False if stable_hw_cfg and bool(getattr(stable_hw_cfg, "enabled", True)) else True,
        "stable_hw_lambda_hw": float(stable_hw_state.get("lambda_hw", 0.0)),
        "stable_hw_refs_inited": bool(stable_hw_state.get("refs_inited", False)),
        "stable_hw_ref_source": str(stable_hw_state.get("ref_source", "unset")),
        "best_acc1": float(best_acc1) if best_acc1 is not None else 0.0,
        "last_acc1": float(last_acc1) if last_acc1 is not None else 0.0,
        "last_hw_stats": last_hw_stats if last_hw_stats is not None else {},
        "mapping_signature": stable_hw_state.get("discrete_cache", {}).get("mapping_signature"),
        "layout_signature": stable_hw_state.get("discrete_cache", {}).get("layout_signature"),
    }
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if export_layout_input:
        export_dir_path = Path(export_dir or "outputs/P3")
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
            chiplet_slots=chiplet_slots,
            mapping_solver=mapping_solver,
            segments=canonical_segments,
            mapping=mapping_canonical,
            wafer_layout=wafer_layout,
            seed=int(getattr(cfg.train, "seed", 0)),
        )
