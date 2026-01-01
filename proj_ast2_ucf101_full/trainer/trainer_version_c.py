"""Version-C full trainer (SPEC §12.2)."""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import math

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from chiplet.chiplet_lib import ChipletLibrary, ChipletSlots
from utils.data_ucf101 import UCF101Dataset
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


def _as_float(val, name: str) -> float:
    """Convert config values that might be strings into floats with a clear error."""
    try:
        return float(val)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Expected {name} to be numeric, but got {val!r}.") from exc


def build_dataloader(cfg):
    ds = UCF101Dataset(cfg, split="train")
    return DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, num_workers=cfg.data.num_workers)


def _traffic_aware_seed(sites_xy: np.ndarray, traffic: np.ndarray, S: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy placement of hot traffic pairs onto nearest site pairs (SPEC §7.1)."""

    Ns = sites_xy.shape[0]
    assign = np.full(S, -1, dtype=int)
    used_sites: set[int] = set()

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
        for a, b, _ in site_pairs:
            if a in used_sites or b in used_sites:
                continue
            assign[i] = a
            assign[j] = b
            used_sites.update([a, b])
            break

    # fill remaining slots deterministically
    remaining_sites = [s for s in range(Ns) if s not in used_sites]
    for s_idx in range(S):
        if assign[s_idx] == -1 and remaining_sites:
            assign[s_idx] = remaining_sites.pop(0)
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
    S = cfg.hw.num_slots
    Ns = sites_xy.shape[0]
    assign_grid = np.arange(S, dtype=int) % Ns

    eff_specs = chiplet_slots(hard=False)["eff_specs"]
    chip_tdp = eff_specs["tdp_w"].detach().cpu().numpy().astype(float)

    traffic = mapping_solver.build_traffic_matrix(segments, mapping).cpu().numpy().astype(float)
    sigma_mm = float(getattr(cfg.layout_seed, "sigma_mm", 20.0)) if hasattr(cfg, "layout_seed") else 20.0
    rng = np.random.default_rng(seed)
    baseline_eval = LayoutEvaluator(
        sigma_mm=sigma_mm,
        baseline={"L_comm_baseline": 1.0, "L_therm_baseline": 1.0},
        scalar_w={"w_comm": 1.0, "w_therm": 1.0, "w_penalty": 1000.0},
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
        scalar_w={"w_comm": 1.0, "w_therm": 1.0, "w_penalty": 1000.0},
    )
    assign_seed = _traffic_aware_seed(sites_xy, traffic, S, rng)
    layout_state.assign = assign_seed
    assign_seed, micro_stats = _micro_place_seed(assign_seed, sites_xy, baseline_eval, layout_state, traffic, rng=rng)

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
        },
        "baseline": baseline,
        "seed": {"assign_seed": assign_seed.tolist(), "micro_place_stats": micro_stats},
        "objective_cfg": {
            "sigma_mm": sigma_mm,
            "scalar_weights": {"w_comm": 1.0, "w_therm": 1.0, "w_penalty": 1000.0},
        },
    }

    out_path = export_dir / "layout_input.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(layout_input, f, indent=2)

    return out_path


def compute_hw_loss(model, chiplet_slots: ChipletSlots, hw_proxy: LayerHwProxy, mapping_solver: MappingSolver, wafer_layout: WaferLayout, partitioner: PartitionPlanner, hw_cfg: Dict):
    slot_out = chiplet_slots(hard=False)
    alpha = slot_out["alpha"]
    eff_specs = slot_out["eff_specs"]
    chip_used_prob = 1.0 - alpha[:, -1]
    L_chip_count = hw_cfg.lambda_chip * chip_used_prob.sum()

    partition_result = partitioner.plan(model=model, eff_specs=eff_specs, use_fine_split=getattr(hw_cfg, "use_fine_split", True))
    segments = partition_result["segments"]
    mapping = partition_result["mapping"]

    cost = mapping_solver.build_cost_matrix(segments, eff_specs, hw_proxy)
    mapping_result = mapping_solver.solve_mapping(
        segments,
        eff_specs,
        hw_proxy,
        layout_positions=wafer_layout.current_pos,
        strategy=getattr(hw_cfg, "mapping_strategy", "greedy_local"),
        distance_scale_ms=getattr(hw_cfg, "distance_scale_ms", 0.0),
    )
    mapping = mapping_result["mapping"]
    total_latency_ms = torch.tensor(mapping_result["total_latency_ms"], device=alpha.device, dtype=torch.float32)
    comm_ms = torch.tensor(mapping_result["comm_ms"], device=alpha.device, dtype=torch.float32)

    lat_ms = cost["lat_ms"]
    power_w = cost["power_w"]
    K = len(segments)
    total_energy_j = lat_ms.new_tensor(0.0)
    mem_mb = cost["mem_mb"]
    S = alpha.shape[0]
    mem_usage = torch.zeros(S, device=alpha.device)
    for k in range(K):
        d = mapping[k]
        lat_s = lat_ms[k, d] / 1e3
        p = power_w[k, d]
        total_energy_j = total_energy_j + lat_s * p
        mem_usage[d] = torch.maximum(mem_usage[d], mem_mb[k, d])
    peak_mem_mb = mem_usage.max()
    total_area_mm2 = (eff_specs["area_mm2"] * chip_used_prob).sum()
    L_area = hw_cfg.lambda_area * torch.relu(total_area_mm2 - hw_cfg.area_limit_mm2) ** 2

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

    L_hw = hw_cfg.lambda_T * total_latency_ms + hw_cfg.lambda_E * total_energy_j + hw_cfg.lambda_mem * peak_mem_mb + L_area + L_chip_count + L_layout
    hw_stats = {
        "total_latency_ms": total_latency_ms.detach(),
        "total_energy_j": total_energy_j.detach(),
        "peak_mem_mb": peak_mem_mb.detach(),
        "total_area_mm2": total_area_mm2.detach(),
        "chip_count": chip_used_prob.sum().detach(),
        "layout": {k: v.detach() for k, v in layout_stats.items()},
        "comm_ms": comm_ms.detach(),
    }
    return L_hw, hw_stats, mapping, partition_result.get("rewrite_plan")


def train_version_c(cfg, export_layout_input: bool = False, export_dir: Optional[str] = None):
    device = get_device(cfg.train.device)
    device_type = device.type
    logger = setup_logger()
    log_path = Path("logs/version_c_stats.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    loader = build_dataloader(cfg)
    data_iter = iter(loader)

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

    for outer in range(cfg.training.outer_epochs):
        tau = max(cfg.chiplet.tau_min, cfg.chiplet.tau_init * (cfg.chiplet.tau_decay ** outer))
        chiplet_slots.set_tau(tau)
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
                L_task = F.cross_entropy(logits, y)
                L_hw, hw_stats, mapping, rewrite_plan = compute_hw_loss(model, chiplet_slots, hw_proxy, mapping_solver, wafer_layout, partitioner, cfg.hw)
                loss = L_task + cfg.loss.lambda_AST * info["L_AST"] + cfg.loss.lambda_hw * L_hw
            scaler.scale(loss).backward()
            scaler.step(optimizer_model)
            scaler.step(optimizer_alpha)
            scaler.step(optimizer_layout)
            scaler.update()
            last_segments = partitioner.plan(model, chiplet_slots(hard=False)["eff_specs"], use_fine_split=getattr(cfg.hw, "use_fine_split", True))["segments"]
            last_mapping = mapping
            if step % 10 == 0:
                acc1 = (logits.argmax(dim=1) == y).float().mean()
                log_stats(logger, {"outer": outer, "step": step, "loss": loss.item(), "acc1": acc1.item(), "lat_ms": hw_stats["total_latency_ms"].item()})
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "step": int(global_step),
                        "outer": int(outer),
                        "loss": float(loss.item()),
                        "lat_ms": float(hw_stats["total_latency_ms"].item()),
                        "energy_j": float(hw_stats["total_energy_j"].item()),
                        "mem_mb": float(hw_stats["peak_mem_mb"].item()),
                        "area": float(hw_stats["total_area_mm2"].item()),
                        "chip_count": float(hw_stats["chip_count"].item()),
                    }) + "\n")
            global_step += 1

        # Step B: alpha refinement
        for _ in range(cfg.training.inner_steps_alpha):
            L_hw, hw_stats, mapping, _ = compute_hw_loss(model, chiplet_slots, hw_proxy, mapping_solver, wafer_layout, partitioner, cfg.hw)
            optimizer_alpha.zero_grad()
            L_hw.backward()
            optimizer_alpha.step()

        # Step D: layout refinement
        for _ in range(cfg.training.inner_steps_layout):
            slot_out = chiplet_slots(hard=False)
            eff_specs = slot_out["eff_specs"]
            part_res = partitioner.plan(model, eff_specs, use_fine_split=getattr(cfg.hw, "use_fine_split", True))
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

    if export_layout_input:
        export_dir_path = Path(export_dir or "outputs/P3")
        _export_layout_input(
            cfg=cfg,
            export_dir=export_dir_path,
            chiplet_slots=chiplet_slots,
            mapping_solver=mapping_solver,
            segments=last_segments,
            mapping=last_mapping,
            wafer_layout=wafer_layout,
            seed=int(getattr(cfg.train, "seed", 0)),
        )
