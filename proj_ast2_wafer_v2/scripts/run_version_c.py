"""High-level Version-C orchestration script.

This is a *reference* implementation of the alternating optimization loop:

  (1) Train (theta, s) with fixed mapping & layout
  (2) Update mapping m via greedy + local search
  (3) Optimize layout L via gradient descent

For a real experiment you will likely:
  - Increase outer_iters and inner epochs
  - Replace DummyVideoDataset with your real dataset
  - Tune loss weights and search spaces
"""
import argparse
import json
import os
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from utils.config import load_config
from ast2.trainer_ast2_single import build_model, build_hw_proxy, train_one_epoch, evaluate
from mapping.mapper_multi_gpu import (
    Chiplet, build_segments, estimate_segment_costs,
    greedy_initial_mapping, local_search_refine,
)
from layout.wafer_layout import build_layout_inputs

from scripts.run_ast2_single import build_dataloaders  # reuse data pipeline (UCF-101)


def build_chiplets_from_cfg(cfg) -> List[Chiplet]:
    combo = cfg.mapping.chiplet_combo  # list of {name, count, mem_capacity, compute_capacity, link_bw}
    chips: List[Chiplet] = []
    idx = 0
    for item in combo:
        for _ in range(item["count"]):
            chips.append(Chiplet(
                name=item["name"],
                mem_capacity=float(item["mem_capacity"]),
                compute_capacity=float(item["compute_capacity"]),
                link_bw=float(item["link_bw"]),
                idx=idx,
            ))
            idx += 1
    return chips


def build_inter_seg_bytes(segments, layer_metas) -> Dict[Tuple[int, int], float]:
    """Simple linear chain: seg i connects to seg i+1.

    Data size is approximated from output token count & embed_dim.
    """
    bytes_map = {}
    for i in range(len(segments) - 1):
        seg_u = segments[i]
        seg_v = segments[i + 1]
        last_layer_meta = layer_metas[seg_u.layer_indices[-1]]
        L_eff = float(last_layer_meta["L_eff"])
        d = float(last_layer_meta["embed_dim"])
        # bytes = B * L_eff * d * 4, but we ignore B and 4 for relative cost
        bytes_ = L_eff * d * 4.0
        bytes_map[(seg_u.idx, seg_v.idx)] = bytes_
    return bytes_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/ast2_single_gpu.yaml",
        help="Base config (model/data/loss/proxy). Mapping/layout use extra fields.",
    )
    parser.add_argument(
        "--cfg",
        dest="config_alias",
        type=str,
        help="Alias of --config for compatibility with experiment command docs.",
    )
    parser.add_argument(
        "--export_layout_input",
        action="store_true",
        help="Force exporting layout_input.json regardless of config value.",
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        help="Override export directory for layout_input.json.",
    )

    args = parser.parse_args()
    config_path = args.config_alias or args.config
    cfg = load_config(config_path)

    # CLI overrides (kept minimal to avoid surprising config drift)
    if args.export_layout_input:
        cfg.layout.export_layout_input = True
    if args.export_dir:
        cfg.layout.export_dir = args.export_dir

    device = torch.device(cfg.train.device)
    model = build_model(cfg).to(device)
    proxy = build_hw_proxy(cfg)

    # Data: reuse the same data pipeline as single-GPU training (UCF-101 by default)
    train_loader, val_loader = build_dataloaders(cfg)
    loader = train_loader

    chips = build_chiplets_from_cfg(cfg)

    outer_iters = cfg.mapping.get("outer_iters", 1)
    inner_epochs = cfg.mapping.get("inner_epochs", 1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    best_score = -1e9
    best_state = None

    for outer in range(outer_iters):
        print(f"===== Outer Iter {outer+1}/{outer_iters} =====")

        # (1) Train theta, s with fixed mapping
        for ep in range(inner_epochs):
            print(f"  [Step1] Train epoch {ep+1}/{inner_epochs}")
            _ = train_one_epoch(
                model=model,
                proxy=proxy,
                dataloader=loader,
                optimizer=optimizer,
                cfg=cfg,
                epoch=outer * inner_epochs + ep + 1,
            )

        # Build layer metas and segments based on current keep_ratios
        sample, _ = train_loader.dataset[0]
        dummy_video = sample.unsqueeze(0).to(device)
        _ = model(dummy_video)
        layer_metas = model.get_layer_metas(tuple(dummy_video.shape), chip_name="")

        segments = build_segments(layer_metas, layers_per_segment=cfg.mapping.get("layers_per_segment", 2))
        estimate_segment_costs(segments, chips, layer_metas, proxy)
        inter_seg_bytes = build_inter_seg_bytes(segments, layer_metas)

        # (2) Update mapping m via greedy + local search
        print("  [Step2] Update mapping m")
        mapping0 = greedy_initial_mapping(segments, chips)
        mapping = local_search_refine(
            segments, chips, mapping0,
            inter_seg_bytes,
            link_bw=chips[0].link_bw,
            lambda_comm=cfg.mapping.get("lambda_comm", 1.0),
        )
        print(f"    mapping={mapping}")

        # (3) Generate site-based layout seeds and export layout_input.json
        S = len(chips)
        traffic_matrix = np.zeros((S, S), dtype=np.float64)
        for (u, v), bytes_ in inter_seg_bytes.items():
            cu = mapping[u]
            cv = mapping[v]
            if cu == cv:
                continue
            traffic_matrix[cu, cv] += bytes_

        wafer_radius_mm = float(cfg.layout.get("wafer_radius_mm", 150.0))
        margin_mm = float(cfg.layout.get("margin_mm", 1.0))
        chip_w = float(cfg.layout.get("chip_max_width_mm", 20.0))
        chip_h = float(cfg.layout.get("chip_max_height_mm", 20.0))
        sigma_mm = float(cfg.layout.get("sigma_mm", 20.0))
        scalar_weights = cfg.layout.get("scalar_weights", {"w_comm": 0.7, "w_therm": 0.3, "w_penalty": 1000.0})
        chip_tdp_w = np.full((S,), float(cfg.layout.get("power_per_chip", 1.0)), dtype=np.float64)

        layout_inputs = build_layout_inputs(
            S=S,
            wafer_radius_mm=wafer_radius_mm,
            chip_max_width_mm=chip_w,
            chip_max_height_mm=chip_h,
            margin_mm=margin_mm,
            traffic_matrix=traffic_matrix,
            chip_tdp_w=chip_tdp_w,
            sigma_mm=sigma_mm,
            scalar_weights=scalar_weights,
            seed=cfg.layout.get("seed", 0),
        )

        if getattr(cfg.layout, "export_layout_input", False):
            out_dir = getattr(cfg.layout, "export_dir", cfg.log.out_dir)
            os.makedirs(out_dir, exist_ok=True)
            layout_input = {
                "layout_version": "v4.3.2",
                "wafer": {"radius_mm": wafer_radius_mm, "margin_mm": margin_mm},
                "sites": {
                    "method": "square_grid_in_circle",
                    "grid_pitch_mm": None,
                    "sites_xy_mm": layout_inputs["sites_xy_mm"].tolist(),
                },
                "slots": {"S": S, "chip_tdp_w": chip_tdp_w.tolist()},
                "mapping": {
                    "mapping_id": f"outer_{outer}",
                    "traffic_matrix": traffic_matrix.tolist(),
                },
                "baseline": {
                    "assign_grid": layout_inputs["assign_grid"].tolist(),
                    "costs": layout_inputs["eval_grid"],
                },
                "seed": {
                    "assign_seed": layout_inputs["assign_seed"].tolist(),
                    "assign_micro": layout_inputs["assign_micro"].tolist(),
                    "eval_seed": layout_inputs["eval_seed"],
                    "eval_micro": layout_inputs["eval_micro"],
                    "micro_place_stats": layout_inputs["micro_place_stats"].__dict__,
                },
                "objective_cfg": {
                    "sigma_mm": sigma_mm,
                    "scalar_weights": scalar_weights,
                    "baseline": layout_inputs["baseline"],
                },
            }
            with open(os.path.join(out_dir, "layout_input.json"), "w", encoding="utf-8") as f:
                json.dump(layout_input, f, indent=2)
            print(f"  [export] layout_input.json saved to {out_dir}")

        # Evaluate current configuration
        acc = evaluate(model, val_loader, cfg)
        score = acc
        print(f"  [Eval] acc={acc:.4f}, score={score:.4f}")

        if score > best_score:
            best_score = score
            best_state = {
                "model": model.state_dict(),
                "mapping": mapping,
                "acc": acc,
            }
            out_dir = cfg.log.out_dir
            os.makedirs(out_dir, exist_ok=True)
            torch.save(best_state, os.path.join(out_dir, "best_version_c.pth"))
            print(f"  [save] new best score={score:.4f}")


if __name__ == "__main__":
    main()