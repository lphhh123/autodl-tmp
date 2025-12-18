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
import os
from typing import List, Dict, Any, Tuple

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from utils.config import load_config
from ast2.trainer_ast2_single import build_model, build_hw_proxy, train_one_epoch, evaluate
from mapping.mapper_multi_gpu import (
    Chiplet, build_segments, estimate_segment_costs,
    greedy_initial_mapping, local_search_refine,
)
from layout.wafer_layout import ChipGeom, WaferLayout, layout_loss

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


def build_chip_geoms_from_chips(chips: List[Chiplet], cfg) -> List[ChipGeom]:
    area_per_chip = 1.0 / max(len(chips), 1)
    side = (area_per_chip ** 0.5) * 0.5
    geoms = []
    for chip in chips:
        geoms.append(ChipGeom(
            idx=chip.idx,
            width=side,
            height=side,
            power=cfg.layout.get("power_per_chip", 1.0),
        ))
    return geoms


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
        "--config", type=str, default="configs/ast2_single_gpu.yaml",
        help="Base config (model/data/loss/proxy). Mapping/layout use extra fields.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    device = torch.device(cfg.train.device)
    model = build_model(cfg).to(device)
    proxy = build_hw_proxy(cfg)

    # Data: reuse the same data pipeline as single-GPU training (UCF-101 by default)
    train_loader, val_loader = build_dataloaders(cfg)
    loader = train_loader


    chips = build_chiplets_from_cfg(cfg)
    chip_geoms = build_chip_geoms_from_chips(chips, cfg)
    layout = WaferLayout(chip_geoms).to(device)

    outer_iters = cfg.mapping.get("outer_iters", 3)
    inner_epochs = cfg.mapping.get("inner_epochs", 1)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # Layout optimizer uses separate optimizer on layout parameters
    layout_opt = optim.Adam(layout.parameters(), lr=cfg.layout.get("lr", 1e-2))

    best_score = -1e9
    best_state = None

    for outer in range(outer_iters):
        print(f"===== Outer Iter {outer+1}/{outer_iters} =====")

        # (1) Train theta, s with fixed mapping & layout
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
        dummy_video, _ = dataset[0]
        dummy_video = dummy_video.unsqueeze(0).to(device)
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

        # (3) Optimize layout L via gradient descent on layout loss
        print("  [Step3] Optimize layout L")
        layout.train()
        for k in range(cfg.layout.get("gd_steps", 50)):
            layout_opt.zero_grad()
            # Build netlist weights from segment mapping
            netlist = {}
            for (u, v), traffic in inter_seg_bytes.items():
                cu = mapping[u]
                cv = mapping[v]
                if cu == cv:
                    continue
                key = (cu, cv)
                netlist[key] = netlist.get(key, 0.0) + traffic
            loss_L = layout_loss(
                layout,
                netlist=netlist,
                alpha_wire=cfg.layout.get("alpha_wire", 1.0),
                beta_overlap=cfg.layout.get("beta_overlap", 10.0),
                gamma_boundary=cfg.layout.get("gamma_boundary", 10.0),
                delta_thermal=cfg.layout.get("delta_thermal", 1.0),
            )
            loss_L.backward()
            layout_opt.step()
            if (k + 1) % 10 == 0:
                print(f"    [layout gd step {k+1}] loss_L={loss_L.item():.4f}")

        # Evaluate current configuration
        acc = evaluate(model, val_loader, cfg)
        score = acc  # you can extend this to combine latency/power/temp

        print(f"  [Eval] acc={acc:.4f}, score={score:.4f}")

        if score > best_score:
            best_score = score
            best_state = {
                "model": model.state_dict(),
                "layout": layout.state_dict(),
                "mapping": mapping,
                "acc": acc,
            }
            out_dir = cfg.log.out_dir
            os.makedirs(out_dir, exist_ok=True)
            torch.save(best_state, os.path.join(out_dir, "best_version_c.pth"))
            print(f"  [save] new best score={score:.4f}")


if __name__ == "__main__":
    main()