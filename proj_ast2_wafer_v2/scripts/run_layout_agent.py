"""Offline layout agent pipeline (simplified v4.3.2 alignment)."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from layout.coarsen import coarsen_traffic
from layout.detailed_place import run_detailed_place
from layout.evaluator import LayoutEvaluator, LayoutState
from layout.expand import expand_clusters_to_sites
from layout.global_place_region import assign_clusters_to_regions
from layout.legalize import legalize_assign
from layout.pareto import ParetoSet
from layout.regions import build_regions


def load_layout_input(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def stage_pipeline(cfg: dict, layout_input: dict, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sites_xy = np.asarray(layout_input["sites"]["sites_xy_mm"], dtype=np.float64)
    S = int(layout_input["slots"]["S"])
    wafer_radius = float(layout_input["wafer"]["radius_mm"])
    chip_tdp_w = np.asarray(layout_input["slots"]["chip_tdp_w"], dtype=np.float64)
    traffic = np.asarray(layout_input["mapping"]["traffic_matrix"], dtype=np.float64)
    T_sym = traffic + traffic.T

    baseline = layout_input["objective_cfg"]["baseline"]
    scalar_w = layout_input["objective_cfg"]["scalar_weights"]
    sigma_mm = float(layout_input["objective_cfg"]["sigma_mm"])
    evaluator = LayoutEvaluator(sigma_mm=sigma_mm, baseline=baseline, scalar_w=scalar_w)

    # Stage: coarsen
    clusters, W_cluster = coarsen_traffic(
        T_sym=T_sym,
        chip_tdp_w=chip_tdp_w,
        slot_mask_used=np.ones(S, dtype=bool),
        target_num_clusters=cfg.get("target_num_clusters", 4),
        min_merge_traffic=cfg.get("min_merge_traffic", 1e-6),
    )

    # Stage: regions
    regions, site_to_region = build_regions(
        sites_xy_mm=sites_xy,
        wafer_radius_mm=wafer_radius,
        ring_edges_ratio=cfg.get("ring_edges_ratio", [0.0, 0.5, 1.0]),
        sectors_per_ring=cfg.get("sectors_per_ring", [8, 12]),
        capacity_ratio=cfg.get("capacity_ratio", 1.0),
    )

    # Stage: cluster -> region assignment
    cluster_to_region = assign_clusters_to_regions(
        clusters=clusters,
        regions=regions,
        W_cluster=W_cluster,
        lambda_graph=cfg.get("lambda_graph", 1.0),
        lambda_ring=cfg.get("lambda_ring", 1.0),
        lambda_cap=cfg.get("lambda_cap", 10000.0),
        ring_score=cfg.get("ring_score", [1.0, 0.5]),
        refine_steps=cfg.get("refine_steps", 0),
        sa_T0=cfg.get("sa_T0", 1.0),
        sa_alpha=cfg.get("sa_alpha", 0.995),
    )

    # Stage: expand
    assign = expand_clusters_to_sites(
        clusters=clusters,
        cluster_to_region=cluster_to_region,
        regions=regions,
        sites_xy_mm=sites_xy,
        traffic_sym=T_sym,
        intra_refine_steps=cfg.get("intra_refine_steps", 0),
    )

    # Stage: legalize
    assign = legalize_assign(assign, sites_xy, wafer_radius)

    # Stage: detailed place + Pareto
    pareto = ParetoSet(
        eps_comm=cfg.get("eps_comm", 0.01),
        eps_therm=cfg.get("eps_therm", 0.01),
        max_points=cfg.get("max_points", 2000),
    )
    trace = []
    hot_pairs = []
    for i in range(S):
        for j in range(i + 1, S):
            hot_pairs.append((i, j, T_sym[i, j]))
    hot_pairs = [p for p in sorted(hot_pairs, key=lambda x: x[2], reverse=True) if p[2] > 0]
    hot_pairs = [(i, j) for i, j, _ in hot_pairs[: cfg.get("top_pairs_k", 10)]]

    best_assign, best_eval = run_detailed_place(
        assign=assign,
        evaluator=evaluator,
        sites_xy_mm=sites_xy,
        chip_tdp_w=chip_tdp_w,
        traffic_bytes=traffic,
        pareto=pareto,
        steps=cfg.get("steps", 1000),
        sa_T0=cfg.get("sa_T0_detailed", 1.0),
        sa_alpha=cfg.get("sa_alpha_detailed", 0.999),
        action_probs=cfg.get("action_probs", {"swap": 0.6, "relocate": 0.3, "cluster_move": 0.1}),
        hot_pairs=hot_pairs,
        trace=trace,
        seed_id=0,
    )

    knee = pareto.knee_point()
    layout_best = {
        "best": {
            "assign": best_assign.tolist(),
            "objectives": {
                "comm_norm": best_eval["comm_norm"],
                "therm_norm": best_eval["therm_norm"],
                "penalty": best_eval["penalty"],
            },
        },
        "pareto_front": [p.__dict__ for p in pareto.points],
        "selection": {"knee": knee.__dict__ if knee else None, "pareto_size": len(pareto.points)},
        "region_plan": {
            "clusters": [c.__dict__ for c in clusters],
            "cluster_to_region": cluster_to_region.tolist(),
        },
    }

    save_json(layout_best, os.path.join(out_dir, "layout_best.json"))
    save_json(trace, os.path.join(out_dir, "trace.json"))
    print(f"Saved layout_best.json and trace.json to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_input", type=str, required=True, help="Path to layout_input.json")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    cfg = {}
    if args.layout_input is None:
        raise ValueError("layout_input is required")

    layout_input = load_layout_input(args.layout_input)
    stage_pipeline(cfg, layout_input, args.out_dir)


if __name__ == "__main__":
    main()

