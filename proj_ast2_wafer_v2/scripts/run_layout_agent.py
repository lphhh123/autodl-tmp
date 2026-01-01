"""Offline layout agent pipeline following the v4.3.2 spec skeleton."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from layout.coarsen import coarsen_traffic
from layout.detailed_place import run_detailed_place
from layout.evaluator import LayoutEvaluator, LayoutState
from layout.expand import expand_clusters_to_sites
from layout.global_place_region import assign_clusters_to_regions
from layout.legalize import legalize_assign
from layout.pareto import ParetoSet
from layout.regions import Region, build_regions
from utils.config import Config, load_config


def load_layout_input(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _assign_hash(assign: np.ndarray) -> str:
    import hashlib

    return hashlib.sha1(np.asarray(assign, dtype=np.int64).tobytes()).hexdigest()[:16]


def _build_default_region(sites_xy_mm: np.ndarray) -> tuple[List[Region], np.ndarray]:
    region = Region(
        region_id=0,
        ring_idx=0,
        sector_idx=0,
        centroid_xy_mm=(float(np.mean(sites_xy_mm[:, 0])), float(np.mean(sites_xy_mm[:, 1]))),
        site_ids=list(range(sites_xy_mm.shape[0])),
        capacity=sites_xy_mm.shape[0],
    )
    site_to_region = np.zeros(sites_xy_mm.shape[0], dtype=np.int64)
    return [region], site_to_region


def _record_trace(
    trace: List[Dict],
    iter_idx: int,
    stage: str,
    op: str,
    op_args_json: str,
    accepted: int,
    eval_res: Dict,
    pareto_added: int,
    seed_id: int,
):
    trace.append(
        {
            "iter": iter_idx,
            "stage": stage,
            "op": op,
            "op_args_json": op_args_json,
            "accepted": accepted,
            "total_scalar": eval_res["total_scalar"],
            "comm_norm": eval_res["comm_norm"],
            "therm_norm": eval_res["therm_norm"],
            "pareto_added": pareto_added,
            "duplicate_penalty": eval_res["penalty"]["duplicate"],
            "boundary_penalty": eval_res["penalty"]["boundary"],
            "seed_id": seed_id,
            "time_ms": 0,
        }
    )


def _save_csv(path: Path, rows: Iterable[Dict], fieldnames: List[str]):
    if not fieldnames:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def stage_pipeline(cfg: dict | Config, layout_input: dict, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    cfg = Config(cfg) if not isinstance(cfg, Config) else cfg

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

    coarsen_cfg = cfg.get("coarsen", {})
    region_cfg = cfg.get("regions", {})
    expand_cfg = cfg.get("expand", {})
    pareto_cfg = cfg.get("pareto", {})
    detailed_cfg = cfg.get("detailed_place", {})

    pareto = ParetoSet(
        eps_comm=float(pareto_cfg.get("eps_comm", 0.01)),
        eps_therm=float(pareto_cfg.get("eps_therm", 0.01)),
        max_points=int(pareto_cfg.get("max_points", 2000)),
    )
    pareto_points: List[Dict] = []
    trace: List[Dict] = []
    iter_idx = 0

    def eval_and_record(assign: np.ndarray, stage: str, op: str = "none", op_args: Dict | None = None, seed_id: int = 0):
        nonlocal iter_idx
        res = evaluator.evaluate(
            LayoutState(
                S=S,
                Ns=sites_xy.shape[0],
                wafer_radius_mm=wafer_radius,
                sites_xy_mm=sites_xy,
                assign=assign,
                chip_tdp_w=chip_tdp_w,
                traffic_bytes=traffic,
                meta={"stage": stage},
            )
        )
        added = pareto.try_add(
            comm_norm=res["comm_norm"],
            therm_norm=res["therm_norm"],
            total_scalar=res["total_scalar"],
            meta={"stage": stage, "iter": iter_idx, "seed_id": seed_id, "assign_hash": _assign_hash(assign)},
        )
        if added:
            pareto_points.append(
                {
                    "solution_id": len(pareto_points),
                    "comm_norm": res["comm_norm"],
                    "therm_norm": res["therm_norm"],
                    "total_scalar": res["total_scalar"],
                    "stage": stage,
                    "iter": iter_idx,
                    "seed_id": seed_id,
                    "assign_hash": _assign_hash(assign),
                }
            )
        _record_trace(
            trace,
            iter_idx=iter_idx,
            stage=stage,
            op=op,
            op_args_json=json.dumps(op_args or {}),
            accepted=1,
            eval_res=res,
            pareto_added=int(added),
            seed_id=seed_id,
        )
        iter_idx += 1
        return res

    # Stage0: baseline + seed evaluations
    baseline_assign = layout_input.get("baseline", {}).get("assign_grid")
    if baseline_assign is not None:
        eval_and_record(np.asarray(baseline_assign, dtype=np.int64), stage="baseline", op="none")
    seed_assign = layout_input.get("seed", {}).get("assign_seed")
    if seed_assign is not None:
        eval_and_record(np.asarray(seed_assign, dtype=np.int64), stage="seed", op="none")
    micro_assign = layout_input.get("seed", {}).get("assign_micro")
    if micro_assign is not None:
        eval_and_record(np.asarray(micro_assign, dtype=np.int64), stage="micro_place", op="none")

    # Stage: coarsen
    if coarsen_cfg.get("enabled", True):
        clusters, W_cluster = coarsen_traffic(
            T_sym=T_sym,
            chip_tdp_w=chip_tdp_w,
            slot_mask_used=np.ones(S, dtype=bool),
            target_num_clusters=int(coarsen_cfg.get("target_num_clusters", 4)),
            min_merge_traffic=float(coarsen_cfg.get("min_merge_traffic", 1e-6)),
        )
    else:
        clusters, W_cluster = coarsen_traffic(
            T_sym=T_sym,
            chip_tdp_w=chip_tdp_w,
            slot_mask_used=np.ones(S, dtype=bool),
            target_num_clusters=S,
            min_merge_traffic=float("inf"),
        )

    # Stage: regions
    if region_cfg.get("enabled", True):
        regions, site_to_region = build_regions(
            sites_xy_mm=sites_xy,
            wafer_radius_mm=wafer_radius,
            ring_edges_ratio=region_cfg.get("ring_edges_ratio", [0.0, 0.5, 1.0]),
            sectors_per_ring=region_cfg.get("sectors_per_ring", [8, 12]),
            capacity_ratio=float(region_cfg.get("capacity_ratio", 1.0)),
        )
    else:
        regions, site_to_region = _build_default_region(sites_xy)

    # Stage: cluster -> region assignment
    cluster_to_region = assign_clusters_to_regions(
        clusters=clusters,
        regions=regions,
        W_cluster=W_cluster,
        lambda_graph=float(region_cfg.get("lambda_graph", 1.0)),
        lambda_ring=float(region_cfg.get("lambda_ring", 1.0)),
        lambda_cap=float(region_cfg.get("lambda_cap", 10000.0)),
        ring_score=region_cfg.get("ring_score", [1.0, 0.5]),
        refine_steps=int(region_cfg.get("refine_steps", 0)),
        sa_T0=float(region_cfg.get("sa_T0", 1.0)),
        sa_alpha=float(region_cfg.get("sa_alpha", 0.995)),
    )

    # Stage: expand
    assign = expand_clusters_to_sites(
        clusters=clusters,
        cluster_to_region=cluster_to_region,
        regions=regions,
        sites_xy_mm=sites_xy,
        traffic_sym=T_sym,
        intra_refine_steps=int(expand_cfg.get("intra_refine_steps", 0)),
    )
    expand_eval = eval_and_record(assign.copy(), stage="expand", op="cluster_to_site")

    # Stage: legalize
    if cfg.get("legalize", {}).get("enabled", True):
        assign = legalize_assign(assign, sites_xy, wafer_radius)
    legal_eval = eval_and_record(assign.copy(), stage="legalize", op="legalize")

    # Stage: detailed place + Pareto
    hot_pairs = []
    for i in range(S):
        for j in range(i + 1, S):
            hot_pairs.append((i, j, T_sym[i, j]))
    hot_pairs = [p for p in sorted(hot_pairs, key=lambda x: x[2], reverse=True) if p[2] > 0]
    hot_pairs = [(i, j) for i, j, _ in hot_pairs[: int(detailed_cfg.get("top_pairs_k", 10))]]

    best_assign, best_eval = run_detailed_place(
        assign=assign,
        evaluator=evaluator,
        sites_xy_mm=sites_xy,
        chip_tdp_w=chip_tdp_w,
        traffic_bytes=traffic,
        pareto=pareto,
        steps=int(detailed_cfg.get("steps", 1000)),
        sa_T0=float(detailed_cfg.get("sa_T0", 1.0)),
        sa_alpha=float(detailed_cfg.get("sa_alpha", 0.999)),
        action_probs=detailed_cfg.get("action_probs", {"swap": 0.6, "relocate": 0.3, "cluster_move": 0.1}),
        hot_pairs=hot_pairs,
        trace=trace,
        pareto_points=pareto_points,
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
                "total_scalar": best_eval["total_scalar"],
            },
        },
        "pareto_front": [p.__dict__ for p in pareto.points],
        "selection": {"knee": knee.__dict__ if knee else None, "pareto_size": len(pareto.points)},
        "region_plan": {
            "clusters": [c.__dict__ for c in clusters],
            "cluster_to_region": cluster_to_region.tolist(),
        },
        "artifacts": {
            "trace_csv": str(Path(out_dir) / "trace.csv"),
            "pareto_csv": str(Path(out_dir) / "pareto_points.csv"),
        },
    }

    report = {
        "baseline": layout_input.get("baseline", {}),
        "best": layout_best["best"],
        "pareto_size": len(pareto.points),
        "selection": layout_best["selection"],
    }

    save_json(layout_best, os.path.join(out_dir, "layout_best.json"))
    save_json(report, os.path.join(out_dir, "report.json"))
    _save_csv(Path(out_dir) / "trace.csv", trace, fieldnames=list(trace[0].keys()) if trace else [])
    _save_csv(
        Path(out_dir) / "pareto_points.csv",
        pareto_points,
        fieldnames=list(pareto_points[0].keys()) if pareto_points else [],
    )
    print(f"Saved layout_best.json, trace.csv and pareto_points.csv to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_input", type=str, required=True, help="Path to layout_input.json")
    parser.add_argument("--cfg", type=str, default=None, help="Optional YAML config for layout agent")
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    cfg: Dict | Config = {}
    if args.cfg:
        cfg = load_config(args.cfg)

    layout_input = load_layout_input(args.layout_input)
    stage_pipeline(cfg, layout_input, args.out_dir)


if __name__ == "__main__":
    main()

