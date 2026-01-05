"""Offline EDA-Agent driver (SPEC v4.3.2)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from layout.alt_opt import run_alt_opt
from layout.coarsen import coarsen_traffic
from layout.detailed_place import run_detailed_place
from layout.evaluator import LayoutEvaluator, LayoutState
from layout.expand import expand_clusters
from layout.global_place_region import assign_clusters_to_regions
from layout.legalize import legalize_assign
from layout.pareto import ParetoSet
from layout.regions import build_regions
from layout.sites import build_sites
from mapping.mapping_solver import MappingSolver
from mapping.segments import Segment
from utils.config import load_config


def load_layout_input(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_pareto_points(pareto: ParetoSet, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("solution_id,comm_norm,therm_norm,total_scalar,stage,iter,seed_id,assign_hash\n")
        for idx, p in enumerate(pareto.points):
            assign_hash = hash(tuple(p.payload.get("assign", [])))
            total = p.payload.get("total_scalar", p.comm_norm + p.therm_norm)
            stage = p.payload.get("stage", "detailed")
            iter_id = p.payload.get("iter", 0)
            seed_id = p.payload.get("seed", 0)
            f.write(f"{idx},{p.comm_norm},{p.therm_norm},{total},{stage},{iter_id},{seed_id},{assign_hash}\n")


def _parse_segments(raw_segments) -> list[Segment]:
    segments: list[Segment] = []
    if not raw_segments:
        return segments
    for idx, seg in enumerate(raw_segments):
        segments.append(
            Segment(
                id=seg.get("id", idx),
                layer_ids=seg.get("layer_ids", []),
                flops=seg.get("flops", 0.0),
                bytes=seg.get("bytes", 0.0),
                seq_len=seg.get("seq_len", 0),
                embed_dim=seg.get("embed_dim", 0),
                num_heads=seg.get("num_heads", 0),
                mlp_ratio=seg.get("mlp_ratio", 0.0),
                precision=seg.get("precision", 1),
                traffic_in_bytes=seg.get("traffic_in_bytes", 0.0),
                traffic_out_bytes=seg.get("traffic_out_bytes", 0.0),
            )
        )
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_input", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    layout_input = load_layout_input(Path(args.layout_input))
    cfg = load_config(args.cfg)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    wafer_radius = float(layout_input["wafer"]["radius_mm"])
    sites_xy = np.array(layout_input["sites"]["sites_xy"], dtype=np.float32)
    assign_grid = np.array(layout_input["baseline"]["assign_grid"], dtype=int)
    assign_seed = np.array(layout_input["seed"]["assign_seed"], dtype=int)
    chip_tdp = np.array(layout_input["slots"]["tdp"], dtype=float)
    traffic = np.array(layout_input["mapping"]["traffic_matrix"], dtype=float)
    traffic_sym = traffic + traffic.T
    mapping_current = layout_input.get("mapping", {}).get("mapping", list(range(assign_seed.shape[0])))
    segments = _parse_segments(layout_input.get("mapping", {}).get("segments", []))
    S = assign_grid.shape[0]
    evaluator = LayoutEvaluator(
        sigma_mm=float(cfg.objective.sigma_mm),
        baseline={
            "L_comm_baseline": float(layout_input["baseline"].get("L_comm", 1.0)),
            "L_therm_baseline": float(layout_input["baseline"].get("L_therm", 1.0)),
        },
        scalar_w={
            "w_comm": float(cfg.objective.scalar_weights.w_comm),
            "w_therm": float(cfg.objective.scalar_weights.w_therm),
            "w_penalty": float(cfg.objective.scalar_weights.w_penalty),
        },
    )
    layout_state = LayoutState(
        S=S,
        Ns=sites_xy.shape[0],
        wafer_radius_mm=wafer_radius,
        sites_xy_mm=sites_xy,
        assign=assign_seed.copy(),
        chip_tdp_w=chip_tdp,
        traffic_bytes=traffic,
        meta={"stage": "seed"},
    )
    mapping_solver = MappingSolver(
        strategy=str(getattr(getattr(cfg, "alt_opt", {}), "mapping_strategy", "greedy_local")),
        mem_limit_factor=float(getattr(getattr(cfg, "alt_opt", {}), "mem_limit_factor", 1.0)),
    )
    pareto = ParetoSet(
        eps_comm=float(cfg.pareto.get("eps_comm", 0.0)) if hasattr(cfg, "pareto") else 0.0,
        eps_therm=float(cfg.pareto.get("eps_therm", 0.0)) if hasattr(cfg, "pareto") else 0.0,
        max_points=int(cfg.pareto.get("max_points", 2000)) if hasattr(cfg, "pareto") else 2000,
    )

    # Stage0: baseline evaluations
    layout_state.assign = assign_grid
    base_eval = evaluator.evaluate(layout_state)
    pareto.add(
        base_eval["comm_norm"],
        base_eval["therm_norm"],
        {"assign": assign_grid.copy(), "total_scalar": base_eval["total_scalar"], "stage": "baseline", "iter": 0, "seed": -1},
    )
    layout_state.assign = assign_seed
    seed_eval = evaluator.evaluate(layout_state)
    pareto.add(
        seed_eval["comm_norm"],
        seed_eval["therm_norm"],
        {"assign": assign_seed.copy(), "total_scalar": seed_eval["total_scalar"], "stage": "seed", "iter": 0, "seed": 0},
    )

    # Stage1: coarsen
    clusters, W = coarsen_traffic(
        traffic_sym,
        slot_tdp=chip_tdp,
        target_num_clusters=int(cfg.coarsen.target_num_clusters),
        min_merge_traffic=float(cfg.coarsen.min_merge_traffic),
    )

    # Stage2: regions
    regions, site_to_region = build_regions(
        sites_xy_mm=sites_xy,
        wafer_radius_mm=wafer_radius,
        ring_edges_ratio=cfg.regions.ring_edges_ratio,
        sectors_per_ring=cfg.regions.sectors_per_ring,
        ring_score=cfg.regions.ring_score,
        capacity_ratio=float(cfg.regions.capacity_ratio),
    )
    cluster_to_region, J = assign_clusters_to_regions(
        clusters,
        regions,
        W,
        lambda_graph=float(cfg.global_place_region.lambda_graph),
        lambda_ring=float(cfg.global_place_region.lambda_ring),
        lambda_cap=float(cfg.global_place_region.lambda_cap),
        refine_cfg=cfg.global_place_region.get("refine", {}),
    )

    # Stage3: expand
    assign_expand = expand_clusters(
        clusters=clusters,
        cluster_to_region=cluster_to_region,
        regions=regions,
        base_assign=np.full_like(assign_seed, -1),
        traffic_sym=traffic_sym,
        sites_xy=sites_xy,
        intra_refine_steps=int(cfg.expand.get("intra_refine_steps", 0)),
    )

    # Stage4: legalize
    assign_leg = legalize_assign(assign_expand, sites_xy, wafer_radius)
    layout_state.assign = assign_leg

    # Stage5: detailed place
    detailed_cfg = cfg.detailed_place if hasattr(cfg, "detailed_place") else {}
    detailed_steps = int(detailed_cfg.get("steps", 0))
    if detailed_steps <= 0:
        layout_state.assign = assign_leg
        legalize_eval = evaluator.evaluate(layout_state)
        pareto.add(
            legalize_eval["comm_norm"],
            legalize_eval["therm_norm"],
            {
                "assign": assign_leg.copy(),
                "total_scalar": legalize_eval["total_scalar"],
                "stage": "legalize",
                "iter": 0,
                "seed": 0,
            },
        )
        assign_final = assign_leg
    else:
        result = run_detailed_place(
            sites_xy=sites_xy,
            assign_seed=assign_leg,
            evaluator=evaluator,
            layout_state=layout_state,
            traffic_sym=traffic_sym,
            site_to_region=site_to_region,
            regions=regions,
            clusters=clusters,
            cluster_to_region=cluster_to_region,
            pareto=pareto,
            cfg=detailed_cfg,
            trace_path=out_dir / "trace.csv",
            seed_id=int(args.seed),
            chip_tdp=chip_tdp,
            llm_usage_path=out_dir / "llm_usage.jsonl",
        )
        assign_final = result.assign

    # Stage6: alt-opt (optional)
    mapping_final = layout_input["mapping"].get("mapping") if "mapping" in layout_input else None
    if hasattr(cfg, "alt_opt") and cfg.alt_opt.get("enabled", False):
        assign_final, mapping_final = run_alt_opt(
            rounds=int(cfg.alt_opt.get("rounds", 3)),
            mapping_solver=mapping_solver,
            segments=segments,
            mapping_init=mapping_current,
            traffic_sym=traffic_sym,
            sites_xy=sites_xy,
            assign_init=assign_final,
            evaluator=evaluator,
            layout_state=layout_state,
            pareto=pareto,
            cfg=cfg.alt_opt,
            trace_path=out_dir,
            chip_tdp=chip_tdp,
        )

    # Outputs
    _write_pareto_points(pareto, out_dir / "pareto_points.csv")
    best_comm, best_therm, best_payload = pareto.knee_point()
    best_assign = best_payload.get("assign", assign_final)
    layout_state.assign = np.array(best_assign, dtype=int)
    best_eval = evaluator.evaluate(layout_state)
    layout_best = {
        "best": {
            "assign": best_assign.tolist(),
            "pos_xy_mm": sites_xy[best_assign].tolist(),
            "objectives": {"comm_norm": best_comm, "therm_norm": best_therm},
            "raw": {"L_comm": best_eval["L_comm"], "L_therm": best_eval["L_therm"]},
            "penalty": best_eval["penalty"],
            "meta": {"stage": "knee_point"},
        },
        "pareto_front": [
            {"comm_norm": p.comm_norm, "therm_norm": p.therm_norm} for p in pareto.points
        ],
        "selection": {
            "method": cfg.pareto.get("selection", "knee_point_v1") if hasattr(cfg, "pareto") else "knee_point_v1",
            "pareto_size": len(pareto.points),
        },
        "region_plan": {
            "clusters": [c.slots for c in clusters],
            "cluster_to_region": cluster_to_region,
            "J": J,
        },
        "artifacts": {
            "trace_csv": str((out_dir / "trace.csv").absolute()),
            "pareto_csv": str((out_dir / "pareto_points.csv").absolute()),
            "llm_usage_jsonl": str((out_dir / "llm_usage.jsonl").absolute()),
        },
        "alt_opt": {
            "enabled": bool(cfg.alt_opt.get("enabled", False)) if hasattr(cfg, "alt_opt") else False,
            "mapping_final": mapping_final,
        },
    }
    with (out_dir / "layout_best.json").open("w", encoding="utf-8") as f:
        json.dump(layout_best, f, indent=2)

    report = {
        "baseline": {"comm_norm": base_eval["comm_norm"], "therm_norm": base_eval["therm_norm"]},
        "knee": {"comm_norm": best_comm, "therm_norm": best_therm},
        "pareto_size": len(pareto.points),
        "alt_opt_rounds": int(cfg.alt_opt.get("rounds", 0)) if hasattr(cfg, "alt_opt") else 0,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
