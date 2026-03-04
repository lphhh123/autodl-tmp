"""Offline EDA-Agent driver (SPEC v5.4)."""
from __future__ import annotations

# --- bootstrap sys.path for both invocation styles ---
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# -----------------------------------------------------

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import time
import uuid
from types import SimpleNamespace
from typing import Any

import numpy as np
from omegaconf import OmegaConf

# Optional progress bar (tqdm). If not installed, fall back to periodic prints.
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

from layout.alt_opt import run_alt_opt
from layout.candidate_pool import signature_from_assign
from layout.coarsen import coarsen_traffic, Cluster
from layout.detailed_place import run_detailed_place
from layout.evaluator import LayoutEvaluator, LayoutState, compute_raw_terms_for_assign, evaluator_version
from layout.expand import expand_clusters
from layout.global_place_region import assign_clusters_to_regions
from layout.legalize import legalize_assign
from layout.pareto import ParetoSet
from layout.pareto_io import write_pareto_points_csv
from layout.regions import build_regions, Region
from layout.sites import build_sites
from layout.trace_metrics import compute_trace_metrics_from_csv
from mapping.mapping_solver import MappingSolver
from mapping.segments import Segment
from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults
from utils.config_utils import get_nested
from utils.contract_seal import assert_cfg_sealed_or_violate
from utils.seed import seed_everything
from utils.stable_hash import stable_hash
from utils.trace_guard import (
    init_trace_dir_v54,
    append_trace_event_v54,
    finalize_trace_dir,
    update_trace_summary,
)
from utils.trace_contract_v54 import TraceContractV54
from utils.trace_signature_v54 import build_signature_v54, REQUIRED_SIGNATURE_FIELDS


def load_layout_input(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_oscillation_metrics(trace_path: Path, window: int, eps_flat: float) -> dict:
    return compute_trace_metrics_from_csv(trace_path, window, eps_flat)


def _prepare_baseline_for_run(
    layout_input: dict,
    sites_xy: np.ndarray,
    assign_grid: np.ndarray,
    chip_tdp: np.ndarray,
    traffic: np.ndarray,
    sigma_target: float,
    policy: str,
) -> tuple[dict, dict]:
    baseline_in = layout_input.get("baseline", {}) or {}
    obj_in = baseline_in.get("objective_cfg", {}) if isinstance(baseline_in, dict) else {}
    sigma_in = None
    ev_ver_in = None
    if isinstance(obj_in, dict):
        try:
            sigma_in = float(obj_in.get("sigma_mm")) if obj_in.get("sigma_mm") is not None else None
        except Exception:
            sigma_in = None
        ev_ver_in = obj_in.get("evaluator_version", None)

    Lc_in = baseline_in.get("L_comm", None) if isinstance(baseline_in, dict) else None
    Lt_in = baseline_in.get("L_therm", None) if isinstance(baseline_in, dict) else None

    need = False
    reasons = []
    if Lc_in is None or Lt_in is None:
        need = True
        reasons.append("missing_baseline_terms")
    if sigma_in is None:
        need = True
        reasons.append("missing_baseline_sigma_mm")
    if sigma_in is not None and abs(float(sigma_in) - float(sigma_target)) > 1e-6:
        need = True
        reasons.append(f"sigma_mismatch:{sigma_in}->{sigma_target}")
    if ev_ver_in is not None and str(ev_ver_in) != str(evaluator_version()):
        need = True
        reasons.append("evaluator_version_mismatch")

    if need and policy == "error":
        raise ValueError(
            "[LayoutBaselineGuard] baseline objective_cfg incompatible with current cfg.objective.sigma_mm. "
            f"policy=error, reasons={reasons}. "
            "Fix by regenerating layout_input baseline with same sigma_mm, or set LAYOUT_BASELINE_POLICY=auto."
        )

    if need:
        raw = compute_raw_terms_for_assign(
            sites_xy_mm=sites_xy,
            assign=assign_grid,
            chip_tdp_w=chip_tdp,
            traffic_bytes=traffic,
            sigma_mm=float(sigma_target),
        )
        L_comm = float(raw["L_comm"])
        L_therm = float(raw["L_therm"])
        source = "recomputed"
    else:
        L_comm = float(Lc_in)
        L_therm = float(Lt_in)
        source = "input"

    baseline_used = {
        "L_comm": float(L_comm),
        "L_therm": float(L_therm),
        "objective_cfg": {
            "objective_version": "v5.4",
            "sigma_mm": float(sigma_target),
            "evaluator_version": evaluator_version(),
            "baseline_schema": "assign_grid+traffic+tdp",
        },
        "source": source,
    }
    baseline_meta = {
        "policy": str(policy),
        "recomputed": bool(need),
        "reasons": reasons,
        "sigma_in": sigma_in,
        "sigma_used": float(sigma_target),
        "L_comm_in": float(Lc_in) if Lc_in is not None else None,
        "L_therm_in": float(Lt_in) if Lt_in is not None else None,
        "L_comm_used": float(L_comm),
        "L_therm_used": float(L_therm),
        "evaluator_version_used": evaluator_version(),
        "evaluator_version_in": str(ev_ver_in) if ev_ver_in is not None else None,
    }
    return baseline_used, baseline_meta


def _is_valid_assign(assign: list[int] | np.ndarray | None, S: int, Ns: int) -> bool:
    if assign is None:
        return False
    if len(assign) != int(S):
        return False
    return all(0 <= int(x) < int(Ns) for x in assign)


def parse_llm_usage(llm_usage_path: Path) -> dict:
    stats = {
        "attempts": 0,
        "ok": 0,
        "status_code_counts": {},
        "error_code_counts": {},
        "fatal_seen": False,
    }
    if not llm_usage_path.exists():
        return stats
    raw = llm_usage_path.read_text(encoding="utf-8")
    chunks = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if len(chunks) <= 1 and raw and "}\\n{" in raw:
        chunks = [x.strip() for x in re.split(r"}\n(?=\{)", raw) if x.strip()]
        chunks = [f"{x}}}" if not x.endswith("}") else x for x in chunks]

    for rec in chunks:
        try:
            obj = json.loads(rec)
        except Exception:
            continue
        if "status_code" not in obj and not obj.get("event", "").startswith("llm_"):
            continue
        stats["attempts"] += 1
        if bool(obj.get("ok", False)):
            stats["ok"] += 1
        sc = str(obj.get("status_code", "") or "")
        if sc:
            stats["status_code_counts"][sc] = int(stats["status_code_counts"].get(sc, 0)) + 1
        ec = str(obj.get("error_code", "") or "")
        if ec:
            stats["error_code_counts"][ec] = int(stats["error_code_counts"].get(ec, 0)) + 1
        if bool(obj.get("fatal", False)):
            stats["fatal_seen"] = True
    return stats


def _append_trace_events_from_csv(trace_events_path: Path, trace_csv: Path, run_id: str) -> int:
    if not trace_csv.exists():
        return 0
    added = 0
    with trace_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stage = str(row.get("stage", "") or "")
            if stage in {"init", "finalize"}:
                continue

            def _num(val, default=0.0):
                try:
                    if val is None or val == "":
                        return default
                    return float(val)
                except Exception:
                    return default

            def _int(val, default=0):
                try:
                    if val is None or val == "":
                        return default
                    return int(float(val))
                except Exception:
                    return default

            iter_id = _int(row.get("iter", 0), 0)
            comm_norm = _num(row.get("comm_norm", 0.0), 0.0)
            therm_norm = _num(row.get("therm_norm", 0.0), 0.0)

            append_trace_event_v54(
                trace_events_path,
                "layout_step",
                payload={
                    "iter": int(iter_id),
                    "stage": stage,
                    "op": str(row.get("op", "")),
                    "op_args_json": str(row.get("op_args_json", "")),
                    "accepted": _int(row.get("accepted", 0), 0),
                    "total_scalar": _num(row.get("total_scalar", 0.0), 0.0),
                    "comm_norm": comm_norm,
                    "therm_norm": therm_norm,
                    "pareto_added": _int(row.get("pareto_added", 0), 0),
                    "duplicate_penalty": _num(row.get("duplicate_penalty", 0.0), 0.0),
                    "boundary_penalty": _num(row.get("boundary_penalty", 0.0), 0.0),
                    "seed_id": _int(row.get("seed_id", 0), 0),
                    "time_ms": _int(row.get("time_ms", 0.0), 0),
                    "signature": str(row.get("signature", "")),
                },
                run_id=str(run_id),
                step=int(iter_id),
            )
            append_trace_event_v54(
                trace_events_path,
                "proxy_sanitize",
                payload={
                    "candidate_id": int(iter_id),
                    "outer_iter": int(iter_id),
                    "inner_step": int(iter_id),
                    "metric": "comm",
                    "raw_value": float(comm_norm),
                    "used_value": float(comm_norm),
                    "penalty_added": 0.0,
                },
                run_id=str(run_id),
                step=int(iter_id),
            )
            append_trace_event_v54(
                trace_events_path,
                "proxy_sanitize",
                payload={
                    "candidate_id": int(iter_id),
                    "outer_iter": int(iter_id),
                    "inner_step": int(iter_id),
                    "metric": "therm",
                    "raw_value": float(therm_norm),
                    "used_value": float(therm_norm),
                    "penalty_added": 0.0,
                },
                run_id=str(run_id),
                step=int(iter_id),
            )
            added += 1
    return added

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


def run_layout_agent(
    cfg,
    out_dir: Path,
    seed: int,
    layout_input_path: str | Path | None = None,
    run_id: str | None = None,
) -> None:
    if layout_input_path is None:
        layout_input_path = getattr(cfg, "layout_input", None)
    if not layout_input_path:
        raise RuntimeError("Missing layout_input: please set cfg.layout_input or pass layout_input_path")
    layout_input = load_layout_input(Path(layout_input_path))
    instance_name = str(layout_input.get("instance_name", "base"))
    baseline_policy = str(os.environ.get("LAYOUT_BASELINE_POLICY", "auto")).strip().lower()
    _sites_xy_for_base = np.array(layout_input["sites"]["sites_xy"], dtype=np.float32)
    _assign_grid_for_base = np.array(layout_input["baseline"]["assign_grid"], dtype=int)
    _chip_tdp_for_base = np.array(layout_input["slots"]["tdp"], dtype=float)
    _traffic_for_base = np.array(layout_input["mapping"]["traffic_matrix"], dtype=float)

    baseline_used, baseline_meta = _prepare_baseline_for_run(
        layout_input=layout_input,
        sites_xy=_sites_xy_for_base,
        assign_grid=_assign_grid_for_base,
        chip_tdp=_chip_tdp_for_base,
        traffic=_traffic_for_base,
        sigma_target=float(cfg.objective.sigma_mm),
        policy=baseline_policy,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_out = Path(getattr(cfg, "output_dir", out_dir))
    base_out.mkdir(parents=True, exist_ok=True)
    # ---- v5.4: canonical trace events (JSONL) ----
    if run_id is None:
        run_id = stable_hash(
            {
                "mode": "layout_agent",
                "cfg_path": str(getattr(cfg, "cfg_path", "")),
                "seed_id": int(seed),
                "layout_input": str(layout_input_path),
            }
        )
    pipeline_mode = str(get_nested(cfg, "layout_agent.pipeline", "full")).strip().lower()
    sa_only = pipeline_mode in ("sa_only", "sa", "detailed_only")
    method_name = str(get_nested(cfg, "layout_agent.method_name", "ours_layout_agent"))

    sig = build_signature_v54(
        cfg,
        method_name=method_name,
        overrides={"seed_global": int(seed), "seed_problem": int(seed)},
    )
    meta = init_trace_dir_v54(
        base_dir=base_out,
        run_id=run_id,
        cfg=cfg,
        signature=sig,
        signature_v54=sig,
        run_meta={
            "heuristic": method_name,
            "pipeline_mode": pipeline_mode,
            "layout_input": str(layout_input_path),
            "mode": "layout",
            "run_id": run_id,
            "seed_id": int(seed),
            "requested": {"method": "layout_agent"},
            "effective": {"method": "layout_agent"},
            "seal_check_policy": {"kind": "per_step", "interval": 1},
            "layout_instance_name": instance_name,
            "baseline_guard": baseline_meta,
        },
        required_signature_fields=REQUIRED_SIGNATURE_FIELDS,
    )
    trace_dir = meta["trace_dir"]
    out_dir = trace_dir
    llm_usage_path = trace_dir / "llm_usage.jsonl"
    llm_usage_path_compat = base_out / "llm_usage.jsonl"
    if not llm_usage_path.exists():
        llm_usage_path.write_text("", encoding="utf-8")
    trace_path = trace_dir / "trace.csv"
    trace_path_compat = base_out / "trace.csv"
    trace_events_path = trace_dir / "trace_events.jsonl"
    with (trace_dir / "trace_header.json").open("r", encoding="utf-8") as f:
        header_payload = json.load(f)
    append_trace_event_v54(
        trace_events_path=trace_events_path,
        run_id=run_id,
        step=0,
        event_type="trace_header",
        payload=header_payload,
        strict=True,
    )
    trace_contract = TraceContractV54
    seal_digest = str(get_nested(cfg, "contract.seal_digest", "") or "")
    assert_cfg_sealed_or_violate(
        cfg=cfg,
        seal_digest=seal_digest,
        trace_events_path=trace_events_path,
        phase="layout_agent",
        step=0,
        fatal=True,
        trace_contract=trace_contract,
    )

    detailed_cfg = cfg.detailed_place if hasattr(cfg, "detailed_place") else {}
    planned_steps = int(detailed_cfg.get("max_steps", detailed_cfg.get("steps", 0)) or 0)
    # ---- budget (SPEC_D/E): evaluator-calls hard budget ----
    budget_cfg = getattr(cfg, "budget", None) or {}
    total_eval_budget = int(getattr(budget_cfg, "total_eval_budget", 0) or 0)
    max_wallclock_sec = float(getattr(budget_cfg, "max_wallclock_sec", 0.0) or 0.0)
    search_eval_budget = total_eval_budget - 1 if total_eval_budget and total_eval_budget > 1 else total_eval_budget
    progress_enabled = bool(get_nested(cfg, "progress.enabled", True))
    progress_poll_s = float(get_nested(cfg, "progress.poll_sec", 0.5))

    start_time = time.time()
    seed_everything(int(seed))
    best_assign = None
    best_total = None
    evaluator = None
    S = 0
    Ns = 0
    ok = False
    pbar = None  # progress bar (tqdm), if enabled
    try:
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
        Ns = sites_xy.shape[0]
        evaluator = LayoutEvaluator(
            sigma_mm=float(cfg.objective.sigma_mm),
            baseline={
                "L_comm_baseline": float(baseline_used.get("L_comm", 1.0)),
                "L_therm_baseline": float(baseline_used.get("L_therm", 1.0)),
            },
            scalar_w={
                "w_comm": float(cfg.objective.scalar_weights.w_comm),
                "w_therm": float(cfg.objective.scalar_weights.w_therm),
                "w_penalty": float(cfg.objective.scalar_weights.w_penalty),
            },
        )
        class _BudgetExceeded(RuntimeError):
            pass

        _orig_eval = evaluator.evaluate

        # tqdm progress bar (eval-calls). Show whenever a hard eval budget is configured.
        last_print_t = 0.0
        if progress_enabled and search_eval_budget and tqdm is not None:
            try:
                # tqdm writes to stderr by default; keep it separate from JSON stdout.
                pbar = tqdm(total=int(search_eval_budget), desc="layout eval_calls", dynamic_ncols=True)
            except Exception:
                pbar = None

        def _budgeted_evaluate(state):
            nonlocal last_print_t
            if search_eval_budget and evaluator.evaluator_calls >= search_eval_budget:
                raise _BudgetExceeded("eval budget exhausted")
            out = _orig_eval(state)
            # Update progress
            if pbar is not None:
                try:
                    pbar.update(1)
                except Exception:
                    pass
            else:
                # Fallback: periodic plain-text heartbeat (every ~progress_poll_s seconds)
                now = time.time()
                if progress_enabled and search_eval_budget and (now - last_print_t) >= max(0.2, progress_poll_s):
                    last_print_t = now
                    cur = int(evaluator.evaluator_calls)
                    print(f"[progress] eval_calls={cur}/{int(search_eval_budget)}")
            return out

        evaluator.evaluate = _budgeted_evaluate
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

        J = 0.0
        if sa_only:
            # ===== SA-only baseline: no coarsen/regions/expand pipeline =====
            Ns = int(sites_xy.shape[0])

            # Trivial single region covering all sites
            centroid = sites_xy.mean(axis=0).astype(np.float32) if Ns > 0 else np.zeros(2, dtype=np.float32)
            regions = [
                Region(
                    region_id=0,
                    ring_idx=0,
                    sector_idx=0,
                    site_ids=[int(i) for i in range(Ns)],
                    capacity=Ns,
                    centroid_xy_mm=centroid,
                    ring_score=1.0,
                )
            ]
            site_to_region = np.zeros((Ns,), dtype=int)

            # Trivial singleton clusters (avoid cluster_move; but keep structure for logging)
            clusters = [Cluster(cluster_id=i, slots=[i], tdp_sum=float(chip_tdp[i])) for i in range(S)]
            cluster_to_region = [0 for _ in range(len(clusters))]

            # Start SA from seed assignment; optionally legalize it (safe)
            if bool(get_nested(cfg, "legalize.enabled", True)):
                assign_leg = legalize_assign(assign_seed.copy(), sites_xy, wafer_radius)
            else:
                assign_leg = assign_seed.copy()

        else:
            # ===== Full pipeline (ours) =====
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
                py_rng=random.Random(int(seed)),
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
        budget_exhausted = False
        try:
            assert_cfg_sealed_or_violate(
                cfg=cfg,
                seal_digest=seal_digest,
                trace_events_path=trace_events_path,
                phase="layout_agent.before_detailed_place",
                step=1,
                fatal=True,
                trace_contract=trace_contract,
            )
            # IMPORTANT(v5.4): pass the FULL sealed cfg into detailed_place, not cfg.detailed_place.
            # detailed_place will read its parameters from cfg.detailed_place internally.
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
                cfg=cfg,
                trace_path=trace_path,
                seed_id=int(seed),
                chip_tdp=chip_tdp,
                llm_usage_path=llm_usage_path,
                recordings_path=out_dir / "recordings.jsonl",
                trace_events_path=trace_events_path,
            )
            assert_cfg_sealed_or_violate(
                cfg=cfg,
                seal_digest=seal_digest,
                trace_events_path=trace_events_path,
                phase="layout_agent.after_detailed_place",
                step=2,
                fatal=True,
                trace_contract=trace_contract,
            )
        except _BudgetExceeded:
            budget_exhausted = True
            # choose best scalar point from pareto and DO NOT revert to assign_leg
            w_comm = float(cfg.objective.get("scalar_weights", {}).get("w_comm", 0.7))
            w_therm = float(cfg.objective.get("scalar_weights", {}).get("w_therm", 0.3))
            _, _, payload = pareto.best_by_scalar(w_comm=w_comm, w_therm=w_therm)
            best_assign = payload.get("assign", None)
            best_total = payload.get("total_scalar", None)

            if best_assign is None:
                best_assign = layout_state.assign.copy()

            result = SimpleNamespace(
                assign=np.array(best_assign, dtype=int),
                policy_meta={"budget_exhausted": True},
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
                trace_path=trace_path,
                chip_tdp=chip_tdp,
            )

        # Outputs
        write_pareto_points_csv(pareto, out_dir / "pareto_points.csv")

        # weights for scalar selection
        w_comm = float(cfg.objective.get("scalar_weights", {}).get("w_comm", 0.7))
        w_therm = float(cfg.objective.get("scalar_weights", {}).get("w_therm", 0.3))

        # knee point (trade-off visualization)
        knee_comm, knee_therm, knee_payload = pareto.knee_point()
        knee_assign = knee_payload.get("assign", assign_final)

        # scalar-best point (for main comparisons)
        sel_comm, sel_therm, sel_payload = pareto.best_by_scalar(w_comm=w_comm, w_therm=w_therm)
        sel_assign = sel_payload.get("assign", assign_final)

        # evaluate selected assign once to get consistent penalties/fields
        layout_state.assign = np.array(sel_assign, dtype=int)
        sel_eval = None
        if sel_assign is not None and (not total_eval_budget or evaluator.evaluator_calls < total_eval_budget):
            evaluator.evaluate = _orig_eval
            sel_eval = evaluator.evaluate(layout_state)

        runtime_s = time.time() - start_time
        selected_total = float(sel_eval.get("total_scalar", 0.0)) if sel_eval else float(sel_payload.get("total_scalar", 0.0))

        # optional: evaluate knee too (only if budget allows; best-effort)
        knee_eval = None
        if knee_assign is not None and (not total_eval_budget or evaluator.evaluator_calls < total_eval_budget):
            layout_state.assign = np.array(knee_assign, dtype=int)
            knee_eval = evaluator.evaluate(layout_state)
        knee_total = float(knee_eval.get("total_scalar", 0.0)) if knee_eval else float(knee_payload.get("total_scalar", 0.0))

        best_assign = sel_assign
        best_total = selected_total

        layout_best = {
            "run_id": run_id,
            "selected": {
                "assign": sel_assign.tolist() if hasattr(sel_assign, "tolist") else sel_assign,
                "comm_norm": float(sel_comm),
                "therm_norm": float(sel_therm),
                "total_scalar": float(selected_total),
            },
            "knee": {
                "assign": knee_assign.tolist() if hasattr(knee_assign, "tolist") else knee_assign,
                "comm_norm": float(knee_comm),
                "therm_norm": float(knee_therm),
                "total_scalar": float(knee_total),
            },
            "pareto_front": [
                {"comm_norm": p.comm_norm, "therm_norm": p.therm_norm} for p in pareto.points
            ],
            "selection": {
                "method": cfg.pareto.get("selection", "scalar_best_v1") if hasattr(cfg, "pareto") else "scalar_best_v1",
                "pareto_size": len(pareto.points),
            },
            "region_plan": {
                "clusters": [c.slots for c in clusters],
                "cluster_to_region": cluster_to_region,
                "J": float(J) if J is not None else None,
            },
            "artifacts": {
                "trace_csv": str(trace_path.absolute()),
                "trace_csv_compat": str(trace_path_compat.absolute()),
                "pareto_csv": str((out_dir / "pareto_points.csv").absolute()),
                "llm_usage_jsonl": str(llm_usage_path.absolute()),
            },
            "alt_opt": {
                "enabled": bool(cfg.alt_opt.get("enabled", False)) if hasattr(cfg, "alt_opt") else False,
                "mapping_final": mapping_final,
            },
        }
        with (out_dir / "layout_best.json").open("w", encoding="utf-8") as f:
            json.dump(layout_best, f, indent=2)

        detailed_cfg = cfg.detailed_place if hasattr(cfg, "detailed_place") else {}
        metrics_window = int(detailed_cfg.get("metrics_window_lastN", 200))
        eps_flat = float(detailed_cfg.get("eps_flat", 1e-4))
        trace_metrics = compute_trace_metrics_from_csv(trace_path, metrics_window, eps_flat)

        trace_min = {}
        for k in (
            "best_total", "best_comm", "best_therm",
            "best_total_iter", "best_total_signature", "best_total_stage",
            "best_total_comm", "best_total_therm",
        ):
            if k in trace_metrics:
                trace_min[k] = trace_metrics.pop(k)

        llm_stats = parse_llm_usage(llm_usage_path)
        llm_used_sum = 0
        if trace_path.exists():
            with trace_path.open("r", encoding="utf-8", newline="") as tf:
                for row in csv.DictReader(tf):
                    try:
                        llm_used_sum += int(float(row.get("llm_used", 0) or 0))
                    except Exception:
                        continue

        report = {
            "run_id": run_id,
            "baseline": {"comm_norm": base_eval["comm_norm"], "therm_norm": base_eval["therm_norm"]},
            # knee is for visualization/trade-off (not used as the main "selected" anymore)
            "knee": {"comm_norm": float(knee_comm), "therm_norm": float(knee_therm), "total_scalar": float(knee_total)},
            # selected is scalar-best under the objective weights (main comparison)
            "selected_total_scalar": float(selected_total),
            "selected_comm_norm": float(sel_eval.get("comm_norm", sel_comm)) if sel_eval else float(sel_comm),
            "selected_therm_norm": float(sel_eval.get("therm_norm", sel_therm)) if sel_eval else float(sel_therm),
            "selected_penalty": sel_eval.get("penalty", {}) if sel_eval else (sel_payload.get("penalty", {}) if isinstance(sel_payload, dict) else {}),
            "pareto_size": len(pareto.points),
            "pareto_front_size": len(pareto.points),
            "alt_opt_rounds": int(cfg.alt_opt.get("rounds", 0)) if hasattr(cfg, "alt_opt") else 0,
            "metrics_window_lastN": metrics_window,
            "eps_flat": eps_flat,
            "runtime_s": float(runtime_s),
            "evaluator_calls": int(getattr(evaluator, "evaluator_calls", getattr(evaluator, "evaluate_calls", 0))),
            "evaluate_calls": int(getattr(evaluator, "evaluate_calls", 0)),
            "policy_switch": result.policy_meta.get("policy_switch") if result.policy_meta else None,
            "cache": result.policy_meta.get("cache") if result.policy_meta else None,
            "mpvs": result.policy_meta.get("mpvs") if getattr(result, "policy_meta", None) else None,
            "trace_min": trace_min,
            "llm": {
                "attempts": int(llm_stats.get("attempts", 0)),
                "ok": int(llm_stats.get("ok", 0)),
                "status_code_counts": llm_stats.get("status_code_counts", {}),
                "error_code_counts": llm_stats.get("error_code_counts", {}),
                "fatal_seen": bool(llm_stats.get("fatal_seen", False)),
                "llm_used_sum": int(llm_used_sum),
            },
            **trace_metrics,
        }
        with (out_dir / "report.json").open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # ---- v5.4: enforce budget fairness artifact (budget.json) ----
        try:
            actual_calls = int(getattr(evaluator, "evaluator_calls", 0) if evaluator else 0)
            budget = {
                "run_id": run_id,
                "primary_limit": {"type": "eval_calls", "limit": int(total_eval_budget)},
                "secondary_limit": {"type": "wall_time_s", "limit": float(max_wallclock_sec)},
                "actual_eval_calls": actual_calls,
                "effective_eval_calls": actual_calls,
                "cache_hits_total": 0,
                "wall_time_s": float(time.time() - start_time),
                "planned_total_eval_budget": int(total_eval_budget),
                "planned_search_eval_budget": int(search_eval_budget),
                "seed_id": int(seed) if "seed" in locals() else None,
                "method": "ours",
                "budget_exhausted": bool(budget_exhausted),
                "actual_evaluator_calls": actual_calls,
            }
            (out_dir / "budget.json").write_text(
                json.dumps(budget, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            raise RuntimeError(f"failed to write budget.json: {exc}") from exc
        ok = True
    except Exception as exc:
        steps_done = int(getattr(evaluator, "evaluator_calls", 0)) if evaluator is not None else 0
        raise
    finally:
        # Close progress bar before writing final logs.
        if pbar is not None:
            try:
                pbar.close()
            except Exception:
                pass
        steps_done = int(getattr(evaluator, "evaluator_calls", 0)) if evaluator is not None else 0
        best_solution_valid = _is_valid_assign(best_assign, S, Ns)
        trace_append_err = None
        try:
            _append_trace_events_from_csv(
                trace_events_path=trace_events_path,
                trace_csv=trace_path,
                run_id=str(run_id),
            )
        except Exception as e:
            trace_append_err = repr(e)
            ok = False
            raise
        finally:
            reason = "ok" if (ok and trace_append_err is None) else f"trace_append_error:{trace_append_err}"
            # prefer actual executed steps from detailed_place meta if available
            steps_exec = 0
            try:
                steps_exec = int(getattr(result, "policy_meta", {}).get("steps_executed", 0))
            except Exception:
                steps_exec = 0
            if steps_exec <= 0:
                steps_exec = int(getattr(evaluator, "evaluator_calls", 0)) if evaluator is not None else 0
            if trace_dir is not None:
                update_trace_summary(
                    trace_dir,
                    ok=bool(ok and trace_append_err is None),
                    reason=reason,
                    steps_done=int(steps_exec),
                    best_solution_valid=bool(ok and best_assign is not None),
                )
                finalize_trace_dir(
                    trace_events_path,
                    reason=reason,
                    steps_done=int(steps_exec),
                    best_solution_valid=bool(ok and best_assign is not None),
                )
            if trace_path.exists():
                shutil.copy2(trace_path, trace_path_compat)
            try:
                if llm_usage_path.exists():
                    llm_usage_path_compat.write_text(
                        llm_usage_path.read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="./configs/layout_agent/layout_L0_heuristic.yaml")
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layout_input", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    OmegaConf.set_struct(cfg, False)
    # auto out_dir if not provided
    cfg_stem = Path(args.cfg).stem
    auto_out = Path("outputs/layout_agent") / f"{cfg_stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir) if args.out_dir else Path(getattr(getattr(cfg, "train", None), "out_dir", "") or auto_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = uuid.uuid4().hex

    # sync cfg.train.out_dir
    if not hasattr(cfg, "train") or cfg.train is None:
        cfg.train = OmegaConf.create({})
    elif not OmegaConf.is_config(cfg.train):
        if isinstance(cfg.train, dict):
            cfg.train = OmegaConf.create(cfg.train)
        else:
            cfg.train = OmegaConf.create(dict(cfg.train))
    cfg.train.out_dir = str(out_dir)
    cfg.out_dir = str(out_dir)

    # ---- v5.4 contract: preserve requested snapshot BEFORE any env overrides ----
    def _strip_contract_local(container: Any) -> Any:
        if not isinstance(container, dict):
            return container
        out = {}
        for k, v in container.items():
            if isinstance(k, str) and (k.startswith("_contract") or k == "contract"):
                continue
            out[k] = _strip_contract_local(v) if isinstance(v, dict) else v
        return out

    # Ensure _contract exists
    if not hasattr(cfg, "_contract") or cfg._contract is None:
        cfg._contract = OmegaConf.create({})

    # Record requested snapshot if missing (pre-override, pre-seal)
    if get_nested(cfg, "_contract.requested_config_snapshot", None) is None:
        raw_req = OmegaConf.to_container(cfg, resolve=False)
        cfg._contract.requested_config_snapshot = _strip_contract_local(raw_req)

    # Ensure overrides list exists
    if get_nested(cfg, "_contract.overrides", None) is None:
        cfg._contract.overrides = []

    # ---- Apply env overrides BEFORE sealing (validate_and_fill_defaults will seal) ----
    budget_override = os.environ.get("TOTAL_EVAL_BUDGET_OVERRIDE", "").strip()
    if budget_override:
        try:
            if not hasattr(cfg, "budget") or cfg.budget is None:
                cfg.budget = OmegaConf.create({})
            old_val = int(getattr(cfg.budget, "total_eval_budget", 0) or 0)
            new_val = int(budget_override)
            cfg.budget.total_eval_budget = new_val

            ov = list(get_nested(cfg, "_contract.overrides", []) or [])
            ov.append(
                {
                    "path": "budget.total_eval_budget",
                    "requested": old_val,
                    "effective": new_val,
                    "reason": "env:TOTAL_EVAL_BUDGET_OVERRIDE",
                }
            )
            cfg._contract.overrides = ov
        except Exception:
            pass

    # NOW seal + fill defaults (seal_digest matches final cfg)
    cfg = validate_and_fill_defaults(cfg, mode="layout")
    seed_everything(int(args.seed))

    # dump config_used
    try:
        (out_dir / "config_used.yaml").write_text(Path(args.cfg).read_text(encoding="utf-8"), encoding="utf-8")
    except Exception:
        pass

    # dump resolved config
    try:
        import omegaconf
        (out_dir / "config_resolved.yaml").write_text(omegaconf.OmegaConf.to_yaml(cfg), encoding="utf-8")
    except Exception:
        (out_dir / "config_resolved.yaml").write_text(str(cfg), encoding="utf-8")

    layout_input_path = args.layout_input or getattr(cfg, "layout_input", None)
    layout_hash = None
    layout_meta = {}
    if layout_input_path:
        try:
            layout_hash = hashlib.sha256(Path(layout_input_path).read_bytes()).hexdigest()
        except Exception:
            layout_hash = None
        try:
            layout_meta = load_layout_input(Path(layout_input_path)).get("meta", {}) or {}
        except Exception:
            layout_meta = {}
    detailed_cfg = cfg.detailed_place if hasattr(cfg, "detailed_place") else {}
    extra_meta = {
        "budget_main_axis": "eval_calls",
        "dataset_id": f"wafer_layout:{layout_meta.get('layout_id', 'unknown')}",
    }
    resolved_text = (out_dir / "config_resolved.yaml").read_text(encoding="utf-8")

    cfg_hash = stable_hash({"cfg": resolved_text})
    from utils.run_manifest import write_run_manifest

    write_run_manifest(
        out_dir=str(out_dir),
        cfg_path=str(args.cfg),
        cfg_hash=str(cfg_hash),
        seed=int(args.seed),
        stable_hw_state={
            "guard_mode": "acc_first_hard_gating",
            "lambda_hw_base": None,
            "lambda_hw_effective": None,
            "discrete_cache": {
                "mapping_signature": str(meta.get("mapping_signature", "")) if "meta" in locals() else "",
                "layout_signature": str(meta.get("layout_signature", "")) if "meta" in locals() else "",
            },
        },
        cfg=cfg,
        extra=extra_meta,
        run_id=run_id,
        spec_version="v5.4",
        command=" ".join(sys.argv),
        code_root=str(_PROJECT_ROOT),
    )

    # run meta
    meta = {
        "argv": sys.argv,
        "out_dir": str(out_dir),
        "cfg_path": str(args.cfg),
        "cfg_hash": str(cfg_hash),
        "seed": int(args.seed),
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    run_layout_agent(
        cfg,
        out_dir=out_dir,
        seed=int(args.seed),
        layout_input_path=args.layout_input,
        run_id=run_id,
    )
    layout_best_path = out_dir / "layout_best.json"
    layout_best = json.loads(layout_best_path.read_text(encoding="utf-8")) if layout_best_path.exists() else {}
    best_assign = layout_best.get("selected", {}).get("assign", []) if isinstance(layout_best, dict) else []
    meta = {
        "mapping_signature": "",
        "layout_signature": signature_from_assign(best_assign) if best_assign else "",
    }

    write_run_manifest(
        out_dir=str(out_dir),
        cfg_path=str(args.cfg),
        cfg_hash=str(cfg_hash),
        seed=int(args.seed),
        stable_hw_state={
            "guard_mode": "acc_first_hard_gating",
            "lambda_hw_base": None,
            "lambda_hw_effective": None,
            "discrete_cache": {
                "mapping_signature": str(meta.get("mapping_signature", "")) if "meta" in locals() else "",
                "layout_signature": str(meta.get("layout_signature", "")) if "meta" in locals() else "",
            },
        },
        cfg=cfg,
        extra=extra_meta,
        run_id=run_id,
        spec_version="v5.4",
        command=" ".join(sys.argv),
        code_root=str(_PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
