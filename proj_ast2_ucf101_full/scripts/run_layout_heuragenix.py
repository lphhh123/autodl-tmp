"""HeurAgenix baseline wrapper for wafer layout (SPEC v5.4).

Note: AMLT_OUTPUT_DIR is treated as the base directory; actual outputs are written under
<AMLT_OUTPUT_DIR>/output/...
"""
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
import math
import os
import random
import subprocess
import time
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

project_root = Path(__file__).resolve().parents[1]

from layout.candidate_pool import signature_from_assign
from layout.evaluator import LayoutEvaluator, LayoutState
from layout.pareto import ParetoSet
from layout.pareto_io import write_pareto_points_csv
from layout.trace_metrics import compute_trace_metrics_from_csv
from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults
from utils.seed import seed_everything
from utils.trace_schema import TRACE_FIELDS

REQUIRED_RECORDING_KEYS = [
    "iter",
    "stage",
    "op",
    "accepted",
    "total_scalar",
    "comm_norm",
    "therm_norm",
    "pareto_added",
    "duplicate_penalty",
    "boundary_penalty",
    "seed_id",
    "time_ms",
]

def compute_oscillation_metrics(trace_path: Path, window: int, eps_flat: float) -> dict:
    return compute_trace_metrics_from_csv(trace_path, window, eps_flat)


def _load_layout_input(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_heuragenix_root(cfg_root: str | None, project_root: Path) -> Path:
    heuragenix_root_cfg = cfg_root or "../HeurAgenix"
    hx = Path(heuragenix_root_cfg).expanduser()

    if not hx.is_absolute():
        cand_parent = (project_root.parent / hx).resolve()
        cand_local = (project_root / hx).resolve()
        if cand_parent.exists():
            heuragenix_root = cand_parent
        else:
            heuragenix_root = cand_local
    else:
        heuragenix_root = hx

    local_dup = project_root / "HeurAgenix"
    parent_dup = project_root.parent / "HeurAgenix"
    if local_dup.exists() and parent_dup.exists():
        print(f"[WARN] Detected two HeurAgenix copies: {local_dup} and {parent_dup}. Use sibling one.")
        heuragenix_root = parent_dup.resolve()

    return heuragenix_root


def _ensure_heuragenix_syspath(heuragenix_root: Path) -> None:
    heuragenix_src = heuragenix_root / "src"
    for p in (heuragenix_root, heuragenix_src):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def _resolve_llm_config_path(llm_config_file: str | None, heuragenix_root: Path, project_root: Path) -> Path | None:
    if not llm_config_file:
        return None
    candidate = Path(llm_config_file).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate
    for base in (heuragenix_root, project_root):
        path = base / candidate
        if path.exists():
            return path
    return None


def _build_objective_cfg(cfg: Any) -> Dict[str, Any]:
    objective = cfg.objective if hasattr(cfg, "objective") else {}
    scalar = objective.get("scalar_weights", {}) if isinstance(objective, dict) else objective.scalar_weights
    return {
        "sigma_mm": float(objective.get("sigma_mm", 20.0)) if isinstance(objective, dict) else float(objective.sigma_mm),
        "scalar_weights": {
            "w_comm": float(scalar.get("w_comm", 0.7)) if isinstance(scalar, dict) else float(scalar.w_comm),
            "w_therm": float(scalar.get("w_therm", 0.3)) if isinstance(scalar, dict) else float(scalar.w_therm),
            "w_penalty": float(scalar.get("w_penalty", 1000.0)) if isinstance(scalar, dict) else float(scalar.w_penalty),
        },
    }


def _build_seed_assign(layout_input: dict, seed: int) -> List[int]:
    seed_payload = layout_input.setdefault("seed", {}) or {}
    assign_seed = seed_payload.get("assign_seed")
    if isinstance(assign_seed, (list, tuple, np.ndarray)) and len(assign_seed) > 0:
        return [int(x) for x in list(assign_seed)]

    slots = layout_input.get("slots", {})
    sites = layout_input.get("sites", {})
    tdp = np.asarray(slots.get("tdp", []), dtype=int)
    S = int(slots.get("S", len(tdp)))
    sites_xy = np.asarray(sites.get("sites_xy", []), dtype=float)
    Ns = int(sites.get("Ns", len(sites_xy)))
    rng = np.random.default_rng(int(seed))
    if Ns <= 0:
        return [0 for _ in range(S)]
    if Ns >= S:
        return rng.choice(Ns, size=S, replace=False).astype(int).tolist()
    return rng.integers(low=0, high=Ns, size=S, dtype=int).tolist()


def _write_test_data(
    case: dict,
    seed: int,
    data_base: Path,
    seed_assign: List[int],
    baseline_cfg: Dict[str, Any],
    problem: str = "wafer_layout",
) -> Tuple[str, str]:
    """
    Write <data_base>/<problem>/test_data/case_seed{seed}.json
    Return:
      case_stem: "case_seed{seed}"
      case_file: "case_seed{seed}.json"
    """
    case_stem = f"case_seed{seed}"
    case_file = f"{case_stem}.json"

    test_dir = data_base / problem / "test_data"
    test_dir.mkdir(parents=True, exist_ok=True)

    payload = dict(case)
    seed_payload = dict(payload.get("seed", {}) or {})
    seed_payload["assign_seed"] = [int(x) for x in seed_assign]
    seed_payload["seed_id"] = int(seed)
    seed_payload["rng_seed"] = int(seed)
    payload["seed"] = seed_payload
    force_main = bool(baseline_cfg.get("force_main_evaluator", True))
    allow_fb = bool(baseline_cfg.get("allow_fallback_evaluator", False))
    require_main = bool(baseline_cfg.get("require_main_evaluator", True))
    payload["force_main_evaluator"] = force_main
    payload["allow_fallback_evaluator"] = allow_fb
    payload["require_main_evaluator"] = require_main

    target = test_dir / case_file
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return case_stem, case_file


def _prepare_work_dir(
    out_dir: Path,
    heuragenix_root: Path,
    case: dict,
    seed: int,
    internal_data_base: Path,
    seed_assign: List[int],
    baseline_cfg: Dict[str, Any],
) -> Tuple[Path, str, str]:
    work_dir = out_dir / "__heuragenix_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    case_stem, case_file = _write_test_data(
        case,
        seed,
        internal_data_base,
        seed_assign,
        baseline_cfg,
        problem="wafer_layout",
    )
    return work_dir, case_stem, case_file


def infer_problem_size(layout_input: dict) -> int:
    chiplets = layout_input.get("chiplets", None)
    if isinstance(chiplets, list) and len(chiplets) > 0:
        return len(chiplets)
    if "S" in layout_input:
        return int(layout_input["S"])
    slots = layout_input.get("slots", None)
    if isinstance(slots, dict) and "S" in slots:
        return int(slots["S"])
    return 1


def _resolve_heuristic_files(heuragenix_root: Path, problem: str, heuristic_dir: str) -> List[str]:
    candidate = Path(heuristic_dir)
    if not candidate.is_absolute():
        candidate = heuragenix_root / "src" / "problems" / problem / "heuristics" / heuristic_dir
    if not candidate.exists():
        raise FileNotFoundError(f"Heuristic directory not found: {candidate}")
    files = [str(path) for path in sorted(candidate.glob("*.py")) if not path.name.startswith("__")]
    if not files:
        raise RuntimeError(f"No heuristics found in {candidate}")
    return files


def _derive_initial_assign(layout_input: dict) -> np.ndarray:
    baseline = layout_input.get("baseline", {})
    seed = layout_input.get("seed", {})
    slots = layout_input.get("slots", {})
    tdp = np.asarray(slots.get("tdp", []), dtype=int)
    S = int(slots.get("S", len(tdp)))
    sites_xy = np.asarray(layout_input.get("sites", {}).get("sites_xy", []), dtype=float)
    Ns = int(layout_input.get("sites", {}).get("Ns", len(sites_xy)))

    init_assign = None
    if seed.get("assign_seed") is not None:
        init_assign = np.asarray(seed.get("assign_seed"), dtype=int)
    elif seed.get("seed_assign") is not None:
        init_assign = np.asarray(seed.get("seed_assign"), dtype=int)
    elif baseline.get("assign_grid") is not None:
        init_assign = np.asarray(baseline.get("assign_grid"), dtype=int)
    elif layout_input.get("baseline_assign_grid") is not None:
        init_assign = np.asarray(layout_input.get("baseline_assign_grid"), dtype=int)
    if init_assign is None:
        if Ns <= 0:
            init_assign = np.zeros(S, dtype=int)
        else:
            init_assign = np.arange(S, dtype=int) % max(1, Ns)
    return init_assign


def _build_evaluator(layout_input: dict, cfg: Any) -> Tuple[LayoutState, LayoutEvaluator]:
    wafer = layout_input.get("wafer", {})
    slots = layout_input.get("slots", {})
    sites = layout_input.get("sites", {})
    baseline = layout_input.get("baseline", {})

    sites_xy = np.asarray(sites.get("sites_xy", []), dtype=np.float32)
    tdp = np.asarray(slots.get("tdp", []), dtype=float)
    S = int(slots.get("S", len(tdp)))
    Ns = int(len(sites_xy))
    traffic = np.asarray(layout_input.get("mapping", {}).get("traffic_matrix", []), dtype=float)
    if traffic.size == 0:
        traffic = np.zeros((S, S), dtype=float)

    objective_cfg = _build_objective_cfg(cfg)
    evaluator = LayoutEvaluator(
        sigma_mm=float(objective_cfg["sigma_mm"]),
        baseline={
            "L_comm_baseline": float(baseline.get("L_comm", 1.0)),
            "L_therm_baseline": float(baseline.get("L_therm", 1.0)),
        },
        scalar_w={
            "w_comm": float(objective_cfg["scalar_weights"]["w_comm"]),
            "w_therm": float(objective_cfg["scalar_weights"]["w_therm"]),
            "w_penalty": float(objective_cfg["scalar_weights"]["w_penalty"]),
        },
    )

    init_assign = _derive_initial_assign(layout_input)
    base_state = LayoutState(
        S=S,
        Ns=Ns,
        wafer_radius_mm=float(wafer.get("radius_mm", 0.0)),
        sites_xy_mm=sites_xy,
        assign=np.asarray(init_assign, dtype=int).copy(),
        chip_tdp_w=tdp,
        traffic_bytes=traffic,
        meta={"margin_mm": float(wafer.get("margin_mm", 0.0))},
    )
    return base_state, evaluator


def _iter_recordings(recordings_path: Path) -> Iterable[Dict[str, Any]]:
    if not recordings_path.exists():
        # caller already wrote init row; just return empty iterator
        return
    with recordings_path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            missing = [k for k in REQUIRED_RECORDING_KEYS if k not in record]
            if missing:
                raise ValueError(
                    f"[recordings schema error] missing keys at {recordings_path}:{ln}: {missing}. "
                    f"Got keys={sorted(list(record.keys()))}"
                )
            yield record


def _action_from_record(record: Dict[str, Any], prev_assign: np.ndarray) -> Dict[str, Any]:
    op = record.get("op", "noop")
    op_args = record.get("op_args", None)
    if op_args is None:
        op_args = record.get("op_args_json", record.get("op_args_str", {}))

    if isinstance(op_args, str):
        s = op_args.strip()
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                op_args = json.loads(s)
            except Exception:
                op_args = {"raw": op_args}
        else:
            op_args = {"raw": op_args}

    if not isinstance(op_args, dict):
        op_args = {"raw": str(op_args)}
    action = {"op": op, "type": op}
    if op == "swap":
        action.update({"i": int(op_args.get("i", -1)), "j": int(op_args.get("j", -1))})
    elif op == "relocate":
        slot = int(op_args.get("i", -1))
        from_site = op_args.get("from_site")
        if from_site is None and 0 <= slot < len(prev_assign):
            from_site = int(prev_assign[slot])
        action.update({"i": slot, "site_id": int(op_args.get("site_id", -1)), "from_site": from_site})
    elif op == "cluster_move":
        action.update(
            {
                "slots": op_args.get("slots"),
                "target_sites": op_args.get("target_sites"),
                "cluster_id": op_args.get("cluster_id", -1),
                "region_id": op_args.get("region_id", -1),
                "from_region": op_args.get("from_region"),
            }
        )
    elif op == "random_kick":
        action.update({"k": op_args.get("k")})
    action.setdefault("from_region", op_args.get("from_region"))
    return action


def _assign_from_signature(sig: str) -> np.ndarray | None:
    """
    Env signature format: "assign:0,1,2,..."
    Return np.ndarray[int] or None
    """
    if not isinstance(sig, str):
        return None
    if not sig.startswith("assign:"):
        return None
    body = sig[len("assign:") :]
    if not body:
        return None
    parts = [p for p in body.split(",") if p != ""]
    try:
        return np.asarray([int(x) for x in parts], dtype=int)
    except Exception:
        return None


def _apply_action(prev_assign: np.ndarray, action: Dict[str, Any], Ns: int) -> np.ndarray:
    if prev_assign is None:
        return prev_assign
    assign = np.asarray(prev_assign, dtype=int).copy()
    op = action.get("op")
    if op == "swap":
        i = int(action.get("i", -1))
        j = int(action.get("j", -1))
        if 0 <= i < len(assign) and 0 <= j < len(assign):
            assign[i], assign[j] = assign[j], assign[i]
        return assign
    if op == "relocate":
        i = int(action.get("i", -1))
        site_id = int(action.get("site_id", -1))
        if 0 <= i < len(assign) and 0 <= site_id < Ns:
            assign[i] = site_id
        return assign
    if op == "cluster_move":
        slots = action.get("slots")
        target_sites = action.get("target_sites")
        if isinstance(slots, (list, tuple)) and isinstance(target_sites, (list, tuple)):
            for slot, site in zip(slots, target_sites):
                try:
                    slot_idx = int(slot)
                    site_id = int(site)
                except Exception:
                    continue
                if 0 <= slot_idx < len(assign) and 0 <= site_id < Ns:
                    assign[slot_idx] = site_id
        return assign
    if op == "random_kick":
        k = action.get("k")
        try:
            k = int(k)
        except Exception:
            k = 0
        if k > 0 and Ns > 0:
            for idx in range(min(k, len(assign))):
                assign[idx] = (assign[idx] + 1) % Ns
        return assign
    return assign


def _write_trace_and_pareto(
    out_dir: Path,
    seed: int,
    recordings_path: Path,
    layout_input: dict,
    cfg: Any,
    max_steps: int | None,
) -> Tuple[Dict[str, Any], ParetoSet]:
    base_state, evaluator = _build_evaluator(layout_input, cfg)
    pareto_cfg = cfg.get("pareto", {}) if isinstance(cfg, dict) else cfg.pareto
    pareto = ParetoSet(
        eps_comm=float(pareto_cfg.get("eps_comm", 0.0)),
        eps_therm=float(pareto_cfg.get("eps_therm", 0.0)),
        max_points=int(pareto_cfg.get("max_points", 2000)),
    )

    trace_path = out_dir / "trace.csv"
    prev_metrics = evaluator.evaluate(base_state)
    prev_total = float(prev_metrics["total_scalar"])
    prev_comm = float(prev_metrics["comm_norm"])
    prev_therm = float(prev_metrics["therm_norm"])
    prev_assign = _derive_initial_assign(layout_input)

    best_eval: Dict[str, Any] | None = None
    best_assign: List[int] | None = None
    accepts = 0
    best_step = -1
    improve_cnt = 0
    tabu_hits = 0
    inverse_hits = 0
    cooldown_hits = 0
    has_records = False

    # always emit a valid trace with at least one row
    with trace_path.open("w", encoding="utf-8", newline="") as f_trace:
        writer = csv.writer(f_trace)
        writer.writerow(TRACE_FIELDS)
        # ---- v5.4 required init row (even if steps=0 / no recordings) ----
        row = [
            0,
            "init",
            "init",
            json.dumps({"op": "init"}, ensure_ascii=False),
            1,
            prev_total,
            prev_comm,
            prev_therm,
            0,
            float(prev_metrics.get("penalty", {}).get("duplicate", 0.0)),
            float(prev_metrics.get("penalty", {}).get("boundary", 0.0)),
            int(seed),
            0,
            signature_from_assign(prev_assign),
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        if len(row) != len(TRACE_FIELDS):
            raise RuntimeError(f"trace row has {len(row)} cols but TRACE_FIELDS has {len(TRACE_FIELDS)}")
        writer.writerow(row)

        last_idx = -1
        for idx, rec in enumerate(_iter_recordings(recordings_path), start=1):
            # skip init record from heuragenix (we already wrote our own init row)
            try:
                if str(rec.get("op", "")).lower() == "init" and int(rec.get("iter", 0)) == 0:
                    continue
            except Exception:
                pass
            has_records = True
            last_idx = idx
            if max_steps is not None and (idx - 1) >= int(max_steps):
                break

            if "signature" not in rec:
                try:
                    signature = signature_from_assign(rec.get("assign", []))
                except Exception:
                    signature = "assign:unknown"
                rec["signature"] = signature

            step_id = int(rec.get("iter", rec.get("step", idx)))
            stage = str(rec.get("stage", "heuragenix"))
            accepted = int(rec.get("accepted", 1))
            op = rec.get("op", "noop")
            op_args = rec.get("op_args", None)
            if op_args is None:
                op_args = rec.get("op_args_json", rec.get("op_args_str", {}))

            # if it's a JSON string, parse it
            if isinstance(op_args, str):
                s = op_args.strip()
                if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
                    try:
                        import json as _json

                        op_args = _json.loads(s)
                    except Exception:
                        op_args = {"raw": op_args}
                else:
                    op_args = {"raw": op_args}

            if not isinstance(op_args, dict):
                op_args = {"raw": str(op_args)}

            # ---- update assign for this step ----
            assign_arr = None
            if isinstance(rec.get("assign"), list) and rec["assign"]:
                assign_arr = np.asarray(rec["assign"], dtype=int)
            if assign_arr is None:
                assign_arr = _assign_from_signature(rec.get("signature"))
            if assign_arr is None:
                if accepted == 1:
                    action = _action_from_record(rec, prev_assign)
                    assign_arr = _apply_action(prev_assign, action, base_state.Ns)
                else:
                    assign_arr = np.asarray(prev_assign, dtype=int)

            base_state.assign = assign_arr
            eval_out = evaluator.evaluate(base_state)
            signature = signature_from_assign(base_state.assign.tolist())

            d_total = float(eval_out["total_scalar"]) - float(prev_total)
            if accepted == 1:
                accepts += 1

            pareto_added = 0
            if accepted == 1:
                pareto_added = pareto.add(
                    float(eval_out["comm_norm"]),
                    float(eval_out["therm_norm"]),
                    {
                        "assign": base_state.assign.copy(),
                        "total_scalar": float(eval_out["total_scalar"]),
                        "stage": stage,
                        "iter": step_id,
                        "seed": seed,
                    },
                )

            if accepted == 1 and d_total < 0:
                improve_cnt += 1

            meta = rec.get("meta", {}) or {}
            tabu_hit = int(meta.get("tabu_hit", 0))
            inverse_hit = int(meta.get("inverse_hit", 0))
            cooldown_hit = int(meta.get("cooldown_hit", 0))
            tabu_hits += tabu_hit
            inverse_hits += inverse_hit
            cooldown_hits += cooldown_hit

            pen = eval_out.get("penalty", {}) or {}
            dup_pen = float(pen.get("duplicate", 0.0))
            bnd_pen = float(pen.get("boundary", 0.0))

            row = [
                idx,
                stage,
                op,
                json.dumps({"op": op, **op_args}, ensure_ascii=False, sort_keys=True),
                accepted,
                float(eval_out.get("total_scalar", 0.0)),
                float(eval_out.get("comm_norm", 0.0)),
                float(eval_out.get("therm_norm", 0.0)),
                int(pareto_added),
                dup_pen,
                bnd_pen,
                seed,
                int(rec.get("time_ms", 0)),
                signature,
                float(d_total),
                float(float(eval_out["comm_norm"]) - float(prev_comm)),
                float(float(eval_out["therm_norm"]) - float(prev_therm)),
                int(tabu_hit),
                int(inverse_hit),
                int(cooldown_hit),
            ]
            if len(row) != len(TRACE_FIELDS):
                raise RuntimeError(f"trace row has {len(row)} cols but TRACE_FIELDS has {len(TRACE_FIELDS)}")
            writer.writerow(row)

            # update prev
            if accepted == 1:
                prev_assign = base_state.assign.tolist()
                prev_total = float(eval_out["total_scalar"])
                prev_comm = float(eval_out["comm_norm"])
                prev_therm = float(eval_out["therm_norm"])

            if best_eval is None or float(eval_out["total_scalar"]) < float(best_eval["total_scalar"]):
                best_eval = dict(eval_out)
                best_assign = base_state.assign.tolist()
                best_step = step_id

        if not has_records:
            best_eval = dict(prev_metrics)
            best_assign = list(prev_assign)
            best_step = 0

    assert trace_path.exists(), "trace.csv must exist even if steps=0"

    report = {
        "seed": int(seed),
        "has_records": bool(has_records),
        "accepts": int(accepts),
        "improve_cnt": int(improve_cnt),
        "tabu_hits": int(tabu_hits),
        "inverse_hits": int(inverse_hits),
        "cooldown_hits": int(cooldown_hits),
        "best_step": int(best_step),
        "best_total": float(best_eval["total_scalar"]) if best_eval else float(prev_total),
        "best_comm": float(best_eval["comm_norm"]) if best_eval else float(prev_comm),
        "best_therm": float(best_eval["therm_norm"]) if best_eval else float(prev_therm),
    }

    # persist layout_best.json
    if best_assign is not None:
        (out_dir / "layout_best.json").write_text(
            json.dumps({"assign": best_assign, "eval": best_eval, "seed": seed}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    trace_meta = {
        "seed": int(seed),
        "max_steps": int(max_steps) if max_steps is not None else None,
        "signature_version": "v5.4",
    }
    (out_dir / "trace_meta.json").write_text(json.dumps(trace_meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return report, pareto


def _write_layout_best(
    out_dir: Path,
    layout_input: dict,
    cfg: Any,
    pareto: ParetoSet,
    best_assign: List[int],
) -> Dict[str, Any]:
    base_state, evaluator = _build_evaluator(layout_input, cfg)
    base_state.assign = np.asarray(best_assign, dtype=int)
    eval_out = evaluator.evaluate(base_state)
    sites_xy = np.asarray(layout_input.get("sites", {}).get("sites_xy", []), dtype=float)
    pos_xy = sites_xy[best_assign].tolist() if len(best_assign) and len(sites_xy) else []
    best_comm, best_therm, _ = pareto.knee_point()
    layout_best = {
        "best": {
            "assign": list(best_assign),
            "pos_xy_mm": pos_xy,
            "objectives": {"comm_norm": float(eval_out.get("comm_norm", 0.0)), "therm_norm": float(eval_out.get("therm_norm", 0.0))},
            "raw": {"L_comm": eval_out.get("L_comm", 0.0), "L_therm": eval_out.get("L_therm", 0.0)},
            "penalty": eval_out.get("penalty", {}),
            "meta": {"stage": "heuragenix_best"},
        },
        "pareto_front": [{"comm_norm": p.comm_norm, "therm_norm": p.therm_norm} for p in pareto.points],
        "selection": {"method": "knee_point_v1", "pareto_size": len(pareto.points)},
        "knee_point": {"comm_norm": best_comm, "therm_norm": best_therm},
        "artifacts": {
            "trace_csv": str((out_dir / "trace.csv").absolute()),
            "pareto_csv": str((out_dir / "pareto_points.csv").absolute()),
            "llm_usage_jsonl": str((out_dir / "llm_usage.jsonl").absolute()),
        },
    }
    with (out_dir / "layout_best.json").open("w", encoding="utf-8") as f:
        json.dump(layout_best, f, indent=2)
    return layout_best


def _read_best_assign(best_solution_path: Path, fallback_assign: List[int]) -> List[int]:
    if not best_solution_path.exists():
        return fallback_assign
    payload = json.loads(best_solution_path.read_text(encoding="utf-8"))
    return payload.get("best_assign") or payload.get("best", {}).get("assign") or fallback_assign


def _read_best_solution_meta(best_solution_path: Path) -> Dict[str, Any]:
    if not best_solution_path.exists():
        return {}
    try:
        return json.loads(best_solution_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _summarize_llm_usage(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "total": 0,
            "ok": 0,
            "ok_rate": 0.0,
            "disabled_after_failures": False,
            "fallback_used": False,
            "fallback_count": 0,
            "fallback_last_engine": None,
        }
    total = 0
    ok = 0
    disabled_after_failures = False
    fallback_used = False
    fallback_count = 0
    fallback_last_engine = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                if payload.get("ok") is True:
                    ok += 1
                if payload.get("reason") == "llm_disabled_after_failures":
                    disabled_after_failures = True
                method = payload.get("method")
                engine = payload.get("engine_used") or payload.get("selection")
                if method and engine and engine != method:
                    fallback_used = True
                    fallback_count += 1
                    fallback_last_engine = engine
    ok_rate = ok / max(1, total)
    return {
        "total": total,
        "ok": ok,
        "ok_rate": ok_rate,
        "disabled_after_failures": disabled_after_failures,
        "fallback_used": fallback_used,
        "fallback_count": fallback_count,
        "fallback_last_engine": fallback_last_engine,
    }


def _append_llm_usage(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _run_heuragenix_inprocess(
    *,
    heuragenix_root: Path,
    project_root: Path,
    problem: str,
    method: str,
    heuristic_dir: str,
    case_name: str,
    case_file: str,
    internal_data_base: Path,
    output_root: Path,
    seed: int,
    iters_sf: float,
    selection_frequency: int,
    num_candidate_heuristics: int,
    rollout_budget: int,
    llm_config: Path | None,
    max_steps: int | None,
    baseline_cfg: Dict[str, Any],
) -> Path:
    for entry in (heuragenix_root, heuragenix_root / "src", project_root):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))

    import importlib

    from src.pipeline.hyper_heuristics.heuristic_only import HeuristicOnlyHyperHeuristic
    from src.pipeline.hyper_heuristics.llm_selection import LLMSelectionHyperHeuristic
    from src.pipeline.hyper_heuristics.random import RandomHyperHeuristic
    from src.pipeline.hyper_heuristics.single import SingleHyperHeuristic
    from src.util.util import load_function

    env_module = importlib.import_module(f"src.problems.{problem}.env")
    Env = getattr(env_module, "Env")

    data_path = (internal_data_base / problem / "test_data" / case_file).resolve()
    env = Env(str(data_path))

    problem_size = int(getattr(env, "problem_size", None) or getattr(env, "S", None) or 1)
    if max_steps is not None:
        env.max_steps = max(0, int(max_steps))
    else:
        env.max_steps = max(1, int(float(iters_sf) * max(1, problem_size)))

    env.llm_timeout_s = int(baseline_cfg.get("llm_timeout_s", 30))
    env.max_llm_failures = int(baseline_cfg.get("max_llm_failures", 2))
    env.fallback_on_llm_failure = str(baseline_cfg.get("fallback_on_llm_failure", "random_hh"))

    heur_dir = heuragenix_root / "src" / "problems" / problem / "heuristics" / heuristic_dir
    if not heur_dir.exists():
        raise FileNotFoundError(f"heuristic_dir not found: {heur_dir.resolve()}")
    heuristic_pool_files = [
        f.stem
        for f in heur_dir.iterdir()
        if f.is_file() and f.suffix == ".py" and not f.name.startswith("_") and f.name != "__init__.py"
    ]

    out_dir = (output_root / problem / case_name / "result" / method).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    usage_path = out_dir / "llm_usage.jsonl"
    usage_path.parent.mkdir(parents=True, exist_ok=True)
    if not usage_path.exists():
        usage_path.write_text("", encoding="utf-8")
    env.reset(output_dir=str(out_dir))

    if method == "heuristic_only":
        runner = HeuristicOnlyHyperHeuristic(
            heuristic_pool=heuristic_pool_files,
            problem=problem,
            heuristic_dir=str(heur_dir),
            iterations_scale_factor=float(iters_sf),
            selection_frequency=int(selection_frequency),
            output_dir=str(out_dir),
            seed=int(seed),
        )
    elif method == "random_hh":
        runner = RandomHyperHeuristic(
            heuristic_pool=heuristic_pool_files,
            problem=problem,
            iterations_scale_factor=float(iters_sf),
            heuristic_dir=str(heur_dir),
            selection_frequency=int(selection_frequency),
            seed=int(seed),
        )
    elif method == "llm_hh":
        runner = LLMSelectionHyperHeuristic(
            heuristic_pool=heuristic_pool_files,
            problem=problem,
            heuristic_dir=str(heur_dir),
            llm_config_file=str(llm_config) if llm_config else None,
            iterations_scale_factor=float(iters_sf),
            selection_frequency=int(selection_frequency),
            num_candidate_heuristics=int(num_candidate_heuristics),
            rollout_budget=int(rollout_budget),
            output_dir=str(out_dir),
            seed=int(seed),
            llm_timeout_s=int(baseline_cfg.get("llm_timeout_s", 30)),
            max_llm_failures=int(baseline_cfg.get("max_llm_failures", 2)),
            fallback_on_llm_failure=str(baseline_cfg.get("fallback_on_llm_failure", "random_hh")),
        )
    else:
        fn = load_function(method, problem=problem)
        runner = SingleHyperHeuristic(
            heuristic=fn,
            iterations_scale_factor=float(iters_sf),
            output_dir=str(out_dir),
            seed=int(seed),
        )

    runner.run(env)
    env.dump_result()
    return out_dir


def _collect_subprocess_outputs(
    out_dir: Path,
    problem: str,
    case_stem: str,
    engine: str,
    result_dir: str = "result",
) -> Path:
    """
    HeurAgenix subprocess writes into:
      <AMLT_OUTPUT_DIR>/output/{problem}/{case_stem}/{result_dir}/{engine}/
    Mirror outputs into:
      <out_dir>/{problem}/{case_stem}/{result_dir}/{engine}/
    """
    out_dir = Path(out_dir)
    internal_out = out_dir / "heuragenix_internal"
    internal_base = internal_out / "output" / problem / case_stem / result_dir / engine
    if not internal_base.exists():
        raise FileNotFoundError(f"[HeurAgenix] internal output not found: {internal_base}")

    mirror = out_dir / problem / case_stem / result_dir / engine
    mirror.mkdir(parents=True, exist_ok=True)

    for name in ["recordings.jsonl", "best_solution.json", "llm_usage.jsonl"]:
        src = internal_base / name
        if src.exists():
            (mirror / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            if name in ["recordings.jsonl", "best_solution.json"]:
                raise FileNotFoundError(f"[HeurAgenix] missing {name}: {src}")
            (mirror / name).write_text("", encoding="utf-8")

    return mirror


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout_input", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.cfg)
    cfg = validate_and_fill_defaults(cfg, mode="layout")
    cfg_stem = Path(args.cfg).stem
    auto_out = Path("outputs/layout_heuragenix") / f"{cfg_stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.out_dir) if args.out_dir else Path(getattr(getattr(cfg, "train", None), "out_dir", "") or auto_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(cfg, "train"):
        cfg.train.out_dir = str(out_dir)

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
    meta = {"argv": sys.argv, "out_dir": str(out_dir), "cfg_path": str(args.cfg), "seed": int(args.seed)}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    layout_input_path = Path(args.layout_input)
    layout_input = _load_layout_input(layout_input_path)

    seed = int(args.seed)
    seed_everything(seed)
    seed_assign = _build_seed_assign(layout_input, seed)
    layout_input.setdefault("seed", {})["assign_seed"] = list(seed_assign)
    layout_input.setdefault("seed", {})["seed_id"] = int(seed)

    project_root = Path(__file__).resolve().parents[1]
    baseline_cfg = cfg.baseline if hasattr(cfg, "baseline") else cfg.get("baseline", {})
    layout_input["require_main_evaluator"] = bool(baseline_cfg.get("require_main_evaluator", True))
    layout_input["allow_fallback_evaluator"] = bool(baseline_cfg.get("allow_fallback_evaluator", False))
    heuragenix_root = _resolve_heuragenix_root(baseline_cfg.get("heuragenix_root"), project_root)
    if not heuragenix_root.exists():
        raise FileNotFoundError(f"HeurAgenix root not found: {heuragenix_root}")
    _ensure_heuragenix_syspath(heuragenix_root)

    run_mode = baseline_cfg.get("run_mode", "inprocess")
    method = str(baseline_cfg.get("method", "llm_hh"))
    baseline_method = str(method)
    fallback_method = str(baseline_cfg.get("fallback_on_llm_failure", "random_hh"))
    start_time = time.time()
    llm_usage_path = out_dir / "llm_usage.jsonl"

    problem = str(baseline_cfg.get("problem", "wafer_layout"))
    internal_out = out_dir / "heuragenix_internal"
    internal_out.mkdir(parents=True, exist_ok=True)
    internal_data_root = internal_out / "data"
    internal_data_root.mkdir(parents=True, exist_ok=True)
    (internal_data_root / problem / "test_data").mkdir(parents=True, exist_ok=True)
    (internal_out / problem).mkdir(parents=True, exist_ok=True)
    internal_data_base = internal_data_root
    work_dir, case_name, case_file = _prepare_work_dir(
        out_dir,
        heuragenix_root,
        layout_input,
        seed,
        internal_data_base,
        seed_assign,
        baseline_cfg,
    )
    heuristic_dir = str(baseline_cfg.get("heuristic_dir", "basic_heuristics"))
    selection_frequency = int(baseline_cfg.get("selection_frequency", 5))
    num_candidate_heuristics = int(baseline_cfg.get("num_candidate_heuristics", 4))
    rollout_budget = int(baseline_cfg.get("rollout_budget", 0))
    iters_sf = float(baseline_cfg.get("iterations_scale_factor", 2.0))
    max_steps = baseline_cfg.get("max_steps", None)
    S = int(infer_problem_size(layout_input))
    if max_steps is not None and int(max_steps) >= 0:
        max_steps = int(max_steps)
        iters_sf = max(1.0, math.ceil(float(max_steps) / max(1, S)))
    else:
        max_steps = None
    effective_max_steps = (
        int(max_steps)
        if max_steps is not None and int(max_steps) >= 0
        else int(max(1, round(float(iters_sf) * max(1, S))))
    )

    layout_hash = None
    try:
        layout_hash = hashlib.sha256(layout_input_path.read_bytes()).hexdigest()
    except Exception:
        layout_hash = None
    resolved_text = (out_dir / "config_resolved.yaml").read_text(encoding="utf-8")
    objective_cfg = _build_objective_cfg(cfg)
    baseline_payload = dict(baseline_cfg) if isinstance(baseline_cfg, dict) else baseline_cfg
    extra = {
        "repo_root": str(_PROJECT_ROOT),
        "problem_name": str(baseline_cfg.get("problem", "wafer_layout")),
        "layout_input_hash": layout_hash,
        "steps": int(effective_max_steps),
        "budget": int(rollout_budget),
        "case_stem": str(case_name),
        "seed": int(seed),
        "seed_assign": [int(x) for x in seed_assign],
        "objective_cfg": objective_cfg,
        "baseline": baseline_payload,
    }
    from utils.stable_hash import stable_hash

    cfg_hash = stable_hash({"cfg": resolved_text})
    if hasattr(cfg, "train"):
        cfg.train.cfg_hash = cfg_hash
        cfg.train.cfg_path = str(args.cfg)
    try:
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
            extra=extra,
        )
    except Exception:
        pass

    output_root = internal_out / "output"
    output_root.mkdir(parents=True, exist_ok=True)

    fallback_used = False
    log_text = ""
    llm_config = None
    if method == "llm_hh":
        llm_config = _resolve_llm_config_path(baseline_cfg.get("llm_config_file"), heuragenix_root, project_root)
        if llm_config is None and (not bool(baseline_cfg.get("allow_llm_missing", False))):
            raise RuntimeError("method=llm_hh requires baseline.llm_config_file, but it is missing.")
        if llm_config is not None:
            with open(llm_config, "r", encoding="utf-8") as f:
                js = json.load(f)
            if "top-p" in js and "top_p" not in js:
                js["top_p"] = js.pop("top-p")
            adapted = internal_out / "llm_config_adapted.json"
            with open(adapted, "w", encoding="utf-8") as f:
                json.dump(js, f, indent=2, ensure_ascii=False)
            llm_config = adapted
    output_dir = output_root / problem / case_name / "result" / method
    case_stem = case_name

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(project_root), str(heuragenix_root), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    env["AMLT_OUTPUT_DIR"] = str(internal_out)
    env["AMLT_DATA_DIR"] = str(internal_data_root)
    timeout_s = int(baseline_cfg.get("subprocess_timeout_s", 1800))
    result = None
    if run_mode == "inprocess":
        try:
            output_dir = _run_heuragenix_inprocess(
                heuragenix_root=heuragenix_root,
                project_root=project_root,
                problem=problem,
                method=method,
                heuristic_dir=heuristic_dir,
                case_name=case_name,
                case_file=case_file,
                internal_data_base=internal_data_base,
                output_root=output_root,
                seed=seed,
                iters_sf=iters_sf,
                selection_frequency=selection_frequency,
                num_candidate_heuristics=num_candidate_heuristics,
                rollout_budget=rollout_budget,
                llm_config=llm_config,
                max_steps=max_steps,
                baseline_cfg=baseline_cfg,
            )
        except Exception as exc:
            log_text = f"inprocess_failed: {exc!r}"
            _append_llm_usage(
                llm_usage_path,
                {"ok": False, "reason": "inprocess_failed", "engine": method, "error": log_text},
            )
            run_mode = "subprocess"

    if run_mode != "inprocess":
        # llm_config has already been resolved by _resolve_llm_config_path(...)
        if method == "llm_hh" and (llm_config is None or (not Path(llm_config).exists())):
            raise FileNotFoundError(
                str(llm_config) if llm_config is not None else "llm_config is None"
            )
        launch_cmd = [
            sys.executable,
            str(heuragenix_root / "launch_hyper_heuristic.py"),
            "-p",
            problem,
            "-e",
            method,
            "-d",
            heuristic_dir,
            "-t",
            case_file,
            "-n",
            f"{iters_sf}",
            "-m",
            str(selection_frequency),
            "-c",
            str(num_candidate_heuristics),
            "-b",
            str(rollout_budget),
            "-r",
            "result",
            "--seed",
            str(seed),
        ]
        if max_steps is not None:
            launch_cmd.extend(["--max_steps", str(max_steps)])
        if method == "llm_hh" and llm_config is not None:
            launch_cmd.extend(["-l", str(llm_config)])
        # keep subprocess behavior consistent with inprocess
        llm_timeout_s = int(baseline_cfg.get("llm_timeout_s", 30))
        max_llm_failures = int(baseline_cfg.get("max_llm_failures", 2))
        fallback_on_llm_failure = str(baseline_cfg.get("fallback_on_llm_failure", "random_hh"))
        launch_cmd.extend(["--llm_timeout_s", str(llm_timeout_s)])
        launch_cmd.extend(["--max_llm_failures", str(max_llm_failures)])
        launch_cmd.extend(["--fallback_on_llm_failure", fallback_on_llm_failure])
        try:
            result = subprocess.run(
                launch_cmd,
                cwd=str(heuragenix_root),
                capture_output=True,
                text=True,
                env=env,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            log_text = f"timeout after {timeout_s}s"
            _append_llm_usage(
                llm_usage_path,
                {
                    "ok": False,
                    "reason": "subprocess_timeout",
                    "engine": method,
                    "error": log_text,
                    "stdout": (exc.stdout or "").strip(),
                    "stderr": (exc.stderr or "").strip(),
                },
            )
            result = None
        if result is None or result.returncode != 0:
            if result is not None:
                log_text = result.stderr.strip() or result.stdout.strip() or f"returncode={result.returncode}"
            _append_llm_usage(
                llm_usage_path,
                {"ok": False, "reason": "launch_failed", "engine": method, "error": log_text},
            )
            if method != fallback_method:
                fallback_used = True
                method = fallback_method
                output_dir = output_root / problem / case_name / "result" / method
                launch_cmd = [
                    sys.executable,
                    str(heuragenix_root / "launch_hyper_heuristic.py"),
                    "-p",
                    problem,
                    "-e",
                    method,
                    "-d",
                    heuristic_dir,
                    "-t",
                    case_file,
                    "-n",
                    f"{iters_sf}",
                    "-m",
                    str(selection_frequency),
                    "-c",
                    str(num_candidate_heuristics),
                    "-b",
                    str(rollout_budget),
                    "-r",
                    "result",
                    "--seed",
                    str(seed),
                ]
                if max_steps is not None:
                    launch_cmd.extend(["--max_steps", str(max_steps)])
                try:
                    result = subprocess.run(
                        launch_cmd,
                        cwd=str(heuragenix_root),
                        capture_output=True,
                        text=True,
                        env=env,
                        timeout=timeout_s,
                    )
                except subprocess.TimeoutExpired as exc:
                    log_text = f"timeout after {timeout_s}s"
                    _append_llm_usage(
                        llm_usage_path,
                        {
                            "ok": False,
                            "reason": "subprocess_timeout",
                            "engine": method,
                            "error": log_text,
                            "stdout": (exc.stdout or "").strip(),
                            "stderr": (exc.stderr or "").strip(),
                        },
                    )
                    result = None
                if result is None or result.returncode != 0:
                    if result is not None:
                        log_text = result.stderr.strip() or result.stdout.strip() or f"returncode={result.returncode}"
                    _append_llm_usage(
                        llm_usage_path,
                        {"ok": False, "reason": "fallback_launch_failed", "engine": method, "error": log_text},
                    )

    if run_mode == "inprocess":
        cands = list((output_root / problem / case_name / "result" / method).glob("recordings.jsonl"))
        if not cands:
            cands = list((output_root / problem / case_name / "result").glob("*/recordings.jsonl"))
        if cands:
            recordings_path = cands[0]
            output_dir = recordings_path.parent
            method = output_dir.name
            fallback_used = method != baseline_method
        else:
            output_dir = out_dir
            recordings_path = out_dir / "recordings.jsonl"
            _append_llm_usage(
                llm_usage_path,
                {"ok": False, "reason": "missing_recordings", "engine": method},
            )
            recordings_path.parent.mkdir(parents=True, exist_ok=True)
            recordings_path.write_text("", encoding="utf-8")
    else:
        output_dir = _collect_subprocess_outputs(
            out_dir=out_dir,
            problem=problem,
            case_stem=case_stem,
            engine=method,
            result_dir="result",
        )
        recordings_path = output_dir / "recordings.jsonl"
        if not recordings_path.exists():
            _append_llm_usage(
                llm_usage_path,
                {"ok": False, "reason": "missing_recordings", "engine": method},
            )
            recordings_path.write_text("", encoding="utf-8")
    # merge heuragenix internal llm_usage into wrapper llm_usage.jsonl
    heuragenix_usage_path = output_dir / "llm_usage.jsonl"
    if heuragenix_usage_path.exists() and heuragenix_usage_path.stat().st_size > 0:
        with heuragenix_usage_path.open("r", encoding="utf-8") as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    obj = {"raw": line}
                if isinstance(obj, dict):
                    obj.setdefault("source", "heuragenix")
                    obj.setdefault("wrapper_case_name", case_name)
                    obj.setdefault("wrapper_seed_id", int(seed))
                    obj.setdefault("wrapper_method", method)
                    obj.setdefault("wrapper_run_mode", run_mode)
                    merged = obj
                else:
                    merged = {
                        "source": "heuragenix",
                        "wrapper_case_name": case_name,
                        "wrapper_seed_id": int(seed),
                        "wrapper_method": method,
                        "wrapper_run_mode": run_mode,
                        "obj": obj,
                    }
                _append_llm_usage(llm_usage_path, merged)
    else:
        _append_llm_usage(
            llm_usage_path,
            {"ok": False, "reason": "missing_llm_usage", "engine": method},
        )
    if not recordings_path.exists():
        llm_usage_path.write_text(
            json.dumps({"event": "recordings_missing", "path": str(recordings_path)}) + "\n",
            encoding="utf-8",
        )
    trace_info, pareto = _write_trace_and_pareto(
        out_dir,
        seed,
        recordings_path,
        layout_input,
        cfg,
        effective_max_steps,
    )
    write_pareto_points_csv(pareto, out_dir / "pareto_points.csv")

    best_solution_path = output_dir / "best_solution.json"
    best_assign = _read_best_assign(best_solution_path, trace_info.get("best_assign") or _derive_initial_assign(layout_input).tolist())
    best_solution_payload = _read_best_solution_meta(best_solution_path)
    _write_layout_best(out_dir, layout_input, cfg, pareto, best_assign)

    detailed_cfg = cfg.get("detailed_place", {}) if isinstance(cfg, dict) else cfg.detailed_place
    metrics_window = int(detailed_cfg.get("metrics_window_lastN", 200))
    eps_flat = float(detailed_cfg.get("eps_flat", 1e-4))
    trace_metrics = compute_trace_metrics_from_csv(out_dir / "trace.csv", metrics_window, eps_flat)

    best_eval = trace_info.get("best_eval") or {}
    best_meta = best_solution_payload.get("meta", {}) if isinstance(best_solution_payload, dict) else {}
    evaluator_source = str(best_meta.get("evaluator_source", ""))
    evaluator_import_error = str(best_meta.get("evaluator_import_error", ""))
    base_eval = trace_info.get("base_eval", {}) or {}
    knee_comm, knee_therm, _ = pareto.knee_point()
    accept_rate = float(trace_info.get("accepts", 0)) / max(1, int(trace_info.get("num_steps", 0)))
    llm_summary = _summarize_llm_usage(llm_usage_path)
    if llm_summary.get("fallback_used"):
        fallback_used = True
        if method == baseline_method:
            fallback_method = "random_hh"
    problem_size_eff = int(infer_problem_size(layout_input))
    effective_max_steps = (
        int(max_steps)
        if (max_steps is not None and int(max_steps) >= 0)
        else int(max(1, round(float(iters_sf) * max(1, problem_size_eff))))
    )
    report = {
        "success": True,
        "error": "",
        "seed_id": int(seed),
        "method": str(method),
        "run_mode": str(run_mode),
        "cfg_path": str(args.cfg),
        "budget": {
            "problem_size": problem_size_eff,
            "iterations_scale_factor": float(iters_sf),
            # ★按“实际预算”写入，而不是 None->0
            "max_steps": effective_max_steps,
            # ★可选：把用户是否显式指定也写清楚，方便复盘
            "max_steps_requested": int(max_steps) if (max_steps is not None and int(max_steps) >= 0) else None,
            # v5.4 fairness: count actual evaluator calls
            "eval_calls": int(getattr(evaluator, "evaluate_calls", 0)),
        },
        "evaluator": {
            "require_main_evaluator": bool(layout_input.get("require_main_evaluator", True)),
            "allow_fallback_evaluator": bool(layout_input.get("allow_fallback_evaluator", False)),
            "evaluator_source": str(evaluator_source),
            "evaluator_import_error": str(evaluator_import_error),
        },
        "baseline": {
            "total_scalar": float(base_eval.get("total_scalar", 0.0)),
            "comm_norm": float(base_eval.get("comm_norm", 0.0)),
            "therm_norm": float(base_eval.get("therm_norm", 0.0)),
        },
        "best_objective": {
            "total_scalar": float(best_eval.get("total_scalar", 0.0)),
            "comm_norm": float(best_eval.get("comm_norm", 0.0)),
            "therm_norm": float(best_eval.get("therm_norm", 0.0)),
        },
        "knee_point": {"comm_norm": float(knee_comm), "therm_norm": float(knee_therm)},
        "pareto_size": int(len(pareto.points)),
        "n_steps": int(trace_info.get("num_steps", 0)),
        "accept_rate": accept_rate,
        "tabu_hits": int(trace_info.get("tabu_hits", 0)),
        "inverse_hits": int(trace_info.get("inverse_hits", 0)),
        "cooldown_hits": int(trace_info.get("cooldown_hits", 0)),
        "oscillation_rate": float(trace_metrics.get("oscillation_rate", 0.0)),
        "repeat_signature_rate": float(trace_metrics.get("repeat_signature_rate", 0.0)),
        "objective_variance": float(trace_metrics.get("objective_variance", 0.0)),
        "trace_metrics": trace_metrics,
        "fallback_used": fallback_used,
        "fallback_method": fallback_method if fallback_used else None,
        "runtime_s": float(time.time() - start_time),
        "metrics_window_lastN": metrics_window,
        "eps_flat": eps_flat,
        "raw_dir": str(output_dir),
        "llm_usage": llm_summary,
        "llm_fallback_used": bool(llm_summary.get("fallback_used", False)),
        "llm_fallback_count": int(llm_summary.get("fallback_count", 0)),
        "llm_fallback_last_engine": llm_summary.get("fallback_last_engine"),
        "evaluator_calls": int(trace_info.get("evaluator_calls", 0)),
        "evaluate_calls": int(trace_info.get("evaluator_calls", 0)),
    }
    require_main = bool(layout_input.get("require_main_evaluator", True))
    if require_main and evaluator_source != "main_project":
        report["success"] = False
        report["error"] = (
            f"HeurAgenix evaluator fallback detected: evaluator_source={evaluator_source}. "
            f"evaluator_import_error={evaluator_import_error}"
        )
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    try:
        from utils.run_manifest import write_run_manifest

        meta = {
            "mapping_signature": best_solution_payload.get("mapping_signature", "")
            if isinstance(best_solution_payload, dict)
            else "",
            "layout_signature": signature_from_assign(best_assign),
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
            extra=extra,
        )
    except Exception:
        pass
    if require_main and evaluator_source != "main_project":
        raise RuntimeError(report["error"])


if __name__ == "__main__":
    main()
