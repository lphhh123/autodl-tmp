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
import shutil
import time
import uuid
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

project_root = Path(__file__).resolve().parents[1]

from layout.candidate_pool import signature_from_assign
from layout.evaluator import LayoutEvaluator, LayoutState
from layout.pareto import ParetoSet
from layout.trace_metrics import compute_trace_metrics_from_csv
from utils.config import load_config
from utils.config_validate import validate_and_fill_defaults
from utils.seed import seed_everything
from utils.stable_hash import stable_hash
from utils.trace_guard import init_trace_dir, append_trace_event_v54, finalize_trace_dir
from utils.trace_signature_v54 import build_signature_v54, REQUIRED_SIGNATURE_FIELDS
from utils.trace_schema import TRACE_FIELDS

# Minimal requirements for portability across HeurAgenix versions / problem envs:
# - iter: step index
# - op: operator name
# We accept op_args OR op_args_json (either may exist)
REQUIRED_RECORDING_KEYS = [
    "iter",
    "op",
]

FAIRNESS_OVERRIDES = {
    "action_families": [],
    "moves_enabled": [],
    "lookahead_k": 0,
    "policy_switch_mode": "none",
    "cache_enabled": False,
}


def _cfg_get(obj, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _build_run_signature(cfg: Any, method_name: str) -> Dict[str, Any]:
    return build_signature_v54(
        cfg,
        method_name=method_name,
        overrides=FAIRNESS_OVERRIDES,
    )

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


def _build_state_and_evaluator(
    layout_input: dict,
    cfg: Any | None = None,
    objective_cfg: dict | None = None,
) -> Tuple[LayoutState, LayoutEvaluator]:
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

    objective_cfg = objective_cfg or _build_objective_cfg(cfg)
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


def _build_evaluator(
    layout_input: dict,
    cfg: Any | None = None,
    objective_cfg: dict | None = None,
) -> LayoutEvaluator:
    _base_state, evaluator = _build_state_and_evaluator(
        layout_input=layout_input,
        cfg=cfg,
        objective_cfg=objective_cfg,
    )
    return evaluator


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
                    f"[recordings schema error] missing minimal keys at {recordings_path}:{ln}: {missing}. "
                    f"Got keys={sorted(list(record.keys()))}"
                )

            # Normalize optional fields (do NOT require them)
            if "stage" not in record:
                record["stage"] = "heuragenix"
            if "accepted" not in record:
                # default accept=1 if env didn't provide it (some envs only log applied ops)
                record["accepted"] = 1
            if "seed_id" not in record:
                record["seed_id"] = 0
            if "time_ms" not in record:
                record["time_ms"] = 0.0

            # Normalize op args: accept op_args OR op_args_json OR op_args_str
            if "op_args" not in record:
                if "op_args_json" in record:
                    record["op_args"] = record["op_args_json"]
                elif "op_args_str" in record:
                    record["op_args"] = record["op_args_str"]
                else:
                    record["op_args"] = {}

            # signature is strongly preferred; but do not crash if missing
            if "signature" not in record:
                record["signature"] = ""

            yield record


def _count_jsonl_lines(p: Path) -> int:
    if p is None or (not p.exists()):
        return 0
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _summarize_trace_hits(trace_path: Path) -> Dict[str, int]:
    if not trace_path.exists():
        return {"n_rows": 0, "tabu_hits": 0, "inverse_hits": 0, "cooldown_hits": 0}
    n_rows = 0
    tabu_hits = 0
    inverse_hits = 0
    cooldown_hits = 0
    with trace_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_rows += 1
            try:
                tabu_hits += int(float(row.get("tabu_hit", 0) or 0))
                inverse_hits += int(float(row.get("inverse_hit", 0) or 0))
                cooldown_hits += int(float(row.get("cooldown_hit", 0) or 0))
            except Exception:
                continue
    return {
        "n_rows": n_rows,
        "tabu_hits": tabu_hits,
        "inverse_hits": inverse_hits,
        "cooldown_hits": cooldown_hits,
    }


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


def signature_from_action(op: str, op_args: dict | None) -> str:
    if not isinstance(op_args, dict):
        op_args = {"_raw": str(op_args)}
    try:
        payload = json.dumps(op_args, ensure_ascii=False, sort_keys=True)
    except Exception:
        payload = str(op_args)
    return f"{op}:{payload}"


def _write_trace_and_pareto(
    layout_input: dict,
    objective_cfg: dict,
    recordings_path: Path,
    trace_csv_path: Path,
    pareto_csv_path: Path,
) -> dict:
    """
    v5.4: do NOT replay-evaluate. Use metrics already recorded by HeurAgenix env.
    signature column uses op_signature; cache_key uses assignment signature (with objective_hash).
    """
    evaluator = _build_evaluator(layout_input=layout_input, objective_cfg=objective_cfg)
    objective_hash = evaluator.objective_hash()

    pareto = ParetoSet(
        eps_comm=float(objective_cfg.get("eps_comm", 0.0)),
        eps_therm=float(objective_cfg.get("eps_therm", 0.0)),
    )

    # streaming write
    trace_csv_path.parent.mkdir(parents=True, exist_ok=True)
    pareto_csv_path.parent.mkdir(parents=True, exist_ok=True)

    best_total = None
    best_comm = None
    best_therm = None
    best_step = None
    prev_total = None
    prev_comm = None
    prev_therm = None

    with trace_csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(TRACE_FIELDS) + "\n")

        have_any = False
        for rec in _iter_recordings(recordings_path):
            have_any = True
            step = int(rec.get("iter", 0))
            stage = str(rec.get("stage", "heuragenix"))
            op = str(rec.get("op", "unknown"))
            accepted = int(rec.get("accepted", 1))

            total_scalar = float(rec.get("total_scalar", 0.0))
            comm_norm = float(rec.get("comm_norm", 0.0))
            therm_norm = float(rec.get("therm_norm", 0.0))

            op_args = rec.get("op_args", {}) or {}
            if not isinstance(op_args, dict):
                op_args = {"_raw": str(op_args)}

            # HeurAgenix env provides:
            #   rec["signature"] = assignment signature
            #   rec["op_signature"] = op signature
            assign_sig = str(rec.get("signature") or "")
            op_sig = str(rec.get("op_signature") or signature_from_action(op, op_args))
            cache_key = f"obj:{objective_hash}|{assign_sig}" if assign_sig else f"obj:{objective_hash}|"
            cache_hit = int(rec.get("cache_hit", 0))

            meta = rec.get("meta", {}) or {}
            tabu_hit = int(meta.get("tabu_hit", 0))
            inverse_hit = int(meta.get("inverse_hit", 0))
            cooldown_hit = int(meta.get("cooldown_hit", 0))

            duplicate_penalty = float(rec.get("duplicate_penalty", 0.0))
            boundary_penalty = float(rec.get("boundary_penalty", 0.0))
            seed_id = int(rec.get("seed_id", 0))
            time_ms = float(rec.get("time_ms", 0.0))

            pareto_added = 0
            if accepted == 1:
                added = pareto.add(comm_norm, therm_norm, {})
                pareto_added = 1 if added else 0

            if prev_total is None:
                delta_total = 0.0
                delta_comm = 0.0
                delta_therm = 0.0
            else:
                delta_total = float(total_scalar - prev_total)
                delta_comm = float(comm_norm - prev_comm)
                delta_therm = float(therm_norm - prev_therm)

            if accepted == 1:
                prev_total = total_scalar
                prev_comm = comm_norm
                prev_therm = therm_norm

            row = {
                "iter": step,
                "stage": stage,
                "op": op,
                "op_args_json": json.dumps(op_args, ensure_ascii=False),
                "accepted": accepted,
                "total_scalar": total_scalar,
                "comm_norm": comm_norm,
                "therm_norm": therm_norm,
                "pareto_added": pareto_added,
                "duplicate_penalty": duplicate_penalty,
                "boundary_penalty": boundary_penalty,
                "seed_id": seed_id,
                "time_ms": time_ms,
                "signature": op_sig,
                "delta_total": delta_total,
                "delta_comm": delta_comm,
                "delta_therm": delta_therm,
                "tabu_hit": tabu_hit,
                "inverse_hit": inverse_hit,
                "cooldown_hit": cooldown_hit,
                "policy": stage,
                "move": op,
                "lookahead_k": int(meta.get("lookahead_k", 0)),
                "cache_hit": cache_hit,
                "cache_key": cache_key,
            }
            f.write(",".join(str(row[k]) for k in TRACE_FIELDS) + "\n")

            if best_total is None or total_scalar > best_total:
                best_total = total_scalar
                best_comm = comm_norm
                best_therm = therm_norm
                best_step = step

        if not have_any:
            raise RuntimeError(f"empty recordings: {recordings_path}")

    # write pareto points
    with pareto_csv_path.open("w", encoding="utf-8") as fp:
        fp.write("comm_norm,therm_norm\n")
        for p in pareto.points:
            fp.write(f"{p.comm_norm},{p.therm_norm}\n")

    return {
        "objective_hash": objective_hash,
        "best_total_scalar": float(best_total if best_total is not None else 0.0),
        "best_comm_norm": float(best_comm if best_comm is not None else 0.0),
        "best_therm_norm": float(best_therm if best_therm is not None else 0.0),
        "best_step": int(best_step if best_step is not None else 0),
        "n_pareto": int(len(pareto.points)),
    }


def _load_pareto_from_csv(pareto_csv_path: Path, pareto_cfg: dict) -> ParetoSet:
    pareto = ParetoSet(
        eps_comm=float(pareto_cfg.get("eps_comm", 0.0)),
        eps_therm=float(pareto_cfg.get("eps_therm", 0.0)),
        max_points=int(pareto_cfg.get("max_points", 2000)),
    )
    if not pareto_csv_path.exists():
        return pareto
    with pareto_csv_path.open("r", encoding="utf-8") as fp:
        header = fp.readline()
        for line in fp:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 2:
                continue
            try:
                comm = float(parts[0])
                therm = float(parts[1])
            except Exception:
                continue
            pareto.add(comm, therm, {})
    return pareto


def _write_layout_best(
    out_dir: Path,
    layout_input: dict,
    cfg: Any,
    pareto: ParetoSet,
    best_assign: List[int],
    run_id: str,
) -> Dict[str, Any]:
    base_state, evaluator = _build_state_and_evaluator(layout_input, cfg)
    base_state.assign = np.asarray(best_assign, dtype=int)
    eval_out = evaluator.evaluate(base_state)
    sites_xy = np.asarray(layout_input.get("sites", {}).get("sites_xy", []), dtype=float)
    pos_xy = sites_xy[best_assign].tolist() if len(best_assign) and len(sites_xy) else []
    best_comm, best_therm, _ = pareto.knee_point()
    layout_best = {
        "run_id": run_id,
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

    eval_counter_src = internal_base / "eval_counter.json"
    if eval_counter_src.exists():
        shutil.copy2(eval_counter_src, mirror / "eval_counter.json")

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
    run_id = uuid.uuid4().hex
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
        raise FileNotFoundError(
            "Cannot find HeurAgenix directory.\n"
            f"cwd={os.getcwd()}\n"
            f"heuragenix_dir={heuragenix_root}\n"
            "Tip: run with --heuragenix_dir /ABS/PATH/TO/HeurAgenix"
        )
    _ensure_heuragenix_syspath(heuragenix_root)

    run_mode = baseline_cfg.get("run_mode", "inprocess")
    method = str(baseline_cfg.get("method", "llm_hh"))
    baseline_method = str(method)
    method = baseline_method
    method_label = f"heuragenix:{method}"
    baseline_name = str(baseline_cfg.get("name", baseline_method))
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
        "trace_events": str(out_dir / "trace" / "trace_events.jsonl"),
    }
    cfg_hash = stable_hash({"cfg": resolved_text})
    # ---- v5.4: canonical trace events (JSONL) ----
    trace_dir = out_dir / "trace"
    signature = _build_run_signature(cfg, method_name=method_label)
    init_trace_dir(
        trace_dir,
        signature=signature,
        run_meta={"heuristic": method_label, "method": method, "run_id": run_id, "mode": "layout_heuragenix"},
        required_signature_keys=REQUIRED_SIGNATURE_FIELDS,
    )
    trace_events_path = trace_dir / "trace_events.jsonl"
    finalize_state = {
        "finalized": False,
        "reason": "error",
        "steps_done": 0,
        "best_solution_valid": False,
        "best_total": None,
    }

    def _finalize_trace(**overrides):
        if finalize_state["finalized"]:
            return
        finalize_state.update({k: v for k, v in overrides.items() if v is not None})
        finalize_trace_dir(
            trace_dir,
            summary_extra={
                "reason": str(finalize_state.get("reason", "error")),
                "steps_done": int(finalize_state.get("steps_done", 0) or 0),
                "best_solution_valid": bool(finalize_state.get("best_solution_valid", False)),
                "best_total": finalize_state.get("best_total", None),
            },
            run_id=run_id,
            step=int(finalize_state.get("steps_done", 0) or 0),
        )
        finalize_state["finalized"] = True

    def _trace_excepthook(exc_type, exc, tb):
        finalize_state["reason"] = "error"
        _finalize_trace()
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _trace_excepthook
    try:
        append_trace_event_v54(
            trace_events_path,
            "init",
            payload={
                "baseline_signature": signature_from_assign(np.array(_derive_initial_assign(layout_input), dtype=int)),
                "seed_signature": signature_from_assign(np.array(seed_assign, dtype=int)),
                "effective_max_steps": int(effective_max_steps),
                "selection_frequency": int(selection_frequency),
                "num_candidate_heuristics": int(num_candidate_heuristics),
                "rollout_budget": int(rollout_budget),
            },
            run_id=run_id,
            step=0,
        )
    except Exception:
        pass
    if hasattr(cfg, "train"):
        cfg.train.cfg_hash = cfg_hash
        cfg.train.cfg_path = str(args.cfg)
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
        extra={
            "budget_main_axis": "eval_calls",
            "dataset_id": f"wafer_layout:{layout_input.get('meta', {}).get('layout_id', 'unknown')}",
        },
        run_id=run_id,
        spec_version="v5.4",
        command=" ".join(sys.argv),
        code_root=str(_PROJECT_ROOT),
    )

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

    recordings_found = True
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
            recordings_found = False
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
            recordings_found = False
            _append_llm_usage(
                llm_usage_path,
                {"ok": False, "reason": "missing_recordings", "engine": method},
            )
            recordings_path.write_text("", encoding="utf-8")
    eval_counter_src = output_dir / "eval_counter.json"
    if eval_counter_src.exists():
        shutil.copy2(eval_counter_src, out_dir / "eval_counter.json")
    extra["recordings_found"] = bool(recordings_found)
    extra["recordings_path"] = str(recordings_path)
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
    method_label = f"heuragenix:{method}"
    pareto_cfg = cfg.get("pareto", {}) if isinstance(cfg, dict) else cfg.pareto
    objective_cfg = _build_objective_cfg(cfg)
    objective_cfg.update(
        {
            "eps_comm": float(pareto_cfg.get("eps_comm", 0.0)),
            "eps_therm": float(pareto_cfg.get("eps_therm", 0.0)),
        }
    )
    trace_info = _write_trace_and_pareto(
        layout_input=layout_input,
        objective_cfg=objective_cfg,
        recordings_path=recordings_path,
        trace_csv_path=out_dir / "trace.csv",
        pareto_csv_path=out_dir / "pareto_points.csv",
    )
    pareto = _load_pareto_from_csv(out_dir / "pareto_points.csv", pareto_cfg)

    best_solution_path = output_dir / "best_solution.json"
    best_assign = _read_best_assign(best_solution_path, trace_info.get("best_assign") or _derive_initial_assign(layout_input).tolist())
    best_solution_payload = _read_best_solution_meta(best_solution_path)
    _write_layout_best(out_dir, layout_input, cfg, pareto, best_assign, run_id)

    detailed_cfg = cfg.get("detailed_place", {}) if isinstance(cfg, dict) else cfg.detailed_place
    metrics_window = int(detailed_cfg.get("metrics_window_lastN", 200))
    eps_flat = float(detailed_cfg.get("eps_flat", 1e-4))
    trace_metrics = compute_trace_metrics_from_csv(out_dir / "trace.csv", metrics_window, eps_flat)

    best_eval = {
        "total_scalar": float(trace_info.get("best_total_scalar", 0.0)),
        "comm_norm": float(trace_info.get("best_comm_norm", 0.0)),
        "therm_norm": float(trace_info.get("best_therm_norm", 0.0)),
    }
    best_meta = best_solution_payload.get("meta", {}) if isinstance(best_solution_payload, dict) else {}
    evaluator_source = str(best_meta.get("evaluator_source", ""))
    evaluator_import_error = str(best_meta.get("evaluator_import_error", ""))
    base_eval = trace_info.get("base_eval", {}) or {}
    knee_comm, knee_therm, _ = pareto.knee_point()
    trace_summary = _summarize_trace_hits(out_dir / "trace.csv")
    accept_rate = float(trace_metrics.get("accept_rate_overall", 0.0))
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
        "run_id": run_id,
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
            "name": baseline_name,
            "method": baseline_method,
            "heuristic_dir": heuristic_dir,
            "selection_frequency": int(selection_frequency),
            "num_candidate_heuristics": int(num_candidate_heuristics),
            "rollout_budget": int(rollout_budget),
            "iterations_scale_factor": float(iters_sf),
            "metrics": {
                "total_scalar": float(base_eval.get("total_scalar", 0.0)),
                "comm_norm": float(base_eval.get("comm_norm", 0.0)),
                "therm_norm": float(base_eval.get("therm_norm", 0.0)),
            },
        },
        "best_objective": {
            "total_scalar": float(best_eval.get("total_scalar", 0.0)),
            "comm_norm": float(best_eval.get("comm_norm", 0.0)),
            "therm_norm": float(best_eval.get("therm_norm", 0.0)),
        },
        "knee_point": {"comm_norm": float(knee_comm), "therm_norm": float(knee_therm)},
        "pareto_size": int(len(pareto.points)),
        "n_steps": int(trace_summary.get("n_rows", 0)),
        "accept_rate": accept_rate,
        "tabu_hits": int(trace_summary.get("tabu_hits", 0)),
        "inverse_hits": int(trace_summary.get("inverse_hits", 0)),
        "cooldown_hits": int(trace_summary.get("cooldown_hits", 0)),
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
        "evaluator_calls": int(trace_summary.get("n_rows", 0)),
        "evaluate_calls": int(trace_summary.get("n_rows", 0)),
        "fairness_overrides_applied": True,
        "fairness_overrides": FAIRNESS_OVERRIDES,
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
    # ---- v5.4: enforce budget fairness artifact (budget.json) ----
    wall_limit = int(getattr(getattr(cfg, "layout_agent", None), "max_runtime_sec", 0) or 0)
    eval_counter_path = out_dir / "eval_counter.json"
    if not eval_counter_path.exists():
        raise RuntimeError("missing eval_counter.json from HeurAgenix env; budget would be invalid")

    ec = json.loads(eval_counter_path.read_text(encoding="utf-8"))
    actual_eval_calls = int(trace_info.get("evaluator_calls", 0))
    if actual_eval_calls <= 0:
        actual_eval_calls = int(_count_jsonl_lines(recordings_path)) if recordings_path.exists() else 0
    effective_eval_calls_total = int(ec.get("effective_eval_calls_total", actual_eval_calls))
    cache_hits = int(ec.get("cache_hits", 0))

    budget = {
        "budget_main_axis": "eval_calls",
        "primary_limit": {"type": "eval_calls", "value": None},
        "secondary_limit": {"type": "wall_time_s", "value": float(wall_limit) if wall_limit else None},
        "actual_eval_calls": actual_eval_calls,
        "effective_eval_calls_total": effective_eval_calls_total,
        "cache_hits": cache_hits,
        "wall_time_s": float(time.time() - start_time),
    }
    (out_dir / "budget.json").write_text(json.dumps(budget, indent=2) + "\n", encoding="utf-8")
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
        extra={
            "budget_main_axis": "eval_calls",
            "dataset_id": f"wafer_layout:{layout_input.get('meta', {}).get('layout_id', 'unknown')}",
        },
        run_id=run_id,
        spec_version="v5.4",
        command=" ".join(sys.argv),
        code_root=str(_PROJECT_ROOT),
    )
    # ---- v5.4: mandatory finalize event (trace_events.jsonl) ----
    try:
        steps_done = int(report.get("n_steps", 0) or 0)
        reason = (
            "steps0"
            if int(args.iterations_scale_factor) <= 0
            else ("done" if report.get("status") == "ok" else "error")
        )
        _finalize_trace(
            reason=reason,
            steps_done=steps_done,
            best_solution_valid=bool(report.get("best_valid", False)),
            best_total=float(report.get("best_total")) if report.get("best_total") is not None else None,
        )
    except Exception:
        pass
    if require_main and evaluator_source != "main_project":
        raise RuntimeError(report["error"])


if __name__ == "__main__":
    main()
