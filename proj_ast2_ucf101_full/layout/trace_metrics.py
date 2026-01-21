from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from utils.trace_schema import TRACE_FIELDS


def _get_float(row: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return default


def _is_undo_action(prev_action: dict, curr_action: dict, prev_sig: str, curr_sig: str) -> bool:
    if not prev_action or not curr_action:
        return False
    # v5.4：trace.signature 是 assign signature，不应依赖 "swap:" 前缀
    # swap 的“undo”定义：连续两次做同一对 slot 的 swap（顺序无关）即可视为撤销/反复
    if prev_action.get("op") == "swap" and curr_action.get("op") == "swap":
        pi, pj = prev_action.get("i", None), prev_action.get("j", None)
        ci, cj = curr_action.get("i", None), curr_action.get("j", None)
        if None not in (pi, pj, ci, cj):
            return {int(pi), int(pj)} == {int(ci), int(cj)}
    if prev_action.get("op") == "relocate" and curr_action.get("op") == "relocate":
        if int(prev_action.get("i", -1)) != int(curr_action.get("i", -1)):
            return False
        prev_from = prev_action.get("from_site")
        prev_to = prev_action.get("site_id")
        curr_from = curr_action.get("from_site")
        curr_to = curr_action.get("site_id")
        if None in (prev_from, prev_to, curr_from, curr_to):
            return False
        return int(curr_to) == int(prev_from) and int(curr_from) == int(prev_to)
    if prev_action.get("op") == "cluster_move" and curr_action.get("op") == "cluster_move":
        if int(prev_action.get("cluster_id", -1)) != int(curr_action.get("cluster_id", -1)):
            return False
        prev_region = prev_action.get("from_region")
        curr_region = curr_action.get("region_id")
        if prev_region is None or curr_region is None:
            return False
        return int(curr_region) == int(prev_region)
    return False


def compute_trace_metrics_from_csv(trace_path: Path, window: int, eps_flat: float) -> dict:
    if not trace_path.exists():
        return {}
    rows = []
    with trace_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        required = {"accepted", "stage"}
        missing = required - header
        if missing:
            raise ValueError(
                "[trace_metrics] Missing required trace fields "
                f"{sorted(list(missing))} in {trace_path}. "
                f"header={sorted(list(header))} TRACE_FIELDS={TRACE_FIELDS}"
            )
        delta_ok = (
            ("delta_total" in header or "d_total" in header)
            and ("delta_comm" in header or "d_comm" in header)
            and ("delta_therm" in header or "d_therm" in header)
        )
        if not delta_ok:
            raise ValueError(
                "[trace_metrics] Unsupported trace schema: missing delta/d fields in "
                f"{trace_path}. header={sorted(list(header))} TRACE_FIELDS={TRACE_FIELDS}"
            )
        for row in reader:
            rows.append(row)
    if not rows:
        return {}

    totals = [float(r.get("total_scalar", 0.0)) for r in rows]
    comm_vals = [float(r.get("comm_norm", 0.0)) for r in rows]
    therm_vals = [float(r.get("therm_norm", 0.0)) for r in rows]
    signatures = [r.get("signature", "") or "" for r in rows]
    accepted = [int(float(r.get("accepted", 0) or 0)) for r in rows]
    d_total = [_get_float(r, "delta_total", "d_total", default=0.0) for r in rows]
    d_comm = [_get_float(r, "delta_comm", "d_comm", default=0.0) for r in rows]
    d_therm = [_get_float(r, "delta_therm", "d_therm", default=0.0) for r in rows]

    start = max(0, len(rows) - window)
    window_indices = list(range(start, len(rows)))
    window_len = len(window_indices)

    seen = set()
    repeat = 0
    for idx in window_indices:
        sig = signatures[idx]
        if not sig:
            continue
        if sig in seen:
            repeat += 1
        else:
            seen.add(sig)
    repeat_signature_rate = repeat / max(1, window_len)

    seen_all = set()
    repeat_all = 0
    for sig in signatures:
        if not sig:
            continue
        if sig in seen_all:
            repeat_all += 1
        else:
            seen_all.add(sig)
    repeat_signature_rate_overall = repeat_all / max(1, len(rows))

    actions = []
    for row in rows:
        try:
            actions.append(json.loads(row.get("op_args_json", "{}")))
        except json.JSONDecodeError:
            actions.append({})

    undo = 0
    for idx in window_indices[1:]:
        if _is_undo_action(actions[idx - 1], actions[idx], signatures[idx - 1], signatures[idx]):
            undo += 1
    undo_rate = undo / max(1, window_len - 1)

    undo_all = 0
    for idx in range(1, len(rows)):
        if _is_undo_action(actions[idx - 1], actions[idx], signatures[idx - 1], signatures[idx]):
            undo_all += 1
    undo_rate_overall = undo_all / max(1, len(rows) - 1)

    obj_arr = np.array(totals, dtype=np.float64)
    objective_variance = float(np.var(obj_arr)) if obj_arr.size else 0.0
    obj_last = obj_arr[start:] if obj_arr.size else obj_arr
    objective_variance_lastN = float(np.var(obj_last)) if obj_last.size else 0.0
    objective_std_lastN = float(np.std(obj_last)) if obj_last.size else 0.0
    best_total_lastN = float(np.min(obj_last)) if obj_last.size else 0.0
    mean_lastN = float(np.mean(obj_last)) if obj_last.size else 0.0

    v_norm = min(1.0, objective_std_lastN / (abs(best_total_lastN) + 1e-9)) if obj_last.size else 0.0
    oscillation_rate = 0.4 * undo_rate + 0.4 * repeat_signature_rate + 0.2 * v_norm

    accept_rate_overall = sum(accepted) / max(1, len(accepted))
    accept_rate_lastN = sum(accepted[start:]) / max(1, window_len)

    improve_steps = 0
    flat_steps = 0
    for idx in window_indices:
        if accepted[idx] and d_total[idx] < 0:
            improve_steps += 1
        if (
            abs(d_total[idx]) < eps_flat
            and abs(d_comm[idx]) < eps_flat
            and abs(d_therm[idx]) < eps_flat
        ):
            flat_steps += 1

    improve_step_ratio = improve_steps / max(1, window_len)
    flat_step_ratio = flat_steps / max(1, window_len)

    return {
        "accept_rate_overall": accept_rate_overall,
        "accept_rate_lastN": accept_rate_lastN,
        "repeat_signature_rate": repeat_signature_rate,
        "repeat_signature_rate_overall": repeat_signature_rate_overall,
        "undo_rate": undo_rate,
        "undo_rate_overall": undo_rate_overall,
        "objective_variance": objective_variance,
        "objective_variance_lastN": objective_variance_lastN,
        "objective_std_lastN": objective_std_lastN,
        "best_total_lastN": best_total_lastN,
        "mean_lastN": mean_lastN,
        "v_norm": v_norm,
        "oscillation_rate": oscillation_rate,
        "improve_step_ratio": improve_step_ratio,
        "flat_step_ratio": flat_step_ratio,
        "last_total": float(totals[-1]) if totals else 0.0,
        "last_comm": float(comm_vals[-1]) if comm_vals else 0.0,
        "last_therm": float(therm_vals[-1]) if therm_vals else 0.0,
        "best_total": float(np.min(obj_arr)) if obj_arr.size else 0.0,
        "best_comm": float(np.min(comm_vals)) if comm_vals else 0.0,
        "best_therm": float(np.min(therm_vals)) if therm_vals else 0.0,
    }
