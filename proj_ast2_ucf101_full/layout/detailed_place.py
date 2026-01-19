"""Detailed placement with SA and Pareto updates (SPEC v4.3.2 §8.6).

Key guarantees:
- No NameError on mixed/llm path (planner_type/llm_provider/llm_init_error always defined)
- planner.type == mixed actually calls LLM every_n_steps when available
- If LLM init fails (e.g., missing ARK_API_KEY), gracefully fall back to heuristic and
  write a one-line init_failed record to llm_usage.jsonl (if provided).
"""
from __future__ import annotations

import copy
import csv
import json
import math
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from layout.coarsen import Cluster
from layout.candidate_pool import signature_from_assign
from layout.evaluator import LayoutEvaluator, LayoutState
from layout.pareto import ParetoSet
from layout.llm_provider import HeuristicProvider, VolcArkProvider, LLMProvider
from layout.candidate_pool import (
    Candidate,
    build_candidate_pool,
    build_state_summary,
    inverse_signature,
    pick_ids_to_actions_sequential,
    _signature_for_action,
)


@dataclass
class DetailedPlaceResult:
    assign: np.ndarray
    pareto: ParetoSet
    trace_path: Path


def _cfg_get(cfg: Any, key: str, default=None):
    """Support dict / OmegaConf(DictConfig) / simple objects."""
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)  # OmegaConf-like
    except Exception:
        return getattr(cfg, key, default)


def _compute_top_pairs(traffic_sym: np.ndarray, k: int) -> List[Tuple[int, int, float]]:
    pairs: List[Tuple[int, int, float]] = []
    S = int(traffic_sym.shape[0])
    for i in range(S):
        for j in range(i + 1, S):
            pairs.append((i, j, float(traffic_sym[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:k]


def _apply_swap(assign: np.ndarray, i: int, j: int):
    assign[i], assign[j] = assign[j], assign[i]


def _apply_relocate(assign: np.ndarray, i: int, site_id: int):
    assign[i] = site_id


def _apply_cluster_move(assign: np.ndarray, cluster: Cluster, target_sites: Optional[List[int]]):
    if not target_sites or len(target_sites) < len(cluster.slots):
        return
    for slot, site in zip(cluster.slots, target_sites):
        assign[int(slot)] = int(site)


def signature_for_assign(assign: np.ndarray) -> str:
    return signature_from_assign(assign.tolist())


def _select_cluster_target_sites(
    assign: np.ndarray,
    cluster: Cluster,
    region_id: int,
    site_to_region: np.ndarray,
    sites_xy: np.ndarray,
) -> List[int]:
    used_sites = set(int(x) for x in assign.tolist())
    region_sites = [s for s, r in enumerate(site_to_region) if int(r) == int(region_id)]
    empties = [sid for sid in region_sites if sid not in used_sites]
    if len(empties) < len(cluster.slots):
        return []
    cluster_pos = np.array([sites_xy[int(assign[int(slot)])] for slot in cluster.slots], dtype=np.float32)
    centroid = cluster_pos.mean(axis=0) if cluster_pos.size else np.zeros(2, dtype=np.float32)
    empties_sorted = sorted(empties, key=lambda s: float(np.linalg.norm(sites_xy[int(s)] - centroid)))
    return [int(s) for s in empties_sorted[: len(cluster.slots)]]


def _sample_action(
    cfg: Any,
    traffic_sym: np.ndarray,
    site_to_region: np.ndarray,
    regions,
    clusters: List[Cluster],
    assign: np.ndarray,
    sites_xy: np.ndarray,
    chip_tdp: Optional[np.ndarray],
    cluster_to_region: List[int],
    py_rng: random.Random,
) -> Dict:
    probs = _cfg_get(cfg, "action_probs", {}) or {}

    # 1) swap: prioritize hot pairs
    if py_rng.random() < float(probs.get("swap", 0.5)):
        hot_cfg = _cfg_get(cfg, "hot_sampling", {}) or {}
        top_k = int(hot_cfg.get("top_pairs_k", 10))
        top_pairs = _compute_top_pairs(traffic_sym, top_k)
        if top_pairs:
            i, j, _ = py_rng.choice(top_pairs)
            return {"op": "swap", "i": int(i), "j": int(j)}

    # 2) relocate: prefer empty sites within same region
    if py_rng.random() < float(probs.get("relocate", 0.3)):
        reloc_cfg = _cfg_get(cfg, "relocate", {}) or {}
        same_region_prob = float(reloc_cfg.get("same_region_prob", 0.8))
        neighbor_k = int(reloc_cfg.get("neighbor_k", 30))

        slot_scores = traffic_sym.sum(axis=1)
        if chip_tdp is not None and len(chip_tdp) == slot_scores.shape[0]:
            slot_scores = slot_scores + chip_tdp

        slot = int(np.argmax(slot_scores)) if float(slot_scores.sum()) > 0 else py_rng.randrange(traffic_sym.shape[0])

        # NOTE: assign[slot] must be valid to index site_to_region
        cur_site = int(assign[slot])
        if cur_site < 0 or cur_site >= site_to_region.shape[0]:
            return {"op": "none"}

        current_region = int(site_to_region[cur_site])
        used = set(int(x) for x in assign.tolist())
        empty_sites = [s for s in range(site_to_region.shape[0]) if s not in used]
        if not empty_sites:
            return {"op": "none"}

        candidates = [s for s in empty_sites if int(site_to_region[s]) == current_region]
        if (not candidates) or (py_rng.random() > same_region_prob):
            candidates = empty_sites

        # choose nearest candidate sites
        dists = [(sid, float(np.linalg.norm(sites_xy[cur_site] - sites_xy[sid]))) for sid in candidates]
        dists.sort(key=lambda x: x[1])
        chosen = dists[: max(1, min(neighbor_k, len(dists)))]
        site_id = int(py_rng.choice(chosen)[0])
        return {"op": "relocate", "i": int(slot), "site_id": int(site_id)}

    # 3) cluster_move: move a heavy cluster to a different region
    if clusters:
        clusters_sorted = sorted(clusters, key=lambda c: float(getattr(c, "tdp_sum", 0.0)), reverse=True)
        c = clusters_sorted[0]
        target_region = regions[-1] if regions else None
        if target_region is not None and cluster_to_region:
            cur_region = int(cluster_to_region[c.cluster_id]) if c.cluster_id < len(cluster_to_region) else -1
            region_options = [r for r in regions if int(r.region_id) != cur_region]
            if region_options:
                target_region = py_rng.choice(region_options)
        if target_region is not None:
            return {"op": "cluster_move", "cluster_id": int(c.cluster_id), "region_id": int(target_region.region_id)}

    return {"op": "none"}


def run_detailed_place(
    sites_xy: np.ndarray,
    assign_seed: np.ndarray,
    evaluator: LayoutEvaluator,
    layout_state: LayoutState,
    traffic_sym: np.ndarray,
    site_to_region: np.ndarray,
    regions,
    clusters: List[Cluster],
    cluster_to_region: List[int],
    pareto: ParetoSet,
    cfg: Any,
    trace_path: Path,
    seed_id: int,
    chip_tdp: Optional[np.ndarray] = None,
    llm_usage_path: Optional[Path] = None,
) -> DetailedPlaceResult:
    # ---- deterministic seeds ----
    base_seed = int(_cfg_get(cfg, "seed", 0)) + int(seed_id)
    rng = np.random.default_rng(base_seed)
    py_rng = random.Random(base_seed)

    # ---- llm logging safety ----
    raw_text: str = ""
    llm_provider: Optional[LLMProvider] = None
    llm_init_error: Optional[str] = None

    # ---- init assign ----
    assign = np.array(assign_seed, dtype=int).copy()
    layout_state.assign = assign

    # ---- anti-oscillation params ----
    anti_cfg = _cfg_get(cfg, "anti_oscillation", {}) or {}
    tabu_tenure = int(_cfg_get(anti_cfg, "tabu_tenure", 8))
    inverse_tenure = int(_cfg_get(anti_cfg, "inverse_tenure", 6))
    per_slot_cooldown = int(_cfg_get(anti_cfg, "per_slot_cooldown", 6))
    aspiration_delta = float(_cfg_get(anti_cfg, "aspiration_delta", 1e-4))

    # ---- planner config ----
    planner_cfg = _cfg_get(cfg, "planner", {"type": "heuristic"}) or {"type": "heuristic"}
    planner_type = str(_cfg_get(planner_cfg, "type", "heuristic"))
    mixed_cfg = _cfg_get(planner_cfg, "mixed", {}) or {}
    mixed_every = int(_cfg_get(mixed_cfg, "every_n_steps", 200)) if planner_type == "mixed" else 0
    k_actions = int(_cfg_get(mixed_cfg, "k_actions", 4))
    queue_enabled = bool(_cfg_get(planner_cfg, "queue_enabled", True))
    feasibility_check = bool(_cfg_get(planner_cfg, "feasibility_check", True))

    timeout_sec = int(_cfg_get(planner_cfg, "timeout_sec", 90))
    max_retry = int(_cfg_get(planner_cfg, "max_retry", 1))
    stage_label = str(_cfg_get(cfg, "stage_label", f"detailed_{planner_type}"))

    # Providers: always have heuristic; LLM optional
    heuristic_provider: LLMProvider = HeuristicProvider()
    if planner_type in ("llm", "mixed"):
        try:
            llm_provider = VolcArkProvider(timeout_sec=timeout_sec, max_retry=max_retry)
        except Exception as e:
            llm_provider = None
            llm_init_error = repr(e)

    # ---- SA params ----
    steps = int(_cfg_get(cfg, "steps", 0))
    T = float(_cfg_get(cfg, "sa_T0", 1.0))
    alpha = float(_cfg_get(cfg, "sa_alpha", 0.999))

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = trace_path.parent
    usage_fp = llm_usage_path.open("a", encoding="utf-8") if llm_usage_path else None
    start_time = time.time()

    # If LLM is requested but init failed, log once and continue with heuristic
    if usage_fp and planner_type in ("llm", "mixed") and llm_provider is None and llm_init_error:
        json.dump({"event": "llm_init_failed", "planner_type": planner_type, "error": llm_init_error}, usage_fp)
        usage_fp.write("\n")
        usage_fp.flush()

    with trace_path.open("w", encoding="utf-8", newline="") as f_trace:
        writer = csv.writer(f_trace)
        writer.writerow(
            [
                "iter",
                "stage",
                "op",
                "op_args_json",
                "accepted",
                "total_scalar",
                "comm_norm",
                "therm_norm",
                "pareto_added",
                "duplicate_penalty",
                "boundary_penalty",
                "seed_id",
                "time_ms",
                "signature",
                "d_total",
                "d_comm",
                "d_therm",
                "tabu_hit",
                "inverse_hit",
                "cooldown_hit",
            ]
        )

        # initial eval
        eval_out = evaluator.evaluate(layout_state)
        prev_total = float(eval_out.get("total_scalar", 0.0))
        prev_comm = float(eval_out.get("comm_norm", 0.0))
        prev_therm = float(eval_out.get("therm_norm", 0.0))
        prev_assign = assign.copy()

        writer.writerow(
            [
                0,
                "init",
                "init",
                json.dumps({"op": "init"}, ensure_ascii=False),
                1,
                prev_total,
                prev_comm,
                prev_therm,
                0,
                0.0,
                0.0,
                int(seed_id),
                0,
                signature_for_assign(prev_assign.tolist()),
                0,
                0,
                0,
                0,
                0,
                0,
            ]
        )
        pareto.add(
            eval_out["comm_norm"],
            eval_out["therm_norm"],
            {
                "assign": assign.copy(),
                "total_scalar": eval_out["total_scalar"],
                "stage": stage_label,
                "iter": 0,
                "seed": seed_id,
            },
        )

        best_total_seen = float(eval_out.get("total_scalar", 0.0))
        best_assign = assign.copy()
        best_eval = dict(eval_out)
        tabu_signatures: deque[str] = deque(maxlen=tabu_tenure)
        inverse_signatures: deque[str] = deque(maxlen=inverse_tenure)
        last_site_per_slot: Dict[int, int] = {int(i): int(assign[int(i)]) for i in range(len(assign))}
        last_move_step_per_slot: Dict[int, int] = {int(i): -10**6 for i in range(len(assign))}

        def _touched_slots(act: Dict[str, Any]) -> List[int]:
            op_local = act.get("op")
            if op_local == "swap":
                return [int(act.get("i", -1)), int(act.get("j", -1))]
            if op_local == "relocate":
                return [int(act.get("i", -1))]
            if op_local == "cluster_move":
                cid = int(act.get("cluster_id", -1))
                if 0 <= cid < len(clusters):
                    return [int(s) for s in clusters[cid].slots]
            return []

        action_queue: List[Dict] = []
        forbidden_history: List[List[int]] = []
        failed_counts: Dict[int, int] = {}
        recent_failed_ids: set[int] = set()
        consecutive_queue_rejects = 0
        queue_window_steps = int(_cfg_get(planner_cfg, "queue_window_steps", 30))
        refresh_due_to_rejects = False
        progress_every = int(_cfg_get(cfg, "progress_every", 10))
        save_every = int(_cfg_get(cfg, "save_every", 50))
        accepted_steps = 0

        for step in range(steps):
            step_start = time.perf_counter()
            candidate_pool = build_candidate_pool(
                assign,
                eval_out,
                evaluator,
                layout_state,
                traffic_sym,
                sites_xy,
                site_to_region,
                regions,
                clusters,
                cluster_to_region,
                chip_tdp,
                cfg,
                rng,
                debug_out_path=(out_dir / "candidate_pool_debug.json") if step == 0 else None,
            )
            cand_map = {c.id: c for c in candidate_pool}
            candidate_ids = [c.id for c in candidate_pool]
            forbidden_ids = list({pid for recent in forbidden_history[-3:] for pid in recent} | recent_failed_ids)

            use_llm = (planner_type == "llm") or (planner_type == "mixed" and mixed_every > 0 and step % mixed_every == 0)
            need_refresh = use_llm and llm_provider is not None and (not action_queue or refresh_due_to_rejects)
            refresh_due_to_rejects = False

            if need_refresh:
                ss = build_state_summary(
                    step,
                    T,
                    eval_out,
                    traffic_sym,
                    assign,
                    site_to_region,
                    chip_tdp,
                    clusters,
                    regions,
                    candidate_pool,
                    candidate_ids,
                    forbidden_ids,
                    k_actions,
                )
                raw_text = ""
                pick_ids: List[int] = []
                pick_types_count: Dict[str, int] = {}
                best_d_total = None
                try:
                    pick_ids = llm_provider.propose_pick(ss, k_actions) or []
                    for pid in pick_ids:
                        cand = cand_map.get(pid)
                        if cand:
                            pick_types_count[cand.type] = pick_types_count.get(cand.type, 0) + 1
                            if best_d_total is None or cand.est.get("d_total", 0) < best_d_total:
                                best_d_total = cand.est.get("d_total", 0)
                except Exception as e:
                    if usage_fp:
                        json.dump(
                            {
                                "event": "llm_step_failed",
                                "step": int(step),
                                "error": repr(e),
                                "raw_preview": raw_text[:200],
                            },
                            usage_fp,
                        )
                        usage_fp.write("\\n")
                        usage_fp.flush()
                actions: List[Dict[str, Any]] = []
                if pick_ids:
                    forbidden_history.append(pick_ids)
                    if feasibility_check:
                        actions = pick_ids_to_actions_sequential(
                            assign, pick_ids, cand_map, clusters, site_to_region, max_k=k_actions
                        )
                    else:
                        for pid in pick_ids[:k_actions]:
                            cand = cand_map.get(pid)
                            if cand is None:
                                continue
                            action_copy = copy.deepcopy(cand.action)
                            action_copy.setdefault("signature", cand.signature)
                            actions.append(
                                {
                                    **action_copy,
                                    "candidate_id": pid,
                                    "type": cand.type,
                                    "signature": action_copy.get("signature", cand.signature),
                                }
                            )
                if usage_fp and hasattr(llm_provider, "last_usage"):
                    usage = dict(getattr(llm_provider, "last_usage") or {})
                    raw_text = str(usage.get("raw_preview", ""))
                    usage.setdefault("ok", bool(pick_ids))
                    usage.setdefault("n_pick", len(pick_ids))
                    usage["pick_ids"] = pick_ids
                    usage["picked_types"] = pick_types_count
                    usage["best_d_total"] = best_d_total
                    usage["n_queue_push"] = len(actions)
                    usage["step"] = int(step)
                    json.dump(usage, usage_fp)
                    usage_fp.write("\\n")
                    usage_fp.flush()
                    expire_step = step + queue_window_steps
                    if queue_enabled:
                        for act in actions:
                            action_copy = copy.deepcopy(act)
                            if "signature" not in action_copy:
                                action_copy["signature"] = _signature_for_action(action_copy, assign)
                            action_queue.append({"action": action_copy, "expire": expire_step})
                    elif actions:
                        action_queue = [
                            {
                                "action": copy.deepcopy(actions[0]),
                                "expire": step,
                            }
                        ]

            action_queue = [a for a in action_queue if a.get("expire", step) >= step]
            fallback_candidates = [c for c in candidate_pool if c.id not in forbidden_ids]
            fallback_idx = 0

            action = {"op": "none"}
            tabu_hit = inverse_hit = cooldown_hit = 0
            est_total_new = None
            op_signature = "none"
            inverse_sig = "none"
            max_attempts = max(5, len(action_queue) + len(fallback_candidates) + 3)

            for _ in range(max_attempts):
                action_payload = None
                if action_queue:
                    action_payload = action_queue.pop(0).get("action")
                elif fallback_idx < len(fallback_candidates):
                    fb = fallback_candidates[fallback_idx]
                    fallback_idx += 1
                    action_payload = {**copy.deepcopy(fb.action), "candidate_id": fb.id, "type": fb.type, "signature": fb.signature}
                else:
                    action_payload = _sample_action(
                        cfg,
                        traffic_sym,
                        site_to_region,
                        regions,
                        clusters,
                        assign,
                        sites_xy,
                        chip_tdp,
                        cluster_to_region,
                        py_rng,
                    )

                action = copy.deepcopy(action_payload) if action_payload else {"op": "none"}
                op = str(action.get("op", "none"))
                if op == "relocate":
                    action["from_site"] = int(assign[int(action.get("i", -1))])
                if op == "cluster_move":
                    cid = int(action.get("cluster_id", -1))
                    if 0 <= cid < len(clusters) and clusters[cid].slots:
                        slot_id = int(clusters[cid].slots[0])
                        if 0 <= slot_id < len(assign):
                            action["from_region"] = int(site_to_region[int(assign[slot_id])])
                        if "target_sites" not in action:
                            action["target_sites"] = _select_cluster_target_sites(
                                assign, clusters[cid], int(action.get("region_id", -1)), site_to_region, sites_xy
                            )
                if "signature" not in action:
                    action["signature"] = _signature_for_action(action, assign)
                op_signature = str(action.get("signature", "none"))
                inverse_sig = inverse_signature(action, assign)
                est_total_new = None
                cand_ref = None
                cid = action.get("candidate_id")
                cid_int = int(cid) if cid is not None else None
                if cid_int is not None and cid_int in cand_map:
                    cand_ref = cand_map[int(cid_int)]
                    est_total_new = cand_ref.est.get("total_new")
                    action.setdefault("type", cand_ref.type)
                    action.setdefault("signature", cand_ref.signature)

                aspiration = est_total_new is not None and est_total_new < best_total_seen - aspiration_delta
                tabu_hit = 1 if op_signature in tabu_signatures else 0
                inverse_hit = 1 if op_signature in inverse_signatures else 0
                cooldown_hit = 0
                for slot in _touched_slots(action):
                    if step - last_move_step_per_slot.get(int(slot), -10**6) < per_slot_cooldown:
                        cooldown_hit = 1
                        break

                if aspiration or not (tabu_hit or inverse_hit or cooldown_hit):
                    break
            op = str(action.get("op", "none"))

            new_assign = assign.copy()
            if op == "swap":
                _apply_swap(new_assign, int(action.get("i", 0)), int(action.get("j", 0)))
            elif op == "relocate":
                _apply_relocate(new_assign, int(action.get("i", 0)), int(action.get("site_id", 0)))
            elif op == "cluster_move":
                cid = int(action.get("cluster_id", 0))
                rid = int(action.get("region_id", 0))
                if 0 <= cid < len(clusters):
                    cluster = clusters[cid]
                    target_sites = action.get("target_sites")
                    if not target_sites:
                        target_sites = _select_cluster_target_sites(assign, cluster, rid, site_to_region, sites_xy)
                        action["target_sites"] = target_sites
                    _apply_cluster_move(new_assign, cluster, target_sites)

            layout_state.assign = new_assign
            eval_new = evaluator.evaluate(layout_state)
            delta = float(eval_new["total_scalar"] - eval_out["total_scalar"])
            delta_comm = float(eval_new["comm_norm"] - eval_out["comm_norm"])
            delta_therm = float(eval_new["therm_norm"] - eval_out["therm_norm"])
            accept = (delta < 0) or (math.exp(-delta / max(T, 1e-6)) > float(rng.random()))

            if accept:
                assign = new_assign
                eval_out = eval_new
                layout_state.assign = assign
                accepted_steps += 1
                added = pareto.add(
                    eval_out["comm_norm"],
                    eval_out["therm_norm"],
                    {
                        "assign": assign.copy(),
                        "total_scalar": eval_out["total_scalar"],
                        "stage": stage_label,
                        "iter": step + 1,
                        "seed": seed_id,
                    },
                )
                tabu_signatures.append(op_signature)
                inverse_signatures.append(inverse_sig)
                for slot in _touched_slots(action):
                    last_move_step_per_slot[int(slot)] = step
                    if 0 <= int(slot) < len(assign):
                        last_site_per_slot[int(slot)] = int(assign[int(slot)])
                if float(eval_out.get("total_scalar", best_total_seen)) < best_total_seen:
                    best_total_seen = float(eval_out.get("total_scalar", best_total_seen))
                    best_assign = assign.copy()
                    best_eval = dict(eval_out)
                consecutive_queue_rejects = 0
            else:
                layout_state.assign = assign
                added = False
                if action.get("candidate_id") is not None:
                    cid = int(action.get("candidate_id"))
                    failed_counts[cid] = failed_counts.get(cid, 0) + 1
                    if failed_counts[cid] > 10:
                        recent_failed_ids.add(cid)
                consecutive_queue_rejects += 1
                if consecutive_queue_rejects >= 6:
                    refresh_due_to_rejects = True

            T *= alpha

            assign_signature = signature_for_assign(assign)
            time_ms = int((time.perf_counter() - step_start) * 1000)
            writer.writerow(
                [
                    step,
                    stage_label,
                    op,
                    json.dumps(action),
                    int(accept),
                    eval_out["total_scalar"],
                    eval_out["comm_norm"],
                    eval_out["therm_norm"],
                    int(added),
                    eval_out["penalty"]["duplicate"],
                    eval_out["penalty"]["boundary"],
                    seed_id,
                    time_ms,
                    assign_signature,
                    delta,
                    delta_comm,
                    delta_therm,
                    tabu_hit,
                    inverse_hit,
                    cooldown_hit,
                ]
            )
            if step % int(_cfg_get(cfg, "trace_flush_every", 20)) == 0:
                f_trace.flush()
            if progress_every > 0 and step % progress_every == 0:
                heartbeat = {
                    "step": int(step),
                    "elapsed_s": float(time.time() - start_time),
                    "cur_total": float(eval_out.get("total_scalar", 0.0)),
                    "best_total": float(best_total_seen),
                    "accept_rate": float(accepted_steps / max(1, step + 1)),
                    "queue_len": int(len(action_queue)),
                    "last_op": op,
                    "temperature": float(T),
                }
                with (out_dir / "heartbeat.json").open("w", encoding="utf-8") as hb_fp:
                    json.dump(heartbeat, hb_fp, indent=2)
            if save_every > 0 and step % save_every == 0:
                checkpoint = {
                    "step": int(step),
                    "assign": assign.tolist(),
                    "best_assign": best_assign.tolist(),
                    "best_eval": best_eval,
                    "cur_eval": eval_out,
                    "temperature": float(T),
                    "recent_signatures": list(tabu_signatures),
                }
                with (out_dir / "checkpoint_state.json").open("w", encoding="utf-8") as ck_fp:
                    json.dump(checkpoint, ck_fp, indent=2)
    if usage_fp:
        usage_fp.close()

    return DetailedPlaceResult(assign=assign, pareto=pareto, trace_path=trace_path)
