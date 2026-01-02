"""Detailed placement with SA and Pareto updates (SPEC v4.3.2 §8.6).

Key guarantees:
- No NameError on mixed/llm path (planner_type/llm_provider/llm_init_error always defined)
- planner.type == mixed actually calls LLM every_n_steps when available
- If LLM init fails (e.g., missing ARK_API_KEY), gracefully fall back to heuristic and
  write a one-line init_failed record to llm_usage.jsonl (if provided).
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from layout.coarsen import Cluster
from layout.evaluator import LayoutEvaluator, LayoutState
from layout.pareto import ParetoSet
from layout.llm_provider import HeuristicProvider, VolcArkProvider, LLMProvider
from layout.candidate_pool import Candidate, build_candidate_pool, simulate_action_sequence


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


def _apply_cluster_move(assign: np.ndarray, cluster: Cluster, target_sites: List[int]):
    if len(target_sites) < len(cluster.slots):
        return
    for slot, site in zip(cluster.slots, target_sites):
        assign[int(slot)] = int(site)


def _state_summary(
    step: int,
    T: float,
    eval_out: Dict[str, float],
    traffic_sym: np.ndarray,
    assign: np.ndarray,
    site_to_region: np.ndarray,
    chip_tdp: Optional[np.ndarray],
    clusters: List[Cluster],
    regions,
    candidates: List[Candidate],
    candidate_ids: List[int],
    forbidden_ids: List[int],
    k: int,
) -> Dict:
    pairs = _compute_top_pairs(traffic_sym, 5)
    hot_slots = []
    if chip_tdp is not None and len(chip_tdp) == traffic_sym.shape[0]:
        order = np.argsort(chip_tdp)[::-1][:5]
        hot_slots = [
            {"i": int(i), "tdp": float(chip_tdp[i]), "region": int(site_to_region[assign[i]])}
            for i in order
        ]
    cand_payload = []
    for cand in candidates:
        desc = ""
        op = cand.type
        action = cand.action
        if op == "swap":
            desc = f"swap {action.get('i')}-{action.get('j')}"
        elif op == "relocate":
            desc = f"reloc slot{action.get('i')}->site{action.get('site_id')}"
        elif op == "cluster_move":
            desc = f"cluster {action.get('cluster_id')}->reg{action.get('region_id')}"
        radial_to = cand.features.get("radial_to") if op != "swap" else None
        cand_payload.append(
            {
                "id": int(cand.id),
                "type": op,
                "d_total": float(cand.est.get("d_total", 0.0)),
                "d_comm": float(cand.est.get("d_comm", 0.0)),
                "d_therm": float(cand.est.get("d_therm", 0.0)),
                "touch_hot": 1 if cand.features.get("touches_hot_pair") or cand.features.get("touches_hot_slot") else 0,
                "radial_to": None if radial_to is None else float(radial_to),
                "desc": desc[:60],
            }
        )
    return {
        "K": int(k),
        "step": int(step),
        "T": float(T),
        "current": {
            "total": float(eval_out.get("total_scalar", 0.0)),
            "comm": float(eval_out.get("comm_norm", 0.0)),
            "therm": float(eval_out.get("therm_norm", 0.0)),
        },
        "top_hot_pairs": [{"i": int(i), "j": int(j), "traffic": float(t)} for i, j, t in pairs],
        "top_hot_slots": hot_slots,
        "candidate_ids": candidate_ids,
        "forbidden_ids": forbidden_ids,
        "candidates": cand_payload,
        "S": int(traffic_sym.shape[0]),
        "Ns": int(len(site_to_region)),
        "num_clusters": int(len(clusters)),
        "num_regions": int(len(regions)),
    }


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
) -> Dict:
    probs = _cfg_get(cfg, "action_probs", {}) or {}

    # 1) swap: prioritize hot pairs
    if random.random() < float(probs.get("swap", 0.5)):
        hot_cfg = _cfg_get(cfg, "hot_sampling", {}) or {}
        top_k = int(hot_cfg.get("top_pairs_k", 10))
        top_pairs = _compute_top_pairs(traffic_sym, top_k)
        if top_pairs:
            i, j, _ = random.choice(top_pairs)
            return {"op": "swap", "i": int(i), "j": int(j)}

    # 2) relocate: prefer empty sites within same region
    if random.random() < float(probs.get("relocate", 0.3)):
        reloc_cfg = _cfg_get(cfg, "relocate", {}) or {}
        same_region_prob = float(reloc_cfg.get("same_region_prob", 0.8))
        neighbor_k = int(reloc_cfg.get("neighbor_k", 30))

        slot_scores = traffic_sym.sum(axis=1)
        if chip_tdp is not None and len(chip_tdp) == slot_scores.shape[0]:
            slot_scores = slot_scores + chip_tdp

        slot = int(np.argmax(slot_scores)) if float(slot_scores.sum()) > 0 else random.randrange(traffic_sym.shape[0])

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
        if (not candidates) or (random.random() > same_region_prob):
            candidates = empty_sites

        # choose nearest candidate sites
        dists = [(sid, float(np.linalg.norm(sites_xy[cur_site] - sites_xy[sid]))) for sid in candidates]
        dists.sort(key=lambda x: x[1])
        chosen = dists[: max(1, min(neighbor_k, len(dists)))]
        site_id = int(random.choice(chosen)[0])
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
                target_region = random.choice(region_options)
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
    random.seed(base_seed)

    # ---- llm logging safety ----
    raw_text: str = ""
    llm_provider: Optional[LLMProvider] = None
    llm_init_error: Optional[str] = None

    # ---- init assign ----
    assign = np.array(assign_seed, dtype=int).copy()
    layout_state.assign = assign

    # ---- planner config ----
    planner_cfg = _cfg_get(cfg, "planner", {"type": "heuristic"}) or {"type": "heuristic"}
    planner_type = str(_cfg_get(planner_cfg, "type", "heuristic"))
    mixed_cfg = _cfg_get(planner_cfg, "mixed", {}) or {}
    mixed_every = int(_cfg_get(mixed_cfg, "every_n_steps", 200)) if planner_type == "mixed" else 0
    k_actions = int(_cfg_get(mixed_cfg, "k_actions", 4))

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
    usage_fp = llm_usage_path.open("a", encoding="utf-8") if llm_usage_path else None

    # If LLM is requested but init failed, log once and continue with heuristic
    if usage_fp and planner_type in ("llm", "mixed") and llm_provider is None and llm_init_error:
        json.dump({"event": "llm_init_failed", "planner_type": planner_type, "error": llm_init_error}, usage_fp)
        usage_fp.write("\n")
        usage_fp.flush()

    with trace_path.open("w", encoding="utf-8") as f_trace:
        f_trace.write(
            "iter,stage,op,op_args_json,accepted,total_scalar,comm_norm,therm_norm,pareto_added,duplicate_penalty,boundary_penalty,seed_id,time_ms\n"
        )

        # initial eval
        eval_out = evaluator.evaluate(layout_state)
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

        action_queue: List[Dict] = []
        forbidden_history: List[List[int]] = []
        failed_counts: Dict[int, int] = {}
        recent_failed_ids: set[int] = set()
        consecutive_queue_rejects = 0
        queue_window_steps = int(_cfg_get(planner_cfg, "queue_window_steps", 30))
        refresh_due_to_rejects = False

        for step in range(steps):
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
            )
            cand_map = {c.id: c for c in candidate_pool}
            candidate_ids = [c.id for c in candidate_pool]
            forbidden_ids = list({pid for recent in forbidden_history[-3:] for pid in recent} | recent_failed_ids)

            use_llm = (planner_type == "llm") or (planner_type == "mixed" and mixed_every > 0 and step % mixed_every == 0)
            need_refresh = use_llm and llm_provider is not None and (not action_queue or refresh_due_to_rejects)
            refresh_due_to_rejects = False

            if need_refresh:
                ss = _state_summary(
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
                try:
                    pick_ids = llm_provider.propose_picks(ss, k_actions) or []
                    pick_types_count: Dict[str, int] = {}
                    best_d_total = None
                    for pid in pick_ids:
                        cand = cand_map.get(pid)
                        if cand:
                            pick_types_count[cand.type] = pick_types_count.get(cand.type, 0) + 1
                            if best_d_total is None or cand.est.get("d_total", 0) < best_d_total:
                                best_d_total = cand.est.get("d_total", 0)
                    if usage_fp and hasattr(llm_provider, "last_usage"):
                        usage = dict(getattr(llm_provider, "last_usage") or {})
                        raw_text = str(usage.get("raw_preview", ""))
                        usage.setdefault("ok", True)
                        usage.setdefault("n_pick", len(pick_ids))
                        usage["pick_ids"] = pick_ids
                        usage["pick_types_count"] = pick_types_count
                        usage["best_d_total_in_pick"] = best_d_total
                        usage["step"] = int(step)
                        json.dump(usage, usage_fp)
                        usage_fp.write("\\n")
                        usage_fp.flush()
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
                if pick_ids:
                    forbidden_history.append(pick_ids)
                    actions = simulate_action_sequence(assign, pick_ids, cand_map, clusters, site_to_region, max_k=k_actions)
                    expire_step = step + queue_window_steps
                    for act in actions:
                        action_queue.append({"action": act, "expire": expire_step})

            action_queue = [a for a in action_queue if a.get("expire", step) >= step]

            action_payload = None
            if action_queue:
                action_payload = action_queue.pop(0)["action"]
            else:
                fallback = next((c for c in candidate_pool if c.id not in forbidden_ids), None)
                if fallback:
                    action_payload = {**fallback.action, "candidate_id": fallback.id, "type": fallback.type}
                else:
                    action_payload = _sample_action(
                        cfg, traffic_sym, site_to_region, regions, clusters, assign, sites_xy, chip_tdp, cluster_to_region
                    )

            action = action_payload or {"op": "none"}
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
                    target_sites = [s for s, r in enumerate(site_to_region) if int(r) == rid][: len(cluster.slots)]
                    _apply_cluster_move(new_assign, cluster, target_sites)

            layout_state.assign = new_assign
            eval_new = evaluator.evaluate(layout_state)
            delta = float(eval_new["total_scalar"] - eval_out["total_scalar"])
            accept = (delta < 0) or (math.exp(-delta / max(T, 1e-6)) > float(rng.random()))

            if accept:
                assign = new_assign
                eval_out = eval_new
                layout_state.assign = assign
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

            f_trace.write(
                f"{step},{stage_label},{op},{json.dumps(action)},"
                f"{int(accept)},{eval_out['total_scalar']},{eval_out['comm_norm']},{eval_out['therm_norm']},{int(added)},"
                f"{eval_out['penalty']['duplicate']},{eval_out['penalty']['boundary']},{seed_id},0\n"
            )
    if usage_fp:
        usage_fp.close()

    return DetailedPlaceResult(assign=assign, pareto=pareto, trace_path=trace_path)
