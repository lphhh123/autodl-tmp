"""Detailed placement with SA and Pareto updates (SPEC v5.4 §8.6).

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
import os
import random
import re
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
from layout.policy_switch import EvalCache, PolicySwitchController
from layout.delta_eval import ObjectiveParams, estimate_action_seq_delta
from layout.macro_engine import MacroEngine
from layout.verifier_engine import ROITracker, compute_gain
from layout.memory_bank import MemoryBank
from layout.mpvs_controller import MPVSController
from utils.trace_schema import TRACE_FIELDS
from utils.stable_hash import stable_hash
from utils.config_utils import get_nested
from utils.contract_seal import assert_cfg_sealed_or_violate
from utils.trace_contract_v54 import TraceContractV54
from utils.trace_guard import append_trace_event_v54

EVAL_VERSION = "v5.4"
TIME_UNIT = "ms"
DIST_UNIT = "mm"


@dataclass
class DetailedPlaceResult:
    assign: np.ndarray
    pareto: ParetoSet
    trace_path: Path
    policy_meta: Optional[Dict[str, Any]] = None


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


def _normalize_planner_type(t: str) -> str:
    t0 = (t or "").strip().lower()
    alias = {
        "llm_mixed_pick": "mixed",
        "llm_mixed": "mixed",
        "mixed_pick": "mixed",
    }
    return alias.get(t0, t0)


def _build_run_signature(cfg: Any) -> Dict[str, Any]:
    lookahead_cfg = _cfg_get(cfg, "lookahead", {}) or {}
    ps_cfg = _cfg_get(cfg, "policy_switch", None)
    action_families = _cfg_get(ps_cfg, "action_families", None)
    moves_enabled = bool(action_families) if action_families is not None else bool(_cfg_get(cfg, "moves_enabled", False))
    lookahead_k = int(_cfg_get(lookahead_cfg, "topk", _cfg_get(lookahead_cfg, "k", 0) or 0))
    bandit_type = str(_cfg_get(ps_cfg, "bandit_type", "eps_greedy"))
    policy_switch_enabled = bool(_cfg_get(ps_cfg, "enabled", False))
    cache_size = int(_cfg_get(ps_cfg, "cache_size", 0) or 0)
    cache_enabled = bool(cache_size > 0 and policy_switch_enabled)
    cache_key_schema_version = str(_cfg_get(ps_cfg, "cache_key_schema_version", "v5.4"))
    ps_enabled = bool(_cfg_get(ps_cfg, "enabled", False))
    return {
        "moves_enabled": moves_enabled,
        "lookahead_k": lookahead_k,
        "bandit_type": bandit_type,
        "policy_switch_mode": (str(_cfg_get(ps_cfg, "mode", "bandit")) if ps_enabled else "none"),
        "cache_enabled": cache_enabled,
        "cache_key_schema_version": cache_key_schema_version,
    }


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


def _apply_relocate(assign: np.ndarray, i: int, site_id: int) -> int | None:
    """Permutation-safe relocate (swap with occupant if needed)."""
    i = int(i)
    site_id = int(site_id)
    if i < 0 or i >= assign.shape[0]:
        return None
    occ = np.where(assign == site_id)[0]
    j = int(occ[0]) if occ.size > 0 else -1
    if j >= 0 and j != i:
        assign[i], assign[j] = int(assign[j]), int(assign[i])
        return int(j)
    assign[i] = site_id
    return None


def _apply_random_kick(assign: np.ndarray, idxs: List[int], site_ids: List[int]) -> None:
    """In-place kick: for each (idx,site) swap with occupant if exists, else set."""
    if not idxs or not site_ids:
        return
    S = int(assign.shape[0])
    for ii, target in zip(idxs, site_ids):
        i = int(ii)
        if i < 0 or i >= S:
            continue
        t = int(target)
        occ = np.where(assign == t)[0]
        j = int(occ[0]) if occ.size > 0 else -1
        if j >= 0 and j != i:
            assign[i], assign[j] = int(assign[j]), int(assign[i])
        else:
            assign[i] = t


def _apply_cluster_move(assign: np.ndarray, cluster: Cluster, target_sites: Optional[List[int]]):
    """Permutation-safe cluster move: sequentially relocate each cluster slot (swap if occupied)."""
    if not target_sites or len(target_sites) < len(cluster.slots):
        return
    for slot, site in zip(cluster.slots, target_sites):
        _apply_relocate(assign, int(slot), int(site))


def signature_for_assign(assign: Any) -> str:
    """
    Accepts either np.ndarray or a python sequence (list/tuple).
    Returns a stable signature string that starts with 'assign:'.
    """
    if assign is None:
        return "assign:null"
    if hasattr(assign, "tolist"):
        assign_list = assign.tolist()
    else:
        assign_list = list(assign)
    return signature_from_assign(assign_list)


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
    """
    Sample a VALID move. Never return {"op":"none"}.
    - swap: always pick i!=j (hot pairs if available, else random)
    - relocate: always pick to_site != cur_site (prefer empty site in same region, else any other site)
    - cluster_move: optional; if disabled or not available, fall back to swap
    """
    probs = _cfg_get(cfg, "action_probs", {}) or {}
    p_swap = float(probs.get("swap", 0.5))
    p_reloc = float(probs.get("relocate", 0.3))
    p_cmov = float(probs.get("cluster_move", 0.0))
    tot = max(1e-9, p_swap + p_reloc + p_cmov)

    r = py_rng.random() * tot

    S = int(assign.shape[0])
    Ns = int(site_to_region.shape[0])

    # --- swap ---
    def _rand_swap():
        i = py_rng.randrange(S)
        j = py_rng.randrange(S)
        while j == i:
            j = py_rng.randrange(S)
        return {"op": "swap", "i": int(i), "j": int(j)}

    # --- relocate (permutation-safe relocate will swap if occupied) ---
    def _rand_reloc():
        reloc_cfg = _cfg_get(cfg, "relocate", {}) or {}
        same_region_prob = float(reloc_cfg.get("same_region_prob", 0.8))
        neighbor_k = int(reloc_cfg.get("neighbor_k", 30))

        # choose a slot (hotter first if possible)
        slot_scores = np.sum(traffic_sym, axis=1)
        if chip_tdp is not None and len(chip_tdp) == slot_scores.shape[0]:
            slot_scores = slot_scores + np.asarray(chip_tdp, dtype=float)
        slot = int(np.argmax(slot_scores)) if float(np.sum(slot_scores)) > 0 else py_rng.randrange(S)

        cur_site = int(assign[slot])
        if cur_site < 0 or cur_site >= Ns:
            # fallback to swap if state is invalid
            return _rand_swap()

        used = set(int(x) for x in assign.tolist())
        empty_sites = [s for s in range(Ns) if int(s) not in used]
        # if no empty sites, still relocate to a different site_id (perm-safe relocate will swap)
        if not empty_sites:
            to_site = py_rng.randrange(Ns)
            while int(to_site) == int(cur_site):
                to_site = py_rng.randrange(Ns)
            return {"op": "relocate", "i": int(slot), "site_id": int(to_site), "from_site": int(cur_site)}

        cur_region = int(site_to_region[cur_site])
        candidates = [s for s in empty_sites if int(site_to_region[s]) == cur_region]
        if (not candidates) or (py_rng.random() > same_region_prob):
            candidates = empty_sites

        # nearest candidates (optional)
        try:
            dists = [(sid, float(np.linalg.norm(sites_xy[cur_site] - sites_xy[sid]))) for sid in candidates]
            dists.sort(key=lambda x: x[1])
            chosen = dists[: max(1, min(neighbor_k, len(dists)))]
            to_site = int(py_rng.choice(chosen)[0])
        except Exception:
            to_site = int(py_rng.choice(candidates))

        if int(to_site) == int(cur_site):
            # enforce change
            to_site = int(py_rng.choice([s for s in candidates if int(s) != int(cur_site)] or candidates))
        return {"op": "relocate", "i": int(slot), "site_id": int(to_site), "from_site": int(cur_site)}

    # --- cluster_move (only if enabled and feasible) ---
    def _rand_cluster_move():
        if (not clusters) or (not regions):
            return None
        # pick heaviest cluster
        clusters_sorted = sorted(clusters, key=lambda c: float(getattr(c, "tdp_sum", 0.0)), reverse=True)
        c = clusters_sorted[0]
        cid = int(getattr(c, "cluster_id", -1))
        if cid < 0:
            return None
        cur_region = int(cluster_to_region[cid]) if (0 <= cid < len(cluster_to_region)) else -1
        region_options = [r for r in regions if int(getattr(r, "region_id", -1)) != cur_region]
        if not region_options:
            return None
        tgt = py_rng.choice(region_options)
        act = {"op": "cluster_move", "cluster_id": int(cid), "region_id": int(getattr(tgt, "region_id", -1))}
        # add slots fallback if available
        if getattr(c, "slots", None):
            act["cluster_slots"] = [int(x) for x in c.slots]
        return act

    if r < p_swap:
        # prefer hot pairs if available
        hot_cfg = _cfg_get(cfg, "hot_sampling", {}) or {}
        top_k = int(hot_cfg.get("top_pairs_k", 10))
        top_pairs = _compute_top_pairs(traffic_sym, top_k)
        if top_pairs:
            i, j, _ = py_rng.choice(top_pairs)
            if int(i) != int(j):
                return {"op": "swap", "i": int(i), "j": int(j)}
        return _rand_swap()

    if r < p_swap + p_reloc:
        return _rand_reloc()

    # cluster_move branch
    cm = _rand_cluster_move()
    if cm is not None:
        return cm
    return _rand_swap()


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
    recordings_path: Optional[Path] = None,
    trace_events_path: Optional[Path] = None,
) -> DetailedPlaceResult:
    ctr = getattr(cfg, "_contract", None)
    if ctr is None or not bool(getattr(ctr, "stamped_v54", False)):
        raise RuntimeError(
            "v5.4 CONTRACT: cfg not validated/stamped. "
            "Call validate_and_fill_defaults(...) via SPEC_D OneCommand entrypoint."
        )
    if not getattr(cfg, "contract", None) or not getattr(cfg.contract, "seal_digest", None):
        raise RuntimeError("v5.4 CONTRACT: missing seal_digest; boot not completed.")
    trace_contract = TraceContractV54
    seal_digest = str(get_nested(cfg, "contract.seal_digest", "") or "")
    if trace_events_path is not None:
        assert_cfg_sealed_or_violate(
            cfg=cfg,
            seal_digest=seal_digest,
            trace_events_path=trace_events_path,
            phase="layout_detailed_place",
            step=0,
            fatal=True,
            trace_contract=trace_contract,
        )
    # Use detailed_place sub-config for algorithm knobs, while keeping v5.4 seal checks on full cfg.
    dp_cfg = getattr(cfg, "detailed_place", None)
    if dp_cfg is None:
        dp_cfg = cfg
    # ---- deterministic seeds ----
    # P2: split RNG streams so non-main branches don't perturb the main search trajectory.
    base_seed = int(_cfg_get(cfg, "seed", 0)) + int(seed_id)
    rng_main = np.random.default_rng(base_seed + 0)
    rng_macro = np.random.default_rng(base_seed + 1_000_003)
    rng_mem = np.random.default_rng(base_seed + 2_000_003)
    rng_llm = np.random.default_rng(base_seed + 3_000_003)
    rng_verify = np.random.default_rng(base_seed + 4_000_003)
    py_rng_main = random.Random(base_seed + 0)
    py_rng_macro = random.Random(base_seed + 1_000_003)
    py_rng_mem = random.Random(base_seed + 2_000_003)
    py_rng_llm = random.Random(base_seed + 3_000_003)
    py_rng_verify = random.Random(base_seed + 4_000_003)

    # Legacy aliases for main path.
    rng = rng_main
    py_rng = py_rng_main

    # ---- llm logging safety ----
    raw_text: str = ""
    llm_provider: Optional[LLMProvider] = None
    llm_init_error: Optional[str] = None

    # ---- init assign ----
    assign = np.array(assign_seed, dtype=int).copy()
    layout_state.assign = assign
    objective_hash = evaluator.objective_hash()

    def _make_cache_key(assign_sig: str, objective_hash: str) -> str:
        cfg_hash_subset = stable_hash(
            {
                "eval_version": EVAL_VERSION,
                "objective_hash": objective_hash,
                "objective_cfg": _cfg_get(cfg, "objective", {}),
                "scales": _cfg_get(cfg, "scales", {}),
            }
        )
        return stable_hash(
            {
                "assign": assign_sig,
                "objective_hash": objective_hash,
                "eval_version": EVAL_VERSION,
                "cfg_hash_subset": cfg_hash_subset,
            }
        )

    # ---- anti-oscillation params ----
    anti_cfg = _cfg_get(dp_cfg, "anti_oscillation", {}) or {}
    tabu_tenure = int(_cfg_get(anti_cfg, "tabu_tenure", 8))
    inverse_tenure = int(_cfg_get(anti_cfg, "inverse_tenure", 6))
    per_slot_cooldown = int(_cfg_get(anti_cfg, "per_slot_cooldown", 6))
    aspiration_delta = float(_cfg_get(anti_cfg, "aspiration_delta", 1e-4))
    # Search health (soft-block + relax to avoid op=none under heavy blocking)
    soft_block_enabled = bool(_cfg_get(anti_cfg, "soft_block_enabled", True))
    soft_block_min_gain = float(_cfg_get(anti_cfg, "soft_block_min_gain", 0.001))
    soft_block_randw_scale = float(_cfg_get(anti_cfg, "soft_block_randw_scale", 1.5))
    relax_enabled = bool(_cfg_get(anti_cfg, "relax_enabled", True))
    relax_stagnation_ge = int(_cfg_get(anti_cfg, "relax_stagnation_ge", 15))
    relax_max_level = int(_cfg_get(anti_cfg, "relax_max_level", 3))
    relax_only_if_improving = bool(_cfg_get(anti_cfg, "relax_only_if_improving", True))
    relax_min_gain = float(_cfg_get(anti_cfg, "relax_min_gain", soft_block_min_gain))

    # ---- planner config ----
    planner_cfg = _cfg_get(dp_cfg, "planner", {"type": "heuristic"}) or {"type": "heuristic"}

    planner_type_requested = str(_cfg_get(planner_cfg, "type", "heuristic"))
    planner_type = _normalize_planner_type(planner_type_requested)

    _valid_types = {"heuristic", "llm", "mixed"}
    if planner_type not in _valid_types:
        if trace_events_path is not None:
            append_trace_event_v54(
                trace_events_path,
                "contract_override",
                payload={
                    "phase": "layout_detailed_place",
                    "reason": "invalid_planner_type",
                    "requested": {"planner.type": planner_type_requested},
                    "effective": {"planner.type": "heuristic"},
                },
                step=0,
            )
        planner_type = "heuristic"

    mixed_cfg = _cfg_get(planner_cfg, "mixed", {}) or {}
    mixed_every_val = _cfg_get(mixed_cfg, "every_n_steps", None)
    if mixed_every_val is None:
        mixed_every_val = _cfg_get(planner_cfg, "every_n_steps", None)
    mixed_every = int(mixed_every_val) if (planner_type == "mixed" and mixed_every_val is not None) else (200 if planner_type == "mixed" else 0)

    k_actions_val = _cfg_get(mixed_cfg, "k_actions", None)
    if k_actions_val is None:
        k_actions_val = _cfg_get(planner_cfg, "k_actions", None)
    k_actions = int(k_actions_val) if k_actions_val is not None else 4

    queue_enabled = bool(_cfg_get(planner_cfg, "queue_enabled", True))
    feasibility_check = bool(_cfg_get(planner_cfg, "feasibility_check", True))
    if _cfg_get(planner_cfg, "feasibility_check", None) is None and _cfg_get(planner_cfg, "feas_enabled", None) is not None:
        feasibility_check = bool(_cfg_get(planner_cfg, "feas_enabled", True))

    timeout_sec = int(_cfg_get(planner_cfg, "timeout_sec", 90))
    max_retry = int(_cfg_get(planner_cfg, "max_retry", 1))
    stage_label = str(_cfg_get(dp_cfg, "stage_label", f"detailed_{planner_type}"))

    look_cfg = _cfg_get(dp_cfg, "lookahead", {}) or {}
    lookahead_enabled = bool(_cfg_get(look_cfg, "enabled", False))
    lookahead_k = int(_cfg_get(look_cfg, "k", _cfg_get(look_cfg, "topk", 8)))
    lookahead_r = int(_cfg_get(look_cfg, "r", 3))
    lookahead_mc = int(_cfg_get(look_cfg, "mc", 2))
    lookahead_alpha = float(_cfg_get(look_cfg, "alpha", 1.0))

    # ===== v5.4 Ours-B2+ controller (optional) =====
    ps_cfg = _cfg_get(dp_cfg, "policy_switch", None)
    use_ps = bool(ps_cfg and _cfg_get(ps_cfg, "enabled", False))
    eval_cache = None
    controller = None
    if use_ps:
        eval_cache = EvalCache(max_size=int(_cfg_get(ps_cfg, "cache_size", 5000)))
        controller = PolicySwitchController(
            action_families=list(_cfg_get(ps_cfg, "action_families", ["relocate", "swap", "inverse"])),
            policies=list(_cfg_get(ps_cfg, "policies", ["heuristic", "llm"])),
            eps=float(_cfg_get(ps_cfg, "eps", 0.1)),
            seed=int(base_seed),
        )

    mpvs_cfg = _cfg_get(dp_cfg, "mpvs", {}) or {}
    mpvs_enabled = bool(_cfg_get(mpvs_cfg, "enabled", False))

    mpvs_cache = None
    if mpvs_enabled:
        # independent cache for verifier evaluations (do NOT depend on policy_switch)
        mpvs_cache = EvalCache(max_size=int(_cfg_get(mpvs_cfg, "cache_size", 12000)))

    mpvs_mem: List[Dict[str, Any]] = []  # legacy memory; kept for backward compatibility
    # Memory global gate (suppress harmful memory streaks under eval-call budgets)
    mpvs_mem_global_until = 0
    mpvs_mem_hist: List[int] = []  # 1=success, 0=fail for selected mem steps
    mem_bank: Optional[MemoryBank] = None
    if mpvs_enabled:
        try:
            mem_cfg0 = _cfg_get(mpvs_cfg, "memory", {}) or {}
            mem_bank = MemoryBank(
                max_size=int(_cfg_get(mem_cfg0, "max_size", 64)),
                ttl_steps=int(_cfg_get(mem_cfg0, "ttl_steps", 120)),
                max_action_len=int(_cfg_get(mem_cfg0, "max_action_len", 3)),
                hot_slots_k=int(_cfg_get(mem_cfg0, "hot_slots_k", 12)),
                ewma_alpha=float(_cfg_get(mem_cfg0, "roi_ewma_alpha", 0.2)),
                min_similarity=float(_cfg_get(mem_cfg0, "min_similarity", 0.4)),
                fail_cooldown=int(_cfg_get(mem_cfg0, "cooldown_fail", 20)),
                max_fail=int(_cfg_get(mem_cfg0, "max_fail", 6)),
                age_penalty=float(_cfg_get(mem_cfg0, "age_penalty", 0.001)),
            )
            try:
                mem_bank.init_hot_slots(traffic_sym, chip_tdp)
            except Exception:
                pass
        except Exception:
            mem_bank = None

    roi_tracker: Optional[ROITracker] = ROITracker(
        alpha=float(_cfg_get(_cfg_get(mpvs_cfg, "verifier", {}) or {}, "roi_ewma_alpha", 0.2))
    ) if mpvs_enabled else None

    # Unified MPVS multi-component controller (macro/mem/llm). Verifier remains MPVS core.
    mpvs_ctrl: Optional[MPVSController] = None
    if mpvs_enabled:
        try:
            ctrl_cfg0 = _cfg_get(mpvs_cfg, "controller", {}) or {}
            mpvs_ctrl = MPVSController(cfg=ctrl_cfg0 if isinstance(ctrl_cfg0, dict) else {}, instance_tag="")
        except Exception:
            mpvs_ctrl = None
    mpvs_explore_left = 0
    # --- MPVS trigger controller state (BATC v0.5) ---
    mpvs_trigger_state = {
        "recent_sigs": [],   # recent assignment signatures (post-step)
        "recent_calls": [],  # recent eval-calls spent per step
        "recent_calls_cheap": [],
        "recent_repeat": [],
        "recent_improve": [],
        "credit": {"macro": 0.0, "verifier": 0.0},
        "cooldown": {"macro": 0, "verifier": 0},
        "last_fire": {"macro": -10000000, "verifier": -10000000},
    }

    total_eval_budget = int(get_nested(cfg, "budget.total_eval_budget", 0) or 0)

    def _budget_remaining() -> int:
        if total_eval_budget <= 0:
            return 10**18
        try:
            used = int(getattr(evaluator, "evaluator_calls", 0))
        except Exception:
            used = 0
        return max(0, int(total_eval_budget) - int(used))

    mpvs_stats = {
        "enabled": bool(mpvs_enabled),
        "llm_calls": 0,
        "llm_ok": 0,
        "llm_selected": 0,
        "llm_macro_selected": 0,
        "macro_selected": 0,
        "memory_selected": 0,
        "macro_scored": 0,
        "mem_scored": 0,
        "plans_scored": 0,
        "verifier_calls_spent": 0,
        "steps_mpvs": 0,
        "pareto_added": 0,
        "macro_gate_blocked": 0,
        "macro_gate_allowed": 0,
        "macro_precheck_failed": 0,
        "macro_precheck_blocked": 0,
        "macro_precheck_allowed": 0,
        "macro_monotone_blocked": 0,
        "nonheur_current_gate_blocked": 0,
        "nonheur_current_gate_blocked_by_src": {},
        "macro_selected_nonimprove": 0,
        "verifier_candidates_considered": 0,
        "verifier_changed_choice": 0,
        "verifier_changed_accept": 0,
        "verifier_lite_verified": 0,
        "verifier_full_verified": 0,
        "verifier_lite_calls": 0,
        "verifier_full_calls": 0,
        "verifier_roi_sum": 0.0,
        "verifier_roi_count": 0,
        "verifier_roi_by_src": {},
        "verifier_gain_by_src": {},
        "verifier_calls_by_src": {},
        "mem_entries": 0,
        "mem_queries": 0,
        "mem_hits": 0,
        "mem_store": 0,
        "mem_store_skip": 0,
        "mem_verify_fail": 0,
        "mem_prefilter_drop": 0,
        "mem_global_blocked": 0,
        "mem_global_triggered": 0,
        "mem_global_until": 0,
        "mem_global_fail_rate": 0.0,
        "llm_allowed": 0,
        "llm_denied": 0,
        "llm_deny_reason": {},
        "comp_ctrl": {},
        "calls_by_src": {},
        "gain_by_src": {},
        # --- BATC trigger stats ---
        "trig_enabled": 0,
        "trig_steps": 0,
        "trig_repeat_ratio_sum": 0.0,
        "trig_calls_avg_sum": 0.0,
        "trig_repeat_high_effective": None,
        "trig_calls_high_effective": None,
        "trig_calls_baseline": None,
        "trig_macro_allowed": 0,
        "trig_macro_fired": 0,
        "trig_macro_fail": 0,
        "trig_macro_success": 0,
        "trig_ver_allowed": 0,
        "trig_ver_fired": 0,
        "trig_ver_fail": 0,
        "trig_ver_success": 0,
        "macro_precheck_fail_min_gain": 0,
        "macro_precheck_pass_min_gain": 0,
        "macro_probe_seen": 0,
        "macro_probe_pass_heur": 0,
        "macro_probe_pass_cur": 0,
        "macro_trial_sponsored": 0,
        "macro_trial_won": 0,
        "macro_release_activated": 0,
        "macro_release_hit": 0,
        "macro_trial_seed": 0,
        "macro_trial_evidence": 0,
        "macro_trial_candidate": 0,
        "macro_trial_sponsor_reason": {},
        "macro_trial_score_sum": 0.0,
        "macro_trial_score_max": 0.0,
        "macro_candidate_activated": 0,
        "macro_candidate_hit": 0,
        "heuristic_rate_ewma": 0.0,
        # v2.1 audit: count sponsored macro wins that use pre-apply best_total as ticket baseline.
        "ticket_baseline_fix_used": 0,
    }
    prev_release_total = 0
    prev_candidate_total = 0
    prev_candidate_hits_sum = 0

    # ----------------------------
    # MacroEngine (stronger macros, fewer evaluator calls)
    # ----------------------------
    macro_engine: Optional[MacroEngine] = None
    mpvs_obj_params: Optional[ObjectiveParams] = None
    if mpvs_enabled:
        try:
            macro_cfg0 = _cfg_get(mpvs_cfg, "macros", {}) or {}
            obj = ObjectiveParams(
                sigma_mm=float(getattr(evaluator, "sigma_mm", 1.0)),
                L_comm_baseline=float(getattr(evaluator, "baseline", {}).get("L_comm_baseline", 1.0)),
                L_therm_baseline=float(getattr(evaluator, "baseline", {}).get("L_therm_baseline", 1.0)),
                w_comm=float(getattr(evaluator, "scalar_w", {}).get("w_comm", 0.0)),
                w_therm=float(getattr(evaluator, "scalar_w", {}).get("w_therm", 0.0)),
            )
            mpvs_obj_params = obj
            macro_engine = MacroEngine(macro_cfg=macro_cfg0 if isinstance(macro_cfg0, dict) else {}, obj=obj, rng=rng_macro)
        except Exception:
            macro_engine = None
            mpvs_obj_params = None

    # Providers: always have heuristic; LLM optional
    heuristic_provider: LLMProvider = HeuristicProvider()
    llm_cfg_path = _cfg_get(planner_cfg, "llm_config_file", None)

    if planner_type in ("llm", "mixed"):
        try:
            if llm_cfg_path:
                llm_provider = VolcArkProvider.from_config_file(
                    llm_cfg_path, timeout_sec=timeout_sec, max_retry=max_retry
                )
            else:
                llm_provider = VolcArkProvider(timeout_sec=timeout_sec, max_retry=max_retry)
        except Exception as e:
            llm_provider = None
            llm_init_error = repr(e)
    fallback_reason_init = "llm_init_failed" if (planner_type in ("llm", "mixed") and llm_provider is None and llm_init_error) else ""
    llm_fail_count = 1 if fallback_reason_init else 0
    llm_attempt_count = 0
    llm_success_count = 0
    llm_disabled = False
    llm_disabled_reason = ""
    if fallback_reason_init and trace_events_path is not None:
        append_trace_event_v54(
            trace_events_path,
            "contract_override",
            payload={
                "phase": "layout_detailed_place",
                "reason": "planner_fallback",
                "requested": {"planner.type": planner_type},
                "effective": {"planner.type": "heuristic"},
                "details": {"fallback_reason": fallback_reason_init},
            },
            step=0,
        )

    # ---- SA params (P2: scale-aware + reheat) ----
    steps = int(_cfg_get(dp_cfg, "steps", 0))

    # steps<=0 means "run until eval budget is exhausted"
    planned_steps = int(steps)
    if steps <= 0:
        steps = 10**9

    sa_cfg = _cfg_get(dp_cfg, "sa", {}) or {}
    # backward compatible keys
    T0_raw = float(_cfg_get(sa_cfg, "T0", _cfg_get(dp_cfg, "sa_T0", 1.0)))
    alpha = float(_cfg_get(sa_cfg, "alpha", _cfg_get(dp_cfg, "sa_alpha", 0.999)))

    # P2 defaults (can be overridden via dp_cfg.sa.*)
    sa_min_T = float(_cfg_get(sa_cfg, "min_T", 1e-6))
    sa_auto_scale = bool(_cfg_get(sa_cfg, "auto_scale", True))
    sa_auto_scale_frac = float(_cfg_get(sa_cfg, "auto_scale_frac", 0.02))  # 2% of base_total when T0 too small

    reheat_patience = int(_cfg_get(sa_cfg, "reheat_patience", 50))
    reheat_factor = float(_cfg_get(sa_cfg, "reheat_factor", 3.0))
    reheat_max_times = int(_cfg_get(sa_cfg, "reheat_max_times", 8))
    reheat_cooldown = int(_cfg_get(sa_cfg, "reheat_cooldown", 30))

    # runtime variables (set after first eval)
    T = float(T0_raw)

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir = trace_path.parent
    _p = str(trace_path).lower()
    if ("randw" in _p) or ("chain_skip_randw" in _p):
        instance_tag = "randw"
    elif "cluster4" in _p:
        instance_tag = "cluster4"
    else:
        instance_tag = "chain_skip"

    try:
        if mpvs_ctrl is not None:
            mpvs_ctrl.instance_tag = str(instance_tag)
    except Exception:
        pass

    usage_fp = llm_usage_path.open("a", encoding="utf-8") if llm_usage_path else None
    recordings_fp = recordings_path.open("w", encoding="utf-8") if recordings_path else None
    start_time = time.time()
    wall_start = time.perf_counter()
    eval_calls_cum = 0
    budget_exhausted = False
    last_step = -1
    best_solution = None
    report = None
    try:
        # If LLM is requested but init failed, log once and continue with heuristic
        if usage_fp and planner_type in ("llm", "mixed") and llm_provider is None and llm_init_error:
            json.dump({"event": "llm_init_failed", "planner_type": planner_type, "error": llm_init_error}, usage_fp)
            usage_fp.write("\n")
            usage_fp.flush()

        with trace_path.open("w", encoding="utf-8", newline="") as f_trace:
            writer = csv.DictWriter(f_trace, fieldnames=TRACE_FIELDS, restval="", extrasaction="ignore")
            writer.writeheader()

            try:
                if True:
                    # initial eval
                    def _sync_eval_calls():
                        nonlocal eval_calls_cum
                        try:
                            eval_calls_cum = int(getattr(evaluator, "evaluator_calls", eval_calls_cum))
                        except Exception:
                            pass

                    def _evaluate(state: LayoutState) -> Dict[str, Any]:
                        out = evaluator.evaluate(state)
                        _sync_eval_calls()
                        return out

                    def _evaluate_assign_cached(assign_arr: np.ndarray) -> Dict[str, Any]:
                        nonlocal eval_calls_cum
                        if mpvs_cache is None:
                            layout_state.assign = assign_arr
                            out = evaluator.evaluate(layout_state)
                            _sync_eval_calls()
                            return out
                        sig = signature_for_assign(assign_arr)
                        ck = _make_cache_key(sig, objective_hash)
                        hit = mpvs_cache.get(ck)
                        if hit is not None:
                            return dict(hit)
                        layout_state.assign = assign_arr
                        out = evaluator.evaluate(layout_state)
                        _sync_eval_calls()
                        mpvs_cache.put(ck, dict(out))
                        return out

                    eval_out = _evaluate(layout_state)
                    _sync_eval_calls()
                    # total eval-call budget (search budget, excluding finalization call)
                    budget_total_calls = int(getattr(evaluator, "eval_budget_limit", 0) or 0)
                    prev_total = float(eval_out.get("total_scalar", 0.0))
                    prev_comm = float(eval_out.get("comm_norm", 0.0))
                    prev_therm = float(eval_out.get("therm_norm", 0.0))
                    # P2: scale-aware temperature initialization.
                    # If objective scale is large and T0 is tiny, SA degenerates to greedy.
                    if sa_auto_scale:
                        base_total0 = float(prev_total)
                        if base_total0 > 1000.0 and T <= 10.0:
                            T = max(T, sa_auto_scale_frac * base_total0)
                    # enforce min_T
                    T = max(T, sa_min_T)
                    prev_assign = assign.copy()
                    accepted_steps = 0
                    assign_signature = signature_for_assign(prev_assign)
                    op_args_obj = {"op": "init"}
                    op_args_json = json.dumps(op_args_obj, ensure_ascii=False)
                    op_signature = stable_hash(op_args_obj)
                    init_cache_key = _make_cache_key(assign_signature, objective_hash)
                    wall_time_ms = int((time.perf_counter() - wall_start) * 1000)
                    cache_hit_cum = int(eval_cache.hits) if eval_cache is not None else 0
                    cache_miss_cum = int(eval_cache.misses) if eval_cache is not None else 0
                    cache_saved_eval_calls_cum = cache_hit_cum

                    init_row = {
                        "iter": -1,
                        "stage": "init",
                        "op": "init",
                        "op_args_json": op_args_json,
                        "accepted": 1,
                        "total_scalar": prev_total,
                        "comm_norm": prev_comm,
                        "therm_norm": prev_therm,
                        "pareto_added": 0,
                        "duplicate_penalty": float(eval_out.get("penalty", {}).get("duplicate", 0.0)),
                        "boundary_penalty": float(eval_out.get("penalty", {}).get("boundary", 0.0)),
                        "seed_id": int(seed_id),
                        "time_ms": wall_time_ms,
                        "signature": assign_signature,
                        "d_total": 0.0,
                        "d_comm": 0.0,
                        "d_therm": 0.0,
                        "delta_total": 0.0,
                        "delta_comm": 0.0,
                        "delta_therm": 0.0,
                        "tabu_hit": 0,
                        "inverse_hit": 0,
                        "cooldown_hit": 0,
                        "policy": "init",
                        "move": "init",
                        "lookahead_k": int(lookahead_k if lookahead_enabled else 0),
                        "cache_hit": 0,
                        "cache_key": init_cache_key,
                        "objective_hash": objective_hash,
                        "eval_calls_cum": eval_calls_cum,
                        "cache_hit_cum": cache_hit_cum,
                        "cache_miss_cum": cache_miss_cum,
                        "cache_saved_eval_calls_cum": cache_saved_eval_calls_cum,
                        "llm_used": 0,
                        "llm_fail_count": llm_fail_count,
                        "fallback_reason": fallback_reason_init,
                        "wall_time_ms_cum": wall_time_ms,
                        "accepted_steps_cum": accepted_steps,
                        "sim_eval_calls_cum": eval_calls_cum,
                        "lookahead_enabled": int(lookahead_enabled),
                        "lookahead_r": float(lookahead_r) if lookahead_enabled else 0.0,
                        "notes": "",
                    }
                    writer.writerow(init_row)
                    if recordings_fp is not None:
                        pen = eval_out.get("penalty", {}) or {}
                        init_record = {
                            "iter": 0,
                            "stage": "init",
                            "op": "init",
                            "op_args": {},
                            "op_args_json": json.dumps({"op": "init"}, ensure_ascii=False),
                            "accepted": 1,
                            "total_scalar": float(eval_out.get("total_scalar", 0.0)),
                            "comm_norm": float(eval_out.get("comm_norm", 0.0)),
                            "therm_norm": float(eval_out.get("therm_norm", 0.0)),
                            "pareto_added": 0,
                            "duplicate_penalty": float(pen.get("duplicate", 0.0)),
                            "boundary_penalty": float(pen.get("boundary", 0.0)),
                            "seed_id": int(seed_id),
                            "time_ms": 0,
                            "signature": signature_for_assign(prev_assign),
                        }
                        recordings_fp.write(json.dumps(init_record, ensure_ascii=False) + "\n")
                        recordings_fp.flush()
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
                        slots = [int(act.get("i", -1))]
                        j = act.get("degraded_swap_with", None)
                        if j is not None:
                            try:
                                jj = int(j)
                                if jj >= 0:
                                    slots.append(jj)
                            except Exception:
                                pass
                        return slots
                    if op_local == "random_kick":
                        return [int(x) for x in (act.get("idxs", []) or [])]
                    if op_local == "cluster_move":
                        cid = int(act.get("cluster_id", -1))
                        if 0 <= cid < len(clusters):
                            return [int(s) for s in clusters[cid].slots]
                    return []
        
                def _apply_action_inplace(assign_arr: np.ndarray, action: Dict[str, Any], clusters, site_to_region, sites_xy):
                    op_local = str(action.get("op", "none"))
                    if op_local == "swap":
                        i = int(action.get("i", -1))
                        j = int(action.get("j", -1))
                        if 0 <= i < len(assign_arr) and 0 <= j < len(assign_arr) and i != j:
                            _apply_swap(assign_arr, i, j)
                        return
                    if op_local == "relocate":
                        i = int(action.get("i", -1))
                        to_site = int(action.get("site_id", -1))
                        if 0 <= i < len(assign_arr) and 0 <= to_site < len(site_to_region):
                            _apply_relocate(assign_arr, i, to_site)
                        return
                    if op_local == "random_kick":
                        _apply_random_kick(
                            assign_arr,
                            [int(x) for x in (action.get("idxs", []) or [])],
                            [int(x) for x in (action.get("site_ids", []) or [])],
                        )
                        return
                    if op_local == "cluster_move":
                        cid = int(action.get("cluster_id", -1))
                        slots = [int(x) for x in (action.get("cluster_slots", []) or [])]
                        cluster_obj = None
                        if 0 <= cid < len(clusters) and getattr(clusters[cid], "slots", None) and clusters[cid].slots:
                            cluster_obj = clusters[cid]
                        elif slots:
                            cluster_obj = Cluster(cluster_id=-1, slots=slots, tdp_sum=0.0)
                        if cluster_obj is None:
                            return
                        rid = int(action.get("region_id", 0))
                        target_sites = action.get("target_sites")
                        if not target_sites:
                            target_sites = _select_cluster_target_sites(assign_arr, cluster_obj, rid, site_to_region, sites_xy)
                        _apply_cluster_move(assign_arr, cluster_obj, target_sites)
                        return

                def _apply_action_for_candidate(base_assign: np.ndarray, act: Dict[str, Any]) -> np.ndarray:
                    new_assign = base_assign.copy()
                    _apply_action_inplace(new_assign, act, clusters, site_to_region, sites_xy)
                    return new_assign

                def _rollout_avg_total(assign0: np.ndarray, candidate_action: Dict[str, Any], r: int, mc: int) -> float:
                    """Apply first action then r-1 greedy heuristic steps; return avg best total over MC rollouts."""
                    totals: List[float] = []
                    for _ in range(max(1, mc)):
                        cur = assign0.copy()
                        _apply_action_inplace(cur, candidate_action, clusters, site_to_region, sites_xy)
                        layout_state.assign = cur
                        e1 = _evaluate(layout_state)
                        best_total = float(e1.get("total_scalar", 0.0))
                        for _t in range(max(0, r - 1)):
                            sub_pool = build_candidate_pool(
                                cur,
                                e1,
                                evaluator,
                                layout_state,
                                traffic_sym,
                                sites_xy,
                                site_to_region,
                                regions,
                                clusters,
                                cluster_to_region,
                                chip_tdp,
                                dp_cfg,
                                rng,
                                debug_out_path=None,
                            )
                            if not sub_pool:
                                break
                            sub_pool.sort(key=lambda c: float(c.est.get("d_total", 0.0)), reverse=False)
                            act = sub_pool[0].action
                            cur2 = cur.copy()
                            _apply_action_inplace(cur2, act, clusters, site_to_region, sites_xy)
                            layout_state.assign = cur2
                            e2 = _evaluate(layout_state)
                            best_total = min(best_total, float(e2.get("total_scalar", best_total)))
                            cur = cur2
                            e1 = e2
                        totals.append(best_total)
                    layout_state.assign = assign0
                    return float(sum(totals) / max(1, len(totals)))
        
                action_queue: List[Dict] = []
                forbidden_history: List[List[int]] = []
                failed_counts: Dict[int, int] = {}
                recent_failed_ids: set[int] = set()
                consecutive_queue_rejects = 0
                queue_window_steps = int(_cfg_get(planner_cfg, "queue_window_steps", 30))
                refresh_due_to_rejects = False
                # ---- hard LLM scheduling / guardrails ----
                last_llm_call_step = -10**9
                llm_calls_total = 0
                llm_max_calls_total = int(_cfg_get(planner_cfg, "llm_max_calls_total", 120))
                llm_min_gap_steps = int(_cfg_get(planner_cfg, "llm_min_gap_steps", max(1, mixed_every if planner_type == "mixed" else 1)))

                # refresh trigger should only be driven by LLM-sourced rejects
                consecutive_llm_rejects = 0

                # optional: allow mid-cycle refresh in mixed (default False to prevent thrash)
                allow_midcycle_refresh = bool(_cfg_get(mixed_cfg, "allow_midcycle_refresh", False))
                progress_every = int(_cfg_get(dp_cfg, "progress_every", 10))
                save_every = int(_cfg_get(dp_cfg, "save_every", 50))
                require_llm = bool(_cfg_get(planner_cfg, "require_llm", False))
                disable_on_fatal = bool(_cfg_get(planner_cfg, "disable_on_fatal", True))

                # ---- P2/P3: stagnation & forced exploration state ----
                last_best_step = 0
                last_reheat_step = -10**9
                reheat_count = 0

                # P3 defaults (can be overridden via dp_cfg.explore.*)
                explore_cfg = _cfg_get(dp_cfg, "explore", {}) or {}
                stagnation_patience = int(_cfg_get(explore_cfg, "stagnation_patience", 50))
                kick_cooldown_steps = int(_cfg_get(explore_cfg, "kick_cooldown_steps", 50))

                # If stagnating, we force one strong move; prefer kick > cluster_move > therm_swap
                force_explore_until = -1

                def _dp_cfg_with_pool_override(overrides: Dict[str, Any]) -> Dict[str, Any]:
                    # build_candidate_pool expects a dp_cfg-like object; pass plain dict
                    base_cp = _cfg_get(dp_cfg, "candidate_pool", {}) or {}
                    return {
                        "candidate_pool": {**dict(base_cp), **dict(overrides)},
                        "action_probs": dict(_cfg_get(dp_cfg, "action_probs", {}) or {}),
                        "relocate": dict(_cfg_get(dp_cfg, "relocate", {}) or {}),
                        "hot_sampling": dict(_cfg_get(dp_cfg, "hot_sampling", {}) or {}),
                        "operator_bank": dict(_cfg_get(dp_cfg, "operator_bank", {}) or {}),
                    }

                ver_cfg = _cfg_get(mpvs_cfg, "verifier", {}) or {}
                ver_pool_over = dict(_cfg_get(ver_cfg, "pool_override", {}) or {})
                # safe defaults if not provided
                if not ver_pool_over:
                    ver_pool_over = {
                        "raw_target_size": 24,
                        "raw_target_max": 30,
                        "final_size": 10,
                        "diversity_enabled": False,
                        "coverage_min_per_bucket": 0,
                    }
                dp_cfg_ver = _dp_cfg_with_pool_override(ver_pool_over)

                def _pick_best_from_pool(
                    pool: List[Any],
                    prefer_types: Optional[List[str]] = None,
                    score_key: str = "d_total",
                ):
                    if not pool:
                        return None
                    cand_list = pool
                    if prefer_types:
                        filt = [c for c in pool if str(getattr(c, "type", "")) in set(prefer_types)]
                        if filt:
                            cand_list = filt
                    cand_list = sorted(cand_list, key=lambda c: float(getattr(c, "est", {}).get(score_key, 0.0)))
                    return cand_list[0] if cand_list else None

                def _build_small_pool(cur_assign: np.ndarray, cur_eval: Dict[str, Any], rng_local) -> List[Any]:
                    rem = int(_budget_remaining())
                    dp_cfg_use = dp_cfg_ver
                    if rem < 2500:
                        dp_cfg_use = _dp_cfg_with_pool_override({"raw_target_size": 10, "raw_target_max": 12, "final_size": 5, "diversity_enabled": False, "coverage_min_per_bucket": 0})
                    elif rem < 8000:
                        dp_cfg_use = _dp_cfg_with_pool_override({"raw_target_size": 14, "raw_target_max": 18, "final_size": 7, "diversity_enabled": False, "coverage_min_per_bucket": 0})
                    p = build_candidate_pool(
                        cur_assign,
                        cur_eval,
                        evaluator,
                        layout_state,
                        traffic_sym,
                        sites_xy,
                        site_to_region,
                        regions,
                        clusters,
                        cluster_to_region,
                        chip_tdp,
                        dp_cfg_use,
                        rng_local,
                        debug_out_path=None,
                    )
                    _sync_eval_calls()
                    return p

                def _apply_action_inplace_simple(assign_arr: np.ndarray, act: Dict[str, Any]) -> None:
                    op = str(act.get("op", "none")).lower()
                    if op == "swap":
                        _apply_swap(assign_arr, int(act.get("i", 0)), int(act.get("j", 0)))
                        return
                    if op == "relocate":
                        _apply_relocate(assign_arr, int(act.get("i", 0)), int(act.get("site_id", 0)))
                        return
                    if op == "random_kick":
                        _apply_random_kick(assign_arr, [int(x) for x in (act.get("idxs", []) or [])], [int(x) for x in (act.get("site_ids", []) or [])])
                        return
                    if op == "cluster_move":
                        cid = int(act.get("cluster_id", -1))
                        slots = [int(x) for x in (act.get("cluster_slots", []) or [])]
                        cluster_obj = None
                        if 0 <= cid < len(clusters) and getattr(clusters[cid], "slots", None) and clusters[cid].slots:
                            cluster_obj = clusters[cid]
                        elif slots:
                            cluster_obj = Cluster(cluster_id=-1, slots=slots, tdp_sum=0.0)
                        if cluster_obj is None:
                            return
                        target_sites = act.get("target_sites")
                        if not target_sites:
                            rid = int(act.get("region_id", 0))
                            target_sites = _select_cluster_target_sites(assign_arr, cluster_obj, rid, site_to_region, sites_xy)
                        _apply_cluster_move(assign_arr, cluster_obj, target_sites)
                        return

                def _exec_macro(
                    name: str,
                    assign0: np.ndarray,
                    eval0: Dict[str, Any],
                    n_steps: int = 3,
                    rng_local=None,
                ) -> Tuple[np.ndarray, Dict[str, Any], float, List[Dict[str, Any]], np.ndarray]:
                    # Stronger macro engine: delta-based candidates + top-k verification.
                    if macro_engine is None:
                        return assign0.copy(), dict(eval0), float(eval0.get("total_scalar", 0.0)), [], assign0.copy()

                    b_assign, b_eval, b_total, acts, cur_after, info = macro_engine.run_macro(
                        name=str(name),
                        assign0=assign0,
                        eval0=eval0,
                        sites_xy_mm=sites_xy,
                        traffic_sym=traffic_sym,
                        chip_tdp_w=(chip_tdp if chip_tdp is not None else np.zeros(int(assign0.shape[0]), dtype=float)),
                        site_to_region=site_to_region,
                        evaluate_assign=_evaluate_assign_cached,
                        n_steps=int(n_steps),
                    )

                    # Attach signatures for consistency
                    executed: List[Dict[str, Any]] = []

                    if rng_local is None:
                        rng_local = rng_verify
                    cur_tmp = assign0.copy()
                    for act in acts:
                        act2 = copy.deepcopy(act)
                        if "signature" not in act2 or not act2.get("signature"):
                            try:
                                act2["signature"] = _signature_for_action(act2, cur_tmp)
                            except Exception:
                                pass
                        executed.append(act2)
                        try:
                            _apply_action_inplace_simple(cur_tmp, act2)
                        except Exception:
                            pass

                    # record macro engine snapshot for diagnosability
                    try:
                        mpvs_stats["macro_engine"] = {"last": info, "ops": macro_engine.snapshot()}
                    except Exception:
                        pass

                    return np.asarray(b_assign, dtype=int), dict(b_eval), float(b_total), executed, np.asarray(cur_after, dtype=int)

                def _refine_sa(
                    assign_start: np.ndarray,
                    eval_start: Dict[str, Any],
                    n_calls: int,
                    T_ref: float,
                    py_rng_local=None,
                ) -> Tuple[np.ndarray, Dict[str, Any], float]:
                    cur = assign_start.copy()
                    cur_eval = dict(eval_start)
                    best_total = float(cur_eval.get("total_scalar", 0.0))
                    best_assign = cur.copy()
                    best_eval = dict(cur_eval)

                    if py_rng_local is None:
                        py_rng_local = py_rng_verify

                    Tloc = max(sa_min_T, float(T_ref))
                    for _ in range(max(0, int(n_calls))):
                        act = _sample_action(
                            dp_cfg,
                            traffic_sym,
                            site_to_region,
                            regions,
                            clusters,
                            cur,
                            sites_xy,
                            chip_tdp,
                            cluster_to_region,
                            py_rng_local,
                        )
                        trial = cur.copy()
                        _apply_action_inplace_simple(trial, act)
                        trial_eval = _evaluate_assign_cached(trial)
                        dt = float(trial_eval.get("total_scalar", 0.0)) - float(cur_eval.get("total_scalar", 0.0))
                        accept = (dt <= 0.0) or (py_rng_local.random() < math.exp(-dt / max(sa_min_T, Tloc)))
                        if accept:
                            cur = trial
                            cur_eval = trial_eval
                            if float(cur_eval.get("total_scalar", 0.0)) < best_total:
                                best_total = float(cur_eval.get("total_scalar", 0.0))
                                best_assign = cur.copy()
                                best_eval = dict(cur_eval)
                        Tloc = max(sa_min_T, Tloc * 0.999)
                    return best_assign, best_eval, best_total

                def _verify_plan(
                    plan: Dict[str, Any],
                    assign0: np.ndarray,
                    eval0: Dict[str, Any],
                    stagnation: int,
                ) -> Dict[str, Any]:
                    horizon = int(_cfg_get(ver_cfg, "horizon", 3))
                    mc = int(_cfg_get(ver_cfg, "mc", 2))
                    n_steps_macro = int(_cfg_get(_cfg_get(mpvs_cfg, "macros", {}) or {}, "n_steps", 3))
                    refine_calls = int(_cfg_get(ver_cfg, "refine_sa_calls", 20))
                    greedy_topm = int(_cfg_get(ver_cfg, "greedy_topm", 1))

                    macro_mode = str(_cfg_get(ver_cfg, "macro_verify_mode", "lite")).lower()
                    if str(plan.get("kind", "")) == "macro" and macro_mode == "lite":
                        # ultra-light: macro execute only (no horizon rollout), optional tiny refine
                        lite_steps = int(_cfg_get(ver_cfg, "lite_macro_steps", 1))
                        lite_refine = int(_cfg_get(ver_cfg, "lite_refine_sa_calls", 0))
                        n_steps_macro = min(n_steps_macro, max(1, lite_steps))
                        refine_calls = min(refine_calls, max(0, lite_refine))
                        horizon = 1
                        greedy_topm = 1

                    # randw needs more steps; keep verifier lighter at early stagnation
                    if instance_tag == "randw":
                        sw = int(_cfg_get(ver_cfg, "randw_switch_stagnation", 20))
                        if int(stagnation) < sw:
                            horizon = min(horizon, int(_cfg_get(ver_cfg, "randw_early_horizon_cap", 2)))
                            mc = min(mc, int(_cfg_get(ver_cfg, "randw_early_mc_cap", 1)))
                            refine_calls = min(refine_calls, int(_cfg_get(ver_cfg, "randw_early_refine_cap", 8)))
                            n_steps_macro = min(n_steps_macro, int(_cfg_get(ver_cfg, "randw_early_macro_steps_cap", 2)))

                    rem = int(_budget_remaining())
                    # budget-adaptive downgrade to avoid exhausting budget in a few MPVS steps
                    if rem < 2500:
                        horizon = min(horizon, 1)
                        mc = 1
                        refine_calls = 0
                        greedy_topm = 1
                        n_steps_macro = min(n_steps_macro, 1)
                    elif rem < 8000:
                        horizon = min(horizon, 2)
                        mc = min(mc, 1)
                        refine_calls = min(refine_calls, 5)
                        n_steps_macro = min(n_steps_macro, 2)
                    elif rem < 16000:
                        horizon = min(horizon, 3)
                        mc = min(mc, 2)
                        refine_calls = min(refine_calls, 12)

                    best_total_all = float(eval0.get("total_scalar", 0.0))
                    best_assign_all = assign0.copy()
                    best_eval_all = dict(eval0)
                    best_actions_all: List[Dict[str, Any]] = []
                    best_final_assign = assign0.copy()

                    for _mc in range(max(1, mc)):
                        cur = assign0.copy()
                        cur_eval = dict(eval0)
                        actions_exec: List[Dict[str, Any]] = []
                        best_total = float(cur_eval.get("total_scalar", 0.0))
                        best_assign = cur.copy()
                        best_eval = dict(cur_eval)

                        if plan.get("kind") == "macro":
                            b_assign, b_eval, b_total, acts, cur_after = _exec_macro(
                                str(plan.get("name", "macro")),
                                cur,
                                cur_eval,
                                n_steps=n_steps_macro,
                                rng_local=rng_macro,
                            )
                            actions_exec.extend(acts)
                            cur = cur_after
                            cur_eval = _evaluate_assign_cached(cur)
                            if b_total < best_total:
                                best_total, best_assign, best_eval = b_total, b_assign, b_eval
                        else:
                            cand = plan.get("cand", None)
                            act = copy.deepcopy(getattr(cand, "action", {}) or {})
                            act.setdefault("candidate_id", int(getattr(cand, "id", -1)))
                            act.setdefault("type", str(getattr(cand, "type", "")))
                            actions_exec.append(act)
                            _apply_action_inplace_simple(cur, act)
                            cur_eval = _evaluate_assign_cached(cur)
                            v = float(cur_eval.get("total_scalar", best_total))
                            if v < best_total:
                                best_total = v
                                best_assign = cur.copy()
                                best_eval = dict(cur_eval)

                        for _t in range(max(0, horizon - 1)):
                            pool = _build_small_pool(cur, cur_eval, rng_local=rng_verify)
                            if not pool:
                                break
                            pool_sorted = sorted(pool, key=lambda c: float(getattr(c, "est", {}).get("d_total", 0.0)))
                            topm = pool_sorted[: max(1, min(len(pool_sorted), greedy_topm))]
                            chosen = topm[_rng_randint(rng_verify, 0, len(topm))] if len(topm) > 1 else topm[0]
                            act2 = copy.deepcopy(getattr(chosen, "action", {}) or {})
                            act2.setdefault("candidate_id", int(getattr(chosen, "id", -1)))
                            act2.setdefault("type", str(getattr(chosen, "type", "")))
                            actions_exec.append(act2)
                            _apply_action_inplace_simple(cur, act2)
                            cur_eval = _evaluate_assign_cached(cur)
                            v2 = float(cur_eval.get("total_scalar", best_total))
                            if v2 < best_total:
                                best_total = v2
                                best_assign = cur.copy()
                                best_eval = dict(cur_eval)

                        T_ref = max(sa_min_T, 0.2 * float(T))
                        r_assign, r_eval, r_total = _refine_sa(best_assign, best_eval, refine_calls, T_ref=T_ref, py_rng_local=py_rng_verify)
                        if r_total < best_total:
                            best_total, best_assign, best_eval = r_total, r_assign, r_eval

                        if best_total < best_total_all:
                            best_total_all = best_total
                            best_assign_all = best_assign.copy()
                            best_eval_all = dict(best_eval)
                            best_actions_all = list(actions_exec)
                            best_final_assign = cur.copy()

                    return {
                        "verified_best_total": float(best_total_all),
                        "best_assign": best_assign_all,
                        "best_eval": best_eval_all,
                        "actions": best_actions_all,
                        "final_assign": best_final_assign,
                    }

                for step in range(steps):
                    last_step = int(step)
                    # --- BATC trace defaults (always defined) ---
                    mpvs_calls0 = None
                    trig_enabled = False
                    trig_repeat_ratio = 0.0
                    trig_calls_avg = 0.0
                    trig_distress = 0.0
                    trig_allow_macro = False
                    trig_allow_verifier = False
                    if trace_events_path is not None:
                        assert_cfg_sealed_or_violate(
                            cfg=cfg,
                            seal_digest=seal_digest,
                            trace_events_path=trace_events_path,
                            phase="layout_detailed_place",
                            step=int(step) + 1,
                            fatal=True,
                            trace_contract=trace_contract,
                        )
                    step_start = time.perf_counter()
                    # P3: stagnation detection (based on global best)
                    stagnation_steps = int(step - last_best_step)
                    force_explore = False
                    if stagnation_steps >= stagnation_patience and step > force_explore_until:
                        force_explore = True
                        force_explore_until = int(step + kick_cooldown_steps)

                        # P2: reheat on stagnation (with cooldown)
                        if (step - last_reheat_step) >= reheat_cooldown and reheat_count < reheat_max_times:
                            T = max(T, reheat_factor * float(T0_raw))
                            # also keep scale-aware floor if base_total huge
                            if sa_auto_scale and float(eval_out.get("total_scalar", 0.0)) > 1000.0 and T <= 10.0:
                                T = max(T, sa_auto_scale_frac * float(eval_out.get("total_scalar", 0.0)))
                            T = max(T, sa_min_T)
                            last_reheat_step = int(step)
                            reheat_count += 1
                    force_explore_for_llm = False
                    if force_explore and (planner_type in ("llm", "mixed")) and (llm_provider is not None) and (not llm_disabled):
                        # let LLM decide strong move on stagnation; do NOT inject deterministic forced action
                        force_explore_for_llm = True
                    eval_calls_before = int(getattr(evaluator, "evaluator_calls", eval_calls_cum))
                    h0 = eval_cache.hits if eval_cache is not None else 0
                    m0 = eval_cache.misses if eval_cache is not None else 0
                    forced_family = None
                    forced_policy = None
                    if controller is not None:
                        forced_family = controller.choose_action_family()
                        forced_policy = controller.choose_policy()
        
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
                        dp_cfg,
                        rng,
                        debug_out_path=(out_dir / "candidate_pool_debug.json") if step == 0 else None,
                    )
                    _sync_eval_calls()
                    if forced_family:
                        filtered = [
                            c for c in candidate_pool if str(c.action.get("op", "")).lower() == forced_family.lower()
                        ]
                        if filtered:
                            candidate_pool = filtered
                    lookahead_scores: Dict[int, float] = {}
                    if lookahead_enabled and planner_type == "mixed" and candidate_pool:
                        candidate_pool.sort(key=lambda c: float(c.est.get("d_total", 0.0)), reverse=True)
                        top = candidate_pool[: max(1, min(lookahead_k, len(candidate_pool)))]
                        verified: Dict[int, float] = {}
                        for cand in top:
                            try:
                                verified_score = _rollout_avg_total(assign, cand.action, r=lookahead_r, mc=lookahead_mc)
                                verified[int(cand.id)] = float(verified_score)
                            except Exception:
                                continue

                        for cand in candidate_pool:
                            cid = int(cand.id)
                            if cid in verified and isinstance(cand.est, dict):
                                cand.est["verified_total"] = float(verified[cid])
                                cand.est["lookahead_score"] = -lookahead_alpha * float(verified[cid])
                                lookahead_scores[cid] = float(cand.est["lookahead_score"])

                        candidate_pool.sort(key=lambda c: float(c.est.get("verified_total", 1e30)))
                    cand_map = {c.id: c for c in candidate_pool}
                    candidate_ids = [c.id for c in candidate_pool]
                    # forbidden ids (shared by MPVS + legacy LLM pipeline)
                    forbidden_ids = list({pid for recent in forbidden_history[-3:] for pid in recent} | recent_failed_ids)
                    mpvs_exploit_override = False
                    forced_mpvs_policy = ""

                    # ==========================
                    # MPVS (Macro Propose–Verify Search): proposer + verifier decides
                    # ==========================
                    if mpvs_enabled:
                        calls0 = int(getattr(evaluator, "evaluator_calls", 0))
                        mpvs_calls0 = int(calls0)
                        prev_total0 = float(eval_out.get("total_scalar", 0.0))
                        prev_comm0 = float(eval_out.get("comm_norm", 0.0))
                        prev_therm0 = float(eval_out.get("therm_norm", 0.0))
                        mpvs_h0 = int(mpvs_cache.hits) if mpvs_cache is not None else 0
                        mem_cfg = _cfg_get(mpvs_cfg, "memory", {}) or {}
                        mem_enabled = bool(_cfg_get(mem_cfg, "enabled", True))
                        ab_cfg = _cfg_get(mpvs_cfg, "ablation", {}) or {}
                        disable_mem = bool(_cfg_get(ab_cfg, "no_mem", False))
                        if disable_mem:
                            mem_enabled = False
                            mpvs_mem[:] = []
                            try:
                                if mem_bank is not None:
                                    mem_bank.entries = []
                            except Exception:
                                pass
                        mem_ttl = int(_cfg_get(mem_cfg, "ttl_steps", 8))
                        mem_max = int(_cfg_get(mem_cfg, "max_size", 32))
                        if mem_enabled:
                            mpvs_mem[:] = [m for m in mpvs_mem if int(m.get("expire", -1)) >= int(step)]
                            mpvs_mem.sort(key=lambda x: float(x.get("score", 1e30)))
                            mpvs_mem[:] = mpvs_mem[:mem_max]
                        else:
                            mpvs_mem[:] = []

                        prop_cfg = _cfg_get(mpvs_cfg, "proposer", {}) or {}
                        k_heur = int(_cfg_get(prop_cfg, "k_heur", 6))
                        k_llm = int(_cfg_get(prop_cfg, "k_llm", 3))
                        llm_every = int(_cfg_get(prop_cfg, "llm_every_n_steps", 10))

                        stagn = int(step - last_best_step) if "last_best_step" in locals() else int(step)
                        if macro_engine is not None:
                            try:
                                macro_engine.tick()
                            except Exception:
                                pass

                        # MPVS unified multi-component control signals (macro/mem/llm)
                        repeat_ratio_pre = 0.0
                        try:
                            ss0 = (mpvs_trigger_state.get("recent_sigs", []) or [])[-50:]
                            if ss0:
                                repeat_ratio_pre = 1.0 - float(len(set(ss0))) / float(max(1, len(ss0)))
                        except Exception:
                            repeat_ratio_pre = 0.0

                        ctrl_cfg = _cfg_get(mpvs_cfg, "controller", {}) or {}
                        stagn_norm = float(_cfg_get(ctrl_cfg, "stagn_norm", 20.0))
                        rep_hi = float(_cfg_get(ctrl_cfg, "repeat_hi", 0.75))
                        distress_pre = 0.0
                        try:
                            distress_pre = max(distress_pre, float(stagn) / float(max(1.0, stagn_norm)))
                            if float(repeat_ratio_pre) > float(rep_hi):
                                distress_pre = max(distress_pre, (float(repeat_ratio_pre) - float(rep_hi)) / float(max(1e-6, 1.0 - float(rep_hi))))
                            distress_pre = float(min(1.0, max(0.0, distress_pre)))
                        except Exception:
                            distress_pre = 0.0

                        roi_macro = float(roi_tracker.roi("macro", 0.0)) if roi_tracker is not None else 0.0
                        roi_mem = float(roi_tracker.roi("mem", 0.0)) if roi_tracker is not None else 0.0
                        roi_llm = float(roi_tracker.roi("llm", 0.0)) if roi_tracker is not None else 0.0

                        if mpvs_ctrl is not None:
                            try:
                                mpvs_ctrl.tick(int(step))
                            except Exception:
                                pass
                            try:
                                mpvs_ctrl.on_progress(int(eval_calls_cum), int(budget_total_calls), float(best_total_seen))
                            except Exception:
                                pass
                        blocked_ratio_pre = float(mpvs_stats.get("blocked_heuristic", 0)) / float(max(1, int(mpvs_stats.get("steps_mpvs", 0))))
                        step_ctx = {
                            "progress": float(eval_calls_cum) / float(max(1, int(budget_total_calls))) if int(budget_total_calls) > 0 else 0.0,
                            "stage": "mid",
                            "stagn_bucket": 0,
                            "health_bucket": 0,
                            "ctx_key": "mid|stg0|hlth0",
                        }
                        if mpvs_ctrl is not None:
                            try:
                                step_ctx = mpvs_ctrl.build_context(
                                    stagn=int(stagn),
                                    repeat_ratio=float(repeat_ratio_pre),
                                    blocked_ratio=float(blocked_ratio_pre),
                                    used_calls=int(eval_calls_cum),
                                    budget_total=int(budget_total_calls),
                                )
                            except Exception:
                                pass

                        explore_cfg = _cfg_get(mpvs_cfg, "explore", {}) or {}
                        explore_every = int(_cfg_get(explore_cfg, "every_n_steps", 10))
                        explore_stagn = int(_cfg_get(explore_cfg, "when_stagnation_ge", 12))
                        warmup_steps = int(_cfg_get(explore_cfg, "warmup_steps", 30))
                        explore_max_consecutive = int(_cfg_get(explore_cfg, "max_consecutive_steps", 2))
                        explore_mode = str(_cfg_get(explore_cfg, "mode", "hybrid")).lower()

                        do_explore = False
                        if explore_mode == "always":
                            do_explore = True
                        else:
                            if stagn >= explore_stagn:
                                do_explore = True
                            elif (int(step) >= warmup_steps) and (int(step) % max(1, explore_every) == 0):
                                do_explore = True

                        if do_explore and mpvs_explore_left <= 0:
                            mpvs_explore_left = explore_max_consecutive
                        do_explore = do_explore and (mpvs_explore_left > 0)

                        mpvs_exploit_override = bool(mpvs_enabled and (not do_explore))
                        if mpvs_exploit_override:
                            # Exploit phase: disable legacy LLM/queue and clear any leftover queued actions
                            action_queue = []
                            forced_mpvs_policy = "heuristic"
                        else:
                            forced_mpvs_policy = ""

                        if do_explore:
                            # LLM proposer is expensive; MPVS controls it strictly.
                            use_llm_now = False
                            llm_deny_reason = ""
                            if (planner_type in ("llm", "mixed")) and (llm_provider is not None) and (not llm_disabled) and (k_llm > 0):
                                sched_ok = (int(step) % max(1, int(llm_every)) == 0)
                                if mpvs_ctrl is not None:
                                    ok, rr = mpvs_ctrl.allow(
                                        "llm",
                                        step=int(step),
                                        stagn=int(stagn),
                                        distress=float(distress_pre),
                                        repeat_ratio=float(repeat_ratio_pre),
                                        roi=float(roi_llm),
                                        used_calls=int(eval_calls_cum),
                                        budget_total=int(budget_total_calls),
                                        ctx=step_ctx,
                                    )
                                    use_llm_now = bool(ok and sched_ok)
                                    llm_deny_reason = str(rr or "")
                                    if use_llm_now:
                                        mpvs_stats["llm_allowed"] = int(mpvs_stats.get("llm_allowed", 0)) + 1
                                        try:
                                            mpvs_ctrl.fired("llm", step=int(step))
                                        except Exception:
                                            pass
                                    else:
                                        mpvs_stats["llm_denied"] = int(mpvs_stats.get("llm_denied", 0)) + 1
                                        if llm_deny_reason:
                                            d = mpvs_stats.get("llm_deny_reason", {}) or {}
                                            d[llm_deny_reason] = int(d.get(llm_deny_reason, 0)) + 1
                                            mpvs_stats["llm_deny_reason"] = d
                                else:
                                    use_llm_now = bool(sched_ok)

                            forbidden_set = set(forbidden_ids)
                            cand_sorted = sorted([c for c in candidate_pool if int(c.id) not in forbidden_set], key=lambda c: float(c.est.get("d_total", 0.0)))
                            heur_cands = cand_sorted[: max(0, min(k_heur, len(cand_sorted)))]

                            llm_cands: List[Any] = []
                            llm_pick_ids: List[int] = []
                            llm_macro_plans: List[Dict[str, Any]] = []
                            direction_bias: Optional[str] = None
                            allow_dir_bias = bool(_cfg_get(prop_cfg, "llm_allow_direction_bias", False))
                            # Virtual candidates (LLM can pick these IDs to propose macro/direction)
                            # Macro IDs
                            V_MACRO = {
                                900001: "therm",
                                900002: "comm",
                                900003: "escape",
                                900004: "cluster",
                            }
                            # Direction IDs (optional, used as a "bias" tag in verifier scoring)
                            V_DIR = {
                                900101: "bias_comm",
                                900102: "bias_therm",
                            }
                            virtual_candidates = []
                            for vid, name in V_MACRO.items():
                                virtual_candidates.append(
                                    {
                                        "id": int(vid),
                                        "type": "macro",
                                        "signature": f"V:macro:{name}",
                                        "d_total": 0.0,
                                        "d_comm": 0.0,
                                        "d_therm": 0.0,
                                        "op": "macro",
                                        "op_args": {"macro_name": name},
                                    }
                                )
                            for vid, name in V_DIR.items():
                                virtual_candidates.append(
                                    {
                                        "id": int(vid),
                                        "type": "direction",
                                        "signature": f"V:dir:{name}",
                                        "d_total": 0.0,
                                        "d_comm": 0.0,
                                        "d_therm": 0.0,
                                        "op": "direction",
                                        "op_args": {"direction": name},
                                    }
                                )
                            if use_llm_now:
                                ss = build_state_summary(
                                    step=int(step),
                                    T=float(T),
                                    eval_out=eval_out,
                                    traffic_sym=traffic_sym,
                                    assign=assign,
                                    site_to_region=site_to_region,
                                    chip_tdp=chip_tdp,
                                    clusters=clusters,
                                    regions=regions,
                                    candidates=candidate_pool,
                                    candidate_ids=candidate_ids,
                                    forbidden_ids=forbidden_ids,
                                    k_actions=max(1, k_llm),
                                    virtual_candidates=virtual_candidates,
                                )
                                try:
                                    llm_pick_ids = llm_provider.propose_pick(ss, max(1, k_llm)) or []
                                    llm_pick_ids = [int(pid) for pid in llm_pick_ids if int(pid) not in forbidden_set]
                                except Exception:
                                    llm_pick_ids = []
                                if usage_fp is not None:
                                    mpvs_stats["llm_calls"] += 1
                                    # llm_provider may expose last_usage
                                    usage = getattr(llm_provider, "last_usage", None)
                                    rec = {
                                        "event": "mpvs_llm_call",
                                        "step": int(step),
                                        "k": int(max(1, k_llm)),
                                        "picked": [int(x) for x in llm_pick_ids],
                                        "picked_virtual": [int(x) for x in llm_pick_ids if int(x) >= 900000],
                                        "ok": 1 if llm_pick_ids else 0,
                                        "usage": usage,
                                    }
                                    usage_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                                    usage_fp.flush()
                                    if llm_pick_ids:
                                        mpvs_stats["llm_ok"] += 1
                                for pid in llm_pick_ids:
                                    pid = int(pid)
                                    if pid in V_MACRO:
                                        llm_macro_plans.append({"kind": "macro", "name": str(V_MACRO[pid]), "src": "llm_macro"})
                                        continue
                                    if pid in V_DIR:
                                        if allow_dir_bias:
                                            direction_bias = str(V_DIR[pid])
                                        else:
                                            mpvs_stats["llm_dir_ignored"] = int(mpvs_stats.get("llm_dir_ignored", 0)) + 1
                                        continue
                                    cc = cand_map.get(pid)
                                    if cc is not None:
                                        llm_cands.append(cc)

                            macro_cfg = _cfg_get(mpvs_cfg, "macros", {}) or {}
                            macro_enabled = bool(_cfg_get(macro_cfg, "enabled", True))

                            # MPVS controller can suppress macro proposing when ROI is persistently low.
                            allow_macro_src = True
                            if mpvs_ctrl is not None:
                                ok, _rr = mpvs_ctrl.allow(
                                    "macro",
                                    step=int(step),
                                    stagn=int(stagn),
                                    distress=float(distress_pre),
                                    repeat_ratio=float(repeat_ratio_pre),
                                    roi=float(roi_macro),
                                    used_calls=int(eval_calls_cum),
                                    budget_total=int(budget_total_calls),
                                    ctx=step_ctx,
                                )
                                allow_macro_src = bool(ok)
                            macro_enabled = bool(macro_enabled and allow_macro_src)
                            base_macros = list(_cfg_get(macro_cfg, "base", ["therm", "comm"]) or [])
                            extra_macros = list(_cfg_get(macro_cfg, "extra_on_stagnation", ["escape", "cluster"]) or [])

                            base_stagn_th = int(_cfg_get(macro_cfg, "enable_base_when_stagnation_ge", 5))
                            extra_stagn_th = int(_cfg_get(macro_cfg, "enable_extra_when_stagnation_ge", 25))

                            macro_names = []
                            if macro_enabled:
                                if stagn >= base_stagn_th:
                                    macro_names.extend([str(x) for x in base_macros])
                                if stagn >= extra_stagn_th:
                                    macro_names.extend([str(x) for x in extra_macros])

                            macro_max = int(_cfg_get(macro_cfg, "max_macros", 3))
                            macro_names = macro_names[: max(0, min(len(macro_names), macro_max))]
                            if macro_engine is not None and macro_names:
                                try:
                                    macro_names = [m for m in macro_engine.rank_macros(list(macro_names)) if macro_engine.available(m)]
                                except Exception:
                                    pass

                            # v2.1: prioritize active macro families (candidate/released) for this ctx_key.
                            # This avoids candidate being ineffective when allow/quota are called before a family is chosen.
                            try:
                                if mpvs_ctrl is not None and macro_enabled:
                                    _ctx_key = str((step_ctx or {}).get("ctx_key", "") or "")
                                    _stage = str((step_ctx or {}).get("stage", "") or "")
                                    active_fams = []
                                    if _ctx_key:
                                        active_fams = list(mpvs_ctrl.get_active_families(ctx_key=_ctx_key, stage=_stage) or [])
                                    if active_fams:
                                        merged = []
                                        seen = set()
                                        for _f in active_fams:
                                            _f = str(_f or "")
                                            if not _f or _f in seen:
                                                continue
                                            seen.add(_f)
                                            merged.append(_f)
                                        for _m in (macro_names or []):
                                            _m = str(_m or "")
                                            if not _m or _m in seen:
                                                continue
                                            seen.add(_m)
                                            merged.append(_m)
                                        macro_names = merged

                                        # Filter availability again in case active_fams introduced new macros.
                                        if macro_engine is not None and macro_names:
                                            macro_names = [m for m in list(macro_names) if macro_engine.available(m)]
                            except Exception:
                                pass

                            # Enforce cap after priority merge
                            macro_names = list(macro_names or [])
                            macro_names = macro_names[: max(0, min(len(macro_names), macro_max))]

                            try:
                                if mpvs_ctrl is not None and macro_names:
                                    mpvs_ctrl.fired("macro", step=int(step))
                            except Exception:
                                pass

                            # --- plan groups (with src tags) ---
                            heur_plans = [{"kind":"atomic","id":int(c.id),"cand":c,"src":"heuristic"} for c in heur_cands]
                            llm_atomic_plans = [{"kind":"atomic","id":int(c.id),"cand":c,"src":"llm_atomic"} for c in llm_cands]
                            macro_plans = [{"kind":"macro","name":str(mn),"src":"macro"} for mn in macro_names]
                            # llm_macro_plans already built above when use_llm_now
                            if "llm_macro_plans" not in locals():
                                llm_macro_plans = []

                            # LLM shadow mode: keep calling LLM for diagnostics, but do NOT let it compete.
                            try:
                                llm_shadow = bool(mpvs_ctrl is not None and bool(getattr(mpvs_ctrl, "llm_shadow_mode", False)))
                            except Exception:
                                llm_shadow = False
                            if llm_shadow and (llm_atomic_plans or llm_macro_plans):
                                mpvs_stats["llm_shadow_plans"] = int(mpvs_stats.get("llm_shadow_plans", 0)) + int(len(llm_atomic_plans) + len(llm_macro_plans))
                                if mpvs_ctrl is not None:
                                    for _pl in llm_atomic_plans:
                                        try:
                                            _c = _pl.get("cand", None)
                                            if _c is None:
                                                continue
                                            d0 = float((_c.est or {}).get("d_total", 0.0))
                                            est_gain = float(max(0.0, -d0))
                                            mpvs_ctrl.llm_shadow_observe(float(est_gain))
                                        except Exception:
                                            continue
                                llm_atomic_plans = []
                                llm_macro_plans = []

                            # MemoryBank v1
                            mem_plans = []
                            mem_stagn_th = int(_cfg_get(mem_cfg, "enable_when_stagnation_ge", 8))
                            mem_gate_cfg = _cfg_get(mem_cfg, "global_gate", {}) or {}
                            mem_gate_enabled = bool(_cfg_get(mem_gate_cfg, "enabled", False))
                            allow_mem_src = True
                            if mpvs_ctrl is not None:
                                ok, rr = mpvs_ctrl.allow(
                                    "mem",
                                    step=int(step),
                                    stagn=int(stagn),
                                    distress=float(distress_pre),
                                    repeat_ratio=float(repeat_ratio_pre),
                                    roi=float(roi_mem),
                                    used_calls=int(eval_calls_cum),
                                    budget_total=int(budget_total_calls),
                                )
                                allow_mem_src = bool(ok)
                                if str(rr or "") == "mem_global_cooldown":
                                    mpvs_stats["mem_global_blocked"] = int(mpvs_stats.get("mem_global_blocked", 0)) + 1

                            if mem_gate_enabled and int(step) < int(mpvs_mem_global_until):
                                mpvs_stats["mem_global_blocked"] = int(mpvs_stats.get("mem_global_blocked", 0)) + 1
                                allow_mem_src = False

                            if mem_enabled and mem_bank is not None and allow_mem_src:
                                try:
                                    mem_bank.tick(int(step))
                                    mpvs_stats["mem_entries"] = int(len(getattr(mem_bank, "entries", []) or []))
                                except Exception:
                                    pass
                                if int(stagn) >= int(mem_stagn_th):
                                    mem_prop_every = int(_cfg_get(mem_cfg, "propose_every_n_steps", 10))
                                    if mem_prop_every <= 1 or (int(step) % int(mem_prop_every) == 0):
                                        mem_topk = int(_cfg_get(mem_cfg, "propose_topk", 6))
                                        mpvs_stats["mem_queries"] = int(mpvs_stats.get("mem_queries", 0)) + 1
                                        hits = []
                                        try:
                                            bud_prog = float(eval_calls_cum) / float(max(1, int(budget_total_calls))) if int(budget_total_calls) > 0 else 0.0
                                            hits = mem_bank.query(
                                                assign,
                                                site_to_region,
                                                step=int(step),
                                                topk=max(0, mem_topk),
                                                budget_progress=float(bud_prog),
                                                repeat_ratio=float(repeat_ratio_pre),
                                                blocked_ratio=float(blocked_ratio_pre),
                                            )
                                        except Exception:
                                            hits = []
                                        if hits:
                                            mpvs_stats["mem_hits"] = int(mpvs_stats.get("mem_hits", 0)) + 1
                                        for ent, sc in hits:
                                            try:
                                                acts = [copy.deepcopy(a) for a in (ent.actions or [])]
                                                for a in acts:
                                                    a.pop("candidate_id", None)
                                                    a.pop("signature", None)
                                                    a.pop("_src", None)
                                                if not acts:
                                                    continue

                                                # 0-call analytic prefilter (only for swap/relocate sequences)
                                                mem_pref_en = bool(_cfg_get(mem_cfg, "prefilter_enabled", True))
                                                mem_pref_min_gain = float(_cfg_get(mem_cfg, "prefilter_min_gain", 0.0001))
                                                if mem_pref_en and (mpvs_obj_params is not None):
                                                    try:
                                                        chip_w = chip_tdp if chip_tdp is not None else np.zeros(int(assign.shape[0]), dtype=float)
                                                        est = estimate_action_seq_delta(assign, acts, sites_xy, traffic_sym, chip_w, mpvs_obj_params)
                                                        if float(est.get("supported", 0.0)) > 0.0:
                                                            if float(est.get("d_total", 0.0)) > -float(mem_pref_min_gain):
                                                                mpvs_stats["mem_prefilter_drop"] = int(mpvs_stats.get("mem_prefilter_drop", 0)) + 1
                                                                continue
                                                    except Exception:
                                                        pass
                                                mem_plans.append(
                                                    {
                                                        "kind": "atomic",
                                                        "name": "",
                                                        "id": -100000 - int(getattr(ent, "mid", -1)),
                                                        "src": "mem",
                                                        "action_seq": acts,
                                                        "mem_id": int(getattr(ent, "mid", -1)),
                                                        "mem_score": float(sc),
                                                    }
                                                )
                                            except Exception:
                                                continue

                            # --- quota config ---
                            quota_cfg = _cfg_get(prop_cfg, "quota", {}) or {}
                            q_heur = int(quota_cfg.get("heur", k_heur))
                            q_llm_atomic = int(quota_cfg.get("llm_atomic", 2))
                            q_llm_macro = int(quota_cfg.get("llm_macro", 2))
                            q_macro = int(quota_cfg.get("macro", 2))
                            q_mem = int(quota_cfg.get("mem", 1))

                            # dynamic quota adjustment by MPVS controller (ROI-aware)
                            try:
                                if mpvs_ctrl is not None:
                                    q_mem = int(mpvs_ctrl.quota("mem", int(q_mem), roi=float(roi_mem), used_calls=int(eval_calls_cum), budget_total=int(budget_total_calls), ctx=step_ctx))
                                    q_llm_atomic = int(mpvs_ctrl.quota("llm", int(q_llm_atomic), roi=float(roi_llm), used_calls=int(eval_calls_cum), budget_total=int(budget_total_calls), ctx=step_ctx))
                                    q_llm_macro = int(mpvs_ctrl.quota("llm", int(q_llm_macro), roi=float(roi_llm), used_calls=int(eval_calls_cum), budget_total=int(budget_total_calls), ctx=step_ctx))
                                    q_macro = int(mpvs_ctrl.quota("macro", int(q_macro), roi=float(roi_macro), used_calls=int(eval_calls_cum), budget_total=int(budget_total_calls), ctx=step_ctx, family=""))
                            except Exception:
                                pass

                            heur_plans = heur_plans[:max(0, q_heur)]
                            llm_atomic_plans = llm_atomic_plans[:max(0, q_llm_atomic)]
                            llm_macro_plans = llm_macro_plans[:max(0, q_llm_macro)]
                            macro_plans = macro_plans[:max(0, q_macro)]
                            mem_plans = mem_plans[:max(0, q_mem)]

                            # --- interleave plans: ensure each source appears at least once when available ---
                            def _interleave(groups):
                                out = []
                                # first pass: one per group
                                for g in groups:
                                    if g:
                                        out.append(g.pop(0))
                                # round-robin
                                while any(groups):
                                    for g in groups:
                                        if g:
                                            out.append(g.pop(0))
                                return out

                            plans = _interleave([mem_plans, llm_macro_plans, macro_plans, llm_atomic_plans, heur_plans])
                            # de-dup by stable key
                            seen_plan = set()
                            uniq = []
                            for pl in plans:
                                k = f"A:{int(pl.get('id',-1))}" if pl.get("kind")=="atomic" else f"M:{pl.get('src','')}:{pl.get('name','')}"
                                if k in seen_plan:
                                    continue
                                seen_plan.add(k)
                                uniq.append(pl)
                            plans = uniq

                            ab_cfg = _cfg_get(mpvs_cfg, "ablation", {}) or {}
                            disable_verifier = bool(_cfg_get(ab_cfg, "no_verifier", False))
                            disable_macro = bool(_cfg_get(ab_cfg, "no_macro", False))
                            disable_llm = bool(_cfg_get(ab_cfg, "no_llm", False))
                            if disable_llm:
                                plans = [p for p in plans if not str(p.get("src", "")).startswith("llm")]
                            if disable_macro:
                                plans = [p for p in plans if p.get("kind") != "macro"]

                            # normalize plans:
                            # - atomic: accept either cand (from current candidate_pool) OR a replayable "action" (memory)
                            norm_plans = []
                            for pl in plans:
                                if pl.get("kind") == "atomic":
                                    if pl.get("cand") is None:
                                        cid = int(pl.get("id", -1))
                                        if cid >= 0:
                                            pl["cand"] = cand_map.get(cid)
                                    if pl.get("cand") is None and pl.get("action") is None and pl.get("action_seq") is None:
                                        continue
                                norm_plans.append(pl)
                            plans = norm_plans

                            ver_cfg = _cfg_get(mpvs_cfg, "verifier", {}) or {}
                            ver_enable_stagn = int(_cfg_get(ver_cfg, "enable_when_stagnation_ge", 0))
                            disable_verifier_step = bool(disable_verifier) or (int(stagn) < int(ver_enable_stagn))
                            # =========================================================
                            # BATC v0.5: Budget-Aware Trigger Controller (macro/verifier)
                            # =========================================================
                            trig_cfg = _cfg_get(mpvs_cfg, "trigger", {}) or {}
                            trig_enabled = bool(_cfg_get(trig_cfg, "enabled", False))
                            if trig_enabled:
                                mpvs_stats["trig_enabled"] = 1
                                mpvs_stats["trig_steps"] = int(mpvs_stats.get("trig_steps", 0)) + 1

                                W = int(_cfg_get(trig_cfg, "window", 30))
                                W = max(10, min(W, 200))

                                # recent signature diversity -> repeat ratio
                                sigs = (mpvs_trigger_state.get("recent_sigs", []) or [])[-W:]
                                if len(sigs) >= 2:
                                    uniq = len(set(sigs))
                                    trig_repeat_ratio = max(0.0, 1.0 - float(uniq) / float(len(sigs)))
                                else:
                                    trig_repeat_ratio = 0.0

                                # recent calls/iter (from previous steps)
                                calls_hist = (mpvs_trigger_state.get("recent_calls", []) or [])[-W:]
                                trig_calls_avg = float(sum(calls_hist)) / float(len(calls_hist)) if calls_hist else 0.0

                                mpvs_stats["trig_repeat_ratio_sum"] = float(mpvs_stats.get("trig_repeat_ratio_sum", 0.0)) + float(trig_repeat_ratio)
                                mpvs_stats["trig_calls_avg_sum"] = float(mpvs_stats.get("trig_calls_avg_sum", 0.0)) + float(trig_calls_avg)

                                repeat_mode = str(_cfg_get(trig_cfg, "repeat_ratio_high_mode", "fixed")).lower()
                                repeat_high = float(_cfg_get(trig_cfg, "repeat_ratio_high", 0.55))
                                if repeat_mode in {"quantile", "q"}:
                                    q = float(_cfg_get(trig_cfg, "repeat_ratio_high_quantile", 0.80))
                                    q = max(0.5, min(0.95, q))
                                    hist_rr = (mpvs_trigger_state.get("recent_repeat", []) or [])[-max(20, W * 3):]
                                    if len(hist_rr) >= int(_cfg_get(trig_cfg, "repeat_ratio_min_samples", 20)):
                                        try:
                                            rr = np.array(hist_rr, dtype=np.float64)
                                            repeat_high = float(np.quantile(rr, q))
                                        except Exception:
                                            pass
                                    repeat_floor = float(_cfg_get(trig_cfg, "repeat_ratio_high_floor", 0.40))
                                    repeat_cap = float(_cfg_get(trig_cfg, "repeat_ratio_high_cap", 0.90))
                                    repeat_high = max(repeat_floor, min(repeat_cap, float(repeat_high)))

                                calls_mode = str(_cfg_get(trig_cfg, "calls_per_iter_high_mode", "fixed")).lower()
                                calls_high = float(_cfg_get(trig_cfg, "calls_per_iter_high", 180.0))
                                baseline_calls = None
                                if calls_mode in {"relative", "rel"}:
                                    mul = float(_cfg_get(trig_cfg, "calls_per_iter_high_mul", 1.25))
                                    mul = max(1.05, min(3.0, mul))
                                    hist_c = (mpvs_trigger_state.get("recent_calls_cheap", []) or [])[-max(20, W * 3):]
                                    if len(hist_c) >= int(_cfg_get(trig_cfg, "calls_baseline_min_samples", 20)):
                                        try:
                                            baseline_calls = float(np.median(np.array(hist_c, dtype=np.float64)))
                                        except Exception:
                                            baseline_calls = None
                                    if baseline_calls is not None and baseline_calls > 0.0:
                                        calls_high = float(baseline_calls) * float(mul)

                                mpvs_stats["trig_repeat_high_effective"] = float(repeat_high)
                                mpvs_stats["trig_calls_high_effective"] = float(calls_high)
                                mpvs_stats["trig_calls_baseline"] = float(baseline_calls) if baseline_calls is not None else None

                                # Baseline warmup: force both macro/verifier OFF for the first N MPVS steps
                                # so that recent_calls_cheap can be populated and relative calls gate becomes effective.
                                baseline_warmup = int(_cfg_get(trig_cfg, "baseline_warmup_steps", 40))
                                force_baseline = bool(baseline_warmup > 0 and int(step) < int(baseline_warmup))

                                # decay cooldowns
                                try:
                                    for k in ("macro", "verifier"):
                                        cd = int((mpvs_trigger_state.get("cooldown", {}) or {}).get(k, 0))
                                        if cd > 0:
                                            mpvs_trigger_state["cooldown"][k] = cd - 1
                                except Exception:
                                    pass

                                # replenish credits
                                mac_t = _cfg_get(trig_cfg, "macro", {}) or {}
                                ver_t = _cfg_get(trig_cfg, "verifier", {}) or {}

                                mac_add = float(_cfg_get(mac_t, "credit_add", 1.0))
                                mac_max = float(_cfg_get(mac_t, "credit_max", 3.0))
                                ver_add = float(_cfg_get(ver_t, "credit_add", 1.0))
                                ver_max = float(_cfg_get(ver_t, "credit_max", 2.0))

                                mpvs_trigger_state["credit"]["macro"] = min(mac_max, float(mpvs_trigger_state["credit"].get("macro", 0.0)) + mac_add)
                                mpvs_trigger_state["credit"]["verifier"] = min(ver_max, float(mpvs_trigger_state["credit"].get("verifier", 0.0)) + ver_add)

                                # distress (simple, stable): stagn + repeat
                                stag_ref = float(_cfg_get(trig_cfg, "stagn_ref", 20.0))
                                stag_ref = max(1.0, stag_ref)
                                rr_ref = max(1e-6, repeat_high)
                                trig_distress = max(float(stagn) / stag_ref, float(trig_repeat_ratio) / rr_ref)
                                trig_distress = max(0.0, min(1.0, trig_distress))

                                # allow macro?
                                mac_minint = int(_cfg_get(mac_t, "min_interval_steps", 10))
                                mac_cost = float(_cfg_get(mac_t, "credit_cost", 3.0))
                                mac_cd_fail = int(_cfg_get(mac_t, "cooldown_fail", 10))
                                mac_stagn_ge = int(_cfg_get(mac_t, "enable_when_stagnation_ge", int(_cfg_get(ver_cfg, "macro_precheck_stagnation_ge", 20))))

                                last_mac = int((mpvs_trigger_state.get("last_fire", {}) or {}).get("macro", -10000000))
                                cd_mac = int((mpvs_trigger_state.get("cooldown", {}) or {}).get("macro", 0))
                                cred_mac = float((mpvs_trigger_state.get("credit", {}) or {}).get("macro", 0.0))

                                mac_mode = str(_cfg_get(mac_t, "distress_mode", "and")).lower()
                                mac_extra = int(_cfg_get(mac_t, "stagnation_extra", 5))
                                cond_rep = float(trig_repeat_ratio) >= float(repeat_high)
                                cond_st2 = int(stagn) >= int(mac_stagn_ge + mac_extra)
                                distress_ok = (cond_rep and cond_st2) if mac_mode in {"and", "strict"} else (cond_rep or cond_st2)
                                # Calls gate: avoid fixed-tax under eval-call budget
                                calls_gate = bool(_cfg_get(trig_cfg, "calls_gate_enabled", True))
                                calls_ok = (float(trig_calls_avg) <= float(calls_high)) if calls_gate else True
                                trig_allow_macro = (not disable_macro) and (int(stagn) >= mac_stagn_ge) and distress_ok and calls_ok and (
                                    (int(step) - last_mac >= mac_minint) and (cd_mac <= 0) and (cred_mac >= mac_cost)
                                )

                                if trig_allow_macro:
                                    mpvs_stats["trig_macro_allowed"] = int(mpvs_stats.get("trig_macro_allowed", 0)) + 1
                                    mpvs_stats["trig_macro_fired"] = int(mpvs_stats.get("trig_macro_fired", 0)) + 1
                                    mpvs_trigger_state["credit"]["macro"] = max(0.0, cred_mac - mac_cost)
                                    mpvs_trigger_state["last_fire"]["macro"] = int(step)
                                else:
                                    # fail-closed: macro does not enter this step's candidate set
                                    plans = [p for p in plans if p.get("kind") != "macro"]

                                # allow verifier?
                                ver_minint = int(_cfg_get(ver_t, "min_interval_steps", 10))
                                ver_cost = float(_cfg_get(ver_t, "credit_cost", 2.0))
                                ver_cd_fail = int(_cfg_get(ver_t, "cooldown_fail", 6))
                                ver_stagn_ge = int(_cfg_get(ver_t, "enable_when_stagnation_ge", ver_enable_stagn))

                                last_ver = int((mpvs_trigger_state.get("last_fire", {}) or {}).get("verifier", -10000000))
                                cd_ver = int((mpvs_trigger_state.get("cooldown", {}) or {}).get("verifier", 0))
                                cred_ver = float((mpvs_trigger_state.get("credit", {}) or {}).get("verifier", 0.0))

                                trig_allow_verifier = (not disable_verifier_step) and (int(stagn) >= ver_stagn_ge) and (
                                    (float(trig_repeat_ratio) >= repeat_high) or (int(stagn) >= ver_stagn_ge + 8)
                                ) and (int(step) - last_ver >= ver_minint) and (cd_ver <= 0) and (cred_ver >= ver_cost) and (
                                    (trig_calls_avg <= calls_high) or (int(stagn) >= ver_stagn_ge + 20) or (trig_calls_avg <= 0.0)
                                )

                                if force_baseline:
                                    trig_allow_macro = False
                                    trig_allow_verifier = False
                                    mpvs_stats["trig_baseline_warmup_steps"] = int(mpvs_stats.get("trig_baseline_warmup_steps", 0)) + 1

                                if trig_allow_verifier:
                                    mpvs_stats["trig_ver_allowed"] = int(mpvs_stats.get("trig_ver_allowed", 0)) + 1
                                    mpvs_stats["trig_ver_fired"] = int(mpvs_stats.get("trig_ver_fired", 0)) + 1
                                    mpvs_trigger_state["credit"]["verifier"] = max(0.0, cred_ver - ver_cost)
                                    mpvs_trigger_state["last_fire"]["verifier"] = int(step)
                                else:
                                    # verifier disabled this step (prevents fixed tax)
                                    disable_verifier_step = True
                            max_plans = int(_cfg_get(ver_cfg, "max_plans", 12))
                            early_margin = float(_cfg_get(ver_cfg, "early_stop_margin", 0.01))
                            tie_eps = float(_cfg_get(ver_cfg, "tie_eps", 0.002))
                            full_verify_topk = int(_cfg_get(ver_cfg, "full_verify_topk", 3))
                            max_full_verify = int(_cfg_get(ver_cfg, "max_full_verify", 2))
                            if instance_tag == "randw":
                                max_full_verify = int(_cfg_get(ver_cfg, "randw_max_full_verify", max_full_verify))
                                max_full_verify = max(1, min(max_full_verify, 2))

                            max_verify_calls_step = int(_cfg_get(ver_cfg, "max_verify_eval_calls_per_step", 180))
                            if instance_tag == "randw":
                                max_verify_calls_step = int(_cfg_get(ver_cfg, "randw_max_verify_eval_calls_per_step", max_verify_calls_step))

                            fast_macro_steps = int(_cfg_get(ver_cfg, "fast_macro_steps", 1))
                            safe_enabled = bool(_cfg_get(ver_cfg, "safe_enabled", True))
                            safe_eps_early = float(_cfg_get(ver_cfg, "safe_eps_early", 0.0005))
                            safe_eps_late = float(_cfg_get(ver_cfg, "safe_eps_late", 0.002))
                            safe_eps_switch_stagnation = int(_cfg_get(ver_cfg, "safe_eps_switch_stagnation", 15))
                            full_verify_always_include_sources = set(
                                str(x) for x in (_cfg_get(ver_cfg, "full_verify_always_include_sources", ["heuristic"]) or [])
                            )
                            if len(plans) > max_plans:
                                plans = plans[:max_plans]

                            # early-stop gating: must cover enough sources first
                            min_cover = int(_cfg_get(ver_cfg, "min_cover_groups", 3))
                            cover = set()
                            tie_stagn_th = int(_cfg_get(ver_cfg, "tie_explore_when_stagnation_ge", 10))

                            def _src_prio(s: str) -> int:
                                # early: exploit (prefer heuristic), late: explore (prefer mem/llm_macro/macro)
                                if stagn < tie_stagn_th:
                                    if stagn < 10:
                                        return {"heuristic": 5, "llm_atomic": 4, "macro": 2, "llm_macro": 1, "mem": 0}.get(str(s), 0)
                                    return {"heuristic": 5, "llm_atomic": 4, "macro": 3, "llm_macro": 2, "mem": 1}.get(str(s), 0)
                                return {"mem": 5, "llm_macro": 4, "macro": 3, "llm_atomic": 2, "heuristic": 1}.get(str(s), 0)

                            def _plan_prio(pl: Dict[str, Any]) -> float:
                                p = float(_src_prio(pl.get("src", "")))
                                if direction_bias and pl.get("src") in {"llm_atomic", "llm_macro"}:
                                    p += 0.25
                                return p

                            bias_beta = float(_cfg_get(ver_cfg, "direction_bias_beta", 0.02))
                            best_plan = None
                            best_v = None
                            best_res = None
                            best_v_raw = None
                            cur_total = float(eval_out.get("total_scalar", 0.0))
                            # v2.1 IMPORTANT: keep a stable scalar baseline from *before* winner application.
                            # Sponsored macro tickets must use this pre-apply baseline to realize long-horizon ROI.
                            try:
                                best_total_before_step = float(cur_total)
                            except Exception:
                                try:
                                    best_total_before_step = float(best_total_seen)
                                except Exception:
                                    best_total_before_step = None
                            per_plan: List[Dict[str, Any]] = []

                            def _eval_dict_from_cand_est(est: Dict[str, Any]) -> Dict[str, Any]:
                                # cand.est comes from evaluator already; map it back to eval_out-like keys
                                return {
                                    "total_scalar": float(est.get("total_new", 1e30)),
                                    "comm_norm": float(est.get("comm_new", 0.0)),
                                    "therm_norm": float(est.get("therm_new", 0.0)),
                                    "penalty": copy.deepcopy(est.get("penalty", {}) or {}),
                                }

                            def _atomic_res_from_cand(cand_obj: Any, assign0: np.ndarray) -> Dict[str, Any]:
                                act = copy.deepcopy(getattr(cand_obj, "action", {}) or {})
                                act.setdefault("candidate_id", int(getattr(cand_obj, "id", -1)))
                                act.setdefault("type", str(getattr(cand_obj, "type", "")))
                                if "signature" not in act or not act.get("signature"):
                                    try:
                                        act["signature"] = _signature_for_action(act, assign0)
                                    except Exception:
                                        pass
                                new_assign = assign0.copy()
                                _apply_action_inplace_simple(new_assign, act)
                                est = dict(getattr(cand_obj, "est", {}) or {})
                                best_eval_local = _eval_dict_from_cand_est(est)
                                return {
                                    "verified_best_total": float(best_eval_local.get("total_scalar", 1e30)),
                                    "best_assign": new_assign,
                                    "best_eval": best_eval_local,
                                    "actions": [act],
                                }

                            def _atomic_res_from_action(action_obj: Dict[str, Any], assign0: np.ndarray, eval0: Dict[str, Any]) -> Dict[str, Any]:
                                # used for memory-replay atomic plan; must evaluate once (cannot reuse cand.est)
                                act = copy.deepcopy(action_obj or {})
                                act.setdefault("type", str(act.get("type", "")))
                                if "signature" not in act or not act.get("signature"):
                                    try:
                                        act["signature"] = _signature_for_action(act, assign0)
                                    except Exception:
                                        pass
                                new_assign = assign0.copy()
                                _apply_action_inplace_simple(new_assign, act)
                                new_eval = _evaluate_assign_cached(new_assign)
                                return {
                                    "verified_best_total": float(new_eval.get("total_scalar", 1e30)),
                                    "best_assign": new_assign,
                                    "best_eval": dict(new_eval),
                                    "actions": [act],
                                }

                            def _atomic_res_from_action_seq(actions: List[Dict[str, Any]], assign0: np.ndarray, eval0: Dict[str, Any]) -> Dict[str, Any]:
                                # exactly ONE evaluator call at the end (cheap-verify)
                                acts: List[Dict[str, Any]] = []
                                new_assign = assign0.copy()
                                for a0 in (actions or []):
                                    act = copy.deepcopy(a0 or {})
                                    act.setdefault("type", str(act.get("type", "")))
                                    if "signature" not in act or not act.get("signature"):
                                        try:
                                            act["signature"] = _signature_for_action(act, new_assign)
                                        except Exception:
                                            pass
                                    _apply_action_inplace_simple(new_assign, act)
                                    acts.append(act)
                                new_eval = _evaluate_assign_cached(new_assign)
                                return {
                                    "verified_best_total": float(new_eval.get("total_scalar", 1e30)),
                                    "best_assign": new_assign,
                                    "best_eval": dict(new_eval),
                                    "actions": acts,
                                }

                            def _cheap_score(pl: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
                                try:
                                    src = str(pl.get("src", ""))
                                    # memory: verifier-mode must NOT trust historical mem_score for competition
                                    if src == "mem" and pl.get("mem_score") is not None:
                                        if disable_verifier_step:
                                            return float(pl.get("mem_score")), {"mode": "mem_score"}
                                        return 1e30, {"mode": "mem_score_disabled"}
                                    # macro: keep neutral cheap (but unverified non-heuristic will be fail-closed later)
                                    if pl.get("kind") == "macro":
                                        return float(eval_out.get("total_scalar", 0.0)), {"mode": "macro_neutral"}
                                    cand = pl.get("cand", None)
                                    if cand is not None:
                                        return float(getattr(cand, "est", {}).get("total_new", 1e30)), {"mode": "atomic_est"}
                                    if pl.get("action_seq") is not None:
                                        if disable_verifier_step:
                                            return float(eval_out.get("total_scalar", 0.0)), {"mode": "atomic_seq_neutral"}
                                        return 1e30, {"mode": "atomic_seq_needs_verify"}
                                    if pl.get("action") is not None:
                                        # action-only plans (memory replay) must be verified before competing in verifier-mode
                                        if disable_verifier_step:
                                            return float(eval_out.get("total_scalar", 0.0)), {"mode": "atomic_action_neutral"}
                                        return 1e30, {"mode": "atomic_action_needs_verify"}
                                    return 1e30, {"mode": "atomic_missing"}
                                except Exception:
                                    return 1e30, {"mode": "error"}

                            def _macro_min_gain_cur() -> float:
                                base = float(_cfg_get(ver_cfg, "macro_min_gain", 0.0))
                                if instance_tag == "randw":
                                    base = float(_cfg_get(ver_cfg, "macro_min_gain_randw", base))
                                mode = str(_cfg_get(ver_cfg, "macro_min_gain_mode", "fixed")).lower()
                                if mode in {"dynamic", "auto"}:
                                    win = int(_cfg_get(ver_cfg, "macro_min_gain_dynamic_window", 200))
                                    q = float(_cfg_get(ver_cfg, "macro_min_gain_dynamic_quantile", 0.30))
                                    q = max(0.05, min(0.95, q))
                                    scale = float(_cfg_get(ver_cfg, "macro_min_gain_dynamic_scale", 1.0))
                                    floor = float(_cfg_get(ver_cfg, "macro_min_gain_dynamic_floor", 0.0))
                                    cap = float(_cfg_get(ver_cfg, "macro_min_gain_dynamic_cap", 0.02))
                                    hist = (mpvs_trigger_state.get("recent_improve", []) or [])[-max(0, win):]
                                    if len(hist) >= int(_cfg_get(ver_cfg, "macro_min_gain_dynamic_min_samples", 10)):
                                        try:
                                            arr = np.array(hist, dtype=np.float64)
                                            dyn = float(np.quantile(arr, q)) * float(scale)
                                            dyn = max(floor, min(cap, dyn))
                                            base = max(base, dyn)
                                        except Exception:
                                            pass
                                return max(0.0, float(base))

                            # --- macro fast precheck: prevent macro/llm_macro from consuming verify budget unless it can beat heuristic ---
                            try:
                                macro_gate_enabled = bool(_cfg_get(ver_cfg, "macro_gain_gate_enabled", True))
                                macro_precheck_every = int(_cfg_get(ver_cfg, "macro_precheck_every_n_steps", 10))
                                macro_enable_stagn = int(_cfg_get(macro_cfg, "enable_base_when_stagnation_ge", 0))
                                macro_precheck_stagn = int(_cfg_get(ver_cfg, "macro_precheck_stagnation_ge", macro_enable_stagn))
                                do_macro_precheck = bool(trig_allow_macro) and (int(stagn) >= int(macro_precheck_stagn)) and (
                                    macro_precheck_every <= 1 or (int(step) % int(macro_precheck_every) == 0)
                                )
                                if macro_gate_enabled and (not disable_macro) and int(fast_macro_steps) > 0 and do_macro_precheck:
                                    # best heuristic cheap baseline (lower is better)
                                    best_heur_est = None
                                    for _pl in plans:
                                        if str(_pl.get("src", "")) != "heuristic" or _pl.get("kind") != "atomic":
                                            continue
                                        _cand = _pl.get("cand", None)
                                        if _cand is None:
                                            continue
                                        _v = float(getattr(_cand, "est", {}).get("total_new", 1e30))
                                        if best_heur_est is None or _v < best_heur_est:
                                            best_heur_est = _v
                                    if best_heur_est is None:
                                        best_heur_est = float(eval_out.get("total_scalar", 0.0))

                                    gain_sw = int(_cfg_get(ver_cfg, "macro_gain_switch_stagnation", 20))
                                    gain_early = float(_cfg_get(ver_cfg, "macro_gain_margin_early", 0.001))
                                    gain_late = float(_cfg_get(ver_cfg, "macro_gain_margin_late", 0.0002))
                                    req = gain_early if int(stagn) < gain_sw else gain_late
                                    if instance_tag == "randw":
                                        req *= float(_cfg_get(ver_cfg, "macro_gain_randw_scale", 1.5))

                                    kept_plans = []
                                    for _pl in plans:
                                        if _pl.get("kind") == "macro" and str(_pl.get("src", "")) in {"macro", "llm_macro"}:
                                            try:
                                                nm = str(_pl.get("name", "macro"))
                                                n_steps_macro = int(_cfg_get(macro_cfg, "n_steps", 3))
                                                n_fast = max(1, min(int(n_steps_macro), int(fast_macro_steps)))
                                                _c0 = int(getattr(evaluator, "evaluator_calls", 0))
                                                b_assign, b_eval, b_total, acts, _cur = _exec_macro(
                                                    nm, assign.copy(), dict(eval_out), n_steps=n_fast, rng_local=rng_macro
                                                )
                                                _c1 = int(getattr(evaluator, "evaluator_calls", 0))
                                                _spent = max(0, int(_c1) - int(_c0))
                                                try:
                                                    cb = mpvs_stats.setdefault("calls_by_src", {})
                                                    cb["macro_precheck"] = int(cb.get("macro_precheck", 0)) + int(_spent)
                                                except Exception:
                                                    pass
                                                _pl["_macro_fast_total"] = float(b_total)
                                                _pl["_macro_fast_res"] = {
                                                    "verified_best_total": float(b_total),
                                                    "best_assign": b_assign,
                                                    "best_eval": b_eval,
                                                    "actions": acts,
                                                }
                                                min_gain = float(_macro_min_gain_cur())
                                                cur_total = float(eval_out.get("total_scalar", 0.0))
                                                margin_heur = float(best_heur_est) - float(b_total)
                                                margin_cur = float(cur_total) - float(b_total)
                                                pass_heur = bool(float(b_total) <= float(best_heur_est) - float(req))
                                                pass_cur = bool(float(b_total) <= float(cur_total) - float(min_gain))
                                                if mpvs_ctrl is not None:
                                                    try:
                                                        mpvs_ctrl.observe_probe(
                                                            comp="macro",
                                                            family=str(nm),
                                                            ctx_key=str(step_ctx.get("ctx_key", "")),
                                                            margin_heur=float(margin_heur),
                                                            margin_cur=float(margin_cur),
                                                            calls=int(_spent),
                                                            pass_heur=bool(pass_heur),
                                                            pass_cur=bool(pass_cur),
                                                        )
                                                    except Exception:
                                                        pass
                                                mpvs_stats["macro_probe_seen"] = int(mpvs_stats.get("macro_probe_seen", 0)) + 1
                                                if pass_heur:
                                                    mpvs_stats["macro_probe_pass_heur"] = int(mpvs_stats.get("macro_probe_pass_heur", 0)) + 1
                                                if pass_cur:
                                                    mpvs_stats["macro_probe_pass_cur"] = int(mpvs_stats.get("macro_probe_pass_cur", 0)) + 1

                                                # Require macro to beat heuristic by req already at fast-proxy stage
                                                if not pass_heur:
                                                    mpvs_stats["macro_precheck_blocked"] = int(mpvs_stats.get("macro_precheck_blocked", 0)) + 1
                                                    continue
                                                # strict admission: macro must improve current objective to be considered
                                                if not pass_cur:
                                                    mpvs_stats["macro_precheck_fail_min_gain"] = int(mpvs_stats.get("macro_precheck_fail_min_gain", 0)) + 1
                                                    sponsored = False
                                                    reason = ""
                                                    if mpvs_ctrl is not None:
                                                        try:
                                                            sponsored, reason, _meta = mpvs_ctrl.maybe_sponsor_trial(
                                                                comp="macro", family=str(nm), ctx=step_ctx, step=int(step)
                                                            )
                                                        except Exception:
                                                            sponsored, reason = False, ""
                                                    if not sponsored:
                                                        try:
                                                            cd = int(_cfg_get(ver_cfg, "macro_precheck_cooldown_fail", 10))
                                                            if instance_tag == "randw":
                                                                cd = int(_cfg_get(ver_cfg, "macro_precheck_cooldown_fail_randw", cd))
                                                            mpvs_trigger_state["cooldown"]["macro"] = max(int(mpvs_trigger_state["cooldown"].get("macro", 0)), cd)
                                                        except Exception:
                                                            pass
                                                        continue
                                                    _pl["_cec_trial"] = 1
                                                    _pl["_cec_ctx_key"] = str(step_ctx.get("ctx_key", ""))
                                                    _pl["_cec_family"] = str(nm)
                                                    _pl["_cec_trial_reason"] = str(reason or "")
                                                    _pl["_cec_trial_kind"] = str(reason or "")
                                                    _pl["_cec_trial_score"] = float((_meta or {}).get("trial_score", 0.0))
                                                    _pl["_cec_trial_edge_local"] = float((_meta or {}).get("edge_local", 0.0))
                                                    _pl["_cec_trial_edge_family"] = float((_meta or {}).get("edge_family", 0.0))
                                                    _pl["_cec_trial_lambda"] = float((_meta or {}).get("lambda_local", 0.0))
                                                    mpvs_stats["macro_trial_sponsored"] = int(mpvs_stats.get("macro_trial_sponsored", 0)) + 1
                                                    mpvs_stats["macro_trial_score_sum"] = float(mpvs_stats.get("macro_trial_score_sum", 0.0)) + float((_meta or {}).get("trial_score", 0.0))
                                                    mpvs_stats["macro_trial_score_max"] = max(float(mpvs_stats.get("macro_trial_score_max", 0.0)), float((_meta or {}).get("trial_score", 0.0)))
                                                    if str(reason) == "seed_sponsor":
                                                        mpvs_stats["macro_trial_seed"] = int(mpvs_stats.get("macro_trial_seed", 0)) + 1
                                                    if str(reason) == "evidence_sponsor":
                                                        mpvs_stats["macro_trial_evidence"] = int(mpvs_stats.get("macro_trial_evidence", 0)) + 1
                                                    if str(reason) == "candidate_sponsor":
                                                        mpvs_stats["macro_trial_candidate"] = int(mpvs_stats.get("macro_trial_candidate", 0)) + 1
                                                    _rs = mpvs_stats.setdefault("macro_trial_sponsor_reason", {})
                                                    _rs[str(reason or "")] = int(_rs.get(str(reason or ""), 0)) + 1
                                                else:
                                                    mpvs_stats["macro_precheck_pass_min_gain"] = int(mpvs_stats.get("macro_precheck_pass_min_gain", 0)) + 1
                                                mpvs_stats["macro_precheck_allowed"] = int(mpvs_stats.get("macro_precheck_allowed", 0)) + 1
                                            except Exception:
                                                mpvs_stats["macro_precheck_failed"] = int(mpvs_stats.get("macro_precheck_failed", 0)) + 1
                                                continue
                                        kept_plans.append(_pl)
                                    plans = kept_plans
                            except Exception:
                                pass

                            scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
                            for pl in plans:
                                cheap, cmeta = _cheap_score(pl)
                                scored.append((float(cheap), pl, cmeta))

                            sorted_scored = sorted(scored, key=lambda x: float(x[0]))
                            # Verifier v1: lite/full stages
                            lite_verify_keys: set[str] = set()
                            full_verify_keys: set[str] = set()

                            def _plan_key(pl: Dict[str, Any]) -> str:
                                return f"A:{int(pl.get('id', -1))}" if pl.get("kind") == "atomic" else f"M:{pl.get('src', '')}:{pl.get('name', '')}"

                            best_pk_cheap = None
                            best_v_cheap = None
                            try:
                                if sorted_scored:
                                    best_pk_cheap = _plan_key(sorted_scored[0][1])
                                    best_v_cheap = float(sorted_scored[0][0])
                            except Exception:
                                best_pk_cheap = None
                                best_v_cheap = None

                            # --- lite/full config ---
                            lite_topk = int(_cfg_get(ver_cfg, "lite_topk", 8))
                            full_topk = int(_cfg_get(ver_cfg, "full_topk", int(full_verify_topk)))
                            max_lite = int(_cfg_get(ver_cfg, "max_lite_verify", 10))
                            max_full = int(_cfg_get(ver_cfg, "max_full_verify", max_full_verify))

                            for _, pl, _ in sorted_scored[: max(0, lite_topk)]:
                                lite_verify_keys.add(_plan_key(pl))

                            nfull = 0
                            for _, pl, _ in sorted_scored:
                                if str(pl.get("kind", "")) != "macro":
                                    continue
                                pk = _plan_key(pl)
                                if pk in full_verify_keys:
                                    continue
                                full_verify_keys.add(pk)
                                nfull += 1
                                if nfull >= max(0, full_topk):
                                    break

                            if full_verify_always_include_sources:
                                for src in full_verify_always_include_sources:
                                    src_scored = [it for it in sorted_scored if str(it[1].get("src", "")) == src]
                                    if not src_scored:
                                        continue
                                    _, pl, _ = src_scored[0]
                                    full_verify_keys.add(_plan_key(pl))

                            # ensure heuristic best is always in full-verify set (so safe fallback has a usable res)
                            try:
                                best_heur = None
                                for _, pl, _ in sorted_scored:
                                    if str(pl.get("src", "")) == "heuristic" and pl.get("kind") == "atomic":
                                        best_heur = pl
                                        break
                                if best_heur is not None:
                                    lite_verify_keys.add(_plan_key(best_heur))
                            except Exception:
                                pass

                            # ensure memory replay is verifiable, but only sparsely to avoid fixed eval-call tax
                            try:
                                if (not disable_verifier_step) and mem_plans:
                                    mem_verify_every = int(_cfg_get(mem_cfg, "verify_every_n_steps", 10))
                                    if mem_verify_every <= 1 or (int(step) % int(mem_verify_every) == 0):
                                        best_mem = None
                                        for _, pl, _ in sorted_scored:
                                            if str(pl.get("src", "")) == "mem" and pl.get("kind") == "atomic" and (pl.get("action_seq") is not None or pl.get("action") is not None):
                                                best_mem = pl
                                                break
                                        if best_mem is not None:
                                            lite_verify_keys.add(_plan_key(best_mem))
                            except Exception:
                                pass

                            # cap number of lite/full plans
                            if len(lite_verify_keys) > max(0, max_lite):
                                capped = set()
                                for _, pl, _ in sorted_scored:
                                    pk = _plan_key(pl)
                                    if pk in lite_verify_keys:
                                        capped.add(pk)
                                        if len(capped) >= max(0, max_lite):
                                            break
                                lite_verify_keys = capped
                            if len(full_verify_keys) > max(0, max_full):
                                cappedf = set()
                                for _, pl, _ in sorted_scored:
                                    pk = _plan_key(pl)
                                    if pk in full_verify_keys:
                                        cappedf.add(pk)
                                        if len(cappedf) >= max(0, max_full):
                                            break
                                full_verify_keys = cappedf

                            if not disable_verifier_step:
                                mpvs_stats["verifier_candidates_considered"] = int(mpvs_stats.get("verifier_candidates_considered", 0)) + int(len(lite_verify_keys))

                            for cheap, pl, cmeta in sorted_scored:
                                pk = _plan_key(pl)
                                src_pl = str(pl.get("src", ""))
                                cand = pl.get("cand", None)
                                if disable_verifier_step:
                                    # no_verifier: do NOT run evaluator-based verification over candidate list.
                                    # Only allow "atomic cand" (0-call, uses cand.est) to be treated as verified.
                                    do_verify = (str(pl.get("kind")) == "atomic" and cand is not None)
                                else:
                                    do_verify = (pk in lite_verify_keys) or (
                                        str(pl.get("kind")) == "atomic" and cand is not None and src_pl == "heuristic"
                                    )
                                if do_verify:
                                    calls_b = int(getattr(evaluator, "evaluator_calls", 0))
                                    if disable_verifier_step:
                                        # no_verifier: we should not verify macro/mem by evaluator here.
                                        # atomic cand uses cand.est and is safe (0-call).
                                        if cand is not None:
                                            res = _atomic_res_from_cand(cand, assign)
                                        else:
                                            res = None
                                    else:
                                        if pl.get("kind") == "atomic":
                                            if cand is not None:
                                                res = _atomic_res_from_cand(cand, assign)
                                            elif pl.get("action_seq") is not None:
                                                res = _atomic_res_from_action_seq(pl.get("action_seq"), assign, eval_out)
                                            elif pl.get("action") is not None:
                                                res = _atomic_res_from_action(pl.get("action"), assign, eval_out)
                                            else:
                                                res = None
                                        else:
                                            if int(getattr(evaluator, "evaluator_calls", 0)) - calls0 >= max_verify_calls_step:
                                                res = None
                                            else:
                                                if pl.get("kind") == "macro":
                                                    lite_steps = int(_cfg_get(ver_cfg, "lite_macro_steps", 1))
                                                    full_steps = int(_cfg_get(ver_cfg, "full_macro_steps", int(_cfg_get(macro_cfg, "n_steps", 3))))
                                                    nrun = int(full_steps) if (pk in full_verify_keys) else int(lite_steps)
                                                    nrun = max(1, nrun)
                                                    b_assign, b_eval, b_total, acts, _cur = _exec_macro(
                                                        str(pl.get("name", "macro")), assign, eval_out, n_steps=nrun, rng_local=rng_macro
                                                    )
                                                    res = {"verified_best_total": float(b_total), "best_assign": b_assign, "best_eval": b_eval, "actions": acts}
                                                else:
                                                    res = None
                                    calls_a = int(getattr(evaluator, "evaluator_calls", 0))
                                    calls_spent = max(0, calls_a - calls_b)
                                    if res is None:
                                        # IMPORTANT: do NOT allow non-heuristic plans to compete with cheap scores.
                                        # Otherwise stale mem_score / macro_neutral will dominate and cause flat/repeat regressions.
                                        if str(pl.get("src", "")) != "heuristic":
                                            v_raw = 1e30
                                            v_eff = 1e30
                                        else:
                                            v_raw = float(cheap)
                                            v_eff = float(cheap)
                                        calls_spent = 0
                                    else:
                                        v_raw = float(res.get("verified_best_total", 1e30))
                                        v_eff = v_raw

                                    # ROI accounting + memory feedback
                                    try:
                                        if res is not None:
                                            gain0 = compute_gain(cur_total, float(res.get("verified_best_total", 1e30)))
                                            if calls_spent > 0:
                                                mpvs_stats["verifier_calls_by_src"][src_pl] = int(mpvs_stats["verifier_calls_by_src"].get(src_pl, 0)) + int(calls_spent)
                                                mpvs_stats["verifier_gain_by_src"][src_pl] = float(mpvs_stats["verifier_gain_by_src"].get(src_pl, 0.0)) + float(gain0)
                                                roi = float(gain0) / float(max(1, int(calls_spent)))
                                                mpvs_stats["verifier_roi_sum"] = float(mpvs_stats.get("verifier_roi_sum", 0.0)) + float(roi)
                                                mpvs_stats["verifier_roi_count"] = int(mpvs_stats.get("verifier_roi_count", 0)) + 1
                                                mpvs_stats["verifier_roi_by_src"][src_pl] = float(mpvs_stats["verifier_roi_by_src"].get(src_pl, 0.0)) + float(roi)
                                                # update ROI both for raw src and its group (llm_* -> llm)
                                                src_group = "llm" if str(src_pl).startswith("llm") else str(src_pl)
                                                if roi_tracker is not None:
                                                    roi_tracker.update(str(src_pl), gain0, int(calls_spent))
                                                    if src_group != str(src_pl):
                                                        roi_tracker.update(str(src_group), gain0, int(calls_spent))

                                                # MPVS controller feedback (cooldown harmful components)
                                                if mpvs_ctrl is not None and src_group in {"macro", "mem", "llm"}:
                                                    roi_val = float(gain0) / float(max(1, int(calls_spent)))
                                                    ok = False
                                                    try:
                                                        if src_group == "macro":
                                                            ok = bool(float(gain0) >= float(_macro_min_gain_cur()))
                                                        elif src_group == "mem":
                                                            mem_min_gain = float(_cfg_get(mem_cfg, "min_gain_use", 0.0002))
                                                            ok = bool(float(gain0) >= float(mem_min_gain))
                                                        else:
                                                            llm_min_gain = float(_cfg_get(_cfg_get(_cfg_get(mpvs_cfg, "controller", {}) or {}, "llm", {}) or {}, "success_min_gain", 0.0002))
                                                            ok = bool(float(gain0) >= float(llm_min_gain))
                                                    except Exception:
                                                        ok = bool(float(gain0) > 0.0)
                                                    try:
                                                        if ok and src_group == "mem":
                                                            mpvs_ctrl.observe_mem_success(step=int(step))
                                                        mpvs_ctrl.observe(
                                                            src_group,
                                                            step=int(step),
                                                            success=bool(ok),
                                                            roi=float(roi_val),
                                                            gain=float(gain0),
                                                            calls=int(calls_spent),
                                                            used_calls=int(eval_calls_cum),
                                                            budget_total=int(budget_total_calls),
                                                            best_total_seen=float(best_total_seen),
                                                            ctx_key=str(step_ctx.get("ctx_key", "")),
                                                            family=str(pl.get("name", "")) if src_group == "macro" else "",
                                                        )
                                                    except Exception:
                                                        pass
                                            if pk in full_verify_keys and str(pl.get("kind", "")) == "macro":
                                                mpvs_stats["verifier_full_verified"] = int(mpvs_stats.get("verifier_full_verified", 0)) + 1
                                                mpvs_stats["verifier_full_calls"] = int(mpvs_stats.get("verifier_full_calls", 0)) + int(calls_spent)
                                            else:
                                                mpvs_stats["verifier_lite_verified"] = int(mpvs_stats.get("verifier_lite_verified", 0)) + 1
                                                mpvs_stats["verifier_lite_calls"] = int(mpvs_stats.get("verifier_lite_calls", 0)) + int(calls_spent)
                                    except Exception:
                                        pass

                                    try:
                                        if src_pl == "mem" and (mem_bank is not None) and pl.get("mem_id") is not None and res is not None:
                                            mem_min_gain = float(_cfg_get(mem_cfg, "min_gain_use", 0.0002))
                                            gain0 = compute_gain(cur_total, float(res.get("verified_best_total", 1e30)))
                                            ok = bool(gain0 >= mem_min_gain)
                                            mem_bank.mark_result(int(pl.get("mem_id")), step=int(step), success=ok, gain=float(gain0), calls=max(1, int(calls_spent)))
                                            if not ok:
                                                mpvs_stats["mem_verify_fail"] = int(mpvs_stats.get("mem_verify_fail", 0)) + 1
                                    except Exception:
                                        pass
                                    try:
                                        if direction_bias == "bias_comm":
                                            best_comm = float(res.get("best_eval", {}).get("comm_norm", 0.0))
                                            cur_comm = float(eval_out.get("comm_norm", 0.0))
                                            improve = max(0.0, cur_comm - best_comm)
                                            v_eff = v_raw - bias_beta * improve
                                        elif direction_bias == "bias_therm":
                                            best_th = float(res.get("best_eval", {}).get("therm_norm", 0.0))
                                            cur_th = float(eval_out.get("therm_norm", 0.0))
                                            improve = max(0.0, cur_th - best_th)
                                            v_eff = v_raw - bias_beta * improve
                                    except Exception:
                                        v_eff = v_raw
                                else:
                                    res = None
                                    v_raw = float(cheap)
                                    v_eff = float(cheap)
                                    calls_spent = 0
                                    # IMPORTANT (budget-fair): under verifier-mode, do NOT let unverified non-heuristic
                                    # macro/mem/action plans compete with neutral/stale cheap scores.
                                    # Atomic cand plans are OK (cand.est is produced by evaluator already).
                                    if (not disable_verifier_step):
                                        if not (pl.get("kind") == "atomic" and pl.get("cand", None) is not None):
                                            if str(pl.get("src", "")) != "heuristic":
                                                v_raw = 1e30
                                                v_eff = 1e30

                                per_plan.append(
                                    {
                                        "plan": pl,
                                        "res": res,
                                        "v_raw": float(v_raw),
                                        "v_eff": float(v_eff),
                                        "cheap_meta": cmeta,
                                        "full_verified": bool(pk in full_verify_keys),
                                        "calls_spent": int(calls_spent),
                                    }
                                )
                                mpvs_stats["plans_scored"] += 1
                                try:
                                    if int(calls_spent) > 0:
                                        cb = mpvs_stats.setdefault("calls_by_src", {})
                                        _src = str(pl.get("src", "unknown"))
                                        cb[_src] = int(cb.get(_src, 0)) + int(calls_spent)
                                        if float(v_raw) < 1e29:
                                            gb = mpvs_stats.setdefault("gain_by_src", {})
                                            gain = max(0.0, float(cur_total) - float(v_raw))
                                            gb[_src] = float(gb.get(_src, 0.0)) + float(gain)
                                except Exception:
                                    pass
                                cover.add(str(pl.get("src", "")))
                                if pl.get("src") == "macro":
                                    mpvs_stats["macro_scored"] += 1
                                if pl.get("src") == "mem":
                                    mpvs_stats["mem_scored"] += 1

                            best_heur_entry = None
                            for ent in per_plan:
                                if str(ent["plan"].get("src", "")) != "heuristic":
                                    continue
                                if best_heur_entry is None or float(ent["v_raw"]) < float(best_heur_entry["v_raw"]) - 1e-12:
                                    best_heur_entry = ent

                            best_ent = None
                            for ent in per_plan:
                                pl = ent["plan"]
                                v_eff = float(ent["v_eff"])
                                if best_v is None or v_eff < best_v - 1e-12:
                                    best_v = v_eff
                                    best_v_raw = float(ent["v_raw"])
                                    best_plan = pl
                                    best_res = ent["res"]
                                    best_ent = ent
                                elif abs(v_eff - float(best_v)) <= tie_eps and best_plan is not None:
                                    if _plan_prio(pl) > _plan_prio(best_plan):
                                        best_v = v_eff
                                        best_v_raw = float(ent["v_raw"])
                                        best_plan = pl
                                        best_res = ent["res"]
                                        best_ent = ent

                            if (best_v is not None) and (len(cover) >= min_cover) and (best_v <= float(cur_total) - early_margin):
                                pass

                            if safe_enabled and best_plan is not None and best_heur_entry is not None:
                                # For non-heuristic, require it to BEAT heuristic by a margin (no more "slightly worse is ok").
                                req_improve = bool(_cfg_get(ver_cfg, "nonheuristic_require_improve", True))
                                m_early = float(_cfg_get(ver_cfg, "nonheuristic_improve_margin_early", safe_eps_early))
                                m_late = float(_cfg_get(ver_cfg, "nonheuristic_improve_margin_late", safe_eps_late * 0.5))
                                safe_eps = m_early if int(stagn) < safe_eps_switch_stagnation else m_late
                                if instance_tag == "randw":
                                    safe_eps *= float(_cfg_get(ver_cfg, "nonheuristic_improve_randw_scale", 2.0))

                                if str(best_plan.get("src", "")) != "heuristic":
                                    bh = float(best_heur_entry["v_raw"])
                                    bv = float(best_v_raw if best_v_raw is not None else 1e30)
                                    # If not verified (bv will be INF) or not strictly better than heuristic, fallback
                                    if (bv >= 1e29) or (req_improve and (bv > bh - safe_eps)):
                                        best_plan = best_heur_entry["plan"]
                                        best_res = best_heur_entry["res"]
                                        best_v = float(best_heur_entry["v_eff"])
                                        best_v_raw = float(best_heur_entry["v_raw"])
                                        best_ent = best_heur_entry

                            macro_gate_enabled = bool(_cfg_get(ver_cfg, "macro_gain_gate_enabled", True))
                            if macro_gate_enabled and best_plan is not None and best_heur_entry is not None and best_ent is not None:
                                src = str(best_plan.get("src", ""))
                                if src in {"macro", "llm_macro"}:
                                    gain_sw = int(_cfg_get(ver_cfg, "macro_gain_switch_stagnation", 20))
                                    gain_early = float(_cfg_get(ver_cfg, "macro_gain_margin_early", 0.001))
                                    gain_late = float(_cfg_get(ver_cfg, "macro_gain_margin_late", 0.0002))
                                    req = gain_early if int(stagn) < gain_sw else gain_late
                                    if instance_tag == "randw":
                                        req *= float(_cfg_get(ver_cfg, "macro_gain_randw_scale", 1.5))

                                    lam = float(_cfg_get(ver_cfg, "macro_cost_lambda", 2e-7))
                                    cb = float(best_ent.get("calls_spent", 0))
                                    ch = float(best_heur_entry.get("calls_spent", 0))
                                    extra = max(0.0, cb - ch)
                                    req = float(req) + float(lam) * float(extra)

                                    # Require macro to beat heuristic by req (lower is better)
                                    if float(best_v_raw if best_v_raw is not None else 1e30) > float(best_heur_entry["v_raw"]) - float(req):
                                        # fallback to heuristic
                                        mpvs_stats["macro_gate_blocked"] = int(mpvs_stats.get("macro_gate_blocked", 0)) + 1
                                        best_plan = best_heur_entry["plan"]
                                        best_res = best_heur_entry["res"]
                                        best_v = float(best_heur_entry["v_eff"])
                                        best_v_raw = float(best_heur_entry["v_raw"])
                                        best_ent = best_heur_entry
                                    else:
                                        mpvs_stats["macro_gate_allowed"] = int(mpvs_stats.get("macro_gate_allowed", 0)) + 1

                            if (not disable_verifier_step) and best_pk_cheap is not None and best_plan is not None:
                                try:
                                    if _plan_key(best_plan) != str(best_pk_cheap):
                                        mpvs_stats["verifier_changed_choice"] = int(mpvs_stats.get("verifier_changed_choice", 0)) + 1
                                except Exception:
                                    pass

                            if best_plan is not None and best_res is None:
                                if best_plan.get("kind") == "macro":
                                    b_assign, b_eval, b_total, acts, _cur = _exec_macro(
                                        str(best_plan.get("name", "macro")),
                                        assign,
                                        eval_out,
                                        n_steps=int(_cfg_get(macro_cfg, "n_steps", 3)),
                                        rng_local=rng_macro,
                                    )
                                    best_res = {"verified_best_total": float(b_total), "best_assign": b_assign, "best_eval": b_eval, "actions": acts}
                                else:
                                    cand = best_plan.get("cand", None)
                                    if cand is not None:
                                        best_res = _atomic_res_from_cand(cand, assign)
                                    elif best_plan.get("action") is not None:
                                        best_res = _atomic_res_from_action(best_plan.get("action"), assign, eval_out)
                                    else:
                                        best_res = {"verified_best_total": float(cur_total), "best_assign": assign.copy(), "best_eval": dict(eval_out), "actions": []}

                            try:
                                if mpvs_ctrl is not None and best_heur_entry is not None and best_heur_entry.get("res") is not None:
                                    heur_gain = max(0.0, float(cur_total) - float(best_heur_entry.get("v_eff", cur_total)))
                                    heur_calls = max(1, int(best_heur_entry.get("calls_spent", 0)))
                                    mpvs_ctrl.observe_heuristic(float(heur_gain), int(heur_calls))
                                    mpvs_stats["heuristic_rate_ewma"] = float(mpvs_ctrl.snapshot().get("heur_rate_ewma", 0.0))
                            except Exception:
                                pass

                            force_reject = False
                            if best_plan is None or best_res is None:
                                mpvs_stats["no_plan_or_res"] = int(mpvs_stats.get("no_plan_or_res", 0)) + 1

                                # Prefer fallback to heuristic-best (guaranteed safe path)
                                if best_heur_entry is not None and best_heur_entry.get("res") is not None:
                                    best_plan = best_heur_entry["plan"]
                                    best_res = best_heur_entry["res"]
                                    best_v = float(best_heur_entry.get("v_eff", cur_total))
                                    best_v_raw = float(best_heur_entry.get("v_raw", cur_total))
                                    best_ent = best_heur_entry
                                else:
                                    # Hard fallback: no-op (do not move), and do not accept
                                    best_plan = {"src": "none", "kind": "none", "name": "", "id": -1}
                                    best_res = {
                                        "verified_best_total": float(cur_total),
                                        "best_assign": assign.copy(),
                                        "best_eval": dict(eval_out),
                                        "actions": [],
                                    }
                                    best_v = float(cur_total)
                                    best_v_raw = float(cur_total)
                                    best_ent = None
                                    force_reject = True

                            # From here, best_res is guaranteed dict
                            try:
                                if best_plan is not None and best_res is not None and best_heur_entry is not None and best_heur_entry.get("res") is not None:
                                    src_best = str((best_plan or {}).get("src", ""))
                                    if src_best and src_best != "heuristic":
                                        v_non = float(best_res.get("verified_best_total", 1e30))
                                        gain_cur = float(cur_total) - float(v_non)

                                        sw = int(_cfg_get(ver_cfg, "nonheuristic_min_gain_current_switch_stagnation", safe_eps_switch_stagnation))
                                        g_early = float(_cfg_get(ver_cfg, "nonheuristic_min_gain_current_early", 0.0))
                                        g_late = float(_cfg_get(ver_cfg, "nonheuristic_min_gain_current_late", 0.0002))
                                        min_gain_cur = g_early if int(stagn) < int(sw) else g_late
                                        if instance_tag == "randw":
                                            min_gain_cur *= float(_cfg_get(ver_cfg, "nonheuristic_min_gain_current_randw_scale", 1.5))

                                        if src_best == "mem":
                                            min_gain_cur = max(float(min_gain_cur), float(_cfg_get(mem_cfg, "min_gain_use", 0.0002)))
                                        if src_best in {"macro", "llm_macro"}:
                                            min_gain_cur = max(float(min_gain_cur), float(_macro_min_gain_cur()))
                                        if src_best.startswith("llm"):
                                            min_gain_cur = max(float(min_gain_cur), float(_cfg_get(ver_cfg, "llm_min_gain_current", 0.0002)))

                                        # Budget-stage release: controller may relax min_gain_current for macro
                                        try:
                                            if mpvs_ctrl is not None:
                                                comp0 = "llm" if src_best.startswith("llm") else ("mem" if src_best == "mem" else ("macro" if src_best in {"macro", "llm_macro"} else ""))
                                                if comp0:
                                                    min_gain_cur = float(
                                                        mpvs_ctrl.adjust_min_gain_current(
                                                            comp0,
                                                            float(min_gain_cur),
                                                            used_calls=int(eval_calls_cum),
                                                            budget_total=int(budget_total_calls),
                                                            ctx=step_ctx,
                                                            family=str((best_plan or {}).get("name", "")),
                                                            sponsored=bool((best_plan or {}).get("_cec_trial", 0)),
                                                        )
                                                    )
                                        except Exception:
                                            pass

                                        if float(gain_cur) < float(min_gain_cur) - 1e-12:
                                            mpvs_stats["nonheur_current_gate_blocked"] = int(mpvs_stats.get("nonheur_current_gate_blocked", 0)) + 1
                                            d = mpvs_stats.get("nonheur_current_gate_blocked_by_src", {}) or {}
                                            d[src_best] = int(d.get(src_best, 0)) + 1
                                            mpvs_stats["nonheur_current_gate_blocked_by_src"] = d

                                            best_plan = best_heur_entry["plan"]
                                            best_res = best_heur_entry["res"]
                                            best_v = float(best_heur_entry.get("v_eff", cur_total))
                                            best_v_raw = float(best_heur_entry.get("v_raw", cur_total))
                                            best_ent = best_heur_entry
                            except Exception:
                                pass

                            vbest = float(best_res.get("verified_best_total", cur_total))
                            delta_v = vbest - cur_total
                            try:
                                src_best = str((best_plan or {}).get("src", ""))
                                if best_heur_entry is None:
                                    try:
                                        for _ent in per_plan:
                                            _pl = _ent.get("plan", {})
                                            if isinstance(_pl, dict) and str(_pl.get("src", "")) == "heuristic":
                                                if best_heur_entry is None or float(_ent.get("v_eff", 1e30)) < float(best_heur_entry.get("v_eff", 1e30)):
                                                    best_heur_entry = _ent
                                    except Exception:
                                        pass
                                if src_best in {"macro", "llm_macro"} and best_heur_entry is not None and bool(_cfg_get(ver_cfg, "macro_monotone_enabled", True)):
                                    acc_min_gain = float(_cfg_get(ver_cfg, "macro_accept_min_gain", float(_macro_min_gain_cur())))
                                    if acc_min_gain > 0.0 and float(delta_v) > -float(acc_min_gain):
                                        mpvs_stats["macro_monotone_blocked"] = int(mpvs_stats.get("macro_monotone_blocked", 0)) + 1
                                        best_plan = best_heur_entry["plan"]
                                        best_res = best_heur_entry["res"]
                                        best_v = float(best_heur_entry.get("v_eff", cur_total))
                                        best_v_raw = float(best_heur_entry.get("v_raw", cur_total))
                                        best_ent = best_heur_entry
                                        vbest = float((best_res or {}).get("verified_best_total", cur_total))
                                        delta_v = float(vbest) - float(cur_total)
                            except Exception:
                                pass
                            # --- BATC cooldown feedback (lightweight) ---
                            try:
                                if trig_enabled:
                                    mac_t = _cfg_get((_cfg_get(mpvs_cfg, "trigger", {}) or {}), "macro", {}) or {}
                                    ver_t = _cfg_get((_cfg_get(mpvs_cfg, "trigger", {}) or {}), "verifier", {}) or {}
                                    mac_cd_fail = int(_cfg_get(mac_t, "cooldown_fail", 10))
                                    ver_cd_fail = int(_cfg_get(ver_t, "cooldown_fail", 6))

                                    # improvement means delta_v < 0 (lower is better)
                                    improved = (float(delta_v) < -1e-12)

                                    if trig_allow_macro:
                                        if str(best_plan.get("kind","")) == "macro" and improved:
                                            mpvs_stats["trig_macro_success"] = int(mpvs_stats.get("trig_macro_success", 0)) + 1
                                            mpvs_trigger_state["cooldown"]["macro"] = 0
                                        else:
                                            mpvs_stats["trig_macro_fail"] = int(mpvs_stats.get("trig_macro_fail", 0)) + 1
                                            mpvs_trigger_state["cooldown"]["macro"] = max(int(mpvs_trigger_state["cooldown"].get("macro", 0)), mac_cd_fail)

                                    if trig_allow_verifier:
                                        if improved:
                                            mpvs_stats["trig_ver_success"] = int(mpvs_stats.get("trig_ver_success", 0)) + 1
                                            mpvs_trigger_state["cooldown"]["verifier"] = 0
                                        else:
                                            mpvs_stats["trig_ver_fail"] = int(mpvs_stats.get("trig_ver_fail", 0)) + 1
                                            mpvs_trigger_state["cooldown"]["verifier"] = max(int(mpvs_trigger_state["cooldown"].get("verifier", 0)), ver_cd_fail)
                            except Exception:
                                pass

                            acc_cfg = _cfg_get(mpvs_cfg, "accept", {}) or {}
                            use_temp = bool(_cfg_get(acc_cfg, "use_temperature", True))
                            enabled_acc = bool(_cfg_get(acc_cfg, "enabled", True))
                            Tacc = max(sa_min_T, float(T) if use_temp else 1.0)

                            accept = True
                            if enabled_acc:
                                if delta_v <= 0.0:
                                    accept = True
                                else:
                                    accept = (py_rng.random() < math.exp(-delta_v / Tacc))

                            # deterministic improving-flag flip vs cheap scoring (avoid SA randomness)
                            try:
                                if (best_v_cheap is not None) and (not disable_verifier_step):
                                    cheap_improve = (float(best_v_cheap) - float(cur_total)) <= 0.0
                                    ver_improve = float(delta_v) <= 0.0
                                    if bool(cheap_improve) != bool(ver_improve):
                                        mpvs_stats["verifier_changed_accept"] = int(mpvs_stats.get("verifier_changed_accept", 0)) + 1
                            except Exception:
                                pass

                            # Hard no-regret for non-heuristic: never accept a worsening move
                            if str(best_plan.get("src", "")) != "heuristic" and delta_v > 0.0:
                                accept = False

                            # Force reject if we had to synthesize no-op fallback
                            if force_reject:
                                accept = False

                            chosen_actions = best_res.get("actions", []) or []
                            assign_before_mpvs = assign.copy()

                            # --- MPVS anti-loop integration (tabu/inverse/cooldown) ---
                            def _mpvs_action_blocked(actions, assign_before, vbest_val, cur_total_val=None, relax_level: int = 0):
                                # compute aspiration: allow tabu if it beats global best by aspiration_delta
                                try:
                                    aspiration = float(vbest_val) < float(best_total_seen) - float(aspiration_delta)
                                except Exception:
                                    aspiration = False
                                if aspiration:
                                    return (False, {"aspiration": 1, "tabu": 0, "inverse": 0, "cooldown": 0, "soft": 0, "relax": int(relax_level)})
                                try:
                                    if soft_block_enabled and (cur_total_val is not None):
                                        mg = float(soft_block_min_gain)
                                        if instance_tag == "randw":
                                            mg *= float(soft_block_randw_scale)
                                        if float(cur_total_val) - float(vbest_val) >= float(mg):
                                            mpvs_stats["blocked_soft_override"] = int(mpvs_stats.get("blocked_soft_override", 0)) + 1
                                            return (False, {"aspiration": 0, "tabu": 0, "inverse": 0, "cooldown": 0, "soft": 1, "relax": int(relax_level)})
                                except Exception:
                                    pass
                                tabu_hit = 0
                                inverse_hit = 0
                                cooldown_hit = 0
                                for act in (actions or []):
                                    a = act
                                    if "signature" not in a or not a.get("signature"):
                                        try:
                                            a["signature"] = _signature_for_action(a, assign_before)
                                        except Exception:
                                            a["signature"] = ""
                                    sig = str(a.get("signature", ""))
                                    rl = int(relax_level)
                                    if rl < 3 and sig and (sig in tabu_signatures):
                                        tabu_hit = 1
                                    if rl < 2 and sig and (sig in inverse_signatures):
                                        inverse_hit = 1
                                    if rl < 1:
                                        # per-slot cooldown
                                        try:
                                            for slot in _touched_slots(a):
                                                if step - last_move_step_per_slot.get(int(slot), -10**6) < per_slot_cooldown:
                                                    cooldown_hit = 1
                                                    break
                                        except Exception:
                                            pass
                                    if cooldown_hit:
                                        break
                                blocked = bool(tabu_hit or inverse_hit or cooldown_hit)
                                return (blocked, {"aspiration": 0, "tabu": tabu_hit, "inverse": inverse_hit, "cooldown": cooldown_hit, "soft": 0, "relax": int(relax_level)})

                            def _mpvs_update_anti_loop_state(actions, assign_before):
                                for act in (actions or []):
                                    if "signature" not in act or not act.get("signature"):
                                        try:
                                            act["signature"] = _signature_for_action(act, assign_before)
                                        except Exception:
                                            act["signature"] = ""
                                    sig = str(act.get("signature", ""))
                                    if sig:
                                        tabu_signatures.append(sig)
                                        try:
                                            inverse_signatures.append(inverse_signature(act, assign_before))
                                        except Exception:
                                            pass
                                    try:
                                        for slot in _touched_slots(act):
                                            last_move_step_per_slot[int(slot)] = int(step)
                                            last_site_per_slot[int(slot)] = int(assign[int(slot)])
                                    except Exception:
                                        pass

                            blocked, blk_meta = _mpvs_action_blocked(chosen_actions, assign_before_mpvs, vbest, cur_total_val=cur_total, relax_level=0)
                            # treat "none" as blocked under stagnation (avoid wasting eval-call budget)
                            try:
                                none_block_stagn = int(_cfg_get(_cfg_get(mpvs_cfg, "accept", {}) or {}, "forbid_none_when_stagnation_ge", 8))
                                if (not blocked) and int(stagn) >= int(none_block_stagn):
                                    if chosen_actions and str(chosen_actions[0].get("op","")) == "none":
                                        blocked = True
                                        blk_meta = dict(blk_meta or {})
                                        blk_meta["none"] = 1
                                        mpvs_stats["blocked_none"] = int(mpvs_stats.get("blocked_none", 0)) + 1
                            except Exception:
                                pass
                            if blocked:
                                mpvs_stats["blocked_any"] = int(mpvs_stats.get("blocked_any", 0)) + 1
                                if str(best_plan.get("src", "")) == "heuristic":
                                    mpvs_stats["blocked_heuristic"] = int(mpvs_stats.get("blocked_heuristic", 0)) + 1
                                else:
                                    mpvs_stats["blocked_nonheuristic"] = int(mpvs_stats.get("blocked_nonheuristic", 0)) + 1

                                mpvs_stats["blocked_tabu"] = int(mpvs_stats.get("blocked_tabu", 0)) + int(blk_meta.get("tabu", 0))
                                mpvs_stats["blocked_inverse"] = int(mpvs_stats.get("blocked_inverse", 0)) + int(blk_meta.get("inverse", 0))
                                mpvs_stats["blocked_cooldown"] = int(mpvs_stats.get("blocked_cooldown", 0)) + int(blk_meta.get("cooldown", 0))

                                # Pick next-best UNBLOCKED entry (applies to heuristic too). If none, force reject (no-op) to save budget.
                                replaced = False
                                try:
                                    cand_ents = sorted(
                                        per_plan,
                                        key=lambda e: (float(e.get("v_eff", 1e30)), -float(_plan_prio((e.get("plan") or {}))))
                                    )
                                    for ent2 in cand_ents:
                                        pl2 = ent2.get("plan") or {}
                                        # build/ensure a usable res dict without extra evaluator calls when possible
                                        res2 = ent2.get("res", None)
                                        if res2 is None:
                                            cand2 = pl2.get("cand", None)
                                            if cand2 is not None:
                                                res2 = _atomic_res_from_cand(cand2, assign)
                                            else:
                                                # action-only/macro without res: skip here (should have been fail-closed in verifier-mode)
                                                continue
                                        acts2 = (res2 or {}).get("actions", []) or []
                                        v2 = float((res2 or {}).get("verified_best_total", cur_total))
                                        gain2 = float(cur_total) - float(v2)
                                        relax_levels = [0]
                                        if relax_enabled and int(stagn) >= int(relax_stagnation_ge):
                                            if (not relax_only_if_improving) or (gain2 >= float(relax_min_gain)):
                                                relax_levels = list(range(0, max(0, int(relax_max_level)) + 1))
                                        b2 = True
                                        _m2 = None
                                        used_rl = 0
                                        for rl in relax_levels:
                                            b2, _m2 = _mpvs_action_blocked(acts2, assign_before_mpvs, v2, cur_total_val=cur_total, relax_level=int(rl))
                                            used_rl = int(rl)
                                            if not b2:
                                                break
                                        if not b2:
                                            best_plan = pl2
                                            best_res = res2
                                            best_v = float(ent2.get("v_eff", v2))
                                            best_v_raw = float(ent2.get("v_raw", v2))
                                            best_ent = ent2
                                            chosen_actions = acts2
                                            vbest = float(best_res.get("verified_best_total", cur_total))
                                            mpvs_stats["blocked_replaced"] = int(mpvs_stats.get("blocked_replaced", 0)) + 1
                                            if used_rl > 0:
                                                mpvs_stats["blocked_relax_replaced"] = int(mpvs_stats.get("blocked_relax_replaced", 0)) + 1
                                                mpvs_stats["blocked_relax_level_sum"] = int(mpvs_stats.get("blocked_relax_level_sum", 0)) + int(used_rl)
                                            replaced = True
                                            break
                                except Exception:
                                    replaced = False

                                if not replaced:
                                    # If everything is blocked, do not move. This is safer than oscillating and burning eval-call budget.
                                    mpvs_stats["blocked_force_reject"] = int(mpvs_stats.get("blocked_force_reject", 0)) + 1
                                    force_reject = True
                                    chosen_actions = []
                                    vbest = float(cur_total)

                            delta_v = vbest - cur_total
                            if enabled_acc:
                                if delta_v <= 0.0:
                                    accept = True
                                else:
                                    accept = (py_rng.random() < math.exp(-delta_v / Tacc))
                            else:
                                accept = True
                            if str(best_plan.get("src", "")) != "heuristic" and delta_v > 0.0:
                                accept = False

                            op_args_obj = {
                                "op": "macro" if best_plan.get("kind") == "macro" else str((chosen_actions[0] if chosen_actions else {}).get("op","none")),
                                "mpvs_kind": str(best_plan.get("kind")),
                                "macro_name": str(best_plan.get("name","")) if best_plan.get("kind")=="macro" else "",
                                "sub_ops": [str(a.get("op","")) for a in chosen_actions[:6]],
                                "n_sub": int(len(chosen_actions)),
                                "verified_best_total": float(vbest),
                                "src": str(best_plan.get("src","heuristic")),
                            }
                            if best_plan.get("kind") == "macro":
                                op_for_trace = "macro"
                            else:
                                op_for_trace = str((chosen_actions[0] if chosen_actions else {}).get("op", "none"))

                            if accept:
                                assign = np.asarray(best_res.get("best_assign", assign), dtype=int).copy()
                                eval_out = dict(best_res.get("best_eval", eval_out))
                                try:
                                    _mpvs_update_anti_loop_state(chosen_actions, assign_before_mpvs)
                                except Exception:
                                    pass
                                prev_total = float(eval_out.get("total_scalar", prev_total))
                                prev_comm = float(eval_out.get("comm_norm", prev_comm))
                                prev_therm = float(eval_out.get("therm_norm", prev_therm))
                                accepted_steps += 1
                                if prev_total < best_total_seen:
                                    best_total_seen = prev_total
                                    last_best_step = int(step)
                                try:
                                    if str((best_plan or {}).get("src", "")) in {"macro", "llm_macro"} and float(delta_v) >= -1e-12:
                                        mpvs_stats["macro_selected_nonimprove"] = int(mpvs_stats.get("macro_selected_nonimprove", 0)) + 1
                                except Exception:
                                    pass
                                try:
                                    if float(delta_v) < -1e-12:
                                        trig_cfg2 = _cfg_get(mpvs_cfg, "trigger", {}) or {}
                                        W2 = int(_cfg_get(trig_cfg2, "window", 30))
                                        W2 = max(10, min(W2, 200))
                                        Wmax2 = int(max(50, min(400, W2 * 4)))
                                        mpvs_trigger_state["recent_improve"].append(float(-float(delta_v)))
                                        if len(mpvs_trigger_state["recent_improve"]) > Wmax2:
                                            mpvs_trigger_state["recent_improve"] = mpvs_trigger_state["recent_improve"][-Wmax2:]
                                except Exception:
                                    pass
                                # IMPORTANT: MPVS must contribute to pareto, otherwise run_layout_agent selection stays at init
                                try:
                                    added = pareto.add(
                                        eval_out["comm_norm"],
                                        eval_out["therm_norm"],
                                        {
                                            "assign": assign.copy(),
                                            "total_scalar": float(eval_out["total_scalar"]),
                                            "stage": stage_label,
                                            "iter": int(step + 1),
                                            "seed": int(seed_id),
                                        },
                                    )
                                    if added:
                                        mpvs_stats["pareto_added"] += 1
                                except Exception:
                                    added = False

                                if mem_enabled and (delta_v < 0.0) and (mem_bank is not None):
                                    try:
                                        store_min_gain = float(_cfg_get(mem_cfg, "min_gain_store", 0.0005))
                                        max_len = int(_cfg_get(mem_cfg, "max_action_len", 3))
                                        acts = (best_res or {}).get("actions", []) or []
                                        gain = float(-float(delta_v))
                                        if (gain >= store_min_gain) and acts and (len(acts) <= max(1, max_len)):
                                            clean = []
                                            for a0 in acts[: max(1, max_len)]:
                                                a = copy.deepcopy(a0)
                                                a.pop("candidate_id", None)
                                                a.pop("signature", None)
                                                a.pop("_src", None)
                                                clean.append(a)
                                            mid = mem_bank.add(
                                                assign_before=assign_before_mpvs,
                                                site_to_region=site_to_region,
                                                traffic_sym=traffic_sym,
                                                chip_tdp=chip_tdp,
                                                actions=clean,
                                                gain=gain,
                                                step=int(step),
                                                origin_src=str(best_plan.get("src", "")),
                                                budget_progress=float(eval_calls_cum) / float(max(1, int(budget_total_calls))) if int(budget_total_calls) > 0 else 0.0,
                                                repeat_ratio=float(repeat_ratio_pre),
                                                blocked_ratio=float(blocked_ratio_pre),
                                            )
                                            if mid is not None:
                                                mpvs_stats["mem_store"] = int(mpvs_stats.get("mem_store", 0)) + 1
                                        else:
                                            mpvs_stats["mem_store_skip"] = int(mpvs_stats.get("mem_store_skip", 0)) + 1
                                    except Exception:
                                        mpvs_stats["mem_store_skip"] = int(mpvs_stats.get("mem_store_skip", 0)) + 1
                            else:
                                added = False

                            T = max(sa_min_T, float(T) * float(alpha))

                            assign_sig = signature_for_assign(assign)
                            op_args_json = json.dumps(op_args_obj, ensure_ascii=False)
                            wall_time_ms = int((time.perf_counter() - wall_start) * 1000)
                            h1 = int(mpvs_cache.hits) if mpvs_cache is not None else 0
                            m1 = int(mpvs_cache.misses) if mpvs_cache is not None else 0
                            cache_saved_eval_calls_cum = h1
                            calls1 = int(getattr(evaluator, "evaluator_calls", 0))
                            mpvs_stats["verifier_calls_spent"] += max(0, calls1 - calls0)
                            mpvs_stats["steps_mpvs"] += 1
                            if accept:
                                src_sel = str(best_plan.get("src", ""))
                                if src_sel == "llm_macro":
                                    mpvs_stats["llm_macro_selected"] += 1
                                if src_sel == "llm_atomic":
                                    mpvs_stats["llm_selected"] += 1
                                if src_sel == "macro":
                                    mpvs_stats["macro_selected"] += 1
                                    if bool((best_plan or {}).get("_cec_trial", 0)):
                                        mpvs_stats["macro_trial_won"] = int(mpvs_stats.get("macro_trial_won", 0)) + 1
                                if src_sel == "mem":
                                    mpvs_stats["memory_selected"] += 1
                                    # Update mem global gate based on realized gain (selected mem only).
                                    try:
                                        mem_gate_cfg = _cfg_get(mem_cfg, "global_gate", {}) or {}
                                        mem_gate_enabled = bool(_cfg_get(mem_gate_cfg, "enabled", False))
                                        if mem_gate_enabled:
                                            window = int(_cfg_get(mem_gate_cfg, "window", 40))
                                            window = max(10, min(200, window))
                                            fail_rate_hi = float(_cfg_get(mem_gate_cfg, "fail_rate_hi", 0.75))
                                            cooldown_steps = int(_cfg_get(mem_gate_cfg, "cooldown_steps", 30))
                                            min_gain_use = float(_cfg_get(mem_gate_cfg, "min_gain_use", 0.0002))

                                            gain0 = float(cur_total) - float(vbest)
                                            ok = bool(gain0 >= float(min_gain_use))
                                            mpvs_mem_hist.append(1 if ok else 0)
                                            if len(mpvs_mem_hist) > window:
                                                mpvs_mem_hist[:] = mpvs_mem_hist[-window:]
                                            fail_rate = 1.0 - float(sum(mpvs_mem_hist)) / float(max(1, len(mpvs_mem_hist)))
                                            mpvs_stats["mem_global_fail_rate"] = float(fail_rate)
                                            if fail_rate >= float(fail_rate_hi):
                                                mpvs_mem_global_until = max(int(mpvs_mem_global_until), int(step) + int(cooldown_steps))
                                                mpvs_stats["mem_global_triggered"] = int(mpvs_stats.get("mem_global_triggered", 0)) + 1
                                    except Exception:
                                        pass

                                # Horizon-aware credit: attribute delayed improvements to the winning component.
                                try:
                                    if mpvs_ctrl is not None:
                                        src_grp = "llm" if src_sel.startswith("llm") else str(src_sel)
                                        if src_grp in {"macro", "mem", "llm"}:
                                            _is_sponsored = bool((best_plan or {}).get("_cec_trial", 0))
                                            _ctx_key_for_ticket = str((best_plan or {}).get("_cec_ctx_key", step_ctx.get("ctx_key", "")))
                                            _family_for_ticket = str((best_plan or {}).get("_cec_family", (best_plan or {}).get("name", "")))
                                            # default: keep existing behavior
                                            _ticket_best_total_seen = float(best_total_seen)
                                            # v2.1 baseline fix: sponsored macro wins use pre-apply baseline only
                                            if src_grp == "macro" and _is_sponsored and best_total_before_step is not None:
                                                _ticket_best_total_seen = float(best_total_before_step)
                                                mpvs_stats["ticket_baseline_fix_used"] = int(mpvs_stats.get("ticket_baseline_fix_used", 0)) + 1
                                                try:
                                                    best_plan["_cec_ticket_best_total_before_step"] = float(best_total_before_step)
                                                    best_plan["_cec_ticket_best_total_seen_passed"] = float(_ticket_best_total_seen)
                                                except Exception:
                                                    pass
                                            mpvs_ctrl.register_win(
                                                src_grp,
                                                used_calls=int(eval_calls_cum),
                                                budget_total=int(budget_total_calls),
                                                best_total_seen=float(_ticket_best_total_seen),
                                                ctx_key=_ctx_key_for_ticket,
                                                family=_family_for_ticket,
                                                sponsored=bool(_is_sponsored),
                                            )
                                            if src_grp == "macro":
                                                fam0 = str((best_plan or {}).get("_cec_family", (best_plan or {}).get("name", "")) or "")
                                                if bool(mpvs_ctrl.release_active("macro", family=fam0, ctx=step_ctx)):
                                                    mpvs_stats["macro_release_hit"] = int(mpvs_stats.get("macro_release_hit", 0)) + 1
                                except Exception:
                                    pass

                            mpvs_stats["mem_global_until"] = int(mpvs_mem_global_until)
                            try:
                                if mpvs_ctrl is not None:
                                    _snap_step = mpvs_ctrl.snapshot()
                                    _release_now = int(_snap_step.get("cec_release_total", 0))
                                    if _release_now > int(prev_release_total):
                                        mpvs_stats["macro_release_activated"] = int(mpvs_stats.get("macro_release_activated", 0)) + int(_release_now - int(prev_release_total))
                                    prev_release_total = int(_release_now)

                                    # v2.1 candidate soft-release stats (simple diff to avoid double counting)
                                    try:
                                        _cec_ctx = _snap_step.get("cec_ctx", {}) or {}
                                        _cand_now = int(_snap_step.get("cec_candidate_total", 0))
                                        if _cand_now > int(prev_candidate_total):
                                            mpvs_stats["macro_candidate_activated"] = int(mpvs_stats.get("macro_candidate_activated", 0)) + int(_cand_now - int(prev_candidate_total))
                                        prev_candidate_total = int(_cand_now)

                                        _cand_hits_sum = 0
                                        if isinstance(_cec_ctx, dict):
                                            for _v in _cec_ctx.values():
                                                try:
                                                    _cand_hits_sum += int((_v or {}).get("candidate_hits", 0))
                                                except Exception:
                                                    pass
                                        if int(_cand_hits_sum) > int(prev_candidate_hits_sum):
                                            mpvs_stats["macro_candidate_hit"] = int(mpvs_stats.get("macro_candidate_hit", 0)) + int(int(_cand_hits_sum) - int(prev_candidate_hits_sum))
                                        prev_candidate_hits_sum = int(_cand_hits_sum)
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                            # --- BATC window update (MPVS post-step, before continue) ---
                            try:
                                if mpvs_enabled:
                                    trig_cfg2 = _cfg_get(mpvs_cfg, "trigger", {}) or {}
                                    if bool(_cfg_get(trig_cfg2, "enabled", False)):
                                        W2 = int(_cfg_get(trig_cfg2, "window", 30))
                                        W2 = max(10, min(W2, 200))
                                        Wmax2 = int(max(50, min(400, W2 * 4)))

                                        mpvs_trigger_state["recent_sigs"].append(str(assign_sig))
                                        if len(mpvs_trigger_state["recent_sigs"]) > Wmax2:
                                            mpvs_trigger_state["recent_sigs"] = mpvs_trigger_state["recent_sigs"][-Wmax2:]

                                        if mpvs_calls0 is not None:
                                            used_now = int(getattr(evaluator, "evaluator_calls", 0))
                                            calls_used_step = max(0, int(used_now) - int(mpvs_calls0))
                                            mpvs_trigger_state["recent_calls"].append(int(calls_used_step))
                                            if len(mpvs_trigger_state["recent_calls"]) > Wmax2:
                                                mpvs_trigger_state["recent_calls"] = mpvs_trigger_state["recent_calls"][-Wmax2:]

                                            if (not bool(trig_allow_macro)) and (not bool(trig_allow_verifier)):
                                                mpvs_trigger_state["recent_calls_cheap"].append(int(calls_used_step))
                                                if len(mpvs_trigger_state["recent_calls_cheap"]) > Wmax2:
                                                    mpvs_trigger_state["recent_calls_cheap"] = mpvs_trigger_state["recent_calls_cheap"][-Wmax2:]

                                        mpvs_trigger_state["recent_repeat"].append(float(trig_repeat_ratio))
                                        if len(mpvs_trigger_state["recent_repeat"]) > Wmax2:
                                            mpvs_trigger_state["recent_repeat"] = mpvs_trigger_state["recent_repeat"][-Wmax2:]
                            except Exception:
                                pass

                            row = {
                                "iter": int(step),
                                "stage": stage_label,
                                "op": op_for_trace,
                                "op_args_json": op_args_json,
                                "accepted": 1 if accept else 0,
                                "total_scalar": float(eval_out.get("total_scalar", prev_total0)),
                                "comm_norm": float(eval_out.get("comm_norm", prev_comm0)),
                                "therm_norm": float(eval_out.get("therm_norm", prev_therm0)),
                                "pareto_added": 1 if (accept and added) else 0,
                                "duplicate_penalty": float(eval_out.get("penalty", {}).get("duplicate", 0.0)),
                                "boundary_penalty": float(eval_out.get("penalty", {}).get("boundary", 0.0)),
                                "seed_id": int(seed_id),
                                "time_ms": int((time.perf_counter() - step_start) * 1000),
                                "signature": assign_sig,
                                "d_total": float(eval_out.get("total_scalar", prev_total0)) - float(prev_total0),
                                "d_comm": float(eval_out.get("comm_norm", prev_comm0)) - float(prev_comm0),
                                "d_therm": float(eval_out.get("therm_norm", prev_therm0)) - float(prev_therm0),
                                "delta_total": float(eval_out.get("total_scalar", prev_total0)) - float(prev_total0),
                                    "delta_comm": float(eval_out.get("comm_norm", prev_comm0)) - float(prev_comm0),
                                    "delta_therm": float(eval_out.get("therm_norm", prev_therm0)) - float(prev_therm0),
                                    "tabu_hit": 0,
                                    "inverse_hit": 0,
                                    "cooldown_hit": 0,
                                    "policy": "mpvs",
                                    "move": str(best_plan.get("kind")),
                                    "lookahead_k": int(lookahead_k if lookahead_enabled else 0),
                                    "cache_hit": 1 if (mpvs_cache is not None and h1 > mpvs_h0) else 0,
                                    "cache_key": "",
                                    "objective_hash": objective_hash,
                                    "eval_calls_cum": int(eval_calls_cum),
                                    "cache_hit_cum": h1,
                                    "cache_miss_cum": m1,
                                    "cache_saved_eval_calls_cum": cache_saved_eval_calls_cum,
                                    "llm_used": 1 if use_llm_now and (not disable_llm) else 0,
                                    "llm_fail_count": int(llm_fail_count),
                                    "fallback_reason": "",
                                    "wall_time_ms_cum": wall_time_ms,
                                    "accepted_steps_cum": int(accepted_steps),
                                    "sim_eval_calls_cum": int(eval_calls_cum),
                                    "lookahead_enabled": 0,
                                    "lookahead_r": 0,
                                    "notes": json.dumps(
                                        {
                                            "mpvs": {
                                                "n_plans": int(len(plans)),
                                                "best_kind": str(best_plan.get("kind")),
                                                "best_src": str(best_plan.get("src","")) if isinstance(best_plan, dict) else None,
                                                "macro": str(best_plan.get("name","")),
                                                "verified_best_total": float(vbest),
                                                "delta_v": float(delta_v),
                                                "trig_enabled": int(trig_enabled),
                                                "trig_distress": float(trig_distress),
                                                "trig_repeat_ratio": float(trig_repeat_ratio),
                                                "trig_calls_avg": float(trig_calls_avg),
                                                "trig_allow_macro": int(trig_allow_macro),
                                                "trig_allow_verifier": int(trig_allow_verifier),
                                                "macro_min_gain": float(_cfg_get(ver_cfg, "macro_min_gain", 0.0008)),
                                                "macro_precheck_fail_min_gain": int(mpvs_stats.get("macro_precheck_fail_min_gain", 0)),
                                                "macro_precheck_pass_min_gain": int(mpvs_stats.get("macro_precheck_pass_min_gain", 0)),
                                                "horizon": int(_cfg_get(ver_cfg,"horizon",3)),
                                                "mc": int(_cfg_get(ver_cfg,"mc",2)),
                                                "refine_sa_calls": int(_cfg_get(ver_cfg,"refine_sa_calls",20)),
                                                "blocked": int(blocked) if isinstance(blocked, bool) else None,
                                                "blocked_meta": blk_meta if isinstance(blk_meta, dict) else None,
                                                "force_reject": int(force_reject) if isinstance(force_reject, bool) else None,
                                                "fallback_reason": fallback_reason if "fallback_reason" in locals() else None,
                                                "use_llm_now": int(use_llm_now),
                                                "llm_pick_ids": llm_pick_ids[:8],
                                            }
                                        },
                                        ensure_ascii=False,
                                    ),
                                }
                            # --- BATC window update (MPVS step) ---
                            try:
                                if mpvs_enabled:
                                    trig_cfg2 = _cfg_get(mpvs_cfg, "trigger", {}) or {}
                                    if bool(_cfg_get(trig_cfg2, "enabled", False)):
                                        W2 = int(_cfg_get(trig_cfg2, "window", 30))
                                        W2 = max(10, min(W2, 200))
                                        Wmax2 = int(max(50, min(400, W2 * 4)))

                                        # signature history
                                        mpvs_trigger_state["recent_sigs"].append(str(assign_sig))
                                        if len(mpvs_trigger_state["recent_sigs"]) > Wmax2:
                                            mpvs_trigger_state["recent_sigs"] = mpvs_trigger_state["recent_sigs"][-Wmax2:]

                                        # calls used in THIS MPVS step
                                        if mpvs_calls0 is not None:
                                            used_now = int(getattr(evaluator, "evaluator_calls", 0))
                                            calls_used_step = max(0, int(used_now) - int(mpvs_calls0))
                                            mpvs_trigger_state["recent_calls"].append(int(calls_used_step))
                                            if len(mpvs_trigger_state["recent_calls"]) > Wmax2:
                                                mpvs_trigger_state["recent_calls"] = mpvs_trigger_state["recent_calls"][-Wmax2:]

                                            # baseline(cheap) calls when BOTH macro+verifier are disabled
                                            if (not bool(trig_allow_macro)) and (not bool(trig_allow_verifier)):
                                                mpvs_trigger_state["recent_calls_cheap"].append(int(calls_used_step))
                                                if len(mpvs_trigger_state["recent_calls_cheap"]) > Wmax2:
                                                    mpvs_trigger_state["recent_calls_cheap"] = mpvs_trigger_state["recent_calls_cheap"][-Wmax2:]

                                        # repeat ratio history for quantile thresholding
                                        mpvs_trigger_state["recent_repeat"].append(float(trig_repeat_ratio))
                                        if len(mpvs_trigger_state["recent_repeat"]) > Wmax2:
                                            mpvs_trigger_state["recent_repeat"] = mpvs_trigger_state["recent_repeat"][-Wmax2:]
                            except Exception:
                                pass

                            writer.writerow(row)
                            if do_explore:
                                mpvs_explore_left -= 1
                            else:
                                mpvs_explore_left = 0
                            continue

                    if force_explore_for_llm:
                        strong_types = {"kick", "cluster_move", "therm_swap", "relocate_free"}
                        candidate_ids = [c.id for c in candidate_pool if str(c.type) in strong_types]
                    # forbidden_ids computed earlier (shared)
        
                    llm_called = 0
                    llm_model = ""
                    llm_prompt_tokens = 0
                    llm_completion_tokens = 0
                    llm_latency_ms = 0
                    fallback_reason_step = ""
                    llm_status_code_step = 0
                    llm_error_code_step = ""
                    llm_ok_step = 0
                    llm_n_pick_step = 0

                    llm_disabled_effective = bool(llm_disabled or mpvs_exploit_override)
                    queue_enabled_effective = bool(queue_enabled and (not mpvs_exploit_override))
                    if forced_mpvs_policy:
                        forced_policy = forced_mpvs_policy

                    # ---- LLM policy scheduling (hard, auditable) ----
                    llm_policy = False
                    llm_policy_reason = ""

                    if (not llm_disabled_effective) and (llm_provider is not None):
                        if planner_type == "llm":
                            llm_policy = True
                            llm_policy_reason = "planner=llm"
                        elif planner_type == "mixed":
                            # STRICT: obey schedule by default
                            if mixed_every > 0 and (step % mixed_every == 0):
                                llm_policy = True
                                llm_policy_reason = "mixed_schedule"
                            elif allow_midcycle_refresh and refresh_due_to_rejects:
                                llm_policy = True
                                llm_policy_reason = "mixed_midcycle_refresh"

                    # ignore forced_policy for mixed to avoid policy thrash
                    if forced_policy and planner_type == "llm":
                        if forced_policy.lower() == "heuristic":
                            llm_policy = False
                            llm_policy_reason = "forced_policy=heuristic"
                        elif forced_policy.lower() == "llm":
                            llm_policy = (llm_provider is not None) and (not llm_disabled)
                            llm_policy_reason = "forced_policy=llm"

                    # rate limit + cap
                    if llm_policy:
                        if (step - last_llm_call_step) < llm_min_gap_steps:
                            llm_policy = False
                            llm_policy_reason = "rate_limited"
                        elif llm_calls_total >= llm_max_calls_total:
                            llm_policy = False
                            llm_policy_reason = "cap_reached"
                            llm_disabled = True
                            llm_disabled_reason = "llm_max_calls_total"

                    need_refresh = bool(
                        llm_policy
                        and (not action_queue or (refresh_due_to_rejects and allow_midcycle_refresh))
                    )

                    llm_called = 1 if need_refresh else 0
                    if need_refresh:
                        last_llm_call_step = int(step)
                        llm_calls_total += 1

                    # consume refresh flag once per step
                    refresh_due_to_rejects = False

                    if llm_policy:
                        llm_model = getattr(llm_provider, "model", "") if llm_provider is not None else ""
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
                            t0 = time.perf_counter()
                            pick_ids = llm_provider.propose_pick(ss, k_actions) or []
                            # sanitize picks: keep only valid ids in cand_map and not forbidden
                            pick_ids = [int(pid) for pid in (pick_ids or []) if int(pid) in cand_map and int(pid) not in set(forbidden_ids)]
                            llm_latency_ms = int((time.perf_counter() - t0) * 1000)
                            usage = getattr(llm_provider, "last_usage", {}) or {}
                            llm_status_code_step = int(usage.get("status_code", 0) or 0)
                            llm_error_code_step = str(usage.get("error_code", "") or "")
                            if (not llm_error_code_step) and ("resp" in usage):
                                m = re.search(r'"code":"([^"]+)"', str(usage.get("resp", "")))
                                if m:
                                    llm_error_code_step = m.group(1)
                            llm_n_pick_step = int(len(pick_ids))
                            llm_ok_step = 1 if (llm_status_code_step == 200 and llm_n_pick_step > 0) else 0
                            llm_attempt_count += 1
                            if llm_ok_step:
                                llm_success_count += 1
                            else:
                                llm_fail_count += 1
                                if llm_status_code_step == 429 and llm_error_code_step == "SetLimitExceeded":
                                    fallback_reason_step = "llm_fatal:SetLimitExceeded"
                                    if disable_on_fatal:
                                        llm_disabled = True
                                        llm_disabled_reason = "SetLimitExceeded"
                                    if require_llm:
                                        raise RuntimeError("LLM fatal: HTTP429 SetLimitExceeded (require_llm=True)")
                                elif llm_status_code_step != 200:
                                    fallback_reason_step = f"llm_http_{llm_status_code_step}:{llm_error_code_step}"
                                else:
                                    fallback_reason_step = "llm_empty_pick"
                            for pid in pick_ids:
                                cand = cand_map.get(pid)
                                if cand:
                                    pick_types_count[cand.type] = pick_types_count.get(cand.type, 0) + 1
                                    if best_d_total is None or cand.est.get("d_total", 0) < best_d_total:
                                        best_d_total = cand.est.get("d_total", 0)
                            usage = usage or getattr(llm_provider, "last_usage", {}) or {}
                            llm_prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
                            llm_completion_tokens = int(usage.get("completion_tokens", 0) or 0)
                        except Exception as e:
                            llm_fail_count += 1
                            fallback_reason_step = "llm_exception"
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
                                usage_fp.write("\n")
                                usage_fp.flush()
                        # ---- LLM gate: prevent LLM from degrading vs heuristic best ----
                        forbidden_set = set(int(x) for x in forbidden_ids)
                        allowed = [c for c in candidate_pool if int(c.id) not in forbidden_set]
                        allowed.sort(key=lambda c: float(c.est.get("d_total", 0.0)))
                        best_id = int(allowed[0].id) if allowed else None
                        best_d = float(allowed[0].est.get("d_total", 0.0)) if allowed else 0.0

                        gate_cfg = _cfg_get(planner_cfg, "llm_gate", {}) or {}
                        gate_rel = float(_cfg_get(gate_cfg, "rel", 0.15))
                        gate_abs = float(_cfg_get(gate_cfg, "abs", 500.0))
                        gate_allow_uphill = bool(_cfg_get(gate_cfg, "allow_uphill", False))

                        slack = max(gate_abs, gate_rel * max(1.0, abs(best_d)))

                        def _ok_pid(pid: int) -> bool:
                            cand = cand_map.get(int(pid))
                            if cand is None:
                                return False
                            if int(pid) in forbidden_set:
                                return False
                            d = float(cand.est.get("d_total", 0.0))
                            if (not gate_allow_uphill) and d > 0.0 + slack:
                                return False
                            if d > best_d + slack:
                                return False
                            act = cand.action if isinstance(cand.action, dict) else {}
                            if str(cand.type) == "cluster_move":
                                cid = int(act.get("cluster_id", -1))
                                slots = act.get("cluster_slots", []) or []
                                if not (0 <= cid < len(clusters)) and (not slots):
                                    return False
                            return True

                        pick_ids_clean: List[int] = []
                        seen = set()
                        for pid in pick_ids:
                            try:
                                pid_int = int(pid)
                            except Exception:
                                continue
                            if pid_int in seen:
                                continue
                            seen.add(pid_int)
                            if _ok_pid(pid_int):
                                pick_ids_clean.append(pid_int)

                        if (not pick_ids_clean) and (best_id is not None):
                            pick_ids_clean = [best_id]
                            fallback_reason_step = (fallback_reason_step + ";llm_gate_fallback").strip(";")

                        if best_id is not None and best_id not in pick_ids_clean:
                            pick_ids_clean.append(best_id)

                        pick_ids = pick_ids_clean

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
                        # push actions into queue (MUST NOT depend on logging)
                        expire_step = step + queue_window_steps
                        queue_size_before_push = int(len(action_queue))
                        if actions:
                            if queue_enabled_effective:
                                for act in actions:
                                    action_copy = copy.deepcopy(act)
                                    if "signature" not in action_copy:
                                        action_copy["signature"] = _signature_for_action(action_copy, assign)
                                    action_copy["_src"] = "llm"
                                    action_queue.append({"action": action_copy, "expire": expire_step})
                            else:
                                action_copy = copy.deepcopy(actions[0])
                                action_copy["_src"] = "llm"
                                action_queue = [{"action": action_copy, "expire": step}]

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
                            usage["llm_policy_reason"] = llm_policy_reason
                            usage["mixed_every"] = int(mixed_every)
                            usage["queue_enabled"] = int(queue_enabled_effective)
                            usage["queue_size_before"] = int(queue_size_before_push)
                            usage["queue_size_after_push"] = int(len(action_queue))
                            usage["llm_calls_total"] = int(llm_calls_total)
                            json.dump(usage, usage_fp)
                            usage_fp.write("\n")
                            usage_fp.flush()
        
                    action_queue = [a for a in action_queue if a.get("expire", step) >= step]
                    # P3: if stagnating, force a strong exploration move to the front of the queue
                    if force_explore and (not force_explore_for_llm):
                        # choose best available strong move (kick > cluster_move > therm_swap)
                        strong_types = ("kick", "cluster_move", "therm_swap")
                        chosen = None
                        for ty in strong_types:
                            pool = [c for c in candidate_pool if str(c.type) == ty and c.id not in forbidden_ids]
                            if pool:
                                pool.sort(key=lambda c: float(c.est.get("d_total", 0.0)))
                                chosen = pool[0]
                                break
                        if chosen is not None:
                            forced_action = {**copy.deepcopy(chosen.action), "candidate_id": int(chosen.id), "type": str(chosen.type), "signature": str(chosen.signature)}
                            forced_action["_src"] = "forced"
                            forced_action["_forced_by"] = "stagnation"
                            forced_action["_force_explore"] = 1
                            # put to front; expire immediately this step
                            action_queue.insert(0, {"action": forced_action, "expire": int(step)})
                    fallback_candidates = [c for c in candidate_pool if c.id not in forbidden_ids]
                    if lookahead_scores:
                        fallback_candidates.sort(
                            key=lambda c: float(lookahead_scores.get(int(c.id), 0.0)),
                            reverse=True,
                        )
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
                            action_payload = {
                                **copy.deepcopy(fb.action),
                                "candidate_id": int(fb.id),
                                "type": str(fb.type),
                                "signature": str(fb.signature),
                                "_cand_bucket": str(getattr(fb, "bucket", "")),
                                "_cand_d_total": float(getattr(fb, "est", {}).get("d_total", 0.0) if isinstance(getattr(fb, "est", None), dict) else 0.0),
                                "_cand_d_comm": float(getattr(fb, "est", {}).get("d_comm", 0.0) if isinstance(getattr(fb, "est", None), dict) else 0.0),
                                "_cand_d_therm": float(getattr(fb, "est", {}).get("d_therm", 0.0) if isinstance(getattr(fb, "est", None), dict) else 0.0),
                            }
                            action_payload["_src"] = "heuristic"
                        else:
                            action_payload = _sample_action(
                                dp_cfg,
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
                            action_payload["_src"] = "heuristic"
        
                        action = copy.deepcopy(action_payload) if action_payload else {"op": "none"}
                        op = str(action.get("op", "none"))
                        if op == "relocate":
                            action["from_site"] = int(assign[int(action.get("i", -1))])
                        if op == "cluster_move":
                            cid = int(action.get("cluster_id", -1))
                            slots = [int(x) for x in (action.get("cluster_slots", []) or [])]

                            cluster_obj = None
                            if 0 <= cid < len(clusters) and getattr(clusters[cid], "slots", None):
                                if clusters[cid].slots:
                                    cluster_obj = clusters[cid]
                            elif slots:
                                cluster_obj = Cluster(cluster_id=-1, slots=slots, tdp_sum=0.0)
                                action.setdefault("_cluster_id_invalid", int(cid))

                            if cluster_obj is not None and cluster_obj.slots:
                                slot_id = int(cluster_obj.slots[0])
                                if 0 <= slot_id < len(assign):
                                    action["from_region"] = int(site_to_region[int(assign[slot_id])])
                                if "target_sites" not in action:
                                    action["target_sites"] = _select_cluster_target_sites(
                                        assign, cluster_obj, int(action.get("region_id", -1)), site_to_region, sites_xy
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
                        if cand_ref is not None:
                            action.setdefault("_cand_bucket", str(getattr(cand_ref, "bucket", "")))
                            if isinstance(getattr(cand_ref, "est", None), dict):
                                action.setdefault("_cand_d_total", float(cand_ref.est.get("d_total", 0.0)))
                                action.setdefault("_cand_d_comm", float(cand_ref.est.get("d_comm", 0.0)))
                                action.setdefault("_cand_d_therm", float(cand_ref.est.get("d_therm", 0.0)))
        
                        aspiration = est_total_new is not None and est_total_new < best_total_seen - aspiration_delta
                        tabu_hit = 1 if op_signature in tabu_signatures else 0
                        inverse_hit = 1 if op_signature in inverse_signatures else 0
                        cooldown_hit = 0
                        for slot in _touched_slots(action):
                            if step - last_move_step_per_slot.get(int(slot), -10**6) < per_slot_cooldown:
                                cooldown_hit = 1
                                break
        
                        # P3: forced explore bypasses tabu/inverse/cooldown filtering
                        if bool(action.get("_force_explore", 0)) or aspiration or not (tabu_hit or inverse_hit or cooldown_hit):
                            break

                        # blocked by anti-loop: do not permanently discard LLM actions immediately
                        if str(action.get("_src", "")) == "llm" and queue_enabled_effective:
                            sc = int(action.get("_skip_count", 0)) + 1
                            action["_skip_count"] = sc
                            if sc <= 2:
                                action_queue.append({"action": copy.deepcopy(action), "expire": step + max(1, queue_window_steps // 2)})
                                continue
                        continue
                    op = str(action.get("op", "none"))
                    sig_before = signature_for_assign(assign)

                    new_assign = assign.copy()
                    if op == "swap":
                        _apply_swap(new_assign, int(action.get("i", 0)), int(action.get("j", 0)))
                    elif op == "relocate":
                        j = _apply_relocate(new_assign, int(action.get("i", 0)), int(action.get("site_id", 0)))
                        if j is not None and "degraded_swap_with" not in action:
                            action["degraded_swap_with"] = int(j)
                    elif op == "random_kick":
                        _apply_random_kick(
                            new_assign,
                            [int(x) for x in (action.get("idxs", []) or [])],
                            [int(x) for x in (action.get("site_ids", []) or [])],
                        )
                    elif op == "cluster_move":
                        cid = int(action.get("cluster_id", -1))
                        rid = int(action.get("region_id", 0))
                        slots = [int(x) for x in (action.get("cluster_slots", []) or [])]

                        cluster_obj = None
                        if 0 <= cid < len(clusters) and getattr(clusters[cid], "slots", None) and clusters[cid].slots:
                            cluster_obj = clusters[cid]
                        elif slots:
                            cluster_obj = Cluster(cluster_id=-1, slots=slots, tdp_sum=0.0)
                            action.setdefault("_cluster_id_invalid", int(cid))

                        if cluster_obj is not None and cluster_obj.slots:
                            target_sites = action.get("target_sites")
                            if not target_sites:
                                target_sites = _select_cluster_target_sites(assign, cluster_obj, rid, site_to_region, sites_xy)
                                action["target_sites"] = target_sites
                            _apply_cluster_move(new_assign, cluster_obj, target_sites)
        
                    layout_state.assign = new_assign
                    sig2 = signature_for_assign(new_assign)

                    # IMPORTANT: no-op action must be treated as rejected (avoid polluting accept/pareto/knee)
                    noop_action = (sig2 == sig_before)
                    if noop_action:
                        eval_new = dict(eval_out)
                        delta = 0.0
                        delta_comm = 0.0
                        delta_therm = 0.0
                    else:
                        k2 = _make_cache_key(sig2, objective_hash)
                        cached = eval_cache.get(k2) if eval_cache is not None else None
                        if cached is None:
                            eval_new = _evaluate(layout_state)
                            if eval_cache is not None:
                                eval_cache.put(k2, dict(eval_new))
                        else:
                            eval_new = dict(cached)
                        delta = float(eval_new["total_scalar"] - eval_out["total_scalar"])
                        delta_comm = float(eval_new["comm_norm"] - eval_out["comm_norm"])
                        delta_therm = float(eval_new["therm_norm"] - eval_out["therm_norm"])
                    if noop_action:
                        accept = False
                    else:
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
                            last_best_step = int(step)
                        consecutive_queue_rejects = 0
                        consecutive_llm_rejects = 0
                    else:
                        layout_state.assign = assign
                        added = False
                        if action.get("candidate_id") is not None:
                            cid = int(action.get("candidate_id"))
                            failed_counts[cid] = failed_counts.get(cid, 0) + 1
                            if failed_counts[cid] > 10:
                                recent_failed_ids.add(cid)
                        consecutive_queue_rejects += 1

                        # only LLM-sourced rejects should trigger refresh
                        if str(action.get("_src", "")) == "llm":
                            consecutive_llm_rejects += 1
                        else:
                            consecutive_llm_rejects = 0

                        if consecutive_llm_rejects >= 6:
                            refresh_due_to_rejects = True
        
                    T *= alpha
                    T = max(T, sa_min_T)
        
                    assign_signature = signature_for_assign(assign)
                    # --- BATC window update (post-step) ---
                    try:
                        if mpvs_enabled:
                            trig_cfg = _cfg_get(mpvs_cfg, "trigger", {}) or {}
                            if bool(_cfg_get(trig_cfg, "enabled", False)):
                                W = int(_cfg_get(trig_cfg, "window", 30))
                                W = max(10, min(W, 200))
                                # keep a slightly longer buffer than W for stability
                                Wmax = int(max(50, min(400, W * 4)))

                                mpvs_trigger_state["recent_sigs"].append(str(assign_signature))
                                if len(mpvs_trigger_state["recent_sigs"]) > Wmax:
                                    mpvs_trigger_state["recent_sigs"] = mpvs_trigger_state["recent_sigs"][-Wmax:]

                                if mpvs_calls0 is not None:
                                    used_now = int(getattr(evaluator, "evaluator_calls", 0))
                                    calls_used_step = max(0, int(used_now) - int(mpvs_calls0))
                                    mpvs_trigger_state["recent_calls"].append(int(calls_used_step))
                                    if len(mpvs_trigger_state["recent_calls"]) > Wmax:
                                        mpvs_trigger_state["recent_calls"] = mpvs_trigger_state["recent_calls"][-Wmax:]
                                    if (not bool(trig_allow_macro)) and (not bool(trig_allow_verifier)):
                                        mpvs_trigger_state["recent_calls_cheap"].append(int(calls_used_step))
                                        if len(mpvs_trigger_state["recent_calls_cheap"]) > Wmax:
                                            mpvs_trigger_state["recent_calls_cheap"] = mpvs_trigger_state["recent_calls_cheap"][-Wmax:]
                                if bool(trig_enabled):
                                    mpvs_trigger_state["recent_repeat"].append(float(trig_repeat_ratio))
                                    if len(mpvs_trigger_state["recent_repeat"]) > Wmax:
                                        mpvs_trigger_state["recent_repeat"] = mpvs_trigger_state["recent_repeat"][-Wmax:]
                    except Exception:
                        pass
                    op_args_obj = action
                    op_args_json = json.dumps(op_args_obj, ensure_ascii=False)
                    op_signature = stable_hash(
                        json.loads(op_args_json) if isinstance(op_args_json, str) else op_args_json
                    )
                    time_ms = int((time.perf_counter() - step_start) * 1000)
                    wall_time_ms_cum = int((time.perf_counter() - wall_start) * 1000)
                    cache_key = _make_cache_key(assign_signature, objective_hash)
                    h1 = eval_cache.hits if eval_cache is not None else 0
                    m1 = eval_cache.misses if eval_cache is not None else 0
                    cache_hit_step = h1 - h0
                    cache_miss_step = m1 - m0
                    cache_size = eval_cache.size if eval_cache is not None else 0
                    _sync_eval_calls()
                    eval_calls_step = eval_calls_cum - eval_calls_before

                    notes = (
                        f"llm_call={int(llm_called)} status={llm_status_code_step} code={llm_error_code_step} ok={llm_ok_step} n_pick={llm_n_pick_step} "
                        f"llm_ok/att={llm_success_count}/{llm_attempt_count}"
                        if (int(llm_called) == 1 or (fallback_reason_step != ""))
                        else ""
                    )
                    if noop_action:
                        notes = (notes + ";noop_action").strip(";")

                    row = {
                        "iter": int(step),
                        "stage": stage_label,
                        "op": str(op),
                        "op_args_json": op_args_json,
                        "accepted": int(accept),
                        "total_scalar": eval_out["total_scalar"],
                        "comm_norm": eval_out["comm_norm"],
                        "therm_norm": eval_out["therm_norm"],
                        "pareto_added": int(added),
                        "duplicate_penalty": eval_out["penalty"]["duplicate"],
                        "boundary_penalty": eval_out["penalty"]["boundary"],
                        "seed_id": int(seed_id),
                        "time_ms": int(time_ms),
                        "signature": assign_signature,
                        "d_total": float(delta),
                        "d_comm": float(delta_comm),
                        "d_therm": float(delta_therm),
                        "delta_total": float(delta),
                        "delta_comm": float(delta_comm),
                        "delta_therm": float(delta_therm),
                        "objective_hash": objective_hash,
                        "tabu_hit": int(tabu_hit),
                        "inverse_hit": int(inverse_hit),
                        "cooldown_hit": int(cooldown_hit),
                        "policy": "llm" if (action.get("_src", "") == "llm") else "heuristic",
                        "move": str(op),
                        "lookahead_k": int(lookahead_k if lookahead_enabled else 0),
                        "cache_hit": int(cache_hit_step),
                        "cache_hit_cum": int(h1),
                        "cache_miss_cum": int(m1),
                        "cache_saved_eval_calls_cum": int(h1),
                        "cache_key": cache_key,
                        "eval_calls_cum": int(eval_calls_cum),
                        "llm_used": int(llm_called),
                        "llm_fail_count": int(llm_fail_count),
                        "fallback_reason": str(fallback_reason_step or fallback_reason_init),
                        "wall_time_ms_cum": int(wall_time_ms_cum),
                        "accepted_steps_cum": int(accepted_steps),
                        "sim_eval_calls_cum": int(eval_calls_cum),
                        "lookahead_enabled": int(lookahead_enabled),
                        "lookahead_r": float(lookahead_r) if lookahead_enabled else 0.0,
                        "notes": notes,
                    }
                    writer.writerow(row)
                    if recordings_fp is not None:
                        pen = eval_out.get("penalty", {}) or {}
                        record = {
                            "iter": int(step),
                            "stage": str(stage_label),
                            "op": str(op),
                            "op_args": action,
                            "op_args_json": json.dumps(action, ensure_ascii=False),
                            "accepted": int(accept),
                            "total_scalar": float(eval_out.get("total_scalar", 0.0)),
                            "comm_norm": float(eval_out.get("comm_norm", 0.0)),
                            "therm_norm": float(eval_out.get("therm_norm", 0.0)),
                            "pareto_added": int(added),
                            "duplicate_penalty": float(pen.get("duplicate", 0.0)),
                            "boundary_penalty": float(pen.get("boundary", 0.0)),
                            "seed_id": int(seed_id),
                            "time_ms": int(time_ms),
                            "signature": str(assign_signature),
                        }
                        recordings_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
                        if step % int(_cfg_get(dp_cfg, "trace_flush_every", 20)) == 0:
                            recordings_fp.flush()
                    if step % int(_cfg_get(dp_cfg, "trace_flush_every", 20)) == 0:
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
                            "stagnation_steps": int(step - last_best_step),
                            "reheat_count": int(reheat_count),
                            "force_explore": int(force_explore),
                            "policy_switch_enabled": bool(use_ps),
                            "policy_last_action_family": getattr(controller, "last_action_family", None),
                            "policy_last_policy": getattr(controller, "last_policy", None),
                            "cache_hit_rate": float(eval_cache.hit_rate) if eval_cache is not None else None,
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
                    if controller is not None:
                        improved = bool(accept) and float(delta) < 0.0
                        controller.update(improved=improved, delta_total=float(delta))
            finally:
                # --- v5.4: append finalize row (CSV) ---
                try:
                    # Prefer the current in-scope eval_out/assign (the real final state)
                    fin_eval = locals().get("eval_out", None)
                    fin_assign_arr = locals().get("assign", None)

                    if not isinstance(fin_assign_arr, np.ndarray):
                        fin_assign_arr = locals().get("prev_assign", None)

                    if fin_assign_arr is None:
                        fin_assign_arr = np.asarray(assign_seed, dtype=int).copy()

                    fin_sig = signature_for_assign(fin_assign_arr)

                    # If eval_out missing (exception path), re-evaluate once to keep finalize consistent
                    if not isinstance(fin_eval, dict):
                        try:
                            layout_state.assign = np.asarray(fin_assign_arr, dtype=int).copy()
                            fin_eval = evaluator.evaluate(layout_state)
                        except Exception:
                            fin_eval = {
                                "total_scalar": 0.0,
                                "comm_norm": 0.0,
                                "therm_norm": 0.0,
                                "penalty": {"duplicate": 0.0, "boundary": 0.0},
                            }

                    fin_total = float(fin_eval.get("total_scalar", 0.0))
                    fin_comm = float(fin_eval.get("comm_norm", 0.0))
                    fin_therm = float(fin_eval.get("therm_norm", 0.0))
                    pen = fin_eval.get("penalty", {}) or {}
                    fin_dup = float(pen.get("duplicate", 0.0))
                    fin_bnd = float(pen.get("boundary", 0.0))

                    fin_cache_key = _make_cache_key(fin_sig, objective_hash)
                    cache_hit_cum = int(eval_cache.hits) if eval_cache is not None else 0
                    cache_miss_cum = int(eval_cache.misses) if eval_cache is not None else 0

                    try:
                        eval_calls_cum = int(getattr(evaluator, "evaluator_calls", eval_calls_cum))
                    except Exception:
                        pass

                    fin_row = {
                        "iter": int(last_step + 1) if last_step >= 0 else 0,
                        "stage": "finalize",
                        "op": "finalize",
                        "op_args_json": json.dumps({"op": "finalize"}, ensure_ascii=False),
                        # IMPORTANT: finalize must not affect accept_rate
                        "accepted": 0,
                        "total_scalar": fin_total,
                        "comm_norm": fin_comm,
                        "therm_norm": fin_therm,
                        "pareto_added": 0,
                        "duplicate_penalty": fin_dup,
                        "boundary_penalty": fin_bnd,
                        "seed_id": int(seed_id),
                        "time_ms": int((time.time() - start_time) * 1000),
                        "signature": fin_sig,

                        # d_* required by v5.4 schema
                        "d_total": 0.0,
                        "d_comm": 0.0,
                        "d_therm": 0.0,

                        "delta_total": 0.0,
                        "delta_comm": 0.0,
                        "delta_therm": 0.0,
                        "tabu_hit": 0,
                        "inverse_hit": 0,
                        "cooldown_hit": 0,
                        "policy": "finalize",
                        "move": "finalize",
                        "lookahead_k": 0,
                        "cache_hit": 0,
                        "cache_key": fin_cache_key,
                        "objective_hash": objective_hash,
                        "eval_calls_cum": int(eval_calls_cum),
                        "cache_hit_cum": cache_hit_cum,
                        "cache_miss_cum": cache_miss_cum,
                        "cache_saved_eval_calls_cum": cache_hit_cum,
                        "llm_used": 0,
                        "llm_fail_count": llm_fail_count,
                        "fallback_reason": fallback_reason_init,
                        "wall_time_ms_cum": int((time.perf_counter() - wall_start) * 1000),
                        "accepted_steps_cum": int(accepted_steps),
                        "sim_eval_calls_cum": int(eval_calls_cum),
                        "lookahead_enabled": int(lookahead_enabled),
                        "lookahead_r": float(lookahead_r) if lookahead_enabled else 0.0,
                        "notes": "",
                    }
                    writer.writerow(fin_row)
                except Exception:
                    pass
                f_trace.flush()
                os.fsync(f_trace.fileno())
                if recordings_fp is not None:
                    recordings_fp.flush()
                    os.fsync(recordings_fp.fileno())
                if best_solution is not None:
                    (out_dir / "layout_best.json").write_text(
                        json.dumps(best_solution, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                if report is not None:
                    (out_dir / "report.json").write_text(
                        json.dumps(report, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
    except Exception as e:
        if e.__class__.__name__ == "_BudgetExceeded":
            budget_exhausted = True
        else:
            raise
    finally:
        if usage_fp:
            usage_fp.close()
        if recordings_fp is not None:
            recordings_fp.close()

    cache_size = int(_cfg_get(ps_cfg, "cache_size", 0)) if ps_cfg is not None else 0
    if eval_cache is not None:
        cache_size = int(getattr(eval_cache, "max_size", cache_size))

    policy_meta = {
        "planner_type": planner_type,
        "steps": int(steps),
        "steps_planned": int(planned_steps),
        "steps_executed": int(last_step + 1) if last_step >= 0 else 0,
        "budget_exhausted": bool(budget_exhausted),
        "lookahead": {"enabled": bool(lookahead_enabled), "k": int(lookahead_k), "r": int(lookahead_r), "mc": int(lookahead_mc), "alpha": float(lookahead_alpha)},
        "objective": {"hash": objective_hash, "cfg": evaluator.objective_cfg_dict()},
        "policy_switch": {
            "enabled": bool(use_ps),
            "last_action_family": getattr(controller, "last_action_family", None),
            "last_policy": getattr(controller, "last_policy", None),
            "cache_size": int(cache_size),
        },
        "cache": {"hit_rate": float(eval_cache.hit_rate) if eval_cache is not None else None},
        "run_signature": _build_run_signature(dp_cfg),
    }
    if mpvs_enabled:
        try:
            if macro_engine is not None:
                mpvs_stats["macro_ops"] = macro_engine.snapshot()
        except Exception:
            pass
        try:
            if roi_tracker is not None:
                mpvs_stats["verifier_roi_ewma"] = roi_tracker.snapshot()
        except Exception:
            pass
        try:
            if mem_bank is not None:
                mm = int(_cfg_get(_cfg_get(mpvs_cfg, "memory", {}) or {}, "snapshot_max", 50))
                snap = mem_bank.snapshot()
                mpvs_stats["mem_bank"] = snap[: max(0, mm)]
        except Exception:
            pass
        try:
            if mpvs_ctrl is not None:
                snap_ctrl = mpvs_ctrl.snapshot()
                mpvs_stats["comp_ctrl"] = snap_ctrl
                mpvs_stats["heuristic_rate_ewma"] = float(snap_ctrl.get("heur_rate_ewma", mpvs_stats.get("heuristic_rate_ewma", 0.0)))
        except Exception:
            pass
        policy_meta["mpvs"] = mpvs_stats
    (out_dir / "trace_meta.json").write_text(json.dumps(policy_meta, indent=2), encoding="utf-8")

    return DetailedPlaceResult(assign=assign, pareto=pareto, trace_path=trace_path, policy_meta=policy_meta)
