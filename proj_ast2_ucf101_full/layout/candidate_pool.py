"""Candidate pool construction for detailed placement LLM planner."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from layout.coarsen import Cluster
from layout.evaluator import LayoutEvaluator, LayoutState


@dataclass
class Candidate:
    id: int
    action: Dict[str, Any]
    type: str
    features: Dict[str, Any]
    est: Dict[str, float]
    score: float
    signature: str


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _radial_norm(point: np.ndarray, max_r: float) -> float:
    if max_r <= 0:
        return 0.0
    return float(np.linalg.norm(point) / max_r)


def _apply_action(assign: np.ndarray, action: Dict[str, Any], clusters: List[Cluster], site_to_region: np.ndarray):
    op = action.get("op")
    if op == "swap":
        i = int(action.get("i", 0))
        j = int(action.get("j", 0))
        assign[i], assign[j] = assign[j], assign[i]
    elif op == "relocate":
        i = int(action.get("i", 0))
        site_id = int(action.get("site_id", 0))
        assign[i] = site_id
    elif op == "cluster_move":
        cid = int(action.get("cluster_id", -1))
        rid = int(action.get("region_id", -1))
        if 0 <= cid < len(clusters):
            cluster = clusters[cid]
            target_sites = [s for s, r in enumerate(site_to_region) if int(r) == rid][: len(cluster.slots)]
            for slot, site in zip(cluster.slots, target_sites):
                assign[int(slot)] = int(site)


def _signature_for_action(action: Dict[str, Any], assign: Optional[np.ndarray] = None) -> str:
    op = action.get("op")
    if op == "swap":
        i = int(action.get("i", 0))
        j = int(action.get("j", 0))
        ii, jj = sorted([i, j])
        return f"swap:{ii}-{jj}"
    if op == "relocate":
        slot = int(action.get("i", -1))
        to_site = int(action.get("site_id", -1))
        return f"rel:{slot}->{to_site}"
    if op == "cluster_move":
        return f"cl:{int(action.get('cluster_id', -1))}->{int(action.get('region_id', -1))}"
    return "none"


def inverse_signature(action: Dict[str, Any], assign: Optional[np.ndarray] = None) -> str:
    op = action.get("op")
    if op == "swap":
        return _signature_for_action(action, assign)
    if op == "relocate":
        slot = int(action.get("i", -1))
        from_site = action.get("from_site", None)
        if from_site is None and assign is not None and 0 <= slot < len(assign):
            from_site = int(assign[slot])
        return f"rel:{slot}->{int(from_site) if from_site is not None else -1}"
    if op == "cluster_move":
        return _signature_for_action(action, assign)
    return "none"


def _evaluate_action(
    assign: np.ndarray,
    action: Dict[str, Any],
    evaluator: LayoutEvaluator,
    layout_state: LayoutState,
    clusters: List[Cluster],
    site_to_region: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    new_assign = assign.copy()
    _apply_action(new_assign, action, clusters, site_to_region)
    layout_state.assign = new_assign
    eval_new = evaluator.evaluate(layout_state)
    layout_state.assign = assign
    return new_assign, {
        "total_new": float(eval_new.get("total_scalar", 0.0)),
        "comm_new": float(eval_new.get("comm_norm", 0.0)),
        "therm_new": float(eval_new.get("therm_norm", 0.0)),
        "pen_dup": float(eval_new.get("penalty", {}).get("duplicate", 0.0)),
        "pen_bnd": float(eval_new.get("penalty", {}).get("boundary", 0.0)),
    }


def _traffic_gain_hint(traffic_sym: np.ndarray, sites_xy: np.ndarray, assign: np.ndarray, i: int, j: Optional[int] = None, site_id: Optional[int] = None) -> float:
    if j is not None:
        cur = float(traffic_sym[i, j]) * float(np.linalg.norm(sites_xy[assign[i]] - sites_xy[assign[j]]))
        swapped = float(traffic_sym[i, j]) * float(np.linalg.norm(sites_xy[assign[i]] - sites_xy[assign[j]]))
        return cur - swapped
    if site_id is not None:
        cur_site = int(assign[i])
        dist_cur = float(np.linalg.norm(sites_xy[cur_site] - sites_xy[site_id]))
        return -dist_cur
    return 0.0


def build_candidate_pool(
    assign: np.ndarray,
    eval_out: Dict[str, float],
    evaluator: LayoutEvaluator,
    layout_state: LayoutState,
    traffic_sym: np.ndarray,
    sites_xy: np.ndarray,
    site_to_region: np.ndarray,
    regions,
    clusters: List[Cluster],
    cluster_to_region: List[int],
    chip_tdp: Optional[np.ndarray],
    cfg: Any,
    rng: np.random.Generator,
) -> List[Candidate]:
    raw_candidates: List[Candidate] = []
    candidate_id = 0

    anti_cfg = _cfg_get(cfg, "anti_oscillation", {}) or {}
    max_relocate_final = int(_cfg_get(anti_cfg, "max_relocate_per_slot_in_final", 2))

    S = assign.shape[0]
    Ns = site_to_region.shape[0]
    max_r = float(np.max(np.linalg.norm(sites_xy, axis=1))) if len(sites_xy) else 1.0

    base_total = float(eval_out.get("total_scalar", 0.0))
    base_comm = float(eval_out.get("comm_norm", 0.0))
    base_therm = float(eval_out.get("therm_norm", 0.0))

    top_pairs: List[Tuple[int, int, float]] = []
    for i in range(S):
        for j in range(i + 1, S):
            top_pairs.append((i, j, float(traffic_sym[i, j])))
    top_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = top_pairs[: max(5, int(_cfg_get(cfg, "top_pairs_k", 20)))]

    slot_heat = traffic_sym.sum(axis=1)
    if chip_tdp is not None and len(chip_tdp) == S:
        slot_heat = slot_heat + chip_tdp
    hot_slots = list(np.argsort(slot_heat)[::-1])

    used_sites = set(int(x) for x in assign.tolist())
    empty_sites = [s for s in range(Ns) if s not in used_sites]

    def add_candidate(action: Dict[str, Any], ctype: str, touches_hot_pair: bool = False, touches_hot_slot: bool = False, region_from=None, region_to=None, radial_from=None, radial_to=None):
        nonlocal candidate_id
        _, est = _evaluate_action(assign, action, evaluator, layout_state, clusters, site_to_region)
        if est.get("pen_dup", 0) > 0 or est.get("pen_bnd", 0) > 0:
            return

        d_total = est["total_new"] - base_total
        d_comm = est["comm_new"] - base_comm
        d_therm = est["therm_new"] - base_therm
        denom_total = max(abs(base_total), 1e-6)
        denom_comm = max(abs(base_comm), 1e-6)
        denom_therm = max(abs(base_therm), 1e-6)
        d_total_z = float(np.clip(d_total / denom_total, -1, 1))
        d_comm_z = float(np.clip(d_comm / denom_comm, -1, 1))
        d_therm_z = float(np.clip(d_therm / denom_therm, -1, 1))

        if action.get("op") == "relocate":
            action["from_site"] = int(assign[int(action.get("i", -1))])
        action_sig = _signature_for_action(action, assign)
        traffic_hint = _traffic_gain_hint(traffic_sym, sites_xy, assign, int(action.get("i", 0)), action.get("j"), action.get("site_id"))

        features = {
            "touches_hot_pair": touches_hot_pair,
            "touches_hot_slot": touches_hot_slot,
            "op_signature": action_sig,
            "region_from": region_from,
            "region_to": region_to,
            "radial_from": radial_from,
            "radial_to": radial_to,
            "traffic_gain_hint": traffic_hint,
        }
        est_full = {
            **est,
            "d_total": d_total,
            "d_comm": d_comm,
            "d_therm": d_therm,
            "d_total_z": d_total_z,
            "d_comm_z": d_comm_z,
            "d_therm_z": d_therm_z,
        }

        wT = float(_cfg_get(cfg, "w_total", 0.7))
        wC = float(_cfg_get(cfg, "w_comm", 0.2))
        wH = float(_cfg_get(cfg, "w_therm", 0.1))
        score = wT * d_total_z + wC * d_comm_z + wH * d_therm_z
        if ctype == "relocate":
            score -= 0.03
        if ctype == "cluster_move":
            score -= 0.02
        if touches_hot_pair:
            score -= 0.02
        if radial_from is not None and radial_to is not None and radial_to > radial_from and chip_tdp is not None:
            score -= 0.02

        cand = Candidate(
            id=candidate_id,
            action=action,
            type=ctype,
            features=features,
            est=est_full,
            score=float(score),
            signature=action_sig,
        )
        raw_candidates.append(cand)
        candidate_id += 1

    # A) Hot-pair swap
    for i, j, _ in top_pairs[:20]:
        add_candidate({"op": "swap", "i": int(i), "j": int(j)}, "swap", touches_hot_pair=True)

    # B) Hot-slot neighbor swap
    for slot in hot_slots[: min(20, len(hot_slots))]:
        cur_site = int(assign[slot])
        dists = [
            (other, float(np.linalg.norm(sites_xy[cur_site] - sites_xy[int(assign[other])])))
            for other in range(S)
            if other != slot
        ]
        dists.sort(key=lambda x: x[1])
        for other, _ in dists[:8]:
            add_candidate({"op": "swap", "i": int(slot), "j": int(other)}, "swap", touches_hot_slot=True)

    # C) Triangle/cycle swap
    for i, j, _ in top_pairs[:10]:
        for k in range(S):
            if k in (i, j):
                continue
            add_candidate({"op": "swap", "i": int(i), "j": int(k)}, "swap")
            add_candidate({"op": "swap", "i": int(j), "j": int(k)}, "swap")
            break

    # D) Region-aware swap
    for i in hot_slots[: min(15, len(hot_slots))]:
        reg_i = int(site_to_region[int(assign[i])])
        for j in range(S):
            if i == j:
                continue
            reg_j = int(site_to_region[int(assign[j])])
            if reg_i == reg_j:
                add_candidate({"op": "swap", "i": int(i), "j": int(j)}, "swap", region_from=reg_i, region_to=reg_j)

    # E) Relocate same-region empty
    for slot in hot_slots[: min(25, len(hot_slots))]:
        reg = int(site_to_region[int(assign[slot])])
        region_sites = [s for s in empty_sites if int(site_to_region[s]) == reg]
        rng.shuffle(region_sites)
        for site_id in region_sites[:4]:
            rf = _radial_norm(sites_xy[int(assign[slot])], max_r)
            rt = _radial_norm(sites_xy[site_id], max_r)
            add_candidate({"op": "relocate", "i": int(slot), "site_id": int(site_id)}, "relocate", touches_hot_slot=True, region_from=reg, region_to=reg, radial_from=rf, radial_to=rt)

    # F) Relocate outward high-tdp
    if chip_tdp is not None and len(chip_tdp) == S:
        high_slots = list(np.argsort(chip_tdp)[::-1][: min(20, S)])
        for slot in high_slots:
            cur_r = _radial_norm(sites_xy[int(assign[slot])], max_r)
            far_sites = sorted(empty_sites, key=lambda s: _radial_norm(sites_xy[s], max_r), reverse=True)
            for site_id in far_sites[:3]:
                rt = _radial_norm(sites_xy[site_id], max_r)
                add_candidate({"op": "relocate", "i": int(slot), "site_id": int(site_id)}, "relocate", touches_hot_slot=True, radial_from=cur_r, radial_to=rt)

    # G) Relocate to reduce comm
    for i, j, _ in top_pairs[:10]:
        anchor_site = int(assign[j])
        dists = [(s, float(np.linalg.norm(sites_xy[anchor_site] - sites_xy[s]))) for s in empty_sites]
        dists.sort(key=lambda x: x[1])
        for site_id, _ in dists[:3]:
            rf = _radial_norm(sites_xy[int(assign[i])], max_r)
            rt = _radial_norm(sites_xy[site_id], max_r)
            add_candidate({"op": "relocate", "i": int(i), "site_id": int(site_id)}, "relocate", touches_hot_pair=True, radial_from=rf, radial_to=rt)

    # H) Cluster_move empty-first
    for cluster in clusters[:8]:
        cid = int(cluster.cluster_id)
        cur_region = int(cluster_to_region[cid]) if cid < len(cluster_to_region) else -1
        for region in regions:
            rid = int(region.region_id)
            if rid == cur_region:
                continue
            target_sites = [s for s in empty_sites if int(site_to_region[s]) == rid]
            if len(target_sites) < len(cluster.slots):
                continue
            add_candidate({"op": "cluster_move", "cluster_id": cid, "region_id": rid}, "cluster_move", region_from=cur_region, region_to=rid)

    # I) Exploration random
    for _ in range(10):
        if rng.random() < 0.5 and len(empty_sites) > 0:
            slot = int(rng.integers(0, S))
            site_id = int(rng.choice(empty_sites))
            add_candidate({"op": "relocate", "i": slot, "site_id": site_id}, "relocate")
        else:
            i = int(rng.integers(0, S))
            j = int(rng.integers(0, S))
            if i != j:
                add_candidate({"op": "swap", "i": i, "j": j}, "swap")

    # scoring and filtering
    raw_candidates = [c for c in raw_candidates if c.est.get("pen_dup", 0) <= 0 and c.est.get("pen_bnd", 0) <= 0]
    raw_candidates.sort(key=lambda c: c.score)
    topN = min(80, len(raw_candidates))
    candidates = raw_candidates[:topN]

    relocate_by_slot: Dict[int, List[Candidate]] = {}
    filtered_candidates: List[Candidate] = []
    for c in candidates:
        if c.type != "relocate":
            filtered_candidates.append(c)
            continue
        slot = int(c.action.get("i", -1))
        relocate_by_slot.setdefault(slot, []).append(c)
    for slot, cands in relocate_by_slot.items():
        cands_sorted = sorted(cands, key=lambda cc: cc.score)
        filtered_candidates.extend(cands_sorted[:max_relocate_final])
    candidates = filtered_candidates

    # diversity packing
    final: List[Candidate] = []
    sig_seen = set()
    type_quota = {"relocate": 18, "cluster_move": 8, "swap": 18}
    exploration_quota = 6
    per_slot_relocate = {}

    def try_add(c: Candidate):
        if c.features.get("op_signature") in sig_seen:
            return False
        if c.type == "relocate":
            slot = int(c.action.get("i", -1))
            per_slot_relocate[slot] = per_slot_relocate.get(slot, 0) + 1
            if per_slot_relocate[slot] > max_relocate_final:
                return False
        final.append(c)
        sig_seen.add(c.features.get("op_signature"))
        return True

    # enforce quotas
    for t in ["relocate", "cluster_move", "swap"]:
        needed = type_quota.get(t, 0)
        for c in candidates:
            if c.type == t and len([x for x in final if x.type == t]) < needed:
                try_add(c)

    # exploration/random fill
    explore_added = 0
    for c in candidates:
        if explore_added >= exploration_quota:
            break
        if c.type in {"swap", "relocate"} and rng.random() < 0.3:
            if try_add(c):
                explore_added += 1

    for c in candidates:
        if len(final) >= 60:
            break
        try_add(c)

    return final


def simulate_action_sequence(assign: np.ndarray, picks: List[int], cand_map: Dict[int, Candidate], clusters: List[Cluster], site_to_region: np.ndarray, max_k: int = 4) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    temp_assign = assign.copy()
    used_sites = set(int(x) for x in temp_assign.tolist())
    for pid in picks[:max_k]:
        cand = cand_map.get(pid)
        if cand is None:
            continue
        action = copy.deepcopy(cand.action)
        if "signature" not in action:
            action["signature"] = cand.signature if hasattr(cand, "signature") else _signature_for_action(action, assign)
        op = action.get("op")
        if op == "relocate":
            target = int(action.get("site_id", -1))
            if target in used_sites:
                continue
            used_sites.discard(int(temp_assign[int(action.get("i", 0))]))
            used_sites.add(target)
        _apply_action(temp_assign, action, clusters, site_to_region)
        if len(set(temp_assign.tolist())) != len(temp_assign):
            continue
        actions.append({**action, "candidate_id": pid, "type": cand.type, "signature": action.get("signature", cand.signature)})
        if len(actions) >= max_k:
            break
    return actions
