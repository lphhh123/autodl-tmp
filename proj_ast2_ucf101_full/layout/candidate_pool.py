from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ----------------------------
# Assign signature (trace uses assign-signature, not op-signature)
# ----------------------------
def signature_from_assign(assign: Any) -> str:
    """
    Canonical signature for an assignment vector (trace.csv uses this).
    Accepts: list[int] / np.ndarray / anything array-like of ints.
    """
    try:
        a = np.asarray(assign, dtype=int).reshape(-1)
        return "assign:" + ",".join(str(int(x)) for x in a.tolist())
    except Exception:
        return "assign:unknown"


# Back-compat alias (some files call signature_for_assign)
def signature_for_assign(assign: Any) -> str:
    return signature_from_assign(assign)


# ----------------------------
# Candidate dataclass
# ----------------------------
@dataclass
class Candidate:
    id: int
    type: str                # "swap" | "relocate" | "cluster_move"
    action: Dict[str, Any]   # op-args dict (must include "op")
    est: Dict[str, Any]      # evaluation dict (includes d_total/d_comm/d_therm)
    signature: str           # op-signature for tabu/inverse checks
    bucket: str = ""         # diversity bucket


# ----------------------------
# Op signature for tabu (NOT used in trace; trace uses assign signature)
# ----------------------------
def _signature_for_action(action: Dict[str, Any], base_assign: Optional[np.ndarray] = None) -> str:
    op = str(action.get("op", "noop"))
    if op == "swap":
        i = int(action.get("i", -1))
        j = int(action.get("j", -1))
        a, b = (i, j) if i <= j else (j, i)
        return f"swap:{a}:{b}"
    if op == "relocate":
        i = int(action.get("i", -1))
        to_site = int(action.get("site_id", -1))
        from_site = action.get("from_site", None)
        if from_site is None and base_assign is not None and 0 <= i < base_assign.shape[0]:
            from_site = int(base_assign[i])
        from_site = int(from_site) if from_site is not None else -1
        return f"relocate:{i}:{from_site}->{to_site}"
    if op == "cluster_move":
        cid = int(action.get("cluster_id", -1))
        to_region = int(action.get("region_id", -1))
        from_region = action.get("from_region", None)
        from_region = int(from_region) if from_region is not None else -1
        # target_sites may be long; hash deterministically
        tgt = action.get("target_sites", []) or []
        try:
            tgt_arr = np.asarray(tgt, dtype=int).reshape(-1)
            h = int(np.sum((tgt_arr + 17) * 1315423911) % 1000003)
        except Exception:
            h = 0
        return f"cluster_move:{cid}:{from_region}->{to_region}:h{h}"
    return f"{op}:unknown"


def inverse_signature(action: Dict[str, Any], base_assign: np.ndarray) -> str:
    """
    Signature of the inverse action (used by detailed_place for anti-oscillation).
    Must be consistent with _signature_for_action().
    """
    op = str(action.get("op", "noop"))
    if op == "swap":
        # self-inverse
        return _signature_for_action(action, base_assign)
    if op == "relocate":
        inv = dict(action)
        i = int(inv.get("i", -1))
        inv_to = inv.get("site_id", None)
        inv_from = inv.get("from_site", None)
        if inv_from is None and 0 <= i < base_assign.shape[0]:
            inv_from = int(base_assign[i])
        # invert: go back to from_site
        inv["site_id"] = int(inv_from) if inv_from is not None else -1
        inv["from_site"] = int(inv_to) if inv_to is not None else -1
        return _signature_for_action(inv, base_assign)
    if op == "cluster_move":
        inv = dict(action)
        inv_to = inv.get("region_id", None)
        inv_from = inv.get("from_region", None)
        if inv_from is None:
            inv_from = -1
        inv["region_id"] = int(inv_from)
        inv["from_region"] = int(inv_to) if inv_to is not None else -1
        return _signature_for_action(inv, base_assign)
    return "inv:unknown"


# ----------------------------
# Helpers to apply actions to an assign vector
# ----------------------------
def _apply_swap(assign: np.ndarray, i: int, j: int) -> np.ndarray:
    a = assign.copy()
    if i == j:
        return a
    a[i], a[j] = int(a[j]), int(a[i])
    return a


def _apply_relocate(assign: np.ndarray, i: int, site_id: int) -> np.ndarray:
    a = assign.copy()
    a[i] = int(site_id)
    return a


def _apply_cluster_move(assign: np.ndarray, cluster_slots: List[int], target_sites: List[int]) -> np.ndarray:
    a = assign.copy()
    for k, s in enumerate(cluster_slots):
        a[int(s)] = int(target_sites[k])
    return a


def _nearest_sites(sites_xy: np.ndarray, anchor_xy: np.ndarray, cand_sites: np.ndarray, k: int) -> List[int]:
    if cand_sites.size == 0:
        return []
    pts = sites_xy[cand_sites]
    d2 = np.sum((pts - anchor_xy[None, :]) ** 2, axis=1)
    idx = np.argsort(d2)[: max(1, min(int(k), d2.shape[0]))]
    return [int(cand_sites[t]) for t in idx.tolist()]


# ----------------------------
# Build candidate pool
# ----------------------------
def build_candidate_pool(
    assign: np.ndarray,
    eval_out: Dict[str, Any],
    evaluator: Any,
    layout_state: Any,
    traffic_sym: np.ndarray,
    sites_xy: np.ndarray,
    site_to_region: np.ndarray,
    regions: Any,
    clusters: List[Any],
    cluster_to_region: np.ndarray,
    chip_tdp: np.ndarray,
    cfg: Any,
    rng: Any,
    debug_out_path: Optional[Path] = None,
) -> List[Candidate]:
    """
    Must provide: Candidate list with fields used by detailed_place:
      - cand.id / cand.type / cand.action / cand.est / cand.signature
    This implementation is "always-correct" (evaluates candidates by calling evaluator).
    """
    S = int(assign.shape[0])
    Ns = int(sites_xy.shape[0])

    # cfg access helper (dict / OmegaConf / obj)
    def _get(obj, key, default=None):
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        try:
            return obj.get(key, default)
        except Exception:
            return getattr(obj, key, default)

    cpcfg = _get(cfg, "candidate_pool", {}) or {}
    raw_target_size = int(_get(cpcfg, "raw_target_size", 180))
    raw_target_max = int(_get(cpcfg, "raw_target_max", 220))
    final_size = int(_get(cpcfg, "final_size", 60))
    diversity_enabled = bool(_get(cpcfg, "diversity_enabled", True))
    coverage_min_per_bucket = int(_get(cpcfg, "coverage_min_per_bucket", 6))

    dp = cfg  # detailed_place cfg is passed here
    action_probs = _get(dp, "action_probs", {}) or {}
    p_swap = float(_get(action_probs, "swap", 0.55))
    p_reloc = float(_get(action_probs, "relocate", 0.35))
    p_cmov = float(_get(action_probs, "cluster_move", 0.10))
    totp = max(1e-9, p_swap + p_reloc + p_cmov)

    relocate_cfg = _get(dp, "relocate", {}) or {}
    same_region_prob = float(_get(relocate_cfg, "same_region_prob", 0.8))
    neighbor_k = int(_get(relocate_cfg, "neighbor_k", 30))

    hot_cfg = _get(dp, "hot_sampling", {}) or {}
    top_pairs_k = int(_get(hot_cfg, "top_pairs_k", 20))
    top_slots_k = int(_get(hot_cfg, "top_slots_k", 20))

    base_total = float(eval_out.get("total_scalar", 0.0))
    base_comm = float(eval_out.get("comm_norm", 0.0))
    base_therm = float(eval_out.get("therm_norm", 0.0))

    # compute "hot" slot/pairs
    traffic_sym = np.asarray(traffic_sym, dtype=float)
    # top pairs by traffic
    pairs: List[Tuple[int, int, float]] = []
    for i in range(S):
        for j in range(i + 1, S):
            pairs.append((i, j, float(traffic_sym[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = pairs[: max(1, min(len(pairs), top_pairs_k))]

    # top slots by sum traffic + tdp
    slot_score = np.sum(traffic_sym, axis=1) + 1e-6 * np.asarray(chip_tdp, dtype=float)
    top_slots = np.argsort(-slot_score)[: max(1, min(S, top_slots_k))].tolist()

    # precompute region -> site indices
    site_to_region = np.asarray(site_to_region, dtype=int).reshape(-1)
    region_sites: Dict[int, np.ndarray] = {}
    for r in np.unique(site_to_region).tolist():
        region_sites[int(r)] = np.where(site_to_region == int(r))[0].astype(int)

    # build candidates
    cands: List[Candidate] = []
    cid = 0

    def _eval_assign(new_assign: np.ndarray) -> Dict[str, Any]:
        layout_state.assign = new_assign
        eo = evaluator.evaluate(layout_state)
        return eo

    def _push(type_name: str, action: Dict[str, Any], new_assign: np.ndarray, bucket: str):
        nonlocal cid, cands
        eo = _eval_assign(new_assign)
        d_total = float(eo["total_scalar"]) - base_total
        d_comm = float(eo["comm_norm"]) - base_comm
        d_therm = float(eo["therm_norm"]) - base_therm
        est = {
            "total_new": float(eo["total_scalar"]),
            "comm_new": float(eo["comm_norm"]),
            "therm_new": float(eo["therm_norm"]),
            "d_total": d_total,
            "d_comm": d_comm,
            "d_therm": d_therm,
            "penalty": eo.get("penalty", {}),
        }
        sig = _signature_for_action(action, base_assign=assign)
        cands.append(
            Candidate(
                id=cid,
                type=type_name,
                action=action,
                est=est,
                signature=sig,
                bucket=bucket,
            )
        )
        cid += 1

    # target raw counts by probs
    raw_n_swap = int(raw_target_size * (p_swap / totp))
    raw_n_reloc = int(raw_target_size * (p_reloc / totp))
    raw_n_cmov = max(1, raw_target_size - raw_n_swap - raw_n_reloc) if p_cmov > 0 else 0

    # swaps: use top_pairs, plus random fill
    for (i, j, _) in top_pairs[:raw_n_swap]:
        act = {"op": "swap", "i": int(i), "j": int(j), "type": "swap"}
        newa = _apply_swap(assign, int(i), int(j))
        _push("swap", act, newa, bucket="swap:hot")
    while len([c for c in cands if c.type == "swap"]) < raw_n_swap and len(cands) < raw_target_max:
        i = int(rng.randrange(0, S))
        j = int(rng.randrange(0, S))
        if i == j:
            continue
        act = {"op": "swap", "i": int(i), "j": int(j), "type": "swap"}
        newa = _apply_swap(assign, int(i), int(j))
        _push("swap", act, newa, bucket="swap:rand")

    # relocates: pick hot slots, propose nearby sites (same-region preferred)
    for i in top_slots:
        if len([c for c in cands if c.type == "relocate"]) >= raw_n_reloc or len(cands) >= raw_target_max:
            break
        i = int(i)
        cur_site = int(assign[i])
        cur_xy = sites_xy[cur_site]
        cur_r = int(site_to_region[cur_site])
        if rng.random() < same_region_prob:
            cand_sites = region_sites.get(cur_r, np.array([], dtype=int))
        else:
            # random region
            all_regions = list(region_sites.keys())
            rr = int(all_regions[int(rng.randrange(0, len(all_regions)))])
            cand_sites = region_sites.get(rr, np.array([], dtype=int))
        neigh = _nearest_sites(sites_xy, cur_xy, cand_sites, neighbor_k)
        for to_site in neigh:
            if to_site == cur_site:
                continue
            act = {"op": "relocate", "i": i, "site_id": int(to_site), "from_site": int(cur_site), "type": "relocate"}
            newa = _apply_relocate(assign, i, int(to_site))
            _push("relocate", act, newa, bucket="relocate:near")
            if len([c for c in cands if c.type == "relocate"]) >= raw_n_reloc or len(cands) >= raw_target_max:
                break

    # cluster_move: move a whole cluster to a (possibly different) region
    if raw_n_cmov > 0 and clusters:
        for _ in range(raw_n_cmov):
            if len(cands) >= raw_target_max:
                break
            cl = clusters[int(rng.randrange(0, len(clusters)))]
            slots = [int(s) for s in getattr(cl, "slots", [])]
            if not slots:
                continue
            cid_cl = int(getattr(cl, "id", -1))
            from_region = int(cluster_to_region[cid_cl]) if 0 <= cid_cl < int(cluster_to_region.shape[0]) else -1
            # pick target region != from_region when possible
            all_regions = list(region_sites.keys())
            if not all_regions:
                continue
            tgt_region = int(all_regions[int(rng.randrange(0, len(all_regions)))])
            if len(all_regions) > 1 and tgt_region == from_region:
                tgt_region = int(all_regions[(all_regions.index(tgt_region) + 1) % len(all_regions)])
            cand_sites = region_sites.get(tgt_region, np.array([], dtype=int))
            if cand_sites.size < len(slots):
                continue

            # anchor: centroid of current cluster placement
            cur_sites = np.asarray([int(assign[s]) for s in slots], dtype=int)
            anchor = np.mean(sites_xy[cur_sites], axis=0)
            chosen = _nearest_sites(sites_xy, anchor, cand_sites, k=max(len(slots), 3 * len(slots)))
            # ensure enough unique target sites
            uniq = []
            used = set()
            for sid in chosen:
                if sid in used:
                    continue
                used.add(sid)
                uniq.append(int(sid))
                if len(uniq) >= len(slots):
                    break
            if len(uniq) < len(slots):
                continue

            act = {
                "op": "cluster_move",
                "cluster_id": int(cid_cl),
                "from_region": int(from_region),
                "region_id": int(tgt_region),
                "target_sites": uniq,
                "type": "cluster_move",
            }
            newa = _apply_cluster_move(assign, slots, uniq)
            _push("cluster_move", act, newa, bucket="cluster_move:rand")

    # sort by best improvement (d_total ascending)
    cands.sort(key=lambda c: float(c.est.get("d_total", 0.0)))

    # diversity selection across buckets (at least per op-type bucket)
    if diversity_enabled:
        buckets: Dict[str, List[Candidate]] = {}
        for c in cands:
            k = c.type  # op-type bucket is enough for v5.4 fairness/ablation
            buckets.setdefault(k, []).append(c)

        selected: List[Candidate] = []
        used_ids = set()

        # first: minimum per bucket
        for b in ["swap", "relocate", "cluster_move"]:
            if b not in buckets:
                continue
            for c in buckets[b][:coverage_min_per_bucket]:
                if c.id in used_ids:
                    continue
                selected.append(c)
                used_ids.add(c.id)

        # then: fill by global best
        for c in cands:
            if len(selected) >= final_size:
                break
            if c.id in used_ids:
                continue
            selected.append(c)
            used_ids.add(c.id)

        cands = selected[:final_size]
    else:
        cands = cands[:final_size]

    # optional debug dump
    if debug_out_path is not None:
        try:
            debug_out_path.parent.mkdir(parents=True, exist_ok=True)
            rows = []
            for c in cands:
                rows.append(
                    {
                        "id": c.id,
                        "type": c.type,
                        "signature": c.signature,
                        "d_total": float(c.est.get("d_total", 0.0)),
                        "d_comm": float(c.est.get("d_comm", 0.0)),
                        "d_therm": float(c.est.get("d_therm", 0.0)),
                        "action": c.action,
                    }
                )
            debug_out_path.write_text(
                "\n".join([str(r) for r in rows]) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass

    return cands


def build_state_summary(
    assign: np.ndarray,
    eval_out: Dict[str, Any],
    candidates: List[Candidate],
    pick_ids: Optional[List[int]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    LLM prompt/state summary: keep it stable and json-serializable.
    """
    topk = candidates[: min(20, len(candidates))]
    summary = {
        "assign_signature": signature_from_assign(assign),
        "metrics": {
            "total_scalar": float(eval_out.get("total_scalar", 0.0)),
            "comm_norm": float(eval_out.get("comm_norm", 0.0)),
            "therm_norm": float(eval_out.get("therm_norm", 0.0)),
            "penalty": eval_out.get("penalty", {}),
        },
        "candidate_pool_size": int(len(candidates)),
        "top_candidates": [
            {
                "id": int(c.id),
                "type": str(c.type),
                "signature": str(c.signature),
                "d_total": float(c.est.get("d_total", 0.0)),
                "action": c.action,
            }
            for c in topk
        ],
        "pick_ids": [int(x) for x in (pick_ids or [])],
    }
    if extra:
        summary["extra"] = extra
    return summary


def pick_ids_to_actions_sequential(
    base_assign: np.ndarray,
    pick_ids: List[int],
    cand_map: Dict[int, Candidate],
    clusters: List[Any],
    site_to_region: np.ndarray,
    max_k: int,
) -> List[Dict[str, Any]]:
    """
    Convert pick-ids into a sequential executable action list.
    Guarantees:
      - always returns <= max_k actions
      - fills from_site/from_region fields based on the *current* assign
    """
    assign = np.asarray(base_assign, dtype=int).copy()
    site_to_region = np.asarray(site_to_region, dtype=int).reshape(-1)

    # cluster_id -> slots
    cl_slots: Dict[int, List[int]] = {}
    for cl in clusters:
        cid = int(getattr(cl, "id", -1))
        slots = [int(s) for s in getattr(cl, "slots", [])]
        cl_slots[cid] = slots

    actions: List[Dict[str, Any]] = []
    for pid in pick_ids[: max_k]:
        if int(pid) not in cand_map:
            continue
        cand = cand_map[int(pid)]
        act = copy.deepcopy(cand.action)
        op = str(act.get("op", cand.type))

        if op == "relocate":
            i = int(act.get("i", -1))
            if 0 <= i < assign.shape[0]:
                act["from_site"] = int(assign[i])
            act["candidate_id"] = int(cand.id)
            act["type"] = "relocate"
            # apply
            assign = _apply_relocate(assign, int(act["i"]), int(act["site_id"]))
            act["signature"] = _signature_for_action(act, base_assign=assign)

        elif op == "swap":
            i = int(act.get("i", -1))
            j = int(act.get("j", -1))
            act["candidate_id"] = int(cand.id)
            act["type"] = "swap"
            assign = _apply_swap(assign, i, j)
            act["signature"] = _signature_for_action(act, base_assign=assign)

        elif op == "cluster_move":
            cid = int(act.get("cluster_id", -1))
            slots = cl_slots.get(cid, [])
            tgt = act.get("target_sites", []) or []
            if not slots or len(tgt) < len(slots):
                continue
            # refresh from_region by current assign (best-effort)
            cur_regions = []
            for s in slots:
                cur_regions.append(int(site_to_region[int(assign[int(s)])]))
            act["from_region"] = int(max(set(cur_regions), key=cur_regions.count)) if cur_regions else int(act.get("from_region", -1))

            # feasibility: ensure target sites in the declared region_id
            rid = int(act.get("region_id", -1))
            ok = True
            for sid in tgt[: len(slots)]:
                if int(site_to_region[int(sid)]) != rid:
                    ok = False
                    break
            if not ok:
                continue

            act["candidate_id"] = int(cand.id)
            act["type"] = "cluster_move"
            assign = _apply_cluster_move(assign, slots, tgt[: len(slots)])
            act["signature"] = _signature_for_action(act, base_assign=assign)

        else:
            continue

        actions.append(act)

    return actions
