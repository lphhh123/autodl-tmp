import json
import os
import random
from copy import deepcopy
from typing import Dict, Any, List, Tuple

# --- bootstrap sys.path so this script can import project modules ---
import sys
from pathlib import Path
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# ---------------------------------------------------------------

import numpy as np
from layout.evaluator import compute_raw_terms_for_assign, evaluator_version


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _set_used_slots(inp: Dict[str, Any], used: List[int]):
    inp["used_slots"] = [int(x) for x in used]


def _build_traffic_matrix(S: int) -> List[List[int]]:
    return [[0 for _ in range(S)] for _ in range(S)]


def _apply_edges(mat: List[List[int]], edges: List[Tuple[int, int, int]]):
    for u, v, w in edges:
        if 0 <= u < len(mat) and 0 <= v < len(mat):
            mat[u][v] += int(w)


def _chain_skip_edges(S: int, base: int, w1: int, w4: int, w8: int) -> List[Tuple[int, int, int]]:
    edges = []
    for i in range(S):
        if i + 1 < S:
            edges.append((i, i + 1, w1))
        if i + 4 < S:
            edges.append((i, i + 4, w4))
        if i + 8 < S:
            edges.append((i, i + 8, w8))
    return edges


def _chain_skip_randw_edges(S: int, seed: int, base: int) -> List[Tuple[int, int, int]]:
    rng = random.Random(seed)
    edges = []
    for i in range(S):
        if i + 1 < S:
            edges.append((i, i + 1, base + rng.randint(-base // 5, base // 5)))
        if i + 4 < S:
            edges.append((i, i + 4, base + rng.randint(-base // 3, base // 3)))
        if i + 8 < S:
            edges.append((i, i + 8, base + rng.randint(-base // 2, base // 2)))
    # clamp positive
    edges = [(u, v, max(1, w)) for (u, v, w) in edges]
    return edges


def _cluster4_edges(S: int, seed: int, intra_w: int, inter_w: int) -> List[Tuple[int, int, int]]:
    """
    4 clusters, each size S/4 (assume S divisible by 4).
    dense intra, sparse inter, both directed.
    """
    rng = random.Random(seed)
    assert S % 4 == 0
    g = S // 4
    clusters = [list(range(k * g, (k + 1) * g)) for k in range(4)]

    edges = []
    # intra dense
    for c in clusters:
        for u in c:
            for v in c:
                if u == v:
                    continue
                w = intra_w + rng.randint(-intra_w // 5, intra_w // 5)
                edges.append((u, v, max(1, w)))

    # inter sparse: connect representative pairs
    for a in range(4):
        for b in range(4):
            if a == b:
                continue
            # pick 1 random node-pair for each cluster pair
            ua = rng.choice(clusters[a])
            vb = rng.choice(clusters[b])
            w = inter_w + rng.randint(-inter_w // 3, inter_w // 3)
            edges.append((ua, vb, max(1, w)))
    return edges


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="outputs/P3/A3/layout_input.json")
    ap.add_argument("--out_dir", type=str, default="outputs/P3/A3/instances")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma_mm", type=float, default=None,
                    help="sigma_mm used to recompute baseline L_therm (default: read from base baseline.objective_cfg.sigma_mm or 20.0)")
    args = ap.parse_args()

    base = _load_json(args.base)

    sigma_mm = args.sigma_mm
    if sigma_mm is None:
        sigma_mm = float(base.get("baseline", {}).get("objective_cfg", {}).get("sigma_mm", 20.0))

    def _recompute_baseline_fields(x: Dict[str, Any], sigma: float) -> None:
        sites_xy = np.asarray(x.get("sites", {}).get("sites_xy", []), dtype=np.float32)
        assign_grid = np.asarray(x.get("baseline", {}).get("assign_grid", []), dtype=int)
        tdp = np.asarray(x.get("slots", {}).get("tdp", []), dtype=float)
        traffic = np.asarray(x.get("mapping", {}).get("traffic_matrix", []), dtype=float)
        if sites_xy.size == 0 or assign_grid.size == 0 or tdp.size == 0 or traffic.size == 0:
            raise ValueError("[make_layout_inputs] cannot recompute baseline: missing sites_xy/assign_grid/tdp/traffic_matrix")
        raw = compute_raw_terms_for_assign(
            sites_xy_mm=sites_xy,
            assign=assign_grid,
            chip_tdp_w=tdp,
            traffic_bytes=traffic,
            sigma_mm=float(sigma),
        )
        x.setdefault("baseline", {})["L_comm"] = float(raw["L_comm"])
        x.setdefault("baseline", {})["L_therm"] = float(raw["L_therm"])
        x.setdefault("baseline", {})["objective_cfg"] = {
            "objective_version": "v5.4",
            "sigma_mm": float(sigma),
            "evaluator_version": evaluator_version(),
            "baseline_schema": "assign_grid+traffic+tdp",
        }

    inp = deepcopy(base)

    S = int(inp.get("S", 16))
    # A1: used_slots from 12 -> 16 (all slots)
    used = list(range(S))
    _set_used_slots(inp, used)

    # NOTE: do NOT change sites_xy/sites list; only mapping/traffic
    base_weight = 301056

    # Instance 1: chain+skip with fixed weights
    inst1 = deepcopy(inp)
    m1 = _build_traffic_matrix(S)
    edges1 = _chain_skip_edges(S, base_weight, w1=base_weight, w4=base_weight * 2, w8=base_weight * 3)
    _apply_edges(m1, edges1)
    inst1["mapping"]["traffic_matrix"] = m1
    inst1["instance_name"] = "chain_skip"
    _recompute_baseline_fields(inst1, sigma_mm)
    _save_json(inst1, os.path.join(args.out_dir, "layout_input_chain_skip.json"))

    # Instance 2: chain+skip randomized weights
    inst2 = deepcopy(inp)
    m2 = _build_traffic_matrix(S)
    edges2 = _chain_skip_randw_edges(S, seed=args.seed, base=base_weight)
    _apply_edges(m2, edges2)
    inst2["mapping"]["traffic_matrix"] = m2
    inst2["instance_name"] = f"chain_skip_randw_s{args.seed}"
    _recompute_baseline_fields(inst2, sigma_mm)
    _save_json(inst2, os.path.join(args.out_dir, f"layout_input_chain_skip_randw_s{args.seed}.json"))

    # Instance 3: 4-cluster structure
    inst3 = deepcopy(inp)
    m3 = _build_traffic_matrix(S)
    edges3 = _cluster4_edges(S, seed=args.seed, intra_w=base_weight * 3, inter_w=base_weight // 2)
    _apply_edges(m3, edges3)
    inst3["mapping"]["traffic_matrix"] = m3
    inst3["instance_name"] = f"cluster4_s{args.seed}"
    _recompute_baseline_fields(inst3, sigma_mm)
    _save_json(inst3, os.path.join(args.out_dir, f"layout_input_cluster4_s{args.seed}.json"))

    print("[make_layout_inputs] wrote:")
    print(" -", os.path.join(args.out_dir, "layout_input_chain_skip.json"))
    print(" -", os.path.join(args.out_dir, f"layout_input_chain_skip_randw_s{args.seed}.json"))
    print(" -", os.path.join(args.out_dir, f"layout_input_cluster4_s{args.seed}.json"))


if __name__ == "__main__":
    main()
