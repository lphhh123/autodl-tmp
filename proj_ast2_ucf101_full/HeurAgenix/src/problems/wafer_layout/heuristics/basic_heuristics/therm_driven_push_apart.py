from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.problems.wafer_layout.components import RelocateSlot


def therm_driven_push_apart(problem_state: Dict, algorithm_data: Dict, **kwargs) -> Tuple[RelocateSlot, Dict]:
    instance = problem_state["instance_data"]
    slots = instance.get("slots", {})
    tdp = np.asarray(slots.get("tdp", []), dtype=float)
    sites_xy = instance.get("sites_xy")
    if sites_xy is None:
        sites_xy = instance.get("sites", {}).get("sites_xy", [])
    sites_xy = np.asarray(sites_xy, dtype=float)
    if isinstance(slots, dict):
        S = int(slots.get("S", len(tdp)))
    else:
        S = int(len(slots))
    best_pair = (0, 1)
    best_val = -1.0
    for i in range(S):
        for j in range(i + 1, S):
            val = float(tdp[i] * tdp[j])
            if val > best_val:
                best_val = val
                best_pair = (i, j)
    target_slot = best_pair[0]
    radii = np.linalg.norm(sites_xy, axis=1)
    far_site = int(np.argmax(radii)) if radii.size else 0
    return RelocateSlot(target_slot, far_site), {"pair": best_pair, "site": far_site}
