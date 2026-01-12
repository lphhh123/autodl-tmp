from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.problems.wafer_layout.components import SwapSlots


def comm_driven_swap(problem_state: Dict, algorithm_data: Dict, **kwargs) -> Tuple[SwapSlots, Dict]:
    instance = problem_state["instance_data"]
    traffic = np.asarray(instance.get("mapping", {}).get("traffic_matrix", []), dtype=float)
    slots = instance.get("slots", {})
    if isinstance(slots, dict):
        S = int(slots.get("S", len(slots.get("tdp", []) or [])))
    else:
        S = int(len(slots))
    best_pair = (0, 1)
    best_val = -1.0
    for i in range(S):
        for j in range(i + 1, S):
            if traffic[i, j] > best_val:
                best_val = float(traffic[i, j])
                best_pair = (i, j)
    return SwapSlots(best_pair[0], best_pair[1]), {"pair": best_pair, "traffic": best_val}
