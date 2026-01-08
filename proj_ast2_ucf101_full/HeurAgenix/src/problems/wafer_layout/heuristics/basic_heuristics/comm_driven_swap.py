from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from problems.wafer_layout.components import SwapSlots


def comm_driven_swap(problem_state: Dict, algorithm_data: Dict, **kwargs) -> Tuple[SwapSlots, Dict]:
    env = algorithm_data["env"]
    traffic = np.asarray(env.instance_data["mapping"]["traffic_matrix"], dtype=float)
    S = int(env.instance_data["slots"]["S"])
    best_pair = (0, 1)
    best_val = -1.0
    for i in range(S):
        for j in range(i + 1, S):
            if traffic[i, j] > best_val:
                best_val = float(traffic[i, j])
                best_pair = (i, j)
    return SwapSlots(best_pair[0], best_pair[1]), {"pair": best_pair, "traffic": best_val}
