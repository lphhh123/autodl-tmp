"""Assignment legalization utilities (SPEC v5.4 §8.5)."""
from __future__ import annotations

import numpy as np


def legalize_assign(assign: np.ndarray, sites_xy: np.ndarray, wafer_radius_mm: float) -> np.ndarray:
    legalized = assign.copy()
    Ns = sites_xy.shape[0]
    S = legalized.shape[0]
    used = set()
    free_sites = [i for i in range(Ns)]
    # resolve duplicates
    for i in range(S):
        site = int(legalized[i])
        if site < 0 or site >= Ns or site in used:
            # pick nearest free site
            best = None
            best_d = float("inf")
            for fs in free_sites:
                d = float(np.linalg.norm(sites_xy[fs]))
                if d < best_d:
                    best_d = d
                    best = fs
            if best is None:
                best = 0
            legalized[i] = best
            site = best
        used.add(site)
        if site in free_sites:
            free_sites.remove(site)
    # boundary check: move outside points inward to nearest valid site
    for i in range(S):
        pos = sites_xy[legalized[i]]
        if np.linalg.norm(pos) > wafer_radius_mm + 1e-6:
            # pick closest in-bound site
            dists = np.linalg.norm(sites_xy, axis=1)
            legal_mask = dists <= wafer_radius_mm + 1e-6
            if legal_mask.any():
                idx = int(np.argmin(np.where(legal_mask, np.abs(dists - wafer_radius_mm), np.inf)))
                legalized[i] = idx
    return legalized
