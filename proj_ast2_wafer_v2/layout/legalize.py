"""Legalization helpers: fix duplicate/boundary issues."""

from __future__ import annotations

import numpy as np


def legalize_assign(assign: np.ndarray, sites_xy_mm: np.ndarray, wafer_radius_mm: float) -> np.ndarray:
    """Ensure assignment has no duplicate sites and stays within wafer."""

    Ns = sites_xy_mm.shape[0]
    S = assign.shape[0]
    new_assign = assign.copy()

    # fix duplicates
    unique, counts = np.unique(new_assign, return_counts=True)
    duplicate_sites = unique[counts > 1]
    free_sites = [s for s in range(Ns) if s not in unique.tolist()]
    free_iter = iter(free_sites)
    for dup_site in duplicate_sites:
        slots = np.nonzero(new_assign == dup_site)[0]
        # keep first, move others
        for s in slots[1:]:
            try:
                new_assign[s] = next(free_iter)
            except StopIteration:
                break

    # boundary check
    pos = sites_xy_mm[new_assign]
    radii = np.linalg.norm(pos, axis=1)
    for i in range(S):
        if radii[i] > wafer_radius_mm:
            # move to nearest in-bounds site
            dists = np.linalg.norm(sites_xy_mm, axis=1)
            nearest = int(np.argmin(dists))
            new_assign[i] = nearest

    return new_assign

