"""Site generator following spec v4.3.2 square grid in circle."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def build_sites(
    wafer_radius_mm: float,
    chip_max_width_mm: float,
    chip_max_height_mm: float,
    margin_mm: float,
    method: str,
    grid_pitch_mm: Optional[float],
    seed: int,
) -> np.ndarray:
    """Return sites_xy_mm: [Ns,2] float32 on a safe square grid.

    Implements spec §4 with `square_grid_in_circle` as the only supported
    method. The output ordering is deterministic (y-major then x).
    """

    if method != "square_grid_in_circle":
        raise ValueError(f"Unsupported site generation method: {method}")

    max_diag = math.sqrt(chip_max_width_mm ** 2 + chip_max_height_mm ** 2)
    safe_spacing = max_diag + 2 * margin_mm
    if grid_pitch_mm is None:
        grid_pitch_mm = safe_spacing

    # fixed ordering: iterate y first then x for determinism
    coords = []
    xs = np.arange(-wafer_radius_mm, wafer_radius_mm + 1e-9, grid_pitch_mm)
    ys = np.arange(-wafer_radius_mm, wafer_radius_mm + 1e-9, grid_pitch_mm)
    limit = wafer_radius_mm - 0.5 * max_diag - margin_mm
    for y in ys:
        for x in xs:
            if math.hypot(x, y) <= limit:
                coords.append((float(x), float(y)))
    return np.asarray(coords, dtype=np.float32)

