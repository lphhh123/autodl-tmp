"""Site generation utilities for discrete wafer layouts (SPEC v5.4 §4)."""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _square_grid_in_circle(
    wafer_radius_mm: float,
    grid_pitch_mm: float,
    max_diag_mm: float,
    margin_mm: float,
    seed: int,
) -> np.ndarray:
    coords = []
    half_extent = wafer_radius_mm
    x_vals = np.arange(-half_extent, half_extent + 1e-6, grid_pitch_mm)
    y_vals = np.arange(-half_extent, half_extent + 1e-6, grid_pitch_mm)
    # deterministic ordering: iterate y outer, x inner (SPEC v5.4)
    for y in y_vals:
        for x in x_vals:
            if math.sqrt(x * x + y * y) <= wafer_radius_mm - 0.5 * max_diag_mm - margin_mm + 1e-9:
                coords.append((float(x), float(y)))
    return np.array(coords, dtype=np.float32)


def build_sites(
    wafer_radius_mm: float,
    chip_max_width_mm: float,
    chip_max_height_mm: float,
    margin_mm: float = 0.0,
    method: str = "square_grid_in_circle",
    grid_pitch_mm: Optional[float] = None,
    seed: int = 0,
) -> np.ndarray:
    """Build discrete candidate sites for layout.

    Args:
        wafer_radius_mm: wafer radius.
        chip_max_width_mm: maximum chip width for spacing estimation.
        chip_max_height_mm: maximum chip height for spacing estimation.
        margin_mm: additional margin when culling boundary sites.
        method: currently only ``square_grid_in_circle`` is supported.
        grid_pitch_mm: optional pitch override.
        seed: rng seed for shuffling sites.
    Returns:
        np.ndarray of shape [Ns, 2] in millimeters.
    """
    max_diag = math.sqrt(chip_max_width_mm ** 2 + chip_max_height_mm ** 2)
    safe_spacing = max_diag + 2 * margin_mm
    pitch = grid_pitch_mm if grid_pitch_mm is not None else safe_spacing
    if method != "square_grid_in_circle":
        raise ValueError(f"Unsupported site generation method: {method}")
    return _square_grid_in_circle(wafer_radius_mm, pitch, max_diag, margin_mm, seed)
