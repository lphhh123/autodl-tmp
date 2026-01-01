import json
import os

import pytest

np = pytest.importorskip("numpy")

from scripts.run_layout_agent import stage_pipeline


def test_layout_agent_smoke(tmp_path):
    S = 4
    sites = np.array([[0, 0], [10, 0], [0, 10], [10, 10], [20, 0], [0, 20]], dtype=float)
    traffic = np.array(
        [
            [0, 1, 2, 0],
            [1, 0, 0, 3],
            [2, 0, 0, 1],
            [0, 3, 1, 0],
        ],
        dtype=float,
    )
    layout_input = {
        "wafer": {"radius_mm": 100.0, "margin_mm": 1.0},
        "sites": {"method": "square_grid_in_circle", "grid_pitch_mm": None, "sites_xy_mm": sites.tolist()},
        "slots": {"S": S, "chip_tdp_w": [1, 1, 1, 1]},
        "mapping": {"mapping_id": "test", "traffic_matrix": traffic.tolist()},
        "objective_cfg": {
            "sigma_mm": 20.0,
            "scalar_weights": {"w_comm": 0.7, "w_therm": 0.3, "w_penalty": 1000.0},
            "baseline": {"L_comm_baseline": 1.0, "L_therm_baseline": 1.0},
        },
    }
    out_dir = tmp_path / "out"
    stage_pipeline({}, layout_input, str(out_dir))
    assert (out_dir / "layout_best.json").exists()
    assert (out_dir / "trace.csv").exists()
    assert (out_dir / "pareto_points.csv").exists()
    with open(out_dir / "layout_best.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "best" in data

