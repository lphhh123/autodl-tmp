import json
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


def _write_layout_input(tmp: Path):
    layout_input = {
        "layout_version": "v5.4",
        "wafer": {"radius_mm": 50.0, "margin_mm": 1.0},
        "sites": {"method": "square_grid_in_circle", "pitch_mm": 20.0, "sites_xy": [[0, 0], [10, 0], [0, 10], [10, 10]]},
        "slots": {"S": 4, "tdp": [300, 300, 300, 300]},
        "mapping": {"mapping_id": "toy", "traffic_matrix": [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]},
        "baseline": {"assign_grid": [0, 1, 2, 3], "L_comm": 1.0, "L_therm": 1.0},
        "seed": {"assign_seed": [0, 2, 1, 3], "micro_place_stats": {}},
        "objective_cfg": {"sigma_mm": 20.0, "scalar_weights": {"w_comm": 0.7, "w_therm": 0.3, "w_penalty": 1000.0}},
    }
    (tmp / "layout_input.json").write_text(json.dumps(layout_input), encoding="utf-8")


def _write_cfg(tmp: Path):
    yaml_text = """
layout_agent:
  version: v5.4
  seed_list: [0]
  export_trace: true
objective:
  sigma_mm: 20.0
  scalar_weights:
    w_comm: 0.7
    w_therm: 0.3
    w_penalty: 1000.0
coarsen:
  target_num_clusters: 2
  min_merge_traffic: 0.0
regions:
  enabled: true
  ring_edges_ratio: [0.0, 1.0]
  sectors_per_ring: [4]
  ring_score: [1.0]
  capacity_ratio: 1.0
global_place_region:
  lambda_graph: 1.0
  lambda_ring: 1.0
  lambda_cap: 1.0
  refine:
    enabled: false
expand:
  intra_refine_steps: 2
pareto:
  enabled: true
  eps_comm: 0.0
  eps_therm: 0.0
  max_points: 50
  selection: knee_point_v1
detailed_place:
  enabled: true
  steps: 5
  sa_T0: 1.0
  sa_alpha: 0.99
  action_probs:
    swap: 0.6
    relocate: 0.3
    cluster_move: 0.1
"""
    (tmp / "cfg.yaml").write_text(yaml_text.strip() + "\n", encoding="utf-8")


def test_smoke_run_layout_agent():
    if importlib.util.find_spec("numpy") is None:
        pytest.skip("numpy not installed in test environment")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _write_layout_input(tmp)
        _write_cfg(tmp)
        out_dir = tmp / "out"
        cmd = [
            sys.executable,
            "-m",
            "scripts.run_layout_agent",
            "--layout_input",
            str(tmp / "layout_input.json"),
            "--cfg",
            str(tmp / "cfg.yaml"),
            "--out_dir",
            str(out_dir),
        ]
        proj_root = Path(__file__).resolve().parents[1]
        subprocess.check_call(cmd, cwd=str(proj_root))
        assert (out_dir / "layout_best.json").exists()
        assert (out_dir / "trace.csv").exists()
        meta = json.loads((out_dir / "trace_meta.json").read_text(encoding="utf-8"))
        assert "objective" in meta and "hash" in meta["objective"] and len(meta["objective"]["hash"]) >= 8
        trace_text = (out_dir / "trace.csv").read_text(encoding="utf-8")
        assert "cache_key" in trace_text.splitlines()[0]
        assert f"obj:{meta['objective']['hash']}|" in trace_text
        trace_events = out_dir / "trace_events.jsonl"
        assert trace_events.exists()
        lines = trace_events.read_text(encoding="utf-8").splitlines()
        first = lines[0]
        assert '"event_type": "trace_header"' in first
        assert '"signature"' in first
        last = lines[-1]
        assert '"event_type": "trace_finalize"' in last
        assert '"reason"' in last and '"steps_done"' in last and '"best_solution_valid"' in last
