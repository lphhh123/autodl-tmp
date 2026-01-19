#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TMP_DIR="$(mktemp -d)"

cat <<'JSON' > "${TMP_DIR}/layout_input.json"
{
  "layout_version": "v5.4",
  "wafer": {"radius_mm": 50.0, "margin_mm": 1.0},
  "sites": {"method": "square_grid_in_circle", "pitch_mm": 20.0, "sites_xy": [[0, 0], [10, 0], [0, 10], [10, 10]]},
  "slots": {"S": 4, "tdp": [300, 300, 300, 300]},
  "mapping": {"mapping_id": "toy", "traffic_matrix": [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]]},
  "baseline": {"assign_grid": [0, 1, 2, 3], "L_comm": 1.0, "L_therm": 1.0},
  "seed": {"assign_seed": [0, 2, 1, 3], "micro_place_stats": {}},
  "objective_cfg": {"sigma_mm": 20.0, "scalar_weights": {"w_comm": 0.7, "w_therm": 0.3, "w_penalty": 1000.0}}
}
JSON

python "${ROOT_DIR}/scripts/run_layout_heuragenix.py" \
  --layout_input "${TMP_DIR}/layout_input.json" \
  --cfg "${ROOT_DIR}/configs/layout_agent/layout_B0_heuragenix_random_hh.yaml" \
  --out_dir "${TMP_DIR}/out_random" \
  --seed 0

python "${ROOT_DIR}/scripts/run_layout_heuragenix.py" \
  --layout_input "${TMP_DIR}/layout_input.json" \
  --cfg "${ROOT_DIR}/configs/layout_agent/layout_B0_heuragenix_llm_hh.yaml" \
  --out_dir "${TMP_DIR}/out_llm" \
  --seed 0

echo "Expected outputs in each out_dir: trace.csv, report.json, layout_best.json, llm_usage.jsonl"
