#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-42}"

# EXP-A0: Dense baseline
python -m scripts.run_ast2_ucf101 \
  --cfg configs/ast2_ucf101_dense.yaml \
  --out_dir outputs/EXP-A0 \
  --seed "${SEED}"

# EXP-A4: Version-C (StableHW + LockedAccRef)
python -m scripts.run_version_c \
  --cfg configs/vc_phase3_full_ucf101.yaml \
  --out_dir outputs/EXP-A4 \
  --seed "${SEED}"

# EXP-B0: Layout baseline (random hh)
python -m scripts.run_layout_heuragenix \
  --cfg configs/layout_B0_heuragenix_random_hh.yaml \
  --layout_input outputs/P3/A3/layout_input.json \
  --out_dir outputs/EXP-B0-random \
  --seed "${SEED}"

# EXP-B0: Layout ours (llm hh)
python -m scripts.run_layout_agent \
  --cfg configs/layout_agent/wafer_layout_default.yaml \
  --layout_input outputs/P3/A3/layout_input.json \
  --out_dir outputs/EXP-B0-ours \
  --seed "${SEED}"
