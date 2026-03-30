#!/usr/bin/env bash
set -euo pipefail

# Chapter-3 (Innovation A / alloc-search) experiment suite.
# - Runs each experiment with 3-GPU DataParallel (USE_DP=1).
# - Includes the three missing ablations:
#     * w/o gate        : EXP-A2p-alloc-nogate-fast-k92
#     * w/o risk proxy  : EXP-A2p-alloc-norisk-fast-k92
#     * w/o look-ahead  : EXP-A2p-alloc-nolookahead-fast-k92
# - Keeps the same pruning-only protocol (fast20, keep_end=0.92).

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# -------- user knobs --------
GPU_IDS="${GPU_IDS:-0,1,2}"          # 3 GPUs for DP (comma-separated)
SEEDS_STR="${SEEDS_STR:-0 1 2}"      # default: three seeds used in the paper
INSTANCE="${INSTANCE:-base}"         # keep as base for A-line pruning-only
RUN_TAG="${RUN_TAG:-ch3}"            # optional suffix to avoid collisions

# DP options (match your usual 3-GPU runs)
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export USE_DP=1
export DP_SCALE_BATCH="${DP_SCALE_BATCH:-1}"
export DP_SCALE_LR="${DP_SCALE_LR:-linear}"

# Avoid CPU oversubscription (important when DP is on)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

export PYTHONPATH=.
export PYTHONUNBUFFERED=1

EXPS=(
  # Baselines
  EXP-A2p-fast-k92              # BASE (aligned pruning-only)
  EXP-A2p-hwloss-fast-k92       # HWLOSS
  EXP-A2p-hwsurr-fast-k92       # HWSURR

  # Our method + ablations
  EXP-A2p-alloc-fast-k92                # ours
  EXP-A2p-alloc-nogate-fast-k92         # w/o gate
  EXP-A2p-alloc-norisk-fast-k92         # w/o risk proxy
  EXP-A2p-alloc-nolookahead-fast-k92    # w/o look-ahead

  # Gradient-free baselines (budget sweep)
  EXP-A2p-cem-fast-k92          # CEM@60
  EXP-A2p-cem15-fast-k92        # CEM@90
  EXP-A2p-cem20-fast-k92        # CEM@120
)

echo "[CH3] GPU_IDS=$GPU_IDS (USE_DP=1)"
echo "[CH3] SEEDS=$SEEDS_STR"
echo "[CH3] RUN_TAG=$RUN_TAG"
echo "[CH3] EXPS: ${EXPS[*]}"

for SEED in $SEEDS_STR; do
  for EXP in "${EXPS[@]}"; do
    echo "==== $(date '+%F %T') START ${EXP} seed=${SEED} GPUs=${GPU_IDS} ===="
    INSTANCE="$INSTANCE" RUN_TAG="$RUN_TAG" \
      bash scripts/experiments_version_c.sh "$EXP" "$SEED"
    echo "==== $(date '+%F %T') END   ${EXP} seed=${SEED} ===="
  done
done

echo "[CH3] ALL DONE"
