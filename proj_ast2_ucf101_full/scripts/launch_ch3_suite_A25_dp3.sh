#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GPU_IDS="${GPU_IDS:-0,1,2}"
SEEDS_STR="${SEEDS_STR:-0 1 2}"
RUN_TAG="${RUN_TAG:-ch3A25_halpnoprobe}"
INSTANCE="${INSTANCE:-base}"

for SEED in ${SEEDS_STR}; do
  echo "[ch3] seed=${SEED} instance=${INSTANCE} gpus=${GPU_IDS}"
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  export USE_DP=1 DP_SCALE_BATCH=1 DP_SCALE_LR=linear AUTO_RESUME=0 FRESH_RUN=1
  export INSTANCE RUN_TAG

  echo "[ch3] warmup EXP-A2p25-warm15-prep-k90"
  bash scripts/experiments_version_c.sh EXP-A2p25-warm15-prep-k90 "${SEED}"
  WARM_OUT="outputs/P3/A2p25/EXP-A2p25-warm15-prep-k90-${INSTANCE}-s${SEED}-${RUN_TAG}"
  WARM_LOG="${WARM_OUT}/stdout.log"
  WARM_CKPT="${WARM_OUT}/checkpoints/last.pth"

  REFS="$(python scripts/extract_ch3_common_refs.py --log_path "${WARM_LOG}" --epoch_min 0 --epoch_max 4 --format shell)"
  eval "${REFS}"
  export HW_REF_LAT_MS HW_REF_MEM_MB HW_REF_COMM_MS
  export INIT_CKPT_PATH="${WARM_CKPT}"
  echo "[ch3] refs LAT=${HW_REF_LAT_MS} MEM=${HW_REF_MEM_MB} COMM=${HW_REF_COMM_MS}"

  for EXP in EXP-A2p25-base-k90 EXP-A2p25-hwloss-k90 EXP-A2p25-cem60-k90 EXP-A2p25-dsfixed-k90 EXP-A2p25-halp-k90 EXP-A2p25-newours-k90 EXP-A2p25-ab-nolookahead-k90; do
    echo "[ch3] run ${EXP} seed=${SEED}"
    bash scripts/experiments_version_c.sh "${EXP}" "${SEED}"
  done
done
