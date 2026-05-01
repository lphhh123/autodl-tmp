#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WARM_RUN_TAG="${WARM_RUN_TAG:-ch3newours25_clean1_}"
RUN_TAG="${RUN_TAG:-ch3newours25_clean2_}"
WARM_SEED="${WARM_SEED:-0}"
SEED="${SEED:-0}"
INSTANCE="${INSTANCE:-base}"
HALP_GPUS="${HALP_GPUS:-0,1,2}"
NOPROBE_GPUS="${NOPROBE_GPUS:-3,4,5}"
STDOUT_AGG_DIR="${STDOUT_AGG_DIR:-${LOG_DIR:-$HOME/runlogs_A}/stdout}"

sanitize_tag() {
  local s="${1:-}"
  if [[ -z "$s" ]]; then
    echo ""
    return 0
  fi
  LC_ALL=C echo "$s" | tr -c 'A-Za-z0-9_.+-' '_'
}

echo "[ch3-fix] ensure warm run exists (tag=${WARM_RUN_TAG}, seed=${WARM_SEED})"
env CUDA_VISIBLE_DEVICES="${HALP_GPUS}" GPU_IDS="${HALP_GPUS}" USE_DP=1 DP_SCALE_LR=linear DP_SCALE_BATCH=1 \
  RUN_TAG="${WARM_RUN_TAG}" INSTANCE="${INSTANCE}" \
  bash scripts/experiments_version_c.sh EXP-A2p25-warm15-prep-k90 "${WARM_SEED}"

WARM_RUN_TAG_SAFE="$(sanitize_tag "${WARM_RUN_TAG}")"
WARM_TAG_SUFFIX=""
if [[ -n "${WARM_RUN_TAG_SAFE}" ]]; then WARM_TAG_SUFFIX="-${WARM_RUN_TAG_SAFE}"; fi
WARM_LINK="${STDOUT_AGG_DIR}/EXP-A2p25-warm15-prep-k90${WARM_TAG_SUFFIX}_seed${WARM_SEED}.log"
WARM_LOG_REAL="$(readlink -f "${WARM_LINK}")"
WARM_OUT="$(dirname "${WARM_LOG_REAL}")"
WARM_LOG="${WARM_OUT}/stdout.log"
WARM_CKPT="${WARM_OUT}/checkpoints/last.pth"

[[ -f "${WARM_LOG}" ]] || { echo "[ERROR] warm log not found: ${WARM_LOG}"; exit 3; }
[[ -f "${WARM_CKPT}" ]] || { echo "[ERROR] warm ckpt not found: ${WARM_CKPT}"; exit 3; }

eval "$(python scripts/extract_ch3_common_refs.py --log_path "${WARM_LOG}" --epoch_min 0 --epoch_max 4 --format shell)"
export INIT_CKPT_PATH="${WARM_CKPT}"
export HW_REF_LAT_MS HW_REF_MEM_MB HW_REF_COMM_MS

echo "[ch3-fix] warm_ckpt=${INIT_CKPT_PATH}"
echo "[ch3-fix] refs LAT=${HW_REF_LAT_MS} MEM=${HW_REF_MEM_MB} COMM=${HW_REF_COMM_MS}"

env CUDA_VISIBLE_DEVICES="${HALP_GPUS}" GPU_IDS="${HALP_GPUS}" USE_DP=1 DP_SCALE_LR=linear DP_SCALE_BATCH=1 \
  RUN_TAG="${RUN_TAG}" INSTANCE="${INSTANCE}" INIT_CKPT_PATH="${INIT_CKPT_PATH}" \
  HW_REF_LAT_MS="${HW_REF_LAT_MS}" HW_REF_MEM_MB="${HW_REF_MEM_MB}" HW_REF_COMM_MS="${HW_REF_COMM_MS}" \
  bash scripts/experiments_version_c.sh EXP-A2p25-halp-k90 "${SEED}" &

env CUDA_VISIBLE_DEVICES="${NOPROBE_GPUS}" GPU_IDS="${NOPROBE_GPUS}" USE_DP=1 DP_SCALE_LR=linear DP_SCALE_BATCH=1 \
  RUN_TAG="${RUN_TAG}" INSTANCE="${INSTANCE}" INIT_CKPT_PATH="${INIT_CKPT_PATH}" \
  HW_REF_LAT_MS="${HW_REF_LAT_MS}" HW_REF_MEM_MB="${HW_REF_MEM_MB}" HW_REF_COMM_MS="${HW_REF_COMM_MS}" \
  bash scripts/experiments_version_c.sh EXP-A2p25-ab-nolookahead-k90 "${SEED}" &

wait

RUN_TAG_SAFE="$(sanitize_tag "${RUN_TAG}")"
TAG_SUFFIX=""
if [[ -n "${RUN_TAG_SAFE}" ]]; then TAG_SUFFIX="-${RUN_TAG_SAFE}"; fi
echo "outputs/stdout_agg/EXP-A2p25-halp-k90${TAG_SUFFIX}_seed${SEED}.log"
echo "outputs/stdout_agg/EXP-A2p25-ab-nolookahead-k90${TAG_SUFFIX}_seed${SEED}.log"
