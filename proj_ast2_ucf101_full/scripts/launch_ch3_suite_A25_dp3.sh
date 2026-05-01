#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GPU_IDS="${GPU_IDS:-0,1,2}"
SEEDS_STR="${SEEDS_STR:-0 1 2}"
RUN_TAG="${RUN_TAG:-ch3A25_halpnoprobe}"
INSTANCE="${INSTANCE:-base}"
STDOUT_AGG_DIR="${STDOUT_AGG_DIR:-${LOG_DIR:-$HOME/runlogs_A}/stdout}"

sanitize_tag() {
  local s="${1:-}"
  if [[ -z "$s" ]]; then
    echo ""
    return 0
  fi
  LC_ALL=C echo "$s" | tr -c 'A-Za-z0-9_.+-' '_'
}

for SEED in ${SEEDS_STR}; do
  echo "[ch3] seed=${SEED} instance=${INSTANCE} gpus=${GPU_IDS}"
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  export USE_DP=1 DP_SCALE_BATCH=1 DP_SCALE_LR=linear AUTO_RESUME=0 FRESH_RUN=1
  export INSTANCE RUN_TAG

  echo "[ch3] warmup EXP-A2p25-warm15-prep-k90"
  bash scripts/experiments_version_c.sh EXP-A2p25-warm15-prep-k90 "${SEED}"
  RUN_TAG_SAFE="$(sanitize_tag "${RUN_TAG}")"
  TAG_SUFFIX=""
  if [[ -n "${RUN_TAG_SAFE}" ]]; then TAG_SUFFIX="-${RUN_TAG_SAFE}"; fi
  WARM_LINK="${STDOUT_AGG_DIR}/EXP-A2p25-warm15-prep-k90${TAG_SUFFIX}_seed${SEED}.log"
  WARM_LOG_REAL="$(readlink -f "${WARM_LINK}")"
  WARM_OUT="$(dirname "${WARM_LOG_REAL}")"
  WARM_LOG="${WARM_OUT}/stdout.log"
  WARM_CKPT="${WARM_OUT}/checkpoints/last.pth"
  [[ -f "${WARM_CKPT}" ]] || { echo "[ERROR] warm ckpt not found: ${WARM_CKPT}"; exit 3; }
  [[ -f "${WARM_LOG}"  ]] || { echo "[ERROR] warm log not found: ${WARM_LOG}"; exit 3; }

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
