#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
GPU_IDS="${GPU_IDS:-0,1,2}"
SEEDS_STR="${SEEDS_STR:-0}"
RUN_TAG="${RUN_TAG:-ch3_timeformer_k90}"
INSTANCE="${INSTANCE:-base}"
STDOUT_AGG_DIR="${STDOUT_AGG_DIR:-${LOG_DIR:-$HOME/runlogs_A}/stdout}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

sanitize_tag() {
  local s="${1:-}"
  if [[ -z "$s" ]]; then echo ""; return 0; fi
  LC_ALL=C echo "$s" | tr -c 'A-Za-z0-9_.+-' '_'
}

for SEED in ${SEEDS_STR}; do
  echo "[ch3-timeformer] seed=${SEED} instance=${INSTANCE} gpus=${GPU_IDS}"
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  TF_PER_GPU_BS="${TF_PER_GPU_BS:-}"
  if [[ -n "${TF_PER_GPU_BS}" ]]; then
    export TRAIN_BATCH_SIZE="${TF_PER_GPU_BS}"
    echo "[ch3-timeformer] override via env: TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} (DP_SCALE_BATCH=1)"
  fi
  export USE_DP=1 DP_SCALE_BATCH=1 DP_SCALE_LR=linear AUTO_RESUME=0 FRESH_RUN=1
  export INSTANCE RUN_TAG STDOUT_AGG_DIR

  bash scripts/experiments_version_c.sh EXP-A2p25-warm15-prep-timeformer-k90 "${SEED}"

  RUN_TAG_SAFE="$(sanitize_tag "${RUN_TAG}")"
  TAG_SUFFIX=""
  if [[ -n "${RUN_TAG_SAFE}" ]]; then TAG_SUFFIX="-${RUN_TAG_SAFE}"; fi

  WARM_LINK="${STDOUT_AGG_DIR}/EXP-A2p25-warm15-prep-timeformer-k90${TAG_SUFFIX}_seed${SEED}.log"
  [[ -L "${WARM_LINK}" || -f "${WARM_LINK}" ]] || { echo "[ERROR] warm stdout link missing: ${WARM_LINK}"; exit 3; }
  WARM_LOG_REAL="$(readlink -f "${WARM_LINK}")"
  WARM_OUT="$(dirname "${WARM_LOG_REAL}")"
  WARM_LOG="${WARM_OUT}/stdout.log"
  WARM_CKPT="${WARM_OUT}/checkpoints/last.pth"
  [[ -f "${WARM_CKPT}" ]] || { echo "[ERROR] warm ckpt not found: ${WARM_CKPT}"; exit 3; }
  [[ -f "${WARM_LOG}" ]] || { echo "[ERROR] warm log not found: ${WARM_LOG}"; exit 3; }

  REFS="$(python scripts/extract_ch3_common_refs.py --log_path "${WARM_LOG}" --epoch_min 0 --epoch_max 4 --format shell)"
  eval "${REFS}"
  [[ -n "${HW_REF_LAT_MS:-}" && -n "${HW_REF_MEM_MB:-}" && -n "${HW_REF_COMM_MS:-}" ]] || { echo "[ERROR] missing HW refs from warm log"; exit 3; }
  export HW_REF_LAT_MS HW_REF_MEM_MB HW_REF_COMM_MS
  export INIT_CKPT_PATH="${WARM_CKPT}"

  for EXP in \
    EXP-A2p25-base-timeformer-k90 \
    EXP-A2p25-newours-tiebreak-timeformer-k90 \
    EXP-A2p25-ab-nolookahead-timeformer-k90 \
    EXP-A2p25-ab-nocandsel-timeformer-k90
  do
    echo "[ch3-timeformer] run ${EXP} seed=${SEED}"
    bash scripts/experiments_version_c.sh "${EXP}" "${SEED}"
  done
done

echo "ALL DONE"
