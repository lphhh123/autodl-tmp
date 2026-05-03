#!/usr/bin/env bash
set -euo pipefail

export CH3_EXPECT_START_OUTER="${CH3_EXPECT_START_OUTER:-8}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

WARM_GPU_IDS="${WARM_GPU_IDS:-0,1,2,3,4,5}"
GPU_IDS_A="${GPU_IDS_A:-0,1,2}"
GPU_IDS_B="${GPU_IDS_B:-3,4,5}"
SEEDS_STR="${SEEDS_STR:-0}"
RUN_TAG="${RUN_TAG:-ch3_timeformer_k90}"
INSTANCE="${INSTANCE:-base}"
STDOUT_AGG_DIR="${STDOUT_AGG_DIR:-${LOG_DIR:-$HOME/runlogs_A}/stdout}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

if [[ -n "${TF_PER_GPU_BS:-}" ]]; then
  export TRAIN_BATCH_SIZE="${TF_PER_GPU_BS}"
  echo "[ch3-timeformer-pairs] override via env: TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE} (DP_SCALE_BATCH=1)"
fi

sanitize_tag() {
  local s="${1:-}"
  if [[ -z "$s" ]]; then echo ""; return 0; fi
  LC_ALL=C echo "$s" | tr -c 'A-Za-z0-9_.+-' '_'
}

PID_A=""
PID_B=""
cleanup() {
  for pid in "${PID_A:-}" "${PID_B:-}"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "[ch3-timeformer-pairs] cleanup: killing lingering pid=${pid}"
      kill "$pid" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT

run_pair() {
  local exp_a="$1"
  local exp_b="$2"
  local seed="$3"
  local log_a="$4"
  local log_b="$5"

  echo "[ch3-timeformer-pairs] start pair seed=${seed}:"
  echo "  A: ${exp_a} on GPU_IDS=${GPU_IDS_A}"
  echo "  B: ${exp_b} on GPU_IDS=${GPU_IDS_B}"

  ( export GPU_IDS="${GPU_IDS_A}"; bash scripts/experiments_version_c.sh "${exp_a}" "${seed}" ) >"${log_a}" 2>&1 &
  PID_A=$!
  ( export GPU_IDS="${GPU_IDS_B}"; bash scripts/experiments_version_c.sh "${exp_b}" "${seed}" ) >"${log_b}" 2>&1 &
  PID_B=$!
  echo "[ch3-timeformer-pairs] launched pid_A=${PID_A} pid_B=${PID_B}"

  set +e
  wait "${PID_A}"
  local rc_a=$?
  wait "${PID_B}"
  local rc_b=$?
  set -e

  if [[ ${rc_a} -ne 0 || ${rc_b} -ne 0 ]]; then
    echo "[ERROR] parallel pair failed: ${exp_a}(rc=${rc_a}) ${exp_b}(rc=${rc_b})"
    if [[ ${rc_a} -ne 0 && -n "${PID_B}" ]] && kill -0 "${PID_B}" 2>/dev/null; then
      kill "${PID_B}" 2>/dev/null || true
    fi
    if [[ ${rc_b} -ne 0 && -n "${PID_A}" ]] && kill -0 "${PID_A}" 2>/dev/null; then
      kill "${PID_A}" 2>/dev/null || true
    fi
    exit 4
  fi

  PID_A=""
  PID_B=""
  echo "[ch3-timeformer-pairs] pair finished seed=${seed}: ${exp_a}, ${exp_b}"
}

for SEED in ${SEEDS_STR}; do
  echo "[ch3-timeformer-pairs] seed=${SEED} instance=${INSTANCE} warm_gpus=${WARM_GPU_IDS}"
  export USE_DP=1 DP_SCALE_BATCH=1 DP_SCALE_LR=linear AUTO_RESUME=0 FRESH_RUN=1
  export INSTANCE RUN_TAG STDOUT_AGG_DIR

  export CUDA_VISIBLE_DEVICES="${WARM_GPU_IDS}"
  export GPU_IDS="${WARM_GPU_IDS}"
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

  PARALLEL_LOG_DIR="${STDOUT_AGG_DIR}/_parallel"
  mkdir -p "${PARALLEL_LOG_DIR}"

  LOG_BASE="${PARALLEL_LOG_DIR}/EXP-A2p25-base-timeformer-k90${TAG_SUFFIX}_seed${SEED}.runlog"
  LOG_OURS="${PARALLEL_LOG_DIR}/EXP-A2p25-newours-tiebreak-timeformer-k90${TAG_SUFFIX}_seed${SEED}.runlog"
  LOG_AB_NOLA="${PARALLEL_LOG_DIR}/EXP-A2p25-ab-nolookahead-timeformer-k90${TAG_SUFFIX}_seed${SEED}.runlog"
  LOG_AB_NOCA="${PARALLEL_LOG_DIR}/EXP-A2p25-ab-nocandsel-timeformer-k90${TAG_SUFFIX}_seed${SEED}.runlog"

  echo "[ch3-timeformer-pairs] batch 1/2 start"
  run_pair \
    "EXP-A2p25-base-timeformer-k90" \
    "EXP-A2p25-newours-tiebreak-timeformer-k90" \
    "${SEED}" \
    "${LOG_BASE}" \
    "${LOG_OURS}"
  echo "[ch3-timeformer-pairs] batch 1/2 done"

  echo "[ch3-timeformer-pairs] batch 2/2 start"
  run_pair \
    "EXP-A2p25-ab-nolookahead-timeformer-k90" \
    "EXP-A2p25-ab-nocandsel-timeformer-k90" \
    "${SEED}" \
    "${LOG_AB_NOLA}" \
    "${LOG_AB_NOCA}"
  echo "[ch3-timeformer-pairs] batch 2/2 done"
done

echo "ALL DONE"
