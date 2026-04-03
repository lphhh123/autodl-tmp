#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

GPU_GROUP_A="${GPU_GROUP_A:-0,1,2}"
GPU_GROUP_B="${GPU_GROUP_B:-3,4,5}"
SEEDS_STR="${SEEDS_STR:-0 1 2}"
RUN_TAG="${RUN_TAG:-ch3newours25}"
INSTANCE="${INSTANCE:-base}"
REF_STATS_RANGE="${REF_STATS_RANGE:-0-4}"

WARMUP_EXP="EXP-A2p25-warm15-prep-k90"
EXPS=(
  EXP-A2p25-base-k90
  EXP-A2p25-hwloss-k90
  EXP-A2p25-cem60-k90
  EXP-A2p25-dsfixed-k90
  EXP-A2p25-newours-k90
  EXP-A2p25-ab-nomemory-k90
  EXP-A2p25-ab-nolookahead-k90
  EXP-A2p25-ab-nocandsel-k90
)

echo "[CH3-NEWOURS] GPU_GROUP_A=${GPU_GROUP_A} GPU_GROUP_B=${GPU_GROUP_B}"
echo "[CH3-NEWOURS] SEEDS=${SEEDS_STR} RUN_TAG=${RUN_TAG} INSTANCE=${INSTANCE}"

parse_ref_range() {
  local s="$1"
  if [[ "$s" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]}"
  else
    echo "0 4"
  fi
}

run_one() {
  local exp="$1"
  local seed="$2"
  local gpus="$3"
  local ckpt="$4"
  echo "[RUN] exp=${exp} seed=${seed} gpus=${gpus} init_ckpt=${ckpt}"
  (
    export CUDA_VISIBLE_DEVICES="${gpus}"
    export USE_DP=1
    export DP_SCALE_BATCH=1
    export DP_SCALE_LR=linear
    export INIT_CKPT_PATH="${ckpt}"
    export AUTO_RESUME=0
    export FRESH_RUN=1
    export RUN_TAG="${RUN_TAG}"
    export HW_REF_LAT_MS="${HW_REF_LAT_MS}"
    export HW_REF_MEM_MB="${HW_REF_MEM_MB}"
    export HW_REF_COMM_MS="${HW_REF_COMM_MS}"
    INSTANCE="${INSTANCE}" bash scripts/experiments_version_c.sh "${exp}" "${seed}"
  )
}

for SEED in ${SEEDS_STR}; do
  echo "==== [SEED ${SEED}] warmup start ${WARMUP_EXP} on GPU_GROUP_A=${GPU_GROUP_A} ===="
  (
    export CUDA_VISIBLE_DEVICES="${GPU_GROUP_A}"
    export USE_DP=1
    export DP_SCALE_BATCH=1
    export DP_SCALE_LR=linear
    export AUTO_RESUME=0
    export FRESH_RUN=1
    export RUN_TAG="${RUN_TAG}"
    INSTANCE="${INSTANCE}" bash scripts/experiments_version_c.sh "${WARMUP_EXP}" "${SEED}"
  )

  warm_out="outputs/${WARMUP_EXP}-${RUN_TAG}/seed${SEED}"
  if [[ "${INSTANCE}" != "base" ]]; then
    warm_out="${warm_out}-${INSTANCE}"
  fi
  ckpt="${warm_out}/checkpoints/last.pth"
  echo "[SEED ${SEED}] warmup ckpt=${ckpt}"
  if [[ ! -f "${ckpt}" ]]; then
    echo "[ERROR] warmup checkpoint missing: ${ckpt}"
    exit 2
  fi

  warm_log="${warm_out}/stdout.log"
  if [[ ! -f "${warm_log}" ]]; then
    echo "[ERROR] warmup stdout log missing: ${warm_log}"
    exit 2
  fi

  read REF_EPOCH_MIN REF_EPOCH_MAX < <(parse_ref_range "${REF_STATS_RANGE}")
  ref_exports="$(python scripts/extract_ch3_common_refs.py \
      --log_path "${warm_log}" \
      --epoch_min "${REF_EPOCH_MIN}" \
      --epoch_max "${REF_EPOCH_MAX}" \
      --format shell)"
  eval "${ref_exports}"
  echo "[SEED ${SEED}] refs from warmup: lat=${HW_REF_LAT_MS} mem=${HW_REF_MEM_MB} comm=${HW_REF_COMM_MS} range=${REF_STATS_RANGE}"

  for wave in 0 1 2 3; do
    idx_a=$((wave * 2))
    idx_b=$((wave * 2 + 1))
    exp_a="${EXPS[$idx_a]}"
    exp_b="${EXPS[$idx_b]}"
    echo "---- [SEED ${SEED}] wave=$((wave + 1)) A:${exp_a}@${GPU_GROUP_A} B:${exp_b}@${GPU_GROUP_B} ----"

    run_one "${exp_a}" "${SEED}" "${GPU_GROUP_A}" "${ckpt}" &
    pid_a=$!
    run_one "${exp_b}" "${SEED}" "${GPU_GROUP_B}" "${ckpt}" &
    pid_b=$!

    wait "${pid_a}"
    wait "${pid_b}"
  done

done

echo "[CH3-NEWOURS] ALL DONE"
