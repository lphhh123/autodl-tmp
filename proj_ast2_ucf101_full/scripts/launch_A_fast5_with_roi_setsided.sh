#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# Five-way parallel rerun for A-line fast protocol:
#   1) EXP-A2p-fast
#   2) EXP-A4-fast
#   3) EXP-A4-roi-fast
#   4) EXP-A4-acho-fast
#   5) EXP-A4-acho-roi-fast
#
# Notes:
# - This script intentionally does NOT launch / wait for dense baseline.
# - Current A-line fast configs use self_val_lock + self_hw_lock, so reruns can be done with NO_BASELINE=1.
# - Default SKIP_DONE=0 because the intended use is "rerun all five again".

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
GPU4="${GPU4:-4}"

SEED="${SEED:-0}"
SMOKE="${SMOKE:-0}"
INSTANCE="${INSTANCE:-base}"
RUN_TAG="${RUN_TAG:-}"
SKIP_DONE="${SKIP_DONE:-0}"
CLEAN_FIRST="${CLEAN_FIRST:-0}"

LOG_DIR="${LOG_DIR:-$HOME/runlogs_A_fast5_with_roi}"
mkdir -p "${LOG_DIR}"

EXP_PRUNE="${EXP_PRUNE:-EXP-A2p-fast}"
EXP_JOINT="${EXP_JOINT:-EXP-A4-fast}"
EXP_ROI_ONLY="${EXP_ROI_ONLY:-EXP-A4-roi-fast}"
EXP_ACHO="${EXP_ACHO:-EXP-A4-acho-fast}"
EXP_ACHO_ROI="${EXP_ACHO_ROI:-EXP-A4-acho-roi-fast}"

OUT_PREFIX="outputs"
[[ "${SMOKE}" == "1" ]] && OUT_PREFIX="outputs/SMOKE"

seed_outdir() {
  local exp="$1"
  echo "${OUT_PREFIX}/${exp}/seed${SEED}"
}

exp_done() {
  local exp="$1"
  local d; d="$(seed_outdir "$exp")"
  [[ -f "$d/metrics.json" ]] && return 0
  [[ -f "$d/metrics/metrics.json" ]] && return 0
  return 1
}

clear_exp() {
  local exp="$1"
  local d; d="$(seed_outdir "$exp")"
  if [[ -d "${d}" ]]; then
    echo "[CLEAN] removing ${d}"
    rm -rf "${d}"
  fi
}

run_exp() {
  local exp="$1"
  local gpu="$2"
  local log_file="$3"

  {
    echo "==== $(date '+%F %T') START ${exp} seed${SEED} GPU=${gpu} ===="
    if [[ "${SKIP_DONE}" == "1" ]] && exp_done "${exp}"; then
      echo "[SKIP] ${exp} already has metrics -> $(seed_outdir "${exp}")"
    else
      CUDA_VISIBLE_DEVICES="${gpu}" \
      LOG_DIR="${LOG_DIR}" \
      SMOKE="${SMOKE}" \
      INSTANCE="${INSTANCE}" \
      RUN_TAG="${RUN_TAG}" \
      NO_BASELINE=1 \
      bash scripts/experiments_version_c.sh "${exp}" "${SEED}"
    fi
    echo "==== $(date '+%F %T') END   ${exp} seed${SEED} GPU=${gpu} ===="
  } >"${log_file}" 2>&1
}

if [[ "${CLEAN_FIRST}" == "1" ]]; then
  clear_exp "${EXP_PRUNE}"
  clear_exp "${EXP_JOINT}"
  clear_exp "${EXP_ROI_ONLY}"
  clear_exp "${EXP_ACHO}"
  clear_exp "${EXP_ACHO_ROI}"
fi

run_exp "${EXP_PRUNE}"     "${GPU0}" "${LOG_DIR}/A_fast5_g0_prune.log" &
run_exp "${EXP_JOINT}"     "${GPU1}" "${LOG_DIR}/A_fast5_g1_a4.log" &
run_exp "${EXP_ROI_ONLY}"  "${GPU2}" "${LOG_DIR}/A_fast5_g2_a4_roi.log" &
run_exp "${EXP_ACHO}"      "${GPU3}" "${LOG_DIR}/A_fast5_g3_a4_acho.log" &
run_exp "${EXP_ACHO_ROI}"  "${GPU4}" "${LOG_DIR}/A_fast5_g4_a4_acho_roi.log" &

wait
echo "[DONE] all 5 fast A-line runs finished. logs: ${LOG_DIR}"
