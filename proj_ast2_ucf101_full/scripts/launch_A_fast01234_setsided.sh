#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"
GPU3="${GPU3:-3}"
GPU4="${GPU4:-4}"
SEED="${SEED:-0}"
SMOKE="${SMOKE:-0}"
INSTANCE="${INSTANCE:-chain_skip}"
RUN_TAG="${RUN_TAG:-}" 
SKIP_DONE="${SKIP_DONE:-1}"

LOG_DIR="${LOG_DIR:-$HOME/runlogs_A_fast01234}"
mkdir -p "${LOG_DIR}"

BASELINE_WAIT_SEC="${BASELINE_WAIT_SEC:-30}"
BASELINE_TIMEOUT_MIN="${BASELINE_TIMEOUT_MIN:-0}"
REQUIRE_BASELINE="${REQUIRE_BASELINE:-1}"
REQUIRE_BASELINE_DONE="${REQUIRE_BASELINE_DONE:-1}"
REQUIRE_REF_CURVE="${REQUIRE_REF_CURVE:-1}"

EXP_DENSE="${EXP_DENSE:-EXP-A1p-fast}"
EXP_PRUNE="${EXP_PRUNE:-EXP-A2p-fast}"
EXP_JOINT="${EXP_JOINT:-EXP-A4-fast}"
EXP_ACHO="${EXP_ACHO:-EXP-A4-acho-fast}"
EXP_ROI="${EXP_ROI:-EXP-A4-acho-roi-fast}"

OUT_PREFIX="outputs"
[[ "${SMOKE}" == "1" ]] && OUT_PREFIX="outputs/SMOKE"

seed_outdir() {
  local exp="$1"
  echo "${OUT_PREFIX}/${exp}/seed${SEED}"
}

BASELINE_FILE="$(seed_outdir "${EXP_DENSE}")/metrics.json"

exp_done() {
  local exp="$1"
  local d; d="$(seed_outdir "$exp")"
  [[ -f "$d/metrics.json" ]] && return 0
  [[ -f "$d/metrics/metrics.json" ]] && return 0
  return 1
}

wait_for_baseline() {
  if [[ "${REQUIRE_BASELINE}" != "1" ]]; then
    echo "[WARN] REQUIRE_BASELINE=0 (skip baseline wait)"
    return 0
  fi
  local start_ts; start_ts="$(date +%s)"
  local bdir; bdir="$(seed_outdir "${EXP_DENSE}")"
  local done_file="${bdir}/DONE"
  local curve_file="${bdir}/acc_ref_curve.json"

  while [[ ! -f "${BASELINE_FILE}" ]]; do
    echo "[WAIT] baseline not found yet: ${BASELINE_FILE}"
    if [[ "${BASELINE_TIMEOUT_MIN}" != "0" ]]; then
      local now_ts; now_ts="$(date +%s)"
      local elapsed=$((now_ts - start_ts))
      local limit=$((BASELINE_TIMEOUT_MIN * 60))
      if (( elapsed > limit )); then
        echo "[ERR] baseline timeout (${BASELINE_TIMEOUT_MIN} min): ${BASELINE_FILE}"
        exit 1
      fi
    fi
    sleep "${BASELINE_WAIT_SEC}"
  done

  if [[ "${REQUIRE_BASELINE_DONE}" == "1" ]]; then
    while [[ ! -f "${done_file}" ]]; do
      echo "[WAIT] baseline not finished yet (missing DONE): ${done_file}"
      sleep "${BASELINE_WAIT_SEC}"
    done
  fi

  if [[ "${REQUIRE_REF_CURVE}" == "1" ]]; then
    while [[ ! -f "${curve_file}" ]]; do
      echo "[WAIT] acc_ref_curve.json not found yet: ${curve_file}"
      sleep "${BASELINE_WAIT_SEC}"
    done
  fi

  echo "[OK] baseline ready: ${BASELINE_FILE}"
}

run_exp() {
  local exp="$1"
  local gpu="$2"
  local log_file="$3"

  {
    echo "==== $(date '+%F %T') START ${exp} seed${SEED} GPU=${gpu} ===="
    if [[ "${SKIP_DONE}" == "1" ]] && exp_done "${exp}"; then
      if [[ "${exp}" == "${EXP_DENSE}" ]]; then
        local bdir; bdir="$(seed_outdir "${EXP_DENSE}")"
        if [[ "${REQUIRE_REF_CURVE}" == "1" && ! -f "${bdir}/acc_ref_curve.json" ]]; then
          python -m scripts.make_acc_ref_curve --stdout "${bdir}/stdout.log" --out "${bdir}/acc_ref_curve.json" --prefer fast --ema-alpha 0.2 --curve-margin 0.0 || true
        fi
        if [[ "${REQUIRE_BASELINE_DONE}" == "1" && ! -f "${bdir}/DONE" ]]; then
          echo "done" > "${bdir}/DONE" || true
        fi
      fi
      echo "[SKIP] ${exp} already has metrics -> $(seed_outdir "${exp}")"
    else
      CUDA_VISIBLE_DEVICES="${gpu}" LOG_DIR="${LOG_DIR}" SMOKE="${SMOKE}" INSTANCE="${INSTANCE}" RUN_TAG="${RUN_TAG}" \
        bash scripts/experiments_version_c.sh "${exp}" "${SEED}"
    fi
    echo "==== $(date '+%F %T') END   ${exp} seed${SEED} GPU=${gpu} ===="
  } >"${log_file}" 2>&1
}

run_exp "${EXP_DENSE}" "${GPU0}" "${LOG_DIR}/A_fast01234_g0.log" &
wait_for_baseline
run_exp "${EXP_PRUNE}" "${GPU1}" "${LOG_DIR}/A_fast01234_g1.log" &
run_exp "${EXP_JOINT}" "${GPU2}" "${LOG_DIR}/A_fast01234_g2.log" &
run_exp "${EXP_ACHO}" "${GPU3}" "${LOG_DIR}/A_fast01234_g3.log" &
run_exp "${EXP_ROI}" "${GPU4}" "${LOG_DIR}/A_fast01234_g4.log" &

wait
echo "[DONE] all runs finished. logs: ${LOG_DIR}"
