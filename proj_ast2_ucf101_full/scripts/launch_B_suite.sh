#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUDGET="${BUDGET:-50k}"
INSTANCE="${INSTANCE:-all}"
SEEDS="${SEEDS:-0}"
EXPS="${EXPS:-EXP-B1 EXP-B2 EXP-B2-ab-nollm EXP-B2-ab-noverifier EXP-B2-ab-nomacro EXP-B3}"
PARALLEL="${PARALLEL:-0}"
MAX_JOBS="${MAX_JOBS:-8}"

run_one () {
  local seed="$1"
  local exp="$2"
  echo "[B-suite] seed=${seed} exp=${exp} INSTANCE=${INSTANCE} BUDGET=${BUDGET}"
  INSTANCE="${INSTANCE}" BUDGET="${BUDGET}" bash scripts/experiments_version_c.sh "${exp}" "${seed}"
}

if [[ "${PARALLEL}" == "1" ]]; then
  echo "[B-suite] PARALLEL=1 MAX_JOBS=${MAX_JOBS}"
  pids=()
  fail=0
  for seed in ${SEEDS}; do
    for exp in ${EXPS}; do
      run_one "${seed}" "${exp}" &
      pids+=("$!")
      while [[ "$(jobs -rp | wc -l)" -ge "${MAX_JOBS}" ]]; do
        if ! wait -n; then
          echo "[B-suite] ERROR: a job failed."
          fail=1
          break
        fi
      done
      [[ "${fail}" == "0" ]] || break
    done
    [[ "${fail}" == "0" ]] || break
  done

  if [[ "${fail}" == "0" ]]; then
    for pid in "${pids[@]}"; do
      if ! wait "${pid}"; then
        echo "[B-suite] ERROR: job pid=${pid} failed."
        fail=1
      fi
    done
  fi

  [[ "${fail}" == "0" ]] || exit 1
else
  for seed in ${SEEDS}; do
    for exp in ${EXPS}; do
      run_one "${seed}" "${exp}"
    done
  done
fi

if [[ -f scripts/pack_B_outputs.sh ]]; then
  bash scripts/pack_B_outputs.sh
fi
