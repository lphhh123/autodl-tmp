#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUDGET="${BUDGET:-50k}"
INSTANCE="${INSTANCE:-all}"
SEEDS="${SEEDS:-0}"
EXPS="${EXPS:-EXP-B1 EXP-B2 EXP-B2-ab-nollm EXP-B2-ab-noverifier EXP-B2-ab-nomacro EXP-B2-ab-nomem EXP-B3}"
PARALLEL="${PARALLEL:-0}"
MAX_JOBS="${MAX_JOBS:-8}"

# Default behavior per user request:
# - purge legacy outputs/EXP-B* once
# - clean outputs/B before each suite run to avoid historical accumulation
PURGE_LEGACY_B="${PURGE_LEGACY_B:-1}"
CLEAN_NEW_B="${CLEAN_NEW_B:-1}"

if [[ -f scripts/clean_B_outputs.sh ]]; then
  PURGE_LEGACY="${PURGE_LEGACY_B}" CLEAN_NEW="${CLEAN_NEW_B}" bash scripts/clean_B_outputs.sh
fi

if [[ "${PARALLEL}" == "1" ]]; then
  tmp_cmds="$(mktemp -t bsuite_cmds.XXXXXX)"
  cleanup() { rm -f "${tmp_cmds}"; }
  trap cleanup EXIT
  for seed in ${SEEDS}; do
    for exp in ${EXPS}; do
      echo "INSTANCE='${INSTANCE}' BUDGET='${BUDGET}' bash scripts/experiments_version_c.sh '${exp}' '${seed}'" >> "${tmp_cmds}"
    done
  done
  echo "[B-suite] PARALLEL=1 MAX_JOBS=${MAX_JOBS} tasks=$(wc -l < "${tmp_cmds}" | tr -d ' ')"
  cat "${tmp_cmds}" | xargs -P "${MAX_JOBS}" -I{} bash -lc "{}"
else
  for seed in ${SEEDS}; do
    for exp in ${EXPS}; do
      echo "[B-suite] seed=${seed} exp=${exp} INSTANCE=${INSTANCE} BUDGET=${BUDGET}"
      INSTANCE="${INSTANCE}" BUDGET="${BUDGET}" bash scripts/experiments_version_c.sh "${exp}" "${seed}"
    done
  done
fi

if [[ -f scripts/pack_B_outputs.sh ]]; then
  bash scripts/pack_B_outputs.sh
fi
