#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUDGET="${BUDGET:-50k}"
INSTANCE="${INSTANCE:-all}"
SEEDS="${SEEDS:-0}"
# Default suite: paper mainline (nollm baselines)
EXPS="${EXPS:-EXP-B1 EXP-B2-mpvs-only EXP-B2-std-budgetaware EXP-B2-bc2cec EXP-B3}"

# Optional headroom probes (controller=0)
RUN_HEADROOM="${RUN_HEADROOM:-0}"
EXPS_HEADROOM="${EXPS_HEADROOM:-EXP-B2-naive-atomiconly EXP-B2-naive-macroonly EXP-B2-naive-chainonly EXP-B2-naive-ruinonly}"
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

  if [[ "${RUN_HEADROOM}" == "1" ]]; then
    for seed in ${SEEDS}; do
      for exp in ${EXPS_HEADROOM}; do
        echo "INSTANCE='${INSTANCE}' BUDGET='${BUDGET}' bash scripts/experiments_version_c.sh '${exp}' '${seed}'" >> "${tmp_cmds}"
      done
    done
  fi

  echo "[B-suite] PARALLEL=1 MAX_JOBS=${MAX_JOBS} tasks=$(wc -l < "${tmp_cmds}" | tr -d ' ')"
  cat "${tmp_cmds}" | xargs -P "${MAX_JOBS}" -I{} bash -lc "{}"
else
  for seed in ${SEEDS}; do
    for exp in ${EXPS}; do
      echo "[B-suite] seed=${seed} exp=${exp} INSTANCE=${INSTANCE} BUDGET=${BUDGET}"
      INSTANCE="${INSTANCE}" BUDGET="${BUDGET}" bash scripts/experiments_version_c.sh "${exp}" "${seed}"
    done
  done

  if [[ "${RUN_HEADROOM}" == "1" ]]; then
    for seed in ${SEEDS}; do
      for exp in ${EXPS_HEADROOM}; do
        echo "[B-suite] headroom seed=${seed} exp=${exp} INSTANCE=${INSTANCE} BUDGET=${BUDGET}"
        INSTANCE="${INSTANCE}" BUDGET="${BUDGET}" bash scripts/experiments_version_c.sh "${exp}" "${seed}"
      done
    done
  fi
fi

if [[ -f scripts/pack_B_outputs.sh ]]; then
  bash scripts/pack_B_outputs.sh
fi
