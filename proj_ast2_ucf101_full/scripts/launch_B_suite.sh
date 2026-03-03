#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUDGET="${BUDGET:-10k}"
INSTANCE="${INSTANCE:-all}"
SEEDS="${SEEDS:-0}"
EXPS="${EXPS:-EXP-B1 EXP-B2 EXP-B0-strong EXP-B0-best1}"

export HEURAGENIX_INCLUDE_LEGACY="${HEURAGENIX_INCLUDE_LEGACY:-1}"
export V54_LLM_PICK_TEMPERATURE="${V54_LLM_PICK_TEMPERATURE:-0.3}"
export V54_LLM_PICK_TOP_P="${V54_LLM_PICK_TOP_P:-0.9}"

for seed in ${SEEDS}; do
  for exp in ${EXPS}; do
    echo "[B-suite] seed=${seed} exp=${exp} INSTANCE=${INSTANCE} BUDGET=${BUDGET}"
    INSTANCE="${INSTANCE}" BUDGET="${BUDGET}" bash scripts/experiments_version_c.sh "${exp}" "${seed}"
  done
done

if [[ -f scripts/pack_B_outputs.sh ]]; then
  bash scripts/pack_B_outputs.sh
fi
