#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUDGET="${BUDGET:-10k}"
INSTANCE="${INSTANCE:-all}"
SEEDS="${SEEDS:-0}"
EXPS="${EXPS:-EXP-B1 EXP-B2 EXP-B2-ab-noqueue EXP-B2-ab-nofeas EXP-B2-ab-nodiverse EXP-B3}"

for seed in ${SEEDS}; do
  for exp in ${EXPS}; do
    echo "[B-suite] seed=${seed} exp=${exp} INSTANCE=${INSTANCE} BUDGET=${BUDGET}"
    INSTANCE="${INSTANCE}" BUDGET="${BUDGET}" bash scripts/experiments_version_c.sh "${exp}" "${seed}"
  done
done

if [[ -f scripts/pack_B_outputs.sh ]]; then
  bash scripts/pack_B_outputs.sh
fi
