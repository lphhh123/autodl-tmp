#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

INSTANCES="${INSTANCES:-cluster4}"
BUDGETS="${BUDGETS:-160k 240k}"
WEIGHT_PAIRS="${WEIGHT_PAIRS:-0.3,0.7}"
SEEDS_MAIN="${SEEDS_MAIN:-0 1 2}"
MAX_JOBS="${MAX_JOBS:-24}"
PACK_AFTER="${PACK_AFTER:-0}"
RUN_BASELINES="${RUN_BASELINES:-1}"

if [[ "${RUN_BASELINES}" == "1" ]]; then
  EXPS_MAIN="EXP-B2-std-budgetaware EXP-B2-taos-style EXP-B2-bc2cec EXP-B2-bc2cec-nolong EXP-B2-bc2cec-shiftm005 EXP-B2-bc2cec-shiftp005"
else
  EXPS_MAIN="EXP-B2-bc2cec-nolong EXP-B2-bc2cec-shiftm005 EXP-B2-bc2cec-shiftp005"
fi

echo "[launch_B_cd_minimal] RUN_BASELINES=${RUN_BASELINES}"
echo "[launch_B_cd_minimal] EXPS_MAIN=${EXPS_MAIN}"
echo "[launch_B_cd_minimal] INSTANCES=${INSTANCES} BUDGETS=${BUDGETS} WEIGHT_PAIRS=${WEIGHT_PAIRS} SEEDS_MAIN=${SEEDS_MAIN} MAX_JOBS=${MAX_JOBS}"

INSTANCES="${INSTANCES}" BUDGETS="${BUDGETS}" WEIGHT_PAIRS="${WEIGHT_PAIRS}" SEEDS_MAIN="${SEEDS_MAIN}" MAX_JOBS="${MAX_JOBS}" PACK_AFTER="${PACK_AFTER}" EXPS_MAIN="${EXPS_MAIN}" RUN_ABLATIONS=0 RUN_HEADROOM=0 RUN_CTL_EVIDENCE=0 bash scripts/launch_B_grid_parallel.sh
