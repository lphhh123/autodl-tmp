#!/usr/bin/env bash
set -euo pipefail

# One-shot runner for the B mainline + key ablations/headroom.
# Uses scripts/launch_B_grid_parallel.sh under the hood.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Defaults match the paper grid we have been iterating on.
MAX_JOBS="${MAX_JOBS:-24}"
INSTANCES="${INSTANCES:-cluster4 chain_skip chain_skip_randw}"
BUDGETS="${BUDGETS:-20k 40k 80k 160k}"
WEIGHT_PAIRS="${WEIGHT_PAIRS:-0.3,0.7}"
SEEDS_MAIN="${SEEDS_MAIN:-0 1 2}"

# Mainline methods:
#   - mpvs-only (atomic)
#   - std-budgetaware
#   - bc2cec (DEFAULT: probes+atomic CF ON)
#   - bc2cec-noprobe (ablation)
#   - bc2cec-probe-raw (ablation: probes ON, no CF)
EXPS_MAIN="${EXPS_MAIN:-EXP-B2-mpvs-only EXP-B2-std-budgetaware EXP-B2-bc2cec EXP-B2-bc2cec-noprobe EXP-B2-bc2cec-probe-raw}"

# Headroom truth table (for oracle/regret)
RUN_HEADROOM="${RUN_HEADROOM:-1}"
EXPS_HEADROOM="${EXPS_HEADROOM:-EXP-B2-naive-atomiconly EXP-B2-naive-relinkonly EXP-B2-naive-shakeonly EXP-B2-naive-tabuonly}"
SEEDS_HEADROOM="${SEEDS_HEADROOM:-0 1 2}"

PACK_AFTER="${PACK_AFTER:-1}"
PACK_TRACE_CSV="${PACK_TRACE_CSV:-0}"

echo "[B-mainline] MAX_JOBS=${MAX_JOBS}"
echo "[B-mainline] INSTANCES=${INSTANCES}"
echo "[B-mainline] BUDGETS=${BUDGETS}"
echo "[B-mainline] WEIGHT_PAIRS=${WEIGHT_PAIRS}"
echo "[B-mainline] SEEDS_MAIN=${SEEDS_MAIN}"
echo "[B-mainline] EXPS_MAIN=${EXPS_MAIN}"
echo "[B-mainline] RUN_HEADROOM=${RUN_HEADROOM} EXPS_HEADROOM=${EXPS_HEADROOM} SEEDS_HEADROOM=${SEEDS_HEADROOM}"

MAX_JOBS="${MAX_JOBS}" \
INSTANCES="${INSTANCES}" \
BUDGETS="${BUDGETS}" \
WEIGHT_PAIRS="${WEIGHT_PAIRS}" \
SEEDS_MAIN="${SEEDS_MAIN}" \
EXPS_MAIN="${EXPS_MAIN}" \
RUN_HEADROOM="${RUN_HEADROOM}" \
EXPS_HEADROOM="${EXPS_HEADROOM}" \
SEEDS_HEADROOM="${SEEDS_HEADROOM}" \
BUDGETS_HEADROOM="${BUDGETS}" \
WEIGHT_PAIRS_HEADROOM="${WEIGHT_PAIRS}" \
PACK_AFTER="${PACK_AFTER}" PACK_TRACE_CSV="${PACK_TRACE_CSV}" \
bash scripts/launch_B_grid_parallel.sh
