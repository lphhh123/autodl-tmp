#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ------------------------------------------------------------
# Parallel grid runner for B experiments (CPU-heavy; 64 cores OK)
# - Mainline: multi seed × multi budget × multi weights × 3 instances
# - Ablations: single seed (default 0) × multi budget × multi weights × 3 instances
# ------------------------------------------------------------

# Concurrency
MAX_JOBS="${MAX_JOBS:-24}"   # adjust if API rate-limits; 8~24 usually safe
DRY_RUN="${DRY_RUN:-0}"

# Instances (explicit; do not rely on INSTANCE=all to keep parallelism high)
INSTANCES="${INSTANCES:-chain_skip chain_skip_randw cluster4}"

# Budgets: choose interpretable low/mid/high; 50k is the paper baseline; 100k is 2×; 200k is 4× and shows saturation.
# You can override: BUDGETS="30k 50k 80k 120k 200k"
BUDGETS="${BUDGETS:-50k 100k 200k}"

# Weights: format is "w_therm,w_comm"
WEIGHT_PAIRS="${WEIGHT_PAIRS:-0.2,0.8 0.3,0.7 0.5,0.5}"

# Seeds
SEEDS_MAIN="${SEEDS_MAIN:-0 1 2 3 4}"   # strong argument
SEEDS_ABL="${SEEDS_ABL:-0}"             # ablations single seed by default

# Experiments
EXPS_MAIN="${EXPS_MAIN:-EXP-B1 EXP-B2 EXP-B3}"
EXPS_ABL="${EXPS_ABL:-EXP-B2-ab-nollm EXP-B2-ab-noverifier EXP-B2-ab-nomacro EXP-B2-ab-nomem}"
RUN_ABLATIONS="${RUN_ABLATIONS:-1}"     # set 0 to skip ablations

# Thread caps to avoid oversubscription per process
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export LOG_SLIM="${LOG_SLIM:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

echo "[B-grid] MAX_JOBS=${MAX_JOBS}"
echo "[B-grid] INSTANCES=${INSTANCES}"
echo "[B-grid] BUDGETS=${BUDGETS}"
echo "[B-grid] WEIGHT_PAIRS=${WEIGHT_PAIRS}"
echo "[B-grid] SEEDS_MAIN=${SEEDS_MAIN}"
echo "[B-grid] SEEDS_ABL=${SEEDS_ABL}"
echo "[B-grid] EXPS_MAIN=${EXPS_MAIN}"
echo "[B-grid] EXPS_ABL=${EXPS_ABL} (RUN_ABLATIONS=${RUN_ABLATIONS})"

# ------------------------------------------------------------
# Preflight: generate layout inputs once to avoid parallel race
# ------------------------------------------------------------
LI_BASE="outputs/P3/A3/layout_input.json"
if [[ ! -f "${LI_BASE}" ]]; then
  echo "[B-grid][preflight] generating ${LI_BASE} via Version-C Phase3 ..."
  python -m scripts.run_version_c --phase P3 --sub A3 || true
fi

mkdir -p "outputs/P3/A3/instances"

# union seeds (main + ablation)
ALL_SEEDS="${SEEDS_MAIN} ${SEEDS_ABL}"
for seed in ${ALL_SEEDS}; do
  echo "[B-grid][preflight] make_layout_inputs seed=${seed}"
  python -m scripts.make_layout_inputs --base "${LI_BASE}" --out_dir "outputs/P3/A3/instances" --seed "${seed}" || true
done

# ------------------------------------------------------------
# Build task list
# ------------------------------------------------------------
tmp_cmds="$(mktemp -t bgrid_cmds.XXXXXX)"
cleanup() { rm -f "${tmp_cmds}"; }
trap cleanup EXIT

fmt_tag() {
  # "0.3" -> "0p3"
  printf '%s' "${1}" | tr '.' 'p' | tr -c 'A-Za-z0-9p' '_'
}

emit_task() {
  local exp="$1"
  local seed="$2"
  local inst="$3"
  local budget="$4"
  local wpair="$5"   # therm,comm
  local wt="${wpair%,*}"
  local wc="${wpair#*,}"
  local tag="b${budget}_wT$(fmt_tag "${wt}")_wC$(fmt_tag "${wc}")"
  # We pass both: SCALAR_WEIGHTS_OVERRIDE for convenience, and tag for unique out dirs.
  echo "RUN_TAG='${tag}' INSTANCE='${inst}' BUDGET='${budget}' SCALAR_WEIGHTS_OVERRIDE='${wt},${wc}' bash scripts/experiments_version_c.sh '${exp}' '${seed}'" >> "${tmp_cmds}"
}

# Mainline grid
for budget in ${BUDGETS}; do
  for wpair in ${WEIGHT_PAIRS}; do
    for seed in ${SEEDS_MAIN}; do
      for inst in ${INSTANCES}; do
        for exp in ${EXPS_MAIN}; do
          emit_task "${exp}" "${seed}" "${inst}" "${budget}" "${wpair}"
        done
      done
    done
  done
done

# Ablations grid (single seed by default, still multi budget/weights/instances for stronger argument)
if [[ "${RUN_ABLATIONS}" == "1" ]]; then
  for budget in ${BUDGETS}; do
    for wpair in ${WEIGHT_PAIRS}; do
      for seed in ${SEEDS_ABL}; do
        for inst in ${INSTANCES}; do
          for exp in ${EXPS_ABL}; do
            emit_task "${exp}" "${seed}" "${inst}" "${budget}" "${wpair}"
          done
        done
      done
    done
  done
fi

N_TASKS="$(wc -l < "${tmp_cmds}" | tr -d ' ')"
echo "[B-grid] tasks=${N_TASKS}  cmds_file=${tmp_cmds}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[B-grid] DRY_RUN=1 showing first 20 commands:"
  head -n 20 "${tmp_cmds}"
  exit 0
fi

# ------------------------------------------------------------
# Run tasks in parallel
# ------------------------------------------------------------
echo "[B-grid] launching with xargs -P ${MAX_JOBS} ..."
cat "${tmp_cmds}" | xargs -P "${MAX_JOBS}" -I{} bash -lc "{}"

echo "[B-grid] done."
