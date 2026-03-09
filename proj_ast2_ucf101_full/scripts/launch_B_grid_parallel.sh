#!/usr/bin/env bash
set -euo pipefail

export LC_ALL=C
export LANG=C

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
# Mainline (paper table): keep it small and comparable.
#   - EXP-B2 is the "full" MPVS config (may include LLM); prefer the nollm baselines below.
EXPS_MAIN="${EXPS_MAIN:-EXP-B1 EXP-B2-mpvs-only EXP-B2-std-budgetaware EXP-B2-bc2cec EXP-B3}"

# MPVS-only ablations (paper ablations / sanity)
EXPS_ABL="${EXPS_ABL:-EXP-B2-ab-nollm EXP-B2-ab-noverifier EXP-B2-ab-nomacro EXP-B2-ab-nomem}"
RUN_ABLATIONS="${RUN_ABLATIONS:-0}"     # default OFF (avoid runtime explosion)

# Headroom probes (controller=0) for component capability diagnosis.
RUN_HEADROOM="${RUN_HEADROOM:-0}"
EXPS_HEADROOM="${EXPS_HEADROOM:-EXP-B2-naive-atomiconly EXP-B2-naive-relinkonly EXP-B2-naive-shakeonly EXP-B2-naive-tabuonly}"
SEEDS_HEADROOM="${SEEDS_HEADROOM:-0}"
BUDGETS_HEADROOM="${BUDGETS_HEADROOM:-${BUDGETS}}"
WEIGHT_PAIRS_HEADROOM="${WEIGHT_PAIRS_HEADROOM:-${WEIGHT_PAIRS}}"
INSTANCES_HEADROOM="${INSTANCES_HEADROOM:-${INSTANCES}}"

# Evidence suite for "control necessity" (uncontrolled + controller ablations)
# Default OFF to avoid exploding runtime. Turn on with:
#   RUN_CTL_EVIDENCE=1
RUN_CTL_EVIDENCE="${RUN_CTL_EVIDENCE:-0}"
EXPS_CTL_EVIDENCE="${EXPS_CTL_EVIDENCE:-EXP-B2-uncontrolled EXP-B2-ctl-ab-notrigger EXP-B2-ctl-ab-nomacrostrict EXP-B2-ctl-ab-nomemgate}"

if [[ "${RUN_CTL_EVIDENCE}" == "1" ]]; then
  echo "[B-grid] RUN_CTL_EVIDENCE=1 -> append control-evidence experiments to EXPS_ABL"
  EXPS_ABL="${EXPS_ABL} ${EXPS_CTL_EVIDENCE}"
fi

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
echo "[B-grid] RUN_HEADROOM=${RUN_HEADROOM}  EXPS_HEADROOM=${EXPS_HEADROOM}  SEEDS_HEADROOM=${SEEDS_HEADROOM}"
echo "[B-grid] RUN_CTL_EVIDENCE=${RUN_CTL_EVIDENCE}"

# ------------------------------------------------------------
# Optional clean to avoid historical accumulation in outputs/B
#   KEEP_B_OUTPUTS=1 -> skip cleaning
# ------------------------------------------------------------
KEEP_B_OUTPUTS="${KEEP_B_OUTPUTS:-0}"
CLEAN_NEW_B="${CLEAN_NEW_B:-1}"
PURGE_LEGACY_B="${PURGE_LEGACY_B:-0}"   # keep conservative for grid; suite can purge legacy
if [[ "${KEEP_B_OUTPUTS}" != "1" && -f scripts/clean_B_outputs.sh ]]; then
  echo "[B-grid] cleaning outputs/B (CLEAN_NEW_B=${CLEAN_NEW_B} PURGE_LEGACY_B=${PURGE_LEGACY_B})"
  PURGE_LEGACY="${PURGE_LEGACY_B}" CLEAN_NEW="${CLEAN_NEW_B}" bash scripts/clean_B_outputs.sh
else
  echo "[B-grid] KEEP_B_OUTPUTS=1 or clean script missing -> skip cleaning"
fi

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
  # Optional prefix to avoid overwriting across sweeps (e.g., HP00__b20k_...)
  local pfx="${RUN_TAG_PREFIX:-}"
  if [[ -n "${pfx}" ]]; then
    tag="${pfx}__${tag}"
  fi
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

# Headroom grid (optional)
if [[ "${RUN_HEADROOM}" == "1" ]]; then
  for budget in ${BUDGETS_HEADROOM}; do
    for wpair in ${WEIGHT_PAIRS_HEADROOM}; do
      for seed in ${SEEDS_HEADROOM}; do
        for inst in ${INSTANCES_HEADROOM}; do
          for exp in ${EXPS_HEADROOM}; do
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

# ------------------------------------------------------------
# Auto-pack after grid (default ON)
#   PACK_AFTER=0 -> skip packing
#   PACK_TRACE_CSV=1 -> include trace.csv (default 0 excludes)
# ------------------------------------------------------------
PACK_AFTER="${PACK_AFTER:-1}"
if [[ "${PACK_AFTER}" == "1" && -f scripts/pack_B_outputs.sh ]]; then
  echo "[B-grid] packing results ... (PACK_TRACE_CSV=${PACK_TRACE_CSV:-0})"
  PACK_TRACE_CSV="${PACK_TRACE_CSV:-0}" bash scripts/pack_B_outputs.sh || true
else
  echo "[B-grid] PACK_AFTER=0 or pack script missing -> skip packing"
fi

# Quick sanity checks (best-effort)
echo "[B-grid] report.json count under outputs/B:"
find outputs/B -name report.json 2>/dev/null | wc -l || true
echo "[B-grid] scan for common errors under outputs/B (best-effort):"
grep -R "Traceback\|ERROR\|Exception\|RateLimit\|429" -n outputs/B 2>/dev/null || true
