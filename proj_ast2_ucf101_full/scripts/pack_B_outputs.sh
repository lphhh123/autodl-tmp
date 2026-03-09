#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PACK_OUT_DIR="${PACK_OUT_DIR:-_pack_B}"
PACK_CLEAN_OUT_DIR="${PACK_CLEAN_OUT_DIR:-1}"
if [[ "${PACK_CLEAN_OUT_DIR}" == "1" ]]; then
  rm -rf "${PACK_OUT_DIR}"
fi
mkdir -p "${PACK_OUT_DIR}"

# New B output root (post-migration). Can override if needed.
B_OUT_ROOT="${B_OUT_ROOT:-outputs/B}"
PACK_MATCH_TAG="${PACK_MATCH_TAG:-}"
PACK_SKIP_ANALYSIS="${PACK_SKIP_ANALYSIS:-0}"

# Pack only selected experiment prefixes by default (avoid packing all historical EXP-B*)
# Groups:
#   - MAIN: paper table (nollm baselines)
#   - HEADROOM: controller=0 probes (for oracle/regret construction)
#   - ABL: MPVS ablations
#   - EVIDENCE: uncontrolled + controller ablations
PACK_MAIN_DEFAULT="EXP-B1 EXP-B2-mpvs-only EXP-B2-std-budgetaware EXP-B2-bc2cec EXP-B2-bc2cec-noprobe EXP-B2-bc2cec-probe-raw EXP-B3"
PACK_HEADROOM_DEFAULT="EXP-B2-naive-atomiconly EXP-B2-naive-relinkonly EXP-B2-naive-shakeonly EXP-B2-naive-tabuonly"
PACK_ABL_DEFAULT="EXP-B2-ab-nollm EXP-B2-ab-nomacro EXP-B2-ab-noverifier EXP-B2-ab-nomem"
PACK_EVIDENCE_DEFAULT="EXP-B2-uncontrolled EXP-B2-ctl-ab-notrigger EXP-B2-ctl-ab-nomacrostrict EXP-B2-ctl-ab-nomemgate"

PACK_INCLUDE_ABLATIONS="${PACK_INCLUDE_ABLATIONS:-0}"
PACK_INCLUDE_EVIDENCE="${PACK_INCLUDE_EVIDENCE:-0}"

DEFAULT_PACK_EXPS="${PACK_MAIN_DEFAULT} ${PACK_HEADROOM_DEFAULT}"
if [[ "${PACK_INCLUDE_ABLATIONS}" == "1" ]]; then
  DEFAULT_PACK_EXPS="${DEFAULT_PACK_EXPS} ${PACK_ABL_DEFAULT}"
fi
if [[ "${PACK_INCLUDE_EVIDENCE}" == "1" ]]; then
  DEFAULT_PACK_EXPS="${DEFAULT_PACK_EXPS} ${PACK_EVIDENCE_DEFAULT}"
fi

PACK_EXPS="${PACK_EXPS:-$DEFAULT_PACK_EXPS}"
echo "[PACK] PACK_EXPS=${PACK_EXPS}"
echo "[PACK] PACK_INCLUDE_ABLATIONS=${PACK_INCLUDE_ABLATIONS} PACK_INCLUDE_EVIDENCE=${PACK_INCLUDE_EVIDENCE}"
echo "[PACK] PACK_OUT_DIR=${PACK_OUT_DIR} PACK_MATCH_TAG=${PACK_MATCH_TAG} PACK_SKIP_ANALYSIS=${PACK_SKIP_ANALYSIS}"

# Whether to include legacy B0/B0* (default off)
PACK_INCLUDE_B0="${PACK_INCLUDE_B0:-0}"

# Default: do NOT pack huge trace.csv (keeps tgz small for sharing)
PACK_TRACE_CSV="${PACK_TRACE_CSV:-0}"

# Prefer packing a budget-full run when available (avoid half-runs after retries).
_is_budget_full() {
  local run_dir="$1"
  [ -f "$run_dir/budget.json" ] || return 1
  python - <<'PY' "$run_dir/budget.json" >/dev/null 2>&1
import json, sys
p = sys.argv[1]
try:
    b = json.load(open(p, 'r', encoding='utf-8'))
except Exception:
    sys.exit(1)
try:
    exhausted = bool(b.get('budget_exhausted', False))
    actual = int(b.get('actual_eval_calls', -1))
    lim = int((b.get('primary_limit', {}) or {}).get('limit', -1))
except Exception:
    sys.exit(1)
if exhausted and lim > 0 and actual == lim:
    sys.exit(0)
sys.exit(1)
PY
}

_best_run_dir() {
  local seed_dir="$1"
  local dirs
  dirs=$(find "$seed_dir" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk '{print $2}') || true
  local d
  for d in $dirs; do
    [ -f "$d/manifest.json" ] || continue
    if _is_budget_full "$d"; then
      echo "$d"; return 0
    fi
  done
  for d in $dirs; do
    [ -f "$d/manifest.json" ] || continue
    echo "$d"; return 0
  done
  return 1
}

# 0) layout_input
tar -czf "${PACK_OUT_DIR}/layout_input_P3A3.tgz" \
  --ignore-failed-read \
  outputs/P3/A3/layout_input.json \
  outputs/P3/A3/instances \
  2>/dev/null || true

# 1) pack latest run per selected EXP-B* prefixes per seed
exp_dirs=()
for prefix in ${PACK_EXPS}; do
  for d in ${B_OUT_ROOT}/${prefix}*; do
    [ -d "$d" ] || continue
    bn="$(basename "$d")"
    if [[ -n "${PACK_MATCH_TAG}" && "$bn" != *"${PACK_MATCH_TAG}"* ]]; then
      continue
    fi
    exp_dirs+=("$d")
  done
done

# de-dup
mapfile -t exp_dirs < <(printf "%s\n" "${exp_dirs[@]}" | sort -u)

for exp in "${exp_dirs[@]}"; do
  [ -d "$exp" ] || continue
  exp_name=$(basename "$exp")

  if [[ "$PACK_INCLUDE_B0" != "1" ]]; then
    if [[ "$exp_name" == EXP-B0* ]]; then
      continue
    fi
  fi

  for seed in "$exp"/seed*; do
    [ -d "$seed" ] || continue
    seed_name=$(basename "$seed")

    if [[ "$exp_name" == "EXP-B0" || "$exp_name" == "EXP-B0-random" ]]; then
      out="${PACK_OUT_DIR}/${exp_name}_${seed_name}_FULL.tgz"
      if [[ "${PACK_TRACE_CSV}" != "1" ]]; then
        tar -czf "$out" \
          --ignore-failed-read \
          --exclude="**/heuragenix_internal/data/**" \
          --exclude="**/__heuragenix_work/**" \
          --exclude="**/trace.csv" \
          "$seed" \
          2>/dev/null || true
      else
        tar -czf "$out" \
          --ignore-failed-read \
          --exclude="**/heuragenix_internal/data/**" \
          --exclude="**/__heuragenix_work/**" \
          "$seed" \
          2>/dev/null || true
      fi
      echo "[PACK] $out"
      continue
    fi

    latest_run_dir="$(_best_run_dir "$seed" || true)"
    if [ -n "$latest_run_dir" ] && [ -d "$latest_run_dir" ]; then
      run_id=$(basename "$latest_run_dir")
      out="${PACK_OUT_DIR}/${exp_name}_${seed_name}_${run_id}.tgz"
      if [[ "${PACK_TRACE_CSV}" != "1" ]]; then
        tar -czf "$out" \
          --ignore-failed-read \
          --exclude="**/heuragenix_internal/data/**" \
          --exclude="**/recordings.jsonl" \
          --exclude="**/trace_events.jsonl" \
          --exclude="**/candidate_pool_debug.json" \
          --exclude="**/trace.csv" \
          --exclude="**/*.npy" --exclude="**/*.npz" \
          --exclude="**/*.pt" --exclude="**/*.pth" --exclude="**/*.ckpt" \
          --exclude="**/__pycache__/**" \
          "$seed/$run_id" \
          2>/dev/null || true
      else
        tar -czf "$out" \
          --ignore-failed-read \
          --exclude="**/heuragenix_internal/data/**" \
          --exclude="**/recordings.jsonl" \
          --exclude="**/trace_events.jsonl" \
          --exclude="**/candidate_pool_debug.json" \
          --exclude="**/*.npy" --exclude="**/*.npz" \
          --exclude="**/*.pt" --exclude="**/*.pth" --exclude="**/*.ckpt" \
          --exclude="**/__pycache__/**" \
          "$seed/$run_id" \
          2>/dev/null || true
      fi
      echo "[PACK] $out"
    else
      out="${PACK_OUT_DIR}/${exp_name}_${seed_name}_NO_RUNID.tgz"
      seed_files=(
        "$seed/report.json"
        "$seed/layout_best.json"
        "$seed/llm_usage.jsonl"
        "$seed/manifest.json"
        "$seed/effective_config_snapshot.yaml"
        "$seed/budget.json"
        "$seed/trace_meta.json"
      )
      if [[ "${PACK_TRACE_CSV}" == "1" ]]; then
        seed_files+=("$seed/trace.csv")
      fi

      tar -czf "$out" \
        --ignore-failed-read \
        "${seed_files[@]}" \
        2>/dev/null || true
      echo "[PACK] $out (no run_id dir found; packed minimal seed files)"
    fi
  done
done

# 2) scripts and configs
tar -czf "${PACK_OUT_DIR}/B_cfg_and_scripts.tgz" \
  --ignore-failed-read \
  scripts/experiments_version_c.sh \
  scripts/launch_B_grid_parallel.sh \
  scripts/launch_B_mainline.sh \
  scripts/pack_B_outputs.sh \
  scripts/run_layout_agent.py \
  layout \
  configs/layout_agent \
  configs/llm \
  2>/dev/null || true

# 2.5) analysis bundle (oracle/regret + macro utilization)
if [[ "${PACK_SKIP_ANALYSIS}" != "1" && -f scripts/analyze_B_oracle_regret.py ]]; then
  echo "[PACK] running scripts/analyze_B_oracle_regret.py ..."
  python scripts/analyze_B_oracle_regret.py --root "${B_OUT_ROOT}" --out_dir "${B_OUT_ROOT}/_analysis" >/dev/null 2>&1 || true
  tar -czf "${PACK_OUT_DIR}/B_analysis.tgz" \
    --ignore-failed-read \
    "${B_OUT_ROOT}/_analysis" \
    2>/dev/null || true
fi

# 3) final package (exclude self)
rm -f "${PACK_OUT_DIR}/ALL_B_PACKS.tgz"
tar --ignore-failed-read --exclude=ALL_B_PACKS.tgz -czf "${PACK_OUT_DIR}/ALL_B_PACKS.tgz" -C "${PACK_OUT_DIR}" . 2>/dev/null || true

ls -lh "${PACK_OUT_DIR}"
