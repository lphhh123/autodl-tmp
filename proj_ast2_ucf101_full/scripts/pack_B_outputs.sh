#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

rm -rf _pack_B
mkdir -p _pack_B

# New B output root (post-migration). Can override if needed.
B_OUT_ROOT="${B_OUT_ROOT:-outputs/B}"

# Pack only selected experiment prefixes by default (avoid packing all historical EXP-B*)
DEFAULT_PACK_EXPS="EXP-B1 EXP-B2 EXP-B2-ab-nollm EXP-B2-ab-nomacro EXP-B2-ab-noverifier EXP-B3"
PACK_EXPS="${PACK_EXPS:-$DEFAULT_PACK_EXPS}"

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
tar -czf _pack_B/layout_input_P3A3.tgz \
  --ignore-failed-read \
  outputs/P3/A3/layout_input.json \
  outputs/P3/A3/instances \
  2>/dev/null || true

# 1) pack latest run per selected EXP-B* prefixes per seed
exp_dirs=()
for prefix in ${PACK_EXPS}; do
  for d in ${B_OUT_ROOT}/${prefix}*; do
    [ -d "$d" ] || continue
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
      out="_pack_B/${exp_name}_${seed_name}_FULL.tgz"
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
      out="_pack_B/${exp_name}_${seed_name}_${run_id}.tgz"
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
      out="_pack_B/${exp_name}_${seed_name}_NO_RUNID.tgz"
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
tar -czf _pack_B/B_cfg_and_scripts.tgz \
  --ignore-failed-read \
  scripts/experiments_version_c.sh \
  scripts/run_layout_agent.py \
  layout \
  configs/layout_agent \
  configs/llm \
  2>/dev/null || true

# 3) final package (exclude self)
rm -f _pack_B/ALL_B_PACKS.tgz
tar --ignore-failed-read --exclude=ALL_B_PACKS.tgz -czf _pack_B/ALL_B_PACKS.tgz -C _pack_B . 2>/dev/null || true

ls -lh _pack_B
