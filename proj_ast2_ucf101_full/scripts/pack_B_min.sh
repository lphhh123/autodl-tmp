#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   TAG_RAW=nnnw bash scripts/pack_B_min.sh
# Optional env:
#   B_OUT_ROOT=outputs/B
#   PACK_EXPS="EXP-B1 EXP-B2 EXP-B2-ab-noverifier EXP-B2-ab-nomacro EXP-B2-ab-nomem EXP-B2-ab-nollm EXP-B3"
#   TAIL_N=400
#   TRACE_TAIL_N=300
#   TRACE_SAMPLE_TARGET=5000

TAG_RAW="${TAG_RAW:-nnnw}"
TAG="$(printf "%s" "$TAG_RAW" | LC_ALL=C tr -c 'A-Za-z0-9_.+-' '_' )"

B_OUT_ROOT="${B_OUT_ROOT:-outputs/B}"
PACK_EXPS="${PACK_EXPS:-EXP-B1 EXP-B2 EXP-B2-ab-noverifier EXP-B2-ab-nomacro EXP-B2-ab-nomem EXP-B2-ab-nollm EXP-B3}"

TAIL_N="${TAIL_N:-400}"
TRACE_TAIL_N="${TRACE_TAIL_N:-300}"
TRACE_SAMPLE_TARGET="${TRACE_SAMPLE_TARGET:-5000}"

OUT="_pack_B_min_${TAG}_$(date +%Y%m%d_%H%M%S)"
rm -rf "$OUT"
mkdir -p "$OUT/_runs" "$OUT/_meta"

echo "[PACK] OUT=$OUT"
echo "[PACK] B_OUT_ROOT=$B_OUT_ROOT"
echo "[PACK] PACK_EXPS=$PACK_EXPS"

# Helper: return 0 if run is budget-full (strict), else 1.
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

# Helper: find best run_id directory.
# Preference order:
#   (1) newest run with manifest.json AND budget-full
#   (2) newest run with manifest.json
_best_run_dir() {
  local seed_dir="$1"
  local dirs
  dirs=$(find "$seed_dir" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p
' 2>/dev/null | sort -nr | awk '{print $2}') || true
  local d
  # pass 1: budget-full
  for d in $dirs; do
    if [ -f "$d/manifest.json" ] && _is_budget_full "$d"; then
      echo "$d"; return 0
    fi
  done
  # pass 2: latest manifest
  for d in $dirs; do
    if [ -f "$d/manifest.json" ]; then
      echo "$d"; return 0
    fi
  done
  return 1
}

# Helper: generate trace tail & sample WITHOUT packaging full trace.csv
_make_trace_slices() {
  local src="$1"
  local dst="$2"
  local trace="$src/trace.csv"
  [ -f "$trace" ] || return 0

  # tail (with header)
  { head -n 1 "$trace"; tail -n "$TRACE_TAIL_N" "$trace"; } > "$dst/trace_tail_${TRACE_TAIL_N}.csv" || true

  # sample ~ TRACE_SAMPLE_TARGET rows (with header)
  local lines
  lines="$(wc -l < "$trace" 2>/dev/null || echo 0)"
  local k=1
  if [ "$lines" -gt 1 ]; then
    k=$(( lines / TRACE_SAMPLE_TARGET ))
    [ "$k" -lt 1 ] && k=1
  fi
  awk -v k="$k" 'NR==1 || (NR%k==0)' "$trace" > "$dst/trace_sample_${TRACE_SAMPLE_TARGET}.csv" || true
}

for prefix in $PACK_EXPS; do
  for exp in "$B_OUT_ROOT"/${prefix}*; do
    [ -d "$exp" ] || continue
    exp_name="$(basename "$exp")"

    for seed in "$exp"/seed*; do
      [ -d "$seed" ] || continue
      seed_name="$(basename "$seed")"

      latest_run_dir="$(_best_run_dir "$seed" || true)"
      if [ -z "${latest_run_dir:-}" ] || [ ! -d "$latest_run_dir" ]; then
        echo "[PACK][WARN] no run_id dir found for $exp_name/$seed_name (skip)"
        continue
      fi

      run_id="$(basename "$latest_run_dir")"
      src="$seed/$run_id"

      stage="$OUT/_stage/outputs/B/$exp_name/$seed_name/$run_id"
      rm -rf "$OUT/_stage"
      mkdir -p "$stage"

      # Essential small files
      for f in budget.json report.json run_summary.json effective_config_snapshot.yaml manifest.json run_manifest.json; do
        [ -f "$src/$f" ] && cp -a "$src/$f" "$stage/" || true
      done

      # trace_meta.json may have been written to seed root (older behavior).
      # Copy from run_id dir if exists; else fallback to seed root.
      if [ -f "$src/trace_meta.json" ]; then
        cp -a "$src/trace_meta.json" "$stage/" || true
      elif [ -f "$seed/trace_meta.json" ]; then
        cp -a "$seed/trace_meta.json" "$stage/" || true
      fi

      # stdout tail (try log/txt)
      if [ -f "$src/stdout.log" ]; then
        tail -n "$TAIL_N" "$src/stdout.log" > "$stage/stdout_tail_${TAIL_N}.log" || true
      elif [ -f "$src/stdout.txt" ]; then
        tail -n "$TAIL_N" "$src/stdout.txt" > "$stage/stdout_tail_${TAIL_N}.log" || true
      elif [ -f "$seed/stdout.log" ]; then
        tail -n "$TAIL_N" "$seed/stdout.log" > "$stage/stdout_tail_${TAIL_N}.log" || true
      elif [ -f "$seed/stdout.txt" ]; then
        tail -n "$TAIL_N" "$seed/stdout.txt" > "$stage/stdout_tail_${TAIL_N}.log" || true
      fi

      # trace slices (do NOT pack full trace.csv)
      _make_trace_slices "$src" "$stage"

      # MIN-pack self-check (writes pack_validate.json; does not fail the pack step)
      if [ -f "scripts/validate_min_pack.py" ]; then
        python scripts/validate_min_pack.py --run_dir "$stage" --out "$stage/pack_validate.json" >/dev/null 2>&1 || true
      fi

      out_run="$OUT/_runs/${exp_name}_${seed_name}_${run_id}.tgz"
      tar -czf "$out_run" -C "$OUT/_stage" "outputs/B/$exp_name/$seed_name/$run_id" 2>/dev/null || true
      echo "[PACK] $out_run"
    done
  done
done

# minimal meta bundle for reproducibility
tar -czf "$OUT/_meta/B_cfg_and_scripts.tgz" \
  --ignore-failed-read \
  scripts/experiments_version_c.sh \
  scripts/run_layout_agent.py \
  scripts/validate_min_pack.py \
  layout \
  configs/layout_agent \
  configs/llm \
  2>/dev/null || true

FINAL="${TAG}-ALL_B_PACKS_MIN.tgz"
tar -czf "$FINAL" -C "$OUT" . 2>/dev/null || true

echo
echo "[DONE] $FINAL"
ls -lh "$FINAL"

# Make sure script is executable
# (Codex should run chmod in repo, but keep this comment here)
