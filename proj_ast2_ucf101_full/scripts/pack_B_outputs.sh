#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

rm -rf _pack_B
mkdir -p _pack_B

# 0) layout_input
tar -czf _pack_B/layout_input_P3A3.tgz \
  outputs/P3/A3/layout_input.json \
  outputs/P3/A3/layout_input_export \
  2>/dev/null || true

# 1) pack latest run per EXP-B* per seed
for exp in outputs/EXP-B*; do
  [ -d "$exp" ] || continue
  exp_name=$(basename "$exp")

  for seed in "$exp"/seed*; do
    [ -d "$seed" ] || continue
    seed_name=$(basename "$seed")

    latest_run_dir=$(ls -1td "$seed"/*/ 2>/dev/null | head -n 1 | sed 's:/*$::')
    if [ -n "$latest_run_dir" ] && [ -d "$latest_run_dir" ]; then
      run_id=$(basename "$latest_run_dir")
      out="_pack_B/${exp_name}_${seed_name}_${run_id}.tgz"
      tar -czf "$out" \
        --exclude="**/heuragenix_internal/data/**" \
        "$seed/$run_id" \
        2>/dev/null
      echo "[PACK] $out"
    else
      out="_pack_B/${exp_name}_${seed_name}.tgz"
      tar -czf "$out" --exclude="**/heuragenix_internal/data/**" "$seed"
      echo "[PACK] $out (no run_id dir found)"
    fi
  done
done

# 2) scripts and configs
tar -czf _pack_B/B_cfg_and_scripts.tgz \
  scripts/experiments_version_c.sh \
  scripts/run_layout_agent.py \
  layout \
  configs/layout_agent \
  configs/llm \
  2>/dev/null || true

# 3) final package (exclude self)
rm -f _pack_B/ALL_B_PACKS.tgz
tar --exclude=ALL_B_PACKS.tgz -czf _pack_B/ALL_B_PACKS.tgz -C _pack_B .

ls -lh _pack_B
