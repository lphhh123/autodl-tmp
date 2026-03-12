#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Safety: avoid accidental double-expansion / axis conflicts from outer shells.
# This sweep uses VARIANT_TAG (includes M axis), so we MUST NOT also expand
# PROBE_CALLS_MODE(S) here.
unset PROBE_CALLS_MODE || true
unset PROBE_CALLS_MODES || true
unset EFF_CALLS_ALPHA_OVERRIDE || true

FAST_VARIANTS="${FAST_VARIANTS:-0}"
KEEP_B_OUTPUTS="${KEEP_B_OUTPUTS:-1}"
export KEEP_B_OUTPUTS

if [[ "$FAST_VARIANTS" == "1" ]]; then
  mapfile -t VARIANTS < <(python scripts/gen_B_variants_tags.py --fast)
else
  mapfile -t VARIANTS < <(python scripts/gen_B_variants_tags.py)
fi

echo "[variants-sweep] variants=${#VARIANTS[@]} FAST_VARIANTS=${FAST_VARIANTS}"

for v in "${VARIANTS[@]}"; do
  [[ -z "$v" ]] && continue
  echo "[variants-sweep] running ${v}"
  VARIANT_TAG="$v" bash scripts/launch_B_grid_parallel.sh

done
