#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Clean the new B output root (outputs/B) before reruns
CLEAN_NEW="${CLEAN_NEW:-1}"

# Purge legacy B folders that used to live directly under outputs/EXP-B*
PURGE_LEGACY="${PURGE_LEGACY:-1}"

echo "[clean_B_outputs] CLEAN_NEW=${CLEAN_NEW} PURGE_LEGACY=${PURGE_LEGACY}"

if [[ "${PURGE_LEGACY}" == "1" ]]; then
  echo "[clean_B_outputs] Removing legacy B dirs: outputs/EXP-B* and outputs/SMOKE/EXP-B* (if exist)"
  rm -rf outputs/EXP-B* 2>/dev/null || true
  rm -rf outputs/SMOKE/EXP-B* 2>/dev/null || true
fi

if [[ "${CLEAN_NEW}" == "1" ]]; then
  echo "[clean_B_outputs] Cleaning new B output root: outputs/B/"
  rm -rf outputs/B 2>/dev/null || true
  mkdir -p outputs/B/stdout_agg
fi

echo "[clean_B_outputs] done."
