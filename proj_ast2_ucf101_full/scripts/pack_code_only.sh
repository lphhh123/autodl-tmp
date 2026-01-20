#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUT="${1:-AST2_AND_HeurAgenix_CODE_ONLY.tar.gz}"

tar -czvf "${OUT}" \
  --exclude='**/__pycache__' \
  --exclude='**/*.pyc' \
  --exclude='**/.ipynb_checkpoints' \
  --exclude='**/rev' \
  --exclude='proj_ast2_ucf101_full/data' \
  --exclude='proj_ast2_ucf101_full/datasets' \
  --exclude='proj_ast2_ucf101_full/logs' \
  --exclude='proj_ast2_ucf101_full/outputs' \
  --exclude='proj_ast2_ucf101_full/proxy_weights' \
  proj_ast2_ucf101_full HeurAgenix

echo "[OK] wrote ${OUT}"
