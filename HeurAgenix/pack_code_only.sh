#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUT="${1:-HeurAgenix_CODE_ONLY.tar.gz}"

tar -czf "${OUT}" HeurAgenix \
  --exclude="HeurAgenix/.git" \
  --exclude="HeurAgenix/**/__pycache__" \
  --exclude="HeurAgenix/**/*.pyc" \
  --exclude="HeurAgenix/.pytest_cache" \
  --exclude="HeurAgenix/data" \
  --exclude="HeurAgenix/output" \
  --exclude="HeurAgenix/orllm"

echo "[OK] wrote ${OUT}"
