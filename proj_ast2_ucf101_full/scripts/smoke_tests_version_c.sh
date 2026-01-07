#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[SMOKE] Proxy ms/mem"
python -m scripts.run_proxy_ms_mem --cfg configs/proxy_ms_mem.yaml

echo "[SMOKE] Proxy power"
python -m scripts.run_proxy_power --cfg configs/proxy_power.yaml

echo "[SMOKE] AST single-device"
python -m scripts.run_ast2_ucf101 --cfg configs/smoke_ast_ucf101.yaml --seed 0 --out_dir outputs/SMOKE/ast

echo "[SMOKE] Version-C"
python -m scripts.run_version_c --cfg configs/smoke_version_c_ucf101.yaml --seed 0 --out_dir outputs/SMOKE/vc

LAYOUT_INPUT="outputs/SMOKE/vc/layout_input.json"

if [[ ! -f "$LAYOUT_INPUT" ]]; then
  echo "[SMOKE][ERROR] Missing layout_input.json at $LAYOUT_INPUT"
  exit 1
fi

echo "[SMOKE] Layout agent L0"
python -m scripts.run_layout_agent \
  --layout_input "$LAYOUT_INPUT" \
  --cfg configs/layout_agent/layout_L0_heuristic.yaml \
  --out_dir outputs/SMOKE/layout_L0_heuristic

echo "[SMOKE] Layout agent L4 pick-ID (fallback if no key)"
python -m scripts.run_layout_agent \
  --layout_input "$LAYOUT_INPUT" \
  --cfg configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick.yaml \
  --out_dir outputs/SMOKE/layout_L4_pick

echo "[SMOKE DONE]"
