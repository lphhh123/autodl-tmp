#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEED="${SEED:-0}"
BASE_OUT="outputs"
SHARED_DIR="${BASE_OUT}/_shared"
mkdir -p "${SHARED_DIR}"

BASELINE_OUT="${BASE_OUT}/EXP-A0/seed${SEED}"
BASELINE_STATS="${BASELINE_OUT}/metrics.json"

LAYOUT_GEN_OUT="${SHARED_DIR}/gen_layout_input/seed${SEED}"
LAYOUT_INPUT="${SHARED_DIR}/layout_input_seed${SEED}.json"

ensure_baseline() {
  if [[ -f "${BASELINE_STATS}" ]]; then
    echo "[OK] baseline exists: ${BASELINE_STATS}"
    return
  fi
  echo "[GEN] baseline not found, running EXP-A0 (dense baseline) ..."
  python -m scripts.run_ast2_ucf101 \
    --cfg configs/ast2_ucf101_dense.yaml \
    --out_dir "${BASE_OUT}/EXP-A0/seed${SEED}" \
    --seed "${SEED}"
  test -f "${BASELINE_STATS}"
}

ensure_layout_input() {
  if [[ -f "${LAYOUT_INPUT}" ]]; then
    echo "[OK] layout_input exists: ${LAYOUT_INPUT}"
    return
  fi
  echo "[GEN] layout_input not found, exporting from Version-C once ..."
  mkdir -p "${LAYOUT_GEN_OUT}"
  python -m scripts.run_version_c \
    --cfg configs/smoke_version_c_ucf101.yaml \
    --out_dir "${LAYOUT_GEN_OUT}" \
    --seed "${SEED}" \
    --baseline_stats "${BASELINE_STATS}" \
    --export_layout_input true \
    --export_dir "${LAYOUT_GEN_OUT}/exports/layout_input"
  test -f "${LAYOUT_GEN_OUT}/exports/layout_input/layout_input.json"
  cp -f "${LAYOUT_GEN_OUT}/exports/layout_input/layout_input.json" "${LAYOUT_INPUT}"
}

ensure_baseline
ensure_layout_input

echo "[SMOKE] Proxy ms/mem"
python -m scripts.run_proxy_ms_mem --cfg configs/proxy_ms_mem.yaml

echo "[SMOKE] Proxy power"
python -m scripts.run_proxy_power --cfg configs/proxy_power.yaml

echo "[SMOKE] StableHW schema check"
python -m scripts.smoke_check_stable_hw_schema --cfg configs/smoke_version_c_ucf101.yaml

echo "[SMOKE] StableHW gradient check"
python -m scripts.smoke_check_hw_loss_grad --cfg configs/smoke_version_c_ucf101.yaml

echo "[SMOKE] AST single-device"
python -m scripts.run_ast2_ucf101 --cfg configs/smoke_ast_ucf101.yaml

echo "[SMOKE] Version-C"
python -m scripts.run_version_c --cfg configs/smoke_version_c_ucf101.yaml --baseline_stats "${BASELINE_STATS}"

echo "[SMOKE] Layout agent L0"
python -m scripts.run_layout_agent \
  --layout_input "${LAYOUT_INPUT}" \
  --cfg configs/layout_agent/layout_L0_heuristic.yaml \
  --out_dir outputs/SMOKE/layout_L0_heuristic \
  --seed 0

echo "[SMOKE] Layout agent L4 pick-ID (fallback if no key)"
python -m scripts.run_layout_agent \
  --layout_input "${LAYOUT_INPUT}" \
  --cfg configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick.yaml \
  --out_dir outputs/SMOKE/layout_L4_pick \
  --seed 0

echo "[SMOKE] HeurAgenix baseline llm_hh (auto-fallback if no key)"
python -m scripts.run_layout_heuragenix \
  --layout_input "${LAYOUT_INPUT}" \
  --cfg configs/layout_agent/layout_B0_heuragenix_llm_hh.yaml \
  --out_dir outputs/SMOKE/layout_B0_heuragenix_llm_hh \
  --seed 0

echo "[SMOKE DONE]"
