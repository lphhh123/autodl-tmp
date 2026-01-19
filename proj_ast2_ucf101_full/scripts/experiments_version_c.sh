#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EXP_ID="${1:-}"
SEED="${2:-0}"

if [[ -z "$EXP_ID" ]]; then
  echo "Usage: $0 <EXP_ID> [SEED]"
  echo "Known EXP_IDs: EXP-A1 EXP-A2 EXP-A3 EXP-A4 EXP-A5 EXP-A6 EXP-A7 EXP-A-G2-uniform EXP-A-G2-random EXP-A-G2-ours EXP-Abl-time EXP-Abl-space EXP-Abl-vor EXP-Abl-1lvl EXP-Abl-nomodal EXP-Abl-uniform EXP-Abl-random EXP-B0 EXP-B0-random EXP-B1 EXP-B2 EXP-B2PLUS EXP-B3 EXP-B2-ab-noqueue EXP-B2-ab-nofeas EXP-B2-ab-nodiverse EXP-APP-A-DENSE-NOSCALE EXP-APP-AV-DENSE EXP-APP-AV-AST-HW EXP-APP-AV-AST-ONLY EXP-APP-VC-PHASE2-FIXED4"
  exit 1
fi

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

run_ast () {
  local cfg="$1"
  local out="$2"
  test -f "$cfg"
  python -m scripts.run_ast2_ucf101 --cfg "$cfg" --out_dir "$out" --seed "$SEED"
}

run_vc () {
  local cfg="$1"
  local out="$2"
  test -f "$cfg"
  python -m scripts.run_version_c --cfg "$cfg" --out_dir "$out" --seed "$SEED" --baseline_stats "${BASELINE_STATS}"
}

run_layout () {
  local cfg="$1"
  local out="$2"
  test -f "$cfg"
  python -m scripts.run_layout_agent \
    --layout_input "${LAYOUT_INPUT}" \
    --cfg "$cfg" --out_dir "$out" --seed "$SEED"
}

run_layout_heuragenix () {
  local cfg="$1"
  local out="$2"
  test -f "$cfg"
  python -m scripts.run_layout_heuragenix \
    --layout_input "${LAYOUT_INPUT}" \
    --cfg "$cfg" --out_dir "$out" --seed "$SEED"
}

declare -A EXP_CFG=(
  [EXP-A1]="configs/ast2_ucf101_dense.yaml"
  [EXP-A2]="configs/ast2_ucf101_ast_only.yaml"
  [EXP-A3]="configs/ast2_ucf101_ast_hw.yaml"
  [EXP-A4]="configs/vc_phase3_full_ucf101.yaml"
  [EXP-A5]="configs/vc_phase3_twostage_ucf101.yaml"
  [EXP-A6]="configs/vc_phase3_mapping_only_ucf101.yaml"
  [EXP-A7]="configs/vc_phase3_layout_only_ucf101.yaml"
  [EXP-A-G2-uniform]="configs/ablations/ast_uniform_keep.yaml"
  [EXP-A-G2-random]="configs/ablations/ast_random_keep.yaml"
  [EXP-A-G2-ours]="configs/vc_phase3_full_ucf101.yaml"
  [EXP-Abl-time]="configs/ablations/ast_no_time.yaml"
  [EXP-Abl-space]="configs/ablations/ast_no_space.yaml"
  [EXP-Abl-vor]="configs/ablations/ast_no_voronoi.yaml"
  [EXP-Abl-1lvl]="configs/ablations/ast_level1.yaml"
  [EXP-Abl-nomodal]="configs/ablations/ast_no_modal.yaml"
  [EXP-Abl-uniform]="configs/ablations/ast_uniform_keep.yaml"
  [EXP-Abl-random]="configs/ablations/ast_random_keep.yaml"
  [EXP-B0]="configs/layout_agent/layout_B0_heuragenix_llm_hh.yaml"
  [EXP-B0-random]="configs/layout_agent/layout_B0_heuragenix_random_hh.yaml"
  [EXP-B1]="configs/layout_agent/layout_L0_heuristic.yaml"
  [EXP-B2]="configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick.yaml"
  [EXP-B2PLUS]="configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_b2plus.yaml"
  [EXP-B3]="configs/layout_agent/layout_L3_region_pareto_sa.yaml"
  [EXP-B2-ab-noqueue]="configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_noqueue.yaml"
  [EXP-B2-ab-nofeas]="configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_nofeas.yaml"
  [EXP-B2-ab-nodiverse]="configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_nodiverse.yaml"
  [EXP-APP-A-DENSE-NOSCALE]="configs/ast2_ucf101_dense_noscale.yaml"
  [EXP-APP-AV-DENSE]="configs/ast2_ucf101_av_dense.yaml"
  [EXP-APP-AV-AST-HW]="configs/ast2_ucf101_av_ast_hw.yaml"
  [EXP-APP-AV-AST-ONLY]="configs/ast2_ucf101_av_ast_only.yaml"
  [EXP-APP-VC-PHASE2-FIXED4]="configs/vc_phase2_fixed4_big.yaml"
)

declare -A EXP_OUT=(
  [EXP-A1]="outputs/EXP-A1/seed${SEED}"
  [EXP-A2]="outputs/EXP-A2/seed${SEED}"
  [EXP-A3]="outputs/EXP-A3/seed${SEED}"
  [EXP-A4]="outputs/EXP-A4/seed${SEED}"
  [EXP-A5]="outputs/EXP-A5_twostage/seed${SEED}"
  [EXP-A6]="outputs/EXP-A6_mappingonly/seed${SEED}"
  [EXP-A7]="outputs/EXP-A7_layoutonly/seed${SEED}"
  [EXP-A-G2-uniform]="outputs/EXP-A-G2-uniform/seed${SEED}"
  [EXP-A-G2-random]="outputs/EXP-A-G2-random/seed${SEED}"
  [EXP-A-G2-ours]="outputs/EXP-A-G2-ours/seed${SEED}"
  [EXP-Abl-time]="outputs/EXP-Abl-time/seed${SEED}"
  [EXP-Abl-space]="outputs/EXP-Abl-space/seed${SEED}"
  [EXP-Abl-vor]="outputs/EXP-Abl-vor/seed${SEED}"
  [EXP-Abl-1lvl]="outputs/EXP-Abl-1lvl/seed${SEED}"
  [EXP-Abl-nomodal]="outputs/EXP-Abl-nomodal/seed${SEED}"
  [EXP-Abl-uniform]="outputs/EXP-Abl-uniform/seed${SEED}"
  [EXP-Abl-random]="outputs/EXP-Abl-random/seed${SEED}"
  [EXP-B0]="outputs/EXP-B0/seed${SEED}"
  [EXP-B0-random]="outputs/EXP-B0-random/seed${SEED}"
  [EXP-B1]="outputs/EXP-B1/seed${SEED}"
  [EXP-B2]="outputs/EXP-B2/seed${SEED}"
  [EXP-B2PLUS]="outputs/EXP-B2PLUS/seed${SEED}"
  [EXP-B3]="outputs/EXP-B3/seed${SEED}"
  [EXP-B2-ab-noqueue]="outputs/EXP-B2-ab-noqueue/seed${SEED}"
  [EXP-B2-ab-nofeas]="outputs/EXP-B2-ab-nofeas/seed${SEED}"
  [EXP-B2-ab-nodiverse]="outputs/EXP-B2-ab-nodiverse/seed${SEED}"
  [EXP-APP-A-DENSE-NOSCALE]="outputs/EXP-APP-A-DENSE-NOSCALE/seed${SEED}"
  [EXP-APP-AV-DENSE]="outputs/EXP-APP-AV-DENSE/seed${SEED}"
  [EXP-APP-AV-AST-HW]="outputs/EXP-APP-AV-AST-HW/seed${SEED}"
  [EXP-APP-AV-AST-ONLY]="outputs/EXP-APP-AV-AST-ONLY/seed${SEED}"
  [EXP-APP-VC-PHASE2-FIXED4]="outputs/EXP-APP-VC-PHASE2-FIXED4/seed${SEED}"
)

declare -A EXP_RUNNER=(
  [EXP-A1]="run_ast"
  [EXP-A2]="run_ast"
  [EXP-A3]="run_ast"
  [EXP-A4]="run_vc"
  [EXP-A5]="run_vc"
  [EXP-A6]="run_vc"
  [EXP-A7]="run_vc"
  [EXP-A-G2-uniform]="run_ast"
  [EXP-A-G2-random]="run_ast"
  [EXP-A-G2-ours]="run_vc"
  [EXP-Abl-time]="run_ast"
  [EXP-Abl-space]="run_ast"
  [EXP-Abl-vor]="run_ast"
  [EXP-Abl-1lvl]="run_ast"
  [EXP-Abl-nomodal]="run_ast"
  [EXP-Abl-uniform]="run_ast"
  [EXP-Abl-random]="run_ast"
  [EXP-B0]="run_layout_heuragenix"
  [EXP-B0-random]="run_layout_heuragenix"
  [EXP-B1]="run_layout"
  [EXP-B2]="run_layout"
  [EXP-B2PLUS]="run_layout"
  [EXP-B3]="run_layout"
  [EXP-B2-ab-noqueue]="run_layout"
  [EXP-B2-ab-nofeas]="run_layout"
  [EXP-B2-ab-nodiverse]="run_layout"
  [EXP-APP-A-DENSE-NOSCALE]="run_ast"
  [EXP-APP-AV-DENSE]="run_ast"
  [EXP-APP-AV-AST-HW]="run_ast"
  [EXP-APP-AV-AST-ONLY]="run_ast"
  [EXP-APP-VC-PHASE2-FIXED4]="run_vc"
)

run_exp () {
  local exp="$1"
  local cfg="${EXP_CFG[$exp]}"
  local out="${EXP_OUT[$exp]}"
  local runner="${EXP_RUNNER[$exp]}"
  if [[ -z "$cfg" || -z "$out" || -z "$runner" ]]; then
    echo "Unknown EXP_ID=$exp"
    exit 2
  fi
  "$runner" "$cfg" "$out"
}

case "$EXP_ID" in
  # -------------------------
  # Innovation A (Main/Core)
  # -------------------------
  EXP-A1|EXP-A2|EXP-A3|EXP-A4|EXP-A5|EXP-A6|EXP-A7) run_exp "$EXP_ID" ;;

  # A-G2 fairness (same rho_target)
  EXP-A-G2-uniform|EXP-A-G2-random|EXP-A-G2-ours) run_exp "$EXP_ID" ;;

  # A-G3 ablations
  EXP-Abl-time|EXP-Abl-space|EXP-Abl-vor|EXP-Abl-1lvl|EXP-Abl-nomodal|EXP-Abl-uniform|EXP-Abl-random) run_exp "$EXP_ID" ;;

  # -------------------------
  # Innovation B (Layout)
  # -------------------------
  EXP-B0|EXP-B0-random|EXP-B1|EXP-B2|EXP-B2PLUS|EXP-B3|EXP-B2-ab-noqueue|EXP-B2-ab-nofeas|EXP-B2-ab-nodiverse) run_exp "$EXP_ID" ;;

  # -------------------------
  # Appendix / Optional (kept but not required for main table)
  # -------------------------
  EXP-APP-A-DENSE-NOSCALE|EXP-APP-AV-DENSE|EXP-APP-AV-AST-HW|EXP-APP-AV-AST-ONLY|EXP-APP-VC-PHASE2-FIXED4) run_exp "$EXP_ID" ;;

  *) echo "Unknown EXP_ID=$EXP_ID"; exit 2 ;;
esac
