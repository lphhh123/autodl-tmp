#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EXP_ID="${1:-}"
SEED="${2:-0}"

if [[ -z "$EXP_ID" ]]; then
  echo "Usage: $0 <EXP_ID> [SEED]"
  exit 1
fi

LAYOUT_INPUT="outputs/P3/A3/layout_input.json"
if [[ "$EXP_ID" == EXP-B* ]]; then
  if [ ! -f "${LAYOUT_INPUT}" ]; then
    echo "[ERROR] Missing ${LAYOUT_INPUT}"
    echo "Run EXP-A3 (or run_version_c with --export_layout_input) first to generate layout_input.json"
    echo "Example:"
    echo "  python scripts/run_version_c.py --config configs/vc_phase3_full_ucf101.yaml --out_dir outputs/P3/A3 --seed 0 --export_layout_input true"
    exit 2
  fi
fi

run_ast () {
  local cfg="$1"
  local out="$2"
  python -m scripts.run_ast2_ucf101 --cfg "$cfg" --out_dir "$out" --seed "$SEED"
}

run_vc () {
  local cfg="$1"
  local out="$2"
  python -m scripts.run_version_c --cfg "$cfg" --out_dir "$out" --seed "$SEED"
}

run_layout () {
  local cfg="$1"
  local out="$2"
  python -m scripts.run_layout_agent \
    --layout_input outputs/P3/A3/layout_input.json \
    --cfg "$cfg" --out_dir "$out" --seed "$SEED"
}

run_layout_heuragenix () {
  local cfg="$1"
  local out="$2"
  python -m scripts.run_layout_heuragenix \
    --layout_input outputs/P3/A3/layout_input.json \
    --cfg "$cfg" --out_dir "$out" --seed "$SEED"
}

case "$EXP_ID" in
  # -------------------------
  # Innovation A (Main/Core)
  # -------------------------
  EXP-A1) run_ast configs/ast2_ucf101_dense.yaml           "outputs/EXP-A1/seed${SEED}" ;;
  EXP-A2) run_ast configs/ast2_ucf101_ast_only.yaml        "outputs/EXP-A2/seed${SEED}" ;;
  EXP-A3) run_ast configs/ast2_ucf101_ast_hw.yaml          "outputs/EXP-A3/seed${SEED}" ;;
  EXP-A4) run_vc  configs/vc_phase3_full_ucf101.yaml       "outputs/EXP-A4/seed${SEED}" ;;
  EXP-A5) run_vc  configs/vc_phase3_twostage_ucf101.yaml   "outputs/EXP-A5_twostage/seed${SEED}" ;;
  EXP-A6) run_vc  configs/vc_phase3_mapping_only_ucf101.yaml "outputs/EXP-A6_mappingonly/seed${SEED}" ;;
  EXP-A7) run_vc  configs/vc_phase3_layout_only_ucf101.yaml  "outputs/EXP-A7_layoutonly/seed${SEED}" ;;

  # A-G2 fairness (same rho_target)
  EXP-A-G2-uniform) run_ast configs/ablations/ast_uniform_keep.yaml "outputs/EXP-A-G2-uniform/seed${SEED}" ;;
  EXP-A-G2-random)  run_ast configs/ablations/ast_random_keep.yaml  "outputs/EXP-A-G2-random/seed${SEED}" ;;
  EXP-A-G2-ours)    run_vc  configs/vc_phase3_full_ucf101.yaml      "outputs/EXP-A-G2-ours/seed${SEED}" ;;

  # A-G3 ablations
  EXP-Abl-time)     run_ast configs/ablations/ast_no_time.yaml      "outputs/EXP-Abl-time/seed${SEED}" ;;
  EXP-Abl-space)    run_ast configs/ablations/ast_no_space.yaml     "outputs/EXP-Abl-space/seed${SEED}" ;;
  EXP-Abl-vor)      run_ast configs/ablations/ast_no_voronoi.yaml   "outputs/EXP-Abl-vor/seed${SEED}" ;;
  EXP-Abl-1lvl)     run_ast configs/ablations/ast_level1.yaml       "outputs/EXP-Abl-1lvl/seed${SEED}" ;;
  EXP-Abl-nomodal)  run_ast configs/ablations/ast_no_modal.yaml     "outputs/EXP-Abl-nomodal/seed${SEED}" ;;
  EXP-Abl-uniform)  run_ast configs/ablations/ast_uniform_keep.yaml "outputs/EXP-Abl-uniform/seed${SEED}" ;;
  EXP-Abl-random)   run_ast configs/ablations/ast_random_keep.yaml  "outputs/EXP-Abl-random/seed${SEED}" ;;

  # -------------------------
  # Innovation B (Layout)
  # -------------------------
  EXP-B0)        run_layout_heuragenix configs/layout_agent/layout_B0_heuragenix_llm_hh.yaml      "outputs/EXP-B0/seed${SEED}" ;;
  EXP-B0-random) run_layout_heuragenix configs/layout_agent/layout_B0_heuragenix_random_hh.yaml   "outputs/EXP-B0-random/seed${SEED}" ;;

  EXP-B1) run_layout configs/layout_agent/layout_L0_heuristic.yaml "outputs/EXP-B1/seed${SEED}" ;;
  EXP-B2) run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick.yaml "outputs/EXP-B2/seed${SEED}" ;;
  EXP-B3) run_layout configs/layout_agent/layout_L3_region_pareto_sa.yaml "outputs/EXP-B3/seed${SEED}" ;;
  EXP-B2-ab-noqueue)   run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_noqueue.yaml   "outputs/EXP-B2-ab-noqueue/seed${SEED}" ;;
  EXP-B2-ab-nofeas)    run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_nofeas.yaml    "outputs/EXP-B2-ab-nofeas/seed${SEED}" ;;
  EXP-B2-ab-nodiverse) run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_nodiverse.yaml "outputs/EXP-B2-ab-nodiverse/seed${SEED}" ;;

  # -------------------------
  # Appendix / Optional (kept but not required for main table)
  # -------------------------
  EXP-APP-A-DENSE-NOSCALE) run_ast configs/ast2_ucf101_dense_noscale.yaml "outputs/EXP-APP-A-DENSE-NOSCALE/seed${SEED}" ;;
  EXP-APP-AV-DENSE)        run_ast configs/ast2_ucf101_av_dense.yaml      "outputs/EXP-APP-AV-DENSE/seed${SEED}" ;;
  EXP-APP-AV-AST-HW)       run_ast configs/ast2_ucf101_av_ast_hw.yaml     "outputs/EXP-APP-AV-AST-HW/seed${SEED}" ;;
  EXP-APP-AV-AST-ONLY)     run_ast configs/ast2_ucf101_av_ast_only.yaml   "outputs/EXP-APP-AV-AST-ONLY/seed${SEED}" ;;
  EXP-APP-VC-PHASE2-FIXED4) run_vc configs/vc_phase2_fixed4_big.yaml      "outputs/EXP-APP-VC-PHASE2-FIXED4/seed${SEED}" ;;

  *) echo "Unknown EXP_ID=$EXP_ID"; exit 2 ;;
esac
