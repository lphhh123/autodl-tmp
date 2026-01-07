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

case "$EXP_ID" in
  # --- Innovation A ---
  EXP-A1) run_ast configs/ast2_ucf101_dense.yaml        "outputs/EXP-A1/seed${SEED}" ;;
  EXP-A2) run_ast configs/ast2_ucf101_ast_only.yaml     "outputs/EXP-A2/seed${SEED}" ;;
  EXP-A3) run_ast configs/ast2_ucf101_ast_hw.yaml       "outputs/EXP-A3/seed${SEED}" ;;
  EXP-A4) run_vc  configs/vc_phase3_full_ucf101.yaml    "outputs/EXP-A4/seed${SEED}" ;;
  EXP-A5) run_vc  configs/vc_phase3_full_ucf101.yaml    "outputs/EXP-A5_twostage/seed${SEED}" ;;
  EXP-A6) run_vc  configs/vc_phase3_full_ucf101.yaml    "outputs/EXP-A6_mappingonly/seed${SEED}" ;;
  EXP-A7) run_vc  configs/vc_phase3_full_ucf101.yaml    "outputs/EXP-A7_layoutonly/seed${SEED}" ;;

  # --- Innovation B ---
  EXP-B1) run_layout configs/layout_agent/layout_L0_heuristic.yaml "outputs/EXP-B1/seed${SEED}" ;;
  EXP-B2) run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick.yaml "outputs/EXP-B2/seed${SEED}" ;;
  EXP-B3) run_layout configs/layout_agent/layout_L3_region_pareto.yaml "outputs/EXP-B3/seed${SEED}" ;;

  EXP-B2-ab-noqueue)   run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_noqueue.yaml   "outputs/EXP-B2-ab-noqueue/seed${SEED}" ;;
  EXP-B2-ab-nofeas)    run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_nofeas.yaml    "outputs/EXP-B2-ab-nofeas/seed${SEED}" ;;
  EXP-B2-ab-nodiverse) run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_nodiverse.yaml "outputs/EXP-B2-ab-nodiverse/seed${SEED}" ;;

  *) echo "Unknown EXP_ID=$EXP_ID"; exit 2 ;;
esac
