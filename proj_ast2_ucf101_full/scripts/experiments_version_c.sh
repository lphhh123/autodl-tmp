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

# ---- TF32 (speed) ---------------------------------------------------------
# For Innovation A we default-enable TF32 to speed up training on Ampere/Ada.
# You can override per run: ENABLE_TF32=0 bash scripts/experiments_version_c.sh ...
if [[ "${EXP_ID}" == EXP-A* ]]; then
  export ENABLE_TF32="${ENABLE_TF32:-1}"
else
  export ENABLE_TF32="${ENABLE_TF32:-0}"
fi

# ---- ensure A3 middleware exists: outputs/P3/A3/layout_input.json ----
ensure_layout_input() {
  local LI="outputs/P3/A3/layout_input.json"
  if [[ -f "${LI}" ]]; then
    echo "[ensure_layout_input] OK: ${LI}"
    return 0
  fi

  echo "[ensure_layout_input] MISSING: ${LI}"
  echo "[ensure_layout_input] Generating via Version-C Phase3 export (A3)..."

  mkdir -p outputs/P3/A3

  python -m scripts.run_version_c \
    --cfg configs/vc_phase3_full_ucf101.yaml \
    --out_dir outputs/P3/A3 \
    --seed "${SEED}" \
    --export_layout_input \
    --export_dir outputs/P3/A3/layout_input_export

  if [[ ! -f "${LI}" ]]; then
    echo "[ensure_layout_input] FAILED: still missing ${LI}"
    exit 3
  fi
  echo "[ensure_layout_input] GENERATED: ${LI}"
}

run_ast () {
  local cfg="$1"
  local out="$2"
  mkdir -p "$out"
  python -m scripts.run_ast2_ucf101 --cfg "$cfg" --out_dir "$out" --seed "$SEED" 2>&1 | tee "$out/stdout.log"
}

run_vc () {
  local cfg="$1"
  local out="$2"
  mkdir -p "$out"
  python -m scripts.run_version_c --cfg "$cfg" --out_dir "$out" --seed "$SEED" 2>&1 | tee "$out/stdout.log"
}

run_layout () {
  local cfg="$1"
  local out="$2"
  ensure_layout_input
  python -m scripts.run_layout_agent \
    --layout_input outputs/P3/A3/layout_input.json \
    --cfg "$cfg" --out_dir "$out" --seed "$SEED"
}

run_layout_heuragenix () {
  local cfg="$1"
  local out="$2"
  ensure_layout_input
  python -m scripts.run_layout_heuragenix \
    --layout_input outputs/P3/A3/layout_input.json \
    --cfg "$cfg" --out_dir "$out" --seed "$SEED"
}

case "$EXP_ID" in
  # -------------------------
  # Innovation A (Main/Core)
  # -------------------------
  EXP-A1)
    export BASELINE_STATS_EXPORT="outputs/dense_baseline/metrics.json"
    run_ast configs/ast2_ucf101_dense_A1.yaml "outputs/EXP-A1/seed${SEED}"
    ;;
  EXP-A2) run_ast configs/ast2_ucf101_ast_only.yaml              "outputs/EXP-A2/seed${SEED}" ;;
  EXP-A3) run_ast configs/ast2_ucf101_ast_hw.yaml                "outputs/EXP-A3/seed${SEED}" ;;
  EXP-A4) run_vc  configs/vc_phase3_full_ucf101.yaml             "outputs/EXP-A4/seed${SEED}" ;;
  EXP-A5) run_vc  configs/vc_phase3_twostage_ucf101.yaml         "outputs/EXP-A5_twostage/seed${SEED}" ;;
  EXP-A6) run_vc  configs/vc_phase3_mapping_only_ucf101.yaml     "outputs/EXP-A6_mappingonly/seed${SEED}" ;;
  EXP-A7) run_vc  configs/vc_phase3_layout_only_ucf101.yaml      "outputs/EXP-A7_layoutonly/seed${SEED}" ;;

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
  EXP-B0)        run_layout_heuragenix configs/layout_agent/layout_B0_heuragenix_llm_hh_exp2.yaml      "outputs/EXP-B0/seed${SEED}" ;;
  EXP-B0-random) run_layout_heuragenix configs/layout_agent/layout_B0_heuragenix_random_hh_exp2.yaml   "outputs/EXP-B0-random/seed${SEED}" ;;

  EXP-B1) run_layout configs/layout_agent/layout_L0_heuristic_exp.yaml "outputs/EXP-B1/seed${SEED}" ;;
  EXP-B2) run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_exp.yaml "outputs/EXP-B2/seed${SEED}" ;;
  EXP-B3) run_layout configs/layout_agent/layout_L3_region_pareto_sa.yaml "outputs/EXP-B3/seed${SEED}" ;;
  EXP-B2-ab-noqueue)   run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_noqueue_exp.yaml   "outputs/EXP-B2-ab-noqueue/seed${SEED}" ;;
  EXP-B2-ab-nofeas)    run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_nofeas_exp.yaml    "outputs/EXP-B2-ab-nofeas/seed${SEED}" ;;
  EXP-B2-ab-nodiverse) run_layout configs/layout_agent/layout_L4_region_pareto_llm_mixed_pick_ab_nodiverse_exp.yaml "outputs/EXP-B2-ab-nodiverse/seed${SEED}" ;;

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
