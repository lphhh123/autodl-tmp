#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ------------------------------------------------------------
# Sweep goal:
#   Keep the CURRENT repaired B-line structure unchanged, and only tune:
#     1) probe cf_discount_by_stage
#     2) controller cec counterfactual_discount_by_stage
#     3) probe shrinkage.lambda_calls
#
# Experiment protocol must remain IDENTICAL to current main run:
#   - SEEDS_MAIN="0 1 2"
#   - INSTANCES="cluster4 chain_skip chain_skip_randw"
#   - BUDGETS="20k 40k 80k 160k 240k"
#   - WEIGHT_PAIRS="0.3,0.7"
#   - EXPS_MAIN includes std / bc2cec / probe-raw / noprobe
#
# Each variant:
#   - runs into a fresh outputs/B scratch space
#   - is then packed into its own NEW result folder
#   - gets its own uniquely named ALL_B_PACKS_<variant>.tgz
# ------------------------------------------------------------

MAX_JOBS="${MAX_JOBS:-36}"
INSTANCES="${INSTANCES:-cluster4 chain_skip chain_skip_randw}"
BUDGETS="${BUDGETS:-20k 40k 80k 160k 240k}"
WEIGHT_PAIRS="${WEIGHT_PAIRS:-0.3,0.7}"
SEEDS_MAIN="${SEEDS_MAIN:-0 1 2}"
EXPS_MAIN_SWEEP="${EXPS_MAIN_SWEEP:-EXP-B2-std-budgetaware EXP-B2-bc2cec EXP-B2-bc2cec-probe-raw EXP-B2-bc2cec-noprobe}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${SWEEP_ROOT:-_pack_B_cfstage_shrink_${STAMP}}"
CFG_DIR="${CFG_DIR:-configs/layout_agent/_k4_sweeps/_cfstage_shrink}"

mkdir -p "${SWEEP_ROOT}"
mkdir -p "${CFG_DIR}"

PLAN="${SWEEP_ROOT}/sweep_plan.tsv"
echo -e "id\tcfg_bc2cec\tcfg_raw\tcfg_noprobe\tprobe_stage_discount\tcec_stage_discount\tshrink_lambda_calls" > "${PLAN}"

echo "[CF-STAGE] ROOT=${ROOT}"
echo "[CF-STAGE] SWEEP_ROOT=${SWEEP_ROOT}"
echo "[CF-STAGE] CFG_DIR=${CFG_DIR}"
echo "[CF-STAGE] MAX_JOBS=${MAX_JOBS}"
echo "[CF-STAGE] INSTANCES=${INSTANCES}"
echo "[CF-STAGE] BUDGETS=${BUDGETS}"
echo "[CF-STAGE] WEIGHT_PAIRS=${WEIGHT_PAIRS}"
echo "[CF-STAGE] SEEDS_MAIN=${SEEDS_MAIN}"
echo "[CF-STAGE] EXPS_MAIN_SWEEP=${EXPS_MAIN_SWEEP}"

write_variant_cfgs() {
  local vid="$1"
  local probe_stage_discount="$2"
  local cec_stage_discount="$3"
  local shrink_lambda_calls="$4"

  local cfg="${CFG_DIR}/${vid}.yaml"
  local cfg_raw="${CFG_DIR}/${vid}_raw.yaml"
  local cfg_nop="${CFG_DIR}/${vid}_noprobe.yaml"

  cat > "${cfg}" <<YAML
_base_: ../S015_A1_lowtax_B2_cfboost_signedaos_K4_bc2cec.yaml

# Variant ${vid}
# Keep the current repaired B-line structure unchanged.
# Only tune:
#   - detailed_place.mpvs.macros.probe.cf_discount_by_stage
#   - detailed_place.mpvs.controller.cec.counterfactual_discount_by_stage
#   - detailed_place.mpvs.macros.probe.shrinkage.lambda_calls
detailed_place:
  mpvs:
    macros:
      probe:
        cf_discount_by_stage: ${probe_stage_discount}
        shrinkage:
          enabled: true
          lambda_calls: ${shrink_lambda_calls}
    controller:
      cec:
        counterfactual_discount_by_stage: ${cec_stage_discount}
YAML

  cat > "${cfg_raw}" <<YAML
_base_: $(basename "${cfg}")

# Pure probe-CF ablation:
# keep EVERYTHING identical to ${vid}, only remove probe-side CF estimation.
detailed_place:
  mpvs:
    macros:
      probe:
        cf_discount: 0.0
        cf_reliability_tau: 0.0
        weight_metric: raw
        cf_one_sided_update: false
        # keep shrinkage ON so this remains a clean CF-only ablation
YAML

  cat > "${cfg_nop}" <<YAML
_base_: $(basename "${cfg}")

detailed_place:
  mpvs:
    macros:
      probe:
        enabled: false
YAML

  echo -e "${vid}\t${cfg}\t${cfg_raw}\t${cfg_nop}\t${probe_stage_discount}\t${cec_stage_discount}\t${shrink_lambda_calls}" >> "${PLAN}"
}

run_and_pack_variant() {
  local vid="$1"
  local cfg="${CFG_DIR}/${vid}.yaml"
  local cfg_raw="${CFG_DIR}/${vid}_raw.yaml"
  local cfg_nop="${CFG_DIR}/${vid}_noprobe.yaml"

  local out_dir="${SWEEP_ROOT}/${vid}"
  mkdir -p "${out_dir}"
  mkdir -p "${out_dir}/configs"

  cp -f "${cfg}" "${out_dir}/configs/"
  cp -f "${cfg_raw}" "${out_dir}/configs/"
  cp -f "${cfg_nop}" "${out_dir}/configs/"

  cat > "${out_dir}/variant_meta.txt" <<EOF2
variant_id=${vid}
INSTANCES=${INSTANCES}
BUDGETS=${BUDGETS}
WEIGHT_PAIRS=${WEIGHT_PAIRS}
SEEDS_MAIN=${SEEDS_MAIN}
EXPS_MAIN_SWEEP=${EXPS_MAIN_SWEEP}
EOF2

  echo "[CF-STAGE] >>> Running ${vid}"

  # Use outputs/B only as temporary scratch during this variant run.
  # Final kept artifacts are packed into ${out_dir}.
  RUN_TAG_PREFIX="${vid}" \
  MAX_JOBS="${MAX_JOBS}" \
  RUN_HEADROOM=0 \
  EXPS_MAIN="${EXPS_MAIN_SWEEP}" \
  SEEDS_MAIN="${SEEDS_MAIN}" \
  INSTANCES="${INSTANCES}" \
  BUDGETS="${BUDGETS}" \
  WEIGHT_PAIRS="${WEIGHT_PAIRS}" \
  KEEP_B_OUTPUTS=0 \
  CLEAN_NEW_B=1 \
  PURGE_LEGACY_B=0 \
  PACK_AFTER=0 \
  BC2CEC_CFG="${cfg}" \
  BC2CEC_RAW_CFG="${cfg_raw}" \
  BC2CEC_NOPROBE_CFG="${cfg_nop}" \
  bash scripts/launch_B_grid_parallel.sh \
    > "${out_dir}/launcher.log" 2>&1

  PACK_OUT_DIR="${out_dir}" \
  PACK_CLEAN_OUT_DIR=0 \
  PACK_MATCH_TAG="${vid}__" \
  PACK_EXPS="${EXPS_MAIN_SWEEP}" \
  PACK_TRACE_CSV=0 \
  PACK_SKIP_ANALYSIS=0 \
  PACK_FINAL_NAME="ALL_B_PACKS_${vid}.tgz" \
  bash scripts/pack_B_outputs.sh \
    >> "${out_dir}/launcher.log" 2>&1

  # Keep only the unique final tgz; remove compatibility symlink if created.
  if [[ -L "${out_dir}/ALL_B_PACKS.tgz" ]]; then
    rm -f "${out_dir}/ALL_B_PACKS.tgz"
  fi

  echo "[CF-STAGE] <<< Done ${vid} -> ${out_dir}/ALL_B_PACKS_${vid}.tgz"
}

# ------------------------------------------------------------
# Focused 4-variant sweep:
#   P0: current repaired baseline
#   P1: stronger late-stage CF
#   P2: stronger late-stage CF + milder shrinkage
#   P3: stronger late-stage CF + more visible CF/raw separation
# ------------------------------------------------------------
variants=$(
cat <<'EOV'
P0_curStage_l120	{early: 0.10, mid: 0.30, late: 0.45}	{early: 0.20, mid: 0.35, late: 0.50}	120
P1_late055_l120	{early: 0.05, mid: 0.25, late: 0.55}	{early: 0.15, mid: 0.35, late: 0.60}	120
P2_late055_l100	{early: 0.05, mid: 0.25, late: 0.55}	{early: 0.15, mid: 0.35, late: 0.60}	100
P3_late055_l080	{early: 0.05, mid: 0.25, late: 0.55}	{early: 0.15, mid: 0.35, late: 0.60}	80
EOV
)

while IFS=$'\t' read -r vid probe_stage_discount cec_stage_discount shrink_lambda_calls; do
  [[ -z "${vid}" ]] && continue
  write_variant_cfgs "${vid}" "${probe_stage_discount}" "${cec_stage_discount}" "${shrink_lambda_calls}"
done <<< "${variants}"

while IFS=$'\t' read -r vid probe_stage_discount cec_stage_discount shrink_lambda_calls; do
  [[ -z "${vid}" ]] && continue
  run_and_pack_variant "${vid}"
done <<< "${variants}"

echo "[CF-STAGE] done."
echo "[CF-STAGE] sweep plan -> ${PLAN}"
echo "[CF-STAGE] result root -> ${SWEEP_ROOT}"
