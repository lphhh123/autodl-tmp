#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ------------------------------------------------------------
# Overnight wide micro-sweep around the current best line (P1):
#   Keep the repaired B-line structure unchanged.
#   Only tune a few high-value knobs:
#     1) probe cf_discount_by_stage
#     2) controller cec counterfactual_discount_by_stage
#     3) probe cf_cap_mult
#     4) controller cec counterfactual_cap_mult_by_stage (late emphasis)
#     5) probe shrinkage.lambda_calls
#     6) a small increase in probe evidence:
#          - stage_call_budget_frac
#          - atomic_eval_topk
#
# Protocol stays IDENTICAL to current main experiment:
#   - SEEDS_MAIN="0 1 2"
#   - INSTANCES="cluster4 chain_skip chain_skip_randw"
#   - BUDGETS="20k 40k 80k 160k 240k"
#   - WEIGHT_PAIRS="0.3,0.7"
#   - EXPS_MAIN includes std / bc2cec / probe-raw / noprobe
#
# Each variant:
#   - runs with fresh outputs/B scratch
#   - packs into its own NEW result folder
#   - gets a uniquely named ALL_B_PACKS_<variant>.tgz
# ------------------------------------------------------------

MAX_JOBS="${MAX_JOBS:-36}"
INSTANCES="${INSTANCES:-cluster4 chain_skip chain_skip_randw}"
BUDGETS="${BUDGETS:-20k 40k 80k 160k 240k}"
WEIGHT_PAIRS="${WEIGHT_PAIRS:-0.3,0.7}"
SEEDS_MAIN="${SEEDS_MAIN:-0 1 2}"
EXPS_MAIN_SWEEP="${EXPS_MAIN_SWEEP:-EXP-B2-std-budgetaware EXP-B2-bc2cec EXP-B2-bc2cec-probe-raw EXP-B2-bc2cec-noprobe}"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${SWEEP_ROOT:-_pack_B_cfnight_wide_${STAMP}}"
CFG_DIR="${CFG_DIR:-configs/layout_agent/_k4_sweeps/_cfnight_wide}"

mkdir -p "${SWEEP_ROOT}"
mkdir -p "${CFG_DIR}"

PLAN="${SWEEP_ROOT}/sweep_plan.tsv"
echo -e "id\tcfg_bc2cec\tcfg_raw\tcfg_noprobe\tprobe_stage_discount\tcec_stage_discount\tprobe_cap\tcec_late_cap\tlambda_calls\tstage_frac\tatomic_topk" > "${PLAN}"

echo "[CF-NIGHT] ROOT=${ROOT}"
echo "[CF-NIGHT] SWEEP_ROOT=${SWEEP_ROOT}"
echo "[CF-NIGHT] CFG_DIR=${CFG_DIR}"
echo "[CF-NIGHT] MAX_JOBS=${MAX_JOBS}"
echo "[CF-NIGHT] INSTANCES=${INSTANCES}"
echo "[CF-NIGHT] BUDGETS=${BUDGETS}"
echo "[CF-NIGHT] WEIGHT_PAIRS=${WEIGHT_PAIRS}"
echo "[CF-NIGHT] SEEDS_MAIN=${SEEDS_MAIN}"
echo "[CF-NIGHT] EXPS_MAIN_SWEEP=${EXPS_MAIN_SWEEP}"

write_variant_cfgs() {
  local vid="$1"
  local probe_stage_discount="$2"
  local cec_stage_discount="$3"
  local probe_cap="$4"
  local cec_late_cap="$5"
  local lambda_calls="$6"
  local stage_frac="$7"
  local atomic_topk="$8"

  local cfg="${CFG_DIR}/${vid}.yaml"
  local cfg_raw="${CFG_DIR}/${vid}_raw.yaml"
  local cfg_nop="${CFG_DIR}/${vid}_noprobe.yaml"

  cat > "${cfg}" <<YAML
_base_: ../S015_A1_lowtax_B2_cfboost_signedaos_K4_bc2cec.yaml

# Variant ${vid}
# Keep the CURRENT repaired B-line structure unchanged.
# Only tune:
#   - detailed_place.mpvs.macros.probe.cf_discount_by_stage
#   - detailed_place.mpvs.controller.cec.counterfactual_discount_by_stage
#   - detailed_place.mpvs.macros.probe.cf_cap_mult
#   - detailed_place.mpvs.controller.cec.counterfactual_cap_mult_by_stage
#   - detailed_place.mpvs.macros.probe.shrinkage.lambda_calls
#   - detailed_place.mpvs.macros.probe.stage_call_budget_frac
#   - detailed_place.mpvs.macros.probe.atomic_eval_topk
detailed_place:
  mpvs:
    macros:
      probe:
        cf_discount_by_stage: ${probe_stage_discount}
        cf_cap_mult: ${probe_cap}
        stage_call_budget_frac: ${stage_frac}
        atomic_eval_topk: ${atomic_topk}
        shrinkage:
          enabled: true
          lambda_calls: ${lambda_calls}
    controller:
      cec:
        counterfactual_discount_by_stage: ${cec_stage_discount}
        counterfactual_cap_mult_by_stage: {early: 1.2, mid: 1.5, late: ${cec_late_cap}}
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

  echo -e "${vid}\t${cfg}\t${cfg_raw}\t${cfg_nop}\t${probe_stage_discount}\t${cec_stage_discount}\t${probe_cap}\t${cec_late_cap}\t${lambda_calls}\t${stage_frac}\t${atomic_topk}" >> "${PLAN}"
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

  echo "[CF-NIGHT] >>> Running ${vid}"

  # outputs/B is scratch only; final kept artifacts go into ${out_dir}
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

  if [[ -L "${out_dir}/ALL_B_PACKS.tgz" ]]; then
    rm -f "${out_dir}/ALL_B_PACKS.tgz"
  fi

  echo "[CF-NIGHT] <<< Done ${vid} -> ${out_dir}/ALL_B_PACKS_${vid}.tgz"
}

# ------------------------------------------------------------
# 10 overnight variants around the current P1 line
# ------------------------------------------------------------
variants=$(
cat <<'EOV'
Q0_P1_base                     {early: 0.05, mid: 0.25, late: 0.55}   {early: 0.15, mid: 0.35, late: 0.60}   1.20   1.80   120   0.012   10
Q1_cap13_late20               {early: 0.05, mid: 0.25, late: 0.55}   {early: 0.15, mid: 0.35, late: 0.60}   1.30   2.00   120   0.012   10
Q2_cap13_late22               {early: 0.05, mid: 0.25, late: 0.55}   {early: 0.15, mid: 0.35, late: 0.60}   1.30   2.20   120   0.012   10
Q3_cap13_late20_l110          {early: 0.05, mid: 0.25, late: 0.55}   {early: 0.15, mid: 0.35, late: 0.60}   1.30   2.00   110   0.012   10
Q4_cap13_late20_l095          {early: 0.05, mid: 0.25, late: 0.55}   {early: 0.15, mid: 0.35, late: 0.60}   1.30   2.00   95    0.012   10
Q5_late060_cap13_l110         {early: 0.05, mid: 0.25, late: 0.60}   {early: 0.15, mid: 0.35, late: 0.65}   1.30   2.00   110   0.012   10
Q6_evid013_top12              {early: 0.05, mid: 0.25, late: 0.55}   {early: 0.15, mid: 0.35, late: 0.60}   1.20   1.80   120   0.013   12
Q7_evid014_top12_cap13        {early: 0.05, mid: 0.25, late: 0.55}   {early: 0.15, mid: 0.35, late: 0.60}   1.30   2.00   120   0.014   12
Q8_evid013_top12_cap13_l110   {early: 0.05, mid: 0.25, late: 0.55}   {early: 0.15, mid: 0.35, late: 0.60}   1.30   2.00   110   0.013   12
Q9_late060_cap13_l095_e013    {early: 0.05, mid: 0.25, late: 0.60}   {early: 0.15, mid: 0.35, late: 0.65}   1.30   2.20   95    0.013   12
EOV
)

while read -r vid probe_stage_discount cec_stage_discount probe_cap cec_late_cap lambda_calls stage_frac atomic_topk; do
  [[ -z "${vid}" ]] && continue
  write_variant_cfgs "${vid}" "${probe_stage_discount}" "${cec_stage_discount}" "${probe_cap}" "${cec_late_cap}" "${lambda_calls}" "${stage_frac}" "${atomic_topk}"
done <<< "${variants}"

while read -r vid probe_stage_discount cec_stage_discount probe_cap cec_late_cap lambda_calls stage_frac atomic_topk; do
  [[ -z "${vid}" ]] && continue
  run_and_pack_variant "${vid}"
done <<< "${variants}"

echo "[CF-NIGHT] done."
echo "[CF-NIGHT] sweep plan -> ${PLAN}"
echo "[CF-NIGHT] result root -> ${SWEEP_ROOT}"
