#!/usr/bin/env bash
set -euo pipefail

# Combo sweep:
#   Axis-1: CF optimization scheme (4)
#   Axis-2: trigger_fracs sets (3/4/5 probes; multiple layouts per count)
#
# Total combos: 4 * 9 = 36 (<=80).
# Each combo runs: bc2cec + probe-raw + noprobe.
# Combos run serially; inside combo, launch_B_grid_parallel uses MAX_JOBS for parallelism.
# Each combo is packed once with a UNIQUE filename.

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MAX_JOBS="${MAX_JOBS:-36}"
INSTANCES="${INSTANCES:-cluster4 chain_skip chain_skip_randw}"
BUDGETS="${BUDGETS:-20k 40k 80k 160k}"
WEIGHT_PAIRS="${WEIGHT_PAIRS:-0.3,0.7}"
SEEDS_MAIN="${SEEDS_MAIN:-0 1 2}"

SWEEP_DIR="${SWEEP_DIR:-_pack_B_combo_sweep}"
CFG_DIR="${CFG_DIR:-configs/layout_agent/_combo_sweeps}"
mkdir -p "${SWEEP_DIR}" "${CFG_DIR}"

PLAN="${SWEEP_DIR}/combo_plan.tsv"
echo -e "combo_id\tscheme\ttrigger_id\ttrigger_fracs\tcfg_bc2cec" > "${PLAN}"

run_pack_one() {
  local out_dir="$1"
  local match_tag="$2"
  local final_name="$3"
  PACK_OUT_DIR="${out_dir}" PACK_CLEAN_OUT_DIR=1 PACK_MATCH_TAG="${match_tag}" \
  PACK_EXPS="EXP-B2-bc2cec EXP-B2-bc2cec-probe-raw EXP-B2-bc2cec-noprobe" \
  PACK_TRACE_CSV=0 PACK_SKIP_ANALYSIS=1 PACK_FINAL_NAME="${final_name}" \
  bash scripts/pack_B_outputs.sh
}

triggers=$(
cat <<'TRIG'
T3a	[0.30, 0.65, 0.85]
T3b	[0.25, 0.60, 0.88]
T3c	[0.35, 0.70, 0.90]
T4a	[0.25, 0.50, 0.75, 0.90]
T4b	[0.30, 0.55, 0.75, 0.90]
T4c	[0.20, 0.45, 0.70, 0.88]
T5a	[0.20, 0.40, 0.60, 0.80, 0.90]
T5b	[0.25, 0.45, 0.65, 0.80, 0.92]
T5c	[0.30, 0.50, 0.65, 0.80, 0.92]
TRIG
)

emit_scheme_body() {
  local sid="$1"
  case "$sid" in
    S0_base)
      cat <<'YAML'
cf_discount: 0.35
cf_discount_by_stage: {early: 0.20, mid: 0.35, late: 0.50}
cf_cap_mult: 1.5
min_atomic_calls_for_cf: 10
cf_reliability_tau: 0.0
cf_one_sided_update: false
success_boost_scale: 0.50
fail_penalty_scale: 0.25
YAML
      ;;
    S1_soft)
      cat <<'YAML'
cf_discount: 0.35
cf_discount_by_stage: {early: 0.20, mid: 0.35, late: 0.50}
cf_cap_mult: 1.5
min_atomic_calls_for_cf: 0
cf_reliability_tau: 120.0
cf_one_sided_update: false
success_boost_scale: 0.50
fail_penalty_scale: 0.25
YAML
      ;;
    S2_oneside)
      cat <<'YAML'
cf_discount: 0.35
cf_discount_by_stage: {early: 0.20, mid: 0.35, late: 0.50}
cf_cap_mult: 1.5
min_atomic_calls_for_cf: 0
cf_reliability_tau: 120.0
cf_one_sided_update: true
cf_one_sided_penalty_mode: raw
cf_one_sided_penalty_scale: 1.0
success_boost_scale: 0.60
fail_penalty_scale: 0.25
YAML
      ;;
    S3_aggr_late)
      cat <<'YAML'
cf_discount: 0.35
cf_discount_by_stage: {early: 0.10, mid: 0.30, late: 0.65}
cf_cap_mult: 1.5
min_atomic_calls_for_cf: 0
cf_reliability_tau: 80.0
cf_one_sided_update: true
cf_one_sided_penalty_mode: raw
cf_one_sided_penalty_scale: 0.7
success_boost_scale: 0.70
fail_penalty_scale: 0.20
YAML
      ;;
    *)
      echo "[COMBO] unknown scheme id: $sid" >&2
      exit 2
      ;;
  esac
}

SCHEMES=(S0_base S1_soft S2_oneside S3_aggr_late)

combo_idx=0
while IFS=$'\t' read -r trig_id trig_list; do
  [[ -z "${trig_id}" ]] && continue

  for scheme_id in "${SCHEMES[@]}"; do

    combo_id=$(printf "C%03d_%s_%s" "${combo_idx}" "${scheme_id}" "${trig_id}")
    combo_idx=$((combo_idx+1))

    cfg="${CFG_DIR}/${combo_id}_bc2cec.yaml"
    cfg_raw="${CFG_DIR}/${combo_id}_raw.yaml"
    cfg_nop="${CFG_DIR}/${combo_id}_noprobe.yaml"

    cat > "${cfg}" <<YAML
_base_: ../layout_L4_region_pareto_llm_mpvs_bc2cec_nollm_exp.yaml
detailed_place:
  mpvs:
    macros:
      probe:
        enabled: true
        trigger_fracs: ${trig_list}
$(emit_scheme_body "${scheme_id}" | sed 's/^/        /')
YAML

    cat > "${cfg_raw}" <<YAML
_base_: ${combo_id}_bc2cec.yaml
detailed_place:
  mpvs:
    macros:
      probe:
        cf_discount: 0.0
        cf_discount_by_stage: {}
        cf_reliability_tau: 0.0
        cf_one_sided_update: false
        weight_metric: raw
YAML

    cat > "${cfg_nop}" <<YAML
_base_: ${combo_id}_bc2cec.yaml
detailed_place:
  mpvs:
    macros:
      probe:
        enabled: false
YAML

    echo -e "${combo_id}\t${scheme_id}\t${trig_id}\t${trig_list}\t${cfg}" >> "${PLAN}"

    echo "[COMBO] >>> ${combo_id}  scheme=${scheme_id}  triggers=${trig_id} ${trig_list}"
    RUN_TAG_PREFIX="${combo_id}" \
    BC2CEC_CFG="${cfg}" \
    BC2CEC_RAW_CFG="${cfg_raw}" \
    BC2CEC_NOPROBE_CFG="${cfg_nop}" \
    MAX_JOBS="${MAX_JOBS}" \
    RUN_HEADROOM=0 \
    EXPS_MAIN="EXP-B2-bc2cec EXP-B2-bc2cec-probe-raw EXP-B2-bc2cec-noprobe" \
    SEEDS_MAIN="${SEEDS_MAIN}" \
    INSTANCES="${INSTANCES}" \
    BUDGETS="${BUDGETS}" \
    WEIGHT_PAIRS="${WEIGHT_PAIRS}" \
    PACK_AFTER=0 \
    bash scripts/launch_B_grid_parallel.sh

    out_dir="${SWEEP_DIR}/${combo_id}"
    final_name="${combo_id}__PACK.tgz"
    run_pack_one "${out_dir}" "${combo_id}__" "${final_name}"
  done
done <<< "${triggers}"

echo "[COMBO] done. Packs -> ${SWEEP_DIR}"
echo "[COMBO] plan -> ${PLAN}"
