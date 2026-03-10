#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MAX_JOBS="${MAX_JOBS:-36}"
SEEDS_MAIN="${SEEDS_MAIN:-0 1 2}"
WEIGHT_PAIRS="${WEIGHT_PAIRS:-0.3,0.7}"

# Two-stage convenience:
#   FAST=1 -> small grid for screening
#   FAST=0 -> full grid
FAST="${FAST:-1}"
if [[ "${FAST}" == "1" ]]; then
  INSTANCES="${INSTANCES:-cluster4 chain_skip_randw}"
  BUDGETS="${BUDGETS:-20k 160k}"
else
  INSTANCES="${INSTANCES:-cluster4 chain_skip chain_skip_randw}"
  BUDGETS="${BUDGETS:-20k 40k 80k 160k}"
fi

SWEEP_DIR="${SWEEP_DIR:-_pack_B_k4_sweep}"
CFG_DIR="${CFG_DIR:-configs/layout_agent/_k4_sweeps}"
mkdir -p "${SWEEP_DIR}" "${CFG_DIR}"

PLAN="${SWEEP_DIR}/k4_sweep_plan.tsv"
echo -e "combo_id\tA\tB\tC\ttrigger_fracs\tcfg_bc2cec" > "${PLAN}"

# Fixed schedule (DO NOT CHANGE)
TRIG_FRACS='[0.30, 0.55, 0.75, 0.90]'

run_pack_one() {
  local out_dir="$1"
  local match_tag="$2"
  local final_name="$3"
  PACK_OUT_DIR="${out_dir}" PACK_CLEAN_OUT_DIR=1 PACK_MATCH_TAG="${match_tag}" \
  PACK_EXPS="EXP-B2-std-budgetaware EXP-B2-bc2cec EXP-B2-bc2cec-probe-raw EXP-B2-bc2cec-noprobe" \
  PACK_TRACE_CSV=0 PACK_SKIP_ANALYSIS=1 PACK_FINAL_NAME="${final_name}" \
  bash scripts/pack_B_outputs.sh
}

emit_A() {
  local aid="$1"
  case "$aid" in
    A0_base)
      cat <<'YAML'
stage_call_budget_frac: 0.015
max_probe_calls: 5000
YAML
      ;;
    A1_lowtax)
      cat <<'YAML'
stage_call_budget_frac: 0.012
max_probe_calls: 4000
YAML
      ;;
    A2_lowtax2)
      cat <<'YAML'
stage_call_budget_frac: 0.010
max_probe_calls: 3000
YAML
      ;;
    *)
      echo "[K4] unknown A id: $aid" >&2
      exit 2
      ;;
  esac
}

emit_B() {
  local bid="$1"
  case "$bid" in
    B0_hard_on)
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
    B1_soft_start)
      cat <<'YAML'
cf_discount: 0.35
cf_discount_by_stage: {early: 0.20, mid: 0.35, late: 0.50}
cf_cap_mult: 1.5
min_atomic_calls_for_cf: 0
cf_reliability_tau: 50.0
cf_one_sided_update: false
success_boost_scale: 0.50
fail_penalty_scale: 0.25
YAML
      ;;
    B2_safe_soft)
      cat <<'YAML'
cf_discount: 0.35
cf_discount_by_stage: {early: 0.10, mid: 0.30, late: 0.45}
cf_cap_mult: 1.2
min_atomic_calls_for_cf: 0
cf_reliability_tau: 50.0
cf_one_sided_update: true
cf_one_sided_penalty_mode: raw
cf_one_sided_penalty_scale: 1.0
success_boost_scale: 0.60
fail_penalty_scale: 0.20
YAML
      ;;
    *)
      echo "[K4] unknown B id: $bid" >&2
      exit 2
      ;;
  esac
}

emit_C() {
  local cid="$1"
  case "$cid" in
    C0_default)
      cat <<'YAML'
adapt:
  enabled: true
  ewma_alpha: 0.2
  fail_cooldown: 20
  success_cooldown: 0
  weight_floor: 0.1
  weight_cap: 5.0
  success_boost: 0.2
  fail_penalty: 0.2
YAML
      ;;
    C1_gentle)
      cat <<'YAML'
adapt:
  enabled: true
  ewma_alpha: 0.2
  fail_cooldown: 5
  success_cooldown: 0
  weight_floor: 0.1
  weight_cap: 5.0
  success_boost: 0.2
  fail_penalty: 0.1
YAML
      ;;
    C2_off)
      cat <<'YAML'
adapt:
  enabled: false
YAML
      ;;
    *)
      echo "[K4] unknown C id: $cid" >&2
      exit 2
      ;;
  esac
}

A_LIST=(A0_base A1_lowtax A2_lowtax2)
B_LIST=(B0_hard_on B1_soft_start B2_safe_soft)
C_LIST=(C0_default C1_gentle C2_off)

combo_idx=0
for A in "${A_LIST[@]}"; do
  for B in "${B_LIST[@]}"; do
    for C in "${C_LIST[@]}"; do
      combo_id=$(printf "S%03d_%s_%s_%s_K4" "${combo_idx}" "${A}" "${B}" "${C}")
      combo_idx=$((combo_idx + 1))

      cfg="${CFG_DIR}/${combo_id}_bc2cec.yaml"
      cfg_raw="${CFG_DIR}/${combo_id}_raw.yaml"
      cfg_nop="${CFG_DIR}/${combo_id}_noprobe.yaml"

      # bc2cec config (fixed trigger_fracs)
      cat > "${cfg}" <<YAML
_base_: ../layout_L4_region_pareto_llm_mpvs_bc2cec_nollm_exp.yaml
detailed_place:
  mpvs:
    macros:
$(emit_C "${C}" | sed 's/^/      /')
      probe:
        enabled: true
        trigger_fracs: ${TRIG_FRACS}
$(emit_A "${A}" | sed 's/^/        /')
        min_remaining_calls: 1500
$(emit_B "${B}" | sed 's/^/        /')
YAML

      # probe-raw ablation
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

      # noprobe ablation
      cat > "${cfg_nop}" <<YAML
_base_: ${combo_id}_bc2cec.yaml
detailed_place:
  mpvs:
    macros:
      probe:
        enabled: false
YAML

      echo -e "${combo_id}\t${A}\t${B}\t${C}\t${TRIG_FRACS}\t${cfg}" >> "${PLAN}"

      echo "[K4] >>> ${combo_id}"
      RUN_TAG_PREFIX="${combo_id}" \
      BC2CEC_CFG="${cfg}" \
      BC2CEC_RAW_CFG="${cfg_raw}" \
      BC2CEC_NOPROBE_CFG="${cfg_nop}" \
      MAX_JOBS="${MAX_JOBS}" \
      RUN_HEADROOM=0 \
      EXPS_MAIN="EXP-B2-std-budgetaware EXP-B2-bc2cec EXP-B2-bc2cec-probe-raw EXP-B2-bc2cec-noprobe" \
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
  done
done

echo "[K4] done. Packs -> ${SWEEP_DIR}"
echo "[K4] plan -> ${PLAN}"
