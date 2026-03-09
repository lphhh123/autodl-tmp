#!/usr/bin/env bash
set -euo pipefail

# Hyperparameter sweep for probe+counterfactual (BC2CEC cfg override).
# Serial across variants; parallel within each variant (xargs -P MAX_JOBS).
#
# Output:
#   _pack_B_sweep/
#     00_REF/ALL_B_PACKS.tgz   (optional reference pack)
#     01_HP00_xxx/ALL_B_PACKS.tgz
#     ...
#     sweep_plan.tsv
#
# Run:
#   MAX_JOBS=36 RUN_REFERENCE=1 RUN_ABLATIONS_PER_HP=1 bash scripts/launch_B_hp_sweep_probeCF.sh

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MAX_JOBS="${MAX_JOBS:-36}"
INSTANCES="${INSTANCES:-cluster4 chain_skip chain_skip_randw}"
BUDGETS="${BUDGETS:-20k 40k 80k 160k}"
WEIGHT_PAIRS="${WEIGHT_PAIRS:-0.3,0.7}"
SEEDS_MAIN="${SEEDS_MAIN:-0 1 2}"

RUN_REFERENCE="${RUN_REFERENCE:-1}"
RUN_ABLATIONS_PER_HP="${RUN_ABLATIONS_PER_HP:-1}"

if [[ "${RUN_ABLATIONS_PER_HP}" == "1" ]]; then
  EXPS_MAIN_SWEEP="${EXPS_MAIN_SWEEP:-EXP-B2-bc2cec EXP-B2-bc2cec-probe-raw EXP-B2-bc2cec-noprobe}"
else
  EXPS_MAIN_SWEEP="${EXPS_MAIN_SWEEP:-EXP-B2-bc2cec}"
fi

SWEEP_DIR="${SWEEP_DIR:-_pack_B_sweep}"
CFG_DIR="${CFG_DIR:-configs/layout_agent/_sweeps}"
mkdir -p "${SWEEP_DIR}"
mkdir -p "${CFG_DIR}"

PLAN="${SWEEP_DIR}/sweep_plan.tsv"
echo -e "id\tname\tcfg\ttriggers\tstage_budget\tatomic_budget\tperop_budget\tatomic_topk\tcf_d\tcf_stage\tcap_mult\tmin_atomic\talpha\tfail_pen" > "${PLAN}"

run_pack_one() {
  local out_dir="$1"
  local match_tag="$2"
  local pack_exps="$3"
  local skip_analysis="$4"
  PACK_OUT_DIR="${out_dir}" PACK_CLEAN_OUT_DIR=1 PACK_MATCH_TAG="${match_tag}" \
  PACK_EXPS="${pack_exps}" PACK_TRACE_CSV=0 PACK_SKIP_ANALYSIS="${skip_analysis}" \
  bash scripts/pack_B_outputs.sh
}

if [[ "${RUN_REFERENCE}" == "1" ]]; then
  echo "[SWEEP] Running reference once (REF)..."
  RUN_TAG_PREFIX="REF" \
  MAX_JOBS="${MAX_JOBS}" \
  RUN_HEADROOM=1 \
  EXPS_MAIN="EXP-B2-mpvs-only EXP-B2-std-budgetaware" \
  SEEDS_MAIN="${SEEDS_MAIN}" \
  EXPS_HEADROOM="EXP-B2-naive-atomiconly EXP-B2-naive-relinkonly EXP-B2-naive-shakeonly EXP-B2-naive-tabuonly" \
  SEEDS_HEADROOM="${SEEDS_MAIN}" \
  INSTANCES="${INSTANCES}" \
  BUDGETS="${BUDGETS}" \
  BUDGETS_HEADROOM="${BUDGETS}" \
  WEIGHT_PAIRS="${WEIGHT_PAIRS}" \
  WEIGHT_PAIRS_HEADROOM="${WEIGHT_PAIRS}" \
  PACK_AFTER=0 \
  bash scripts/launch_B_grid_parallel.sh

  run_pack_one "${SWEEP_DIR}/00_REF" "REF__" "EXP-B2-mpvs-only EXP-B2-std-budgetaware EXP-B2-naive-atomiconly EXP-B2-naive-relinkonly EXP-B2-naive-shakeonly EXP-B2-naive-tabuonly" 0
fi

variants=$(
cat <<'EOV'
HP00	A2_d25_cap15_m180	[0.25,0.70]	900	300	200	8	0.25	{early: 0.15, mid: 0.25, late: 0.40}	1.5	180	0.20	0.25
HP01	A2_d35_cap15_m180	[0.25,0.70]	900	300	200	8	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	180	0.20	0.25
HP02	A2_d45_cap15_m180	[0.25,0.70]	900	300	200	8	0.45	{early: 0.25, mid: 0.45, late: 0.60}	1.5	180	0.20	0.25
HP03	A2_d35_cap12_m180	[0.25,0.70]	900	300	200	8	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.2	180	0.20	0.25
HP04	A2_d35_cap20_m180	[0.25,0.70]	900	300	200	8	0.35	{early: 0.20, mid: 0.35, late: 0.50}	2.0	180	0.20	0.25
HP05	A2_d35_cap15_m120	[0.25,0.70]	900	300	200	8	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	120	0.20	0.25
HP06	A2_d35_cap15_m240	[0.25,0.70]	900	300	200	8	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	240	0.20	0.25
HP07	A2_d35_topk10	[0.25,0.70]	900	300	200	10	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	180	0.20	0.25
HP08	A2_d35_alpha30	[0.25,0.70]	900	300	200	8	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	180	0.30	0.25
HP09	A2_d35_fail15	[0.25,0.70]	900	300	200	8	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	180	0.20	0.15

HP10	B3_d25_cap15_m180	[0.20,0.60,0.85]	1200	350	250	10	0.25	{early: 0.15, mid: 0.25, late: 0.40}	1.5	180	0.20	0.25
HP11	B3_d35_cap15_m180	[0.20,0.60,0.85]	1200	350	250	10	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	180	0.20	0.25
HP12	B3_d45_cap15_m180	[0.20,0.60,0.85]	1200	350	250	10	0.45	{early: 0.25, mid: 0.45, late: 0.60}	1.5	180	0.20	0.25
HP13	B3_d35_cap12_m180	[0.20,0.60,0.85]	1200	350	250	10	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.2	180	0.20	0.25
HP14	B3_d35_cap20_m180	[0.20,0.60,0.85]	1200	350	250	10	0.35	{early: 0.20, mid: 0.35, late: 0.50}	2.0	180	0.20	0.25
HP15	B3_d35_m120	[0.20,0.60,0.85]	1200	350	250	10	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	120	0.20	0.25
HP16	B3_d35_m240	[0.20,0.60,0.85]	1200	350	250	10	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	240	0.20	0.25
HP17	B3_atomic450	[0.20,0.60,0.85]	1200	450	250	10	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	240	0.20	0.25
HP18	B3_topk12	[0.20,0.60,0.85]	1200	350	250	12	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	180	0.20	0.25
HP19	B3_fail35	[0.20,0.60,0.85]	1200	350	250	10	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	180	0.20	0.35

HP20	C3_d25_cap15_m240	[0.20,0.60,0.85]	1500	450	300	12	0.25	{early: 0.15, mid: 0.25, late: 0.40}	1.5	240	0.20	0.25
HP21	C3_d35_cap15_m240	[0.20,0.60,0.85]	1500	450	300	12	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	240	0.20	0.25
HP22	C3_d45_cap15_m240	[0.20,0.60,0.85]	1500	450	300	12	0.45	{early: 0.25, mid: 0.45, late: 0.60}	1.5	240	0.20	0.25
HP23	C3_d35_cap12_m240	[0.20,0.60,0.85]	1500	450	300	12	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.2	240	0.20	0.25
HP24	C3_d35_cap20_m240	[0.20,0.60,0.85]	1500	450	300	12	0.35	{early: 0.20, mid: 0.35, late: 0.50}	2.0	240	0.20	0.25
HP25	C3_atomic550	[0.20,0.60,0.85]	1500	550	250	12	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	240	0.20	0.25
HP26	C3_alpha15	[0.20,0.60,0.85]	1500	450	300	12	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	240	0.15	0.25
HP27	C3_alpha30	[0.20,0.60,0.85]	1500	450	300	12	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	240	0.30	0.25
HP28	C3_fail15	[0.20,0.60,0.85]	1500	450	300	12	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	240	0.20	0.15
HP29	C3_fail35	[0.20,0.60,0.85]	1500	450	300	12	0.35	{early: 0.20, mid: 0.35, late: 0.50}	1.5	240	0.20	0.35
EOV
)

echo "[SWEEP] MAX_JOBS=${MAX_JOBS}  INSTANCES=${INSTANCES}  BUDGETS=${BUDGETS}  SEEDS=${SEEDS_MAIN}"
echo "[SWEEP] Writing configs to ${CFG_DIR} and packing to ${SWEEP_DIR}"

while IFS=$'\t' read -r id name triggers stage_budget atomic_budget perop_budget atomic_topk cf_d cf_stage cap_mult min_atomic alpha fail_pen; do
  [[ -z "${id}" || "${id}" == \#* ]] && continue
  cfg="${CFG_DIR}/bc2cec_${id}.yaml"
  cfg_raw="${CFG_DIR}/bc2cec_${id}_raw.yaml"
  cfg_nop="${CFG_DIR}/bc2cec_${id}_noprobe.yaml"

  cat > "${cfg}" <<YAML
_base_: ../layout_L4_region_pareto_llm_mpvs_bc2cec_nollm_exp.yaml
detailed_place:
  mpvs:
    macros:
      probe:
        enabled: true
        trigger_fracs: ${triggers}
        stage_call_budget: ${stage_budget}
        atomic_call_budget: ${atomic_budget}
        per_op_call_budget: ${perop_budget}
        atomic_eval_topk: ${atomic_topk}
        cf_discount: ${cf_d}
        cf_discount_by_stage: ${cf_stage}
        cf_cap_mult: ${cap_mult}
        min_atomic_calls_for_cf: ${min_atomic}
        ewma_alpha: ${alpha}
        update_weight: true
        weight_metric: cf
        weight_use_ewma: true
        fail_penalty_scale: ${fail_pen}
YAML

  cat > "${cfg_raw}" <<YAML
_base_: bc2cec_${id}.yaml
detailed_place:
  mpvs:
    macros:
      probe:
        cf_discount: 0.0
        cf_discount_by_stage: {}
        weight_metric: raw
        update_weight: true
        weight_use_ewma: true
YAML

  cat > "${cfg_nop}" <<YAML
_base_: bc2cec_${id}.yaml
detailed_place:
  mpvs:
    macros:
      probe:
        enabled: false
YAML

  echo -e "${id}\t${name}\t${cfg}\t${triggers}\t${stage_budget}\t${atomic_budget}\t${perop_budget}\t${atomic_topk}\t${cf_d}\t${cf_stage}\t${cap_mult}\t${min_atomic}\t${alpha}\t${fail_pen}" >> "${PLAN}"

  echo "[SWEEP] >>> ${id}  ${name}"
  RUN_TAG_PREFIX="${id}" \
  BC2CEC_CFG="${cfg}" \
  BC2CEC_RAW_CFG="${cfg_raw}" \
  BC2CEC_NOPROBE_CFG="${cfg_nop}" \
  MAX_JOBS="${MAX_JOBS}" \
  RUN_HEADROOM=0 \
  EXPS_MAIN="${EXPS_MAIN_SWEEP}" \
  SEEDS_MAIN="${SEEDS_MAIN}" \
  INSTANCES="${INSTANCES}" \
  BUDGETS="${BUDGETS}" \
  WEIGHT_PAIRS="${WEIGHT_PAIRS}" \
  PACK_AFTER=0 \
  bash scripts/launch_B_grid_parallel.sh

  out_dir="${SWEEP_DIR}/01_${id}_${name}"
  run_pack_one "${out_dir}" "${id}__" "EXP-B2-bc2cec EXP-B2-bc2cec-probe-raw EXP-B2-bc2cec-noprobe" 1
done <<< "${variants}"

echo "[SWEEP] done. Packs -> ${SWEEP_DIR}"
echo "[SWEEP] plan -> ${PLAN}"
