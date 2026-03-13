#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# ------------------------------------------------------------
# Micro sweep for current repaired B-line:
#   - Keep EVERYTHING identical to the current main experiment protocol
#   - Only sweep a few CF-related hyperparameters
#   - Save each variant's packed results into a NEW folder tree
#   - Each variant gets its own uniquely named ALL_B_PACKS_*.tgz
# ------------------------------------------------------------

MAX_JOBS="${MAX_JOBS:-36}"
INSTANCES="${INSTANCES:-cluster4 chain_skip chain_skip_randw}"
BUDGETS="${BUDGETS:-20k 40k 80k 160k 240k}"
WEIGHT_PAIRS="${WEIGHT_PAIRS:-0.3,0.7}"
SEEDS_MAIN="${SEEDS_MAIN:-0 1 2}"
EXPS_MAIN_SWEEP="${EXPS_MAIN_SWEEP:-EXP-B2-std-budgetaware EXP-B2-bc2cec EXP-B2-bc2cec-probe-raw EXP-B2-bc2cec-noprobe}"

# New result root (do not store final kept results under old location)
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SWEEP_ROOT="${SWEEP_ROOT:-_pack_B_cfretune_micro_${STAMP}}"
CFG_DIR="${CFG_DIR:-configs/layout_agent/_k4_sweeps/_cfretune_micro}"

mkdir -p "${SWEEP_ROOT}"
mkdir -p "${CFG_DIR}"

PLAN="${SWEEP_ROOT}/sweep_plan.tsv"
echo -e "id\tcfg_bc2cec\tcfg_raw\tcfg_noprobe\ttau\tgain_scale\tmin_real_calls" > "${PLAN}"

echo "[CF-RETUNE] ROOT=${ROOT}"
echo "[CF-RETUNE] SWEEP_ROOT=${SWEEP_ROOT}"
echo "[CF-RETUNE] CFG_DIR=${CFG_DIR}"
echo "[CF-RETUNE] MAX_JOBS=${MAX_JOBS}"
echo "[CF-RETUNE] INSTANCES=${INSTANCES}"
echo "[CF-RETUNE] BUDGETS=${BUDGETS}"
echo "[CF-RETUNE] WEIGHT_PAIRS=${WEIGHT_PAIRS}"
echo "[CF-RETUNE] SEEDS_MAIN=${SEEDS_MAIN}"
echo "[CF-RETUNE] EXPS_MAIN_SWEEP=${EXPS_MAIN_SWEEP}"

write_variant_cfgs() {
  local vid="$1"
  local tau="$2"
  local gain_scale="$3"
  local min_real_calls="$4"

  local cfg="${CFG_DIR}/${vid}.yaml"
  local cfg_raw="${CFG_DIR}/${vid}_raw.yaml"
  local cfg_nop="${CFG_DIR}/${vid}_noprobe.yaml"

  cat > "${cfg}" <<YAML
_base_: ../S015_A1_lowtax_B2_cfboost_signedaos_K4_bc2cec.yaml

# Variant ${vid}
# Keep all experiment protocol and algorithm structure identical to the current repaired line.
# Only sweep:
#   - detailed_place.mpvs.macros.probe.cf_reliability_tau
#   - detailed_place.mpvs.controller.cec.aos.probe_feed.gain_scale
#   - detailed_place.mpvs.controller.cec.counterfactual_min_real_calls
detailed_place:
  mpvs:
    macros:
      probe:
        cf_reliability_tau: ${tau}
    controller:
      cec:
        counterfactual_min_real_calls: ${min_real_calls}
        aos:
          probe_feed:
            gain_scale: ${gain_scale}
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
YAML

  cat > "${cfg_nop}" <<YAML
_base_: $(basename "${cfg}")

detailed_place:
  mpvs:
    macros:
      probe:
        enabled: false
YAML

  echo -e "${vid}\t${cfg}\t${cfg_raw}\t${cfg_nop}\t${tau}\t${gain_scale}\t${min_real_calls}" >> "${PLAN}"
}

run_and_pack_variant() {
  local vid="$1"
  local tau="$2"
  local gain_scale="$3"
  local min_real_calls="$4"

  local cfg="${CFG_DIR}/${vid}.yaml"
  local cfg_raw="${CFG_DIR}/${vid}_raw.yaml"
  local cfg_nop="${CFG_DIR}/${vid}_noprobe.yaml"

  local out_dir="${SWEEP_ROOT}/${vid}"
  mkdir -p "${out_dir}"
  mkdir -p "${out_dir}/configs"

  cp -f "${cfg}" "${out_dir}/configs/"
  cp -f "${cfg_raw}" "${out_dir}/configs/"
  cp -f "${cfg_nop}" "${out_dir}/configs/"

  cat > "${out_dir}/variant_meta.txt" <<EOF
variant_id=${vid}
cf_reliability_tau=${tau}
aos_probe_gain_scale=${gain_scale}
counterfactual_min_real_calls=${min_real_calls}
INSTANCES=${INSTANCES}
BUDGETS=${BUDGETS}
WEIGHT_PAIRS=${WEIGHT_PAIRS}
SEEDS_MAIN=${SEEDS_MAIN}
EXPS_MAIN_SWEEP=${EXPS_MAIN_SWEEP}
EOF

  echo "[CF-RETUNE] >>> Running ${vid}"

  # Use outputs/B only as temporary scratch during this variant run.
  # Final kept results are packed into ${out_dir}.
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

  # Keep only the uniquely named final tgz; remove compatibility symlink if created.
  if [[ -L "${out_dir}/ALL_B_PACKS.tgz" ]]; then
    rm -f "${out_dir}/ALL_B_PACKS.tgz"
  fi

  echo "[CF-RETUNE] <<< Done ${vid} -> ${out_dir}/ALL_B_PACKS_${vid}.tgz"
}

# ------------------------------------------------------------
# 6 focused variants (same structure, only 3 hyperparameters change)
#   A0: current baseline
#   A1/A2: strengthen probe-CF reliability
#   B1/B2: strengthen CF + probe->AOS transmission
#   C1: strengthen CF + transmission + slightly earlier controller CEC activation
# ------------------------------------------------------------
variants=$(
cat <<'EOV'
A0_tau20_g020_m200	20	0.20	200
A1_tau16_g020_m200	16	0.20	200
A2_tau12_g020_m200	12	0.20	200
B1_tau16_g025_m200	16	0.25	200
B2_tau12_g025_m200	12	0.25	200
C1_tau16_g025_m160	16	0.25	160
EOV
)

while IFS=$'\t' read -r vid tau gain_scale min_real_calls; do
  [[ -z "${vid}" ]] && continue
  write_variant_cfgs "${vid}" "${tau}" "${gain_scale}" "${min_real_calls}"
done <<< "${variants}"

while IFS=$'\t' read -r vid tau gain_scale min_real_calls; do
  [[ -z "${vid}" ]] && continue
  run_and_pack_variant "${vid}" "${tau}" "${gain_scale}" "${min_real_calls}"
done <<< "${variants}"

echo "[CF-RETUNE] done."
echo "[CF-RETUNE] sweep plan -> ${PLAN}"
echo "[CF-RETUNE] result root -> ${SWEEP_ROOT}"

