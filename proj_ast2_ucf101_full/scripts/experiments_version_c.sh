#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SMOKE="${SMOKE:-0}"
EXP_ID="${1:-}"
SEED="${2:-0}"
INSTANCE="${INSTANCE:-base}"
# INSTANCE usage:
#   INSTANCE=cluster4 bash scripts/experiments_version_c.sh EXP-B2 0
#   INSTANCE=all      bash scripts/experiments_version_c.sh EXP-B2 0
INSTANCE_LIST_DEFAULT=("chain_skip" "chain_skip_randw" "cluster4")
BUDGET="${BUDGET:-10k}"
# Optional tag appended to output directories/log symlinks to avoid collisions in sweeps
RUN_TAG="${RUN_TAG:-}"

sanitize_tag() {
  local s="${1:-}"
  if [[ -z "$s" ]]; then
    echo ""
    return 0
  fi
  # keep filename-safe charset
  # Use C locale to avoid locale-dependent range parsing in tr.
  # Put '-' at the end so it is treated literally.
  LC_ALL=C echo "$s" | tr -c 'A-Za-z0-9_.+-' '_'
}

TAG_SUFFIX=""
RUN_TAG_SAFE="$(sanitize_tag "${RUN_TAG}")"
if [[ -n "${RUN_TAG_SAFE}" ]]; then
  TAG_SUFFIX="-${RUN_TAG_SAFE}"
fi

# ---- arg validation (avoid accidental extra tokens like "0andom") ----
if [[ $# -gt 2 ]]; then
  echo "[ERROR] Too many args. Usage: $0 <EXP_ID> [SEED]"
  echo "  Got: $*"
  exit 2
fi

if [[ -n "${SEED}" && ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] SEED must be an integer, got: ${SEED}"
  exit 2
fi

# Default slim logging (can be overridden per run)
export LOG_SLIM="${LOG_SLIM:-1}"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"

if [[ -z "$EXP_ID" ]]; then
  echo "Usage: $0 <EXP_ID> [SEED]"
  exit 1
fi

parse_budget_to_int() {
  local b="${1:-}"
  b="$(echo "$b" | tr '[:upper:]' '[:lower:]')"
  # allow raw integer
  if [[ "$b" =~ ^[0-9]+$ ]]; then
    echo "$b"
    return 0
  fi
  # allow 50k / 200k / 2m
  if [[ "$b" =~ ^[0-9]+k$ ]]; then
    echo $((10#${b%k} * 1000))
    return 0
  fi
  if [[ "$b" =~ ^[0-9]+m$ ]]; then
    echo $((10#${b%m} * 1000000))
    return 0
  fi
  echo ""
  return 1
}

TOTAL_EVAL_BUDGET="$(parse_budget_to_int "${BUDGET}")" || true
if [[ -z "${TOTAL_EVAL_BUDGET}" ]]; then
  echo "[ERROR] Unknown BUDGET=${BUDGET}. Examples: 10k, 50k, 100k, 150k, 200k, 50000"
  exit 2
fi

# ---- OUTPUT PREFIX --------------------------------------------------------
# SMOKE=1 -> outputs/SMOKE/... (so it won't overwrite formal runs)
OUT_PREFIX_BASE="outputs"
if [[ "${SMOKE}" == "1" ]]; then
  OUT_PREFIX_BASE="outputs/SMOKE"
  mkdir -p "${OUT_PREFIX_BASE}"
  echo "[SMOKE] enabled: output prefix base -> ${OUT_PREFIX_BASE}"
fi

# Route Innovation-B outputs into a dedicated folder to avoid mixing with historical runs.
#   B runs   -> outputs/B/...
#   B smoke  -> outputs/SMOKE/B/...
# Everything else stays under outputs/... (or outputs/SMOKE/... for smoke)
OUT_PREFIX="${OUT_PREFIX_BASE}"
if [[ "${EXP_ID}" == EXP-B* ]]; then
  OUT_PREFIX="${OUT_PREFIX_BASE}/B"
fi
mkdir -p "${OUT_PREFIX}"

# ---- AUTO RESUME / FRESH RUN ---------------------------------------------
# By default we DO NOT auto-resume.
# If you want crash recovery, opt-in per run:
#   AUTO_RESUME=1 bash scripts/experiments_version_c.sh EXP-A1 0
export AUTO_RESUME="${AUTO_RESUME:-0}"

# When not auto-resuming, default to fresh run semantics:
# move any existing output dir aside so the rerun never mixes with old ckpts.
# Disable if you really want to overwrite in-place:
#   FRESH_RUN=0 ...
export FRESH_RUN="${FRESH_RUN:-1}"

# For SMOKE runs, always disable auto-resume (avoid looping on deterministic errors).
if [[ "${SMOKE}" == "1" ]]; then
  export AUTO_RESUME="0"
fi

# Centralized stdout directory (symlinks) for easier monitoring.
# Defaults to the same LOG_DIR used by launch_A_noabl_* scripts when present.
STDOUT_AGG_DIR="${STDOUT_AGG_DIR:-${LOG_DIR:-$HOME/runlogs_A}/stdout}"
mkdir -p "${STDOUT_AGG_DIR}"

# For B runs, if user didn't override STDOUT_AGG_DIR, keep stdout symlinks inside outputs/B/stdout_agg
if [[ "${EXP_ID}" == EXP-B* ]]; then
  _default_stdout_dir="${LOG_DIR:-$HOME/runlogs_A}/stdout"
  if [[ "${STDOUT_AGG_DIR}" == "${_default_stdout_dir}" ]]; then
    STDOUT_AGG_DIR="${OUT_PREFIX}/stdout_agg"
    mkdir -p "${STDOUT_AGG_DIR}"
  fi
fi

# ---- TF32 (speed) ---------------------------------------------------------
# For Innovation A we default-enable TF32 to speed up training on Ampere/Ada.
# You can override per run: ENABLE_TF32=0 bash scripts/experiments_version_c.sh ...
if [[ "${EXP_ID}" == EXP-A* ]]; then
  export ENABLE_TF32="${ENABLE_TF32:-1}"
else
  export ENABLE_TF32="${ENABLE_TF32:-0}"
fi

resolve_layout_input() {
  case "${INSTANCE}" in
    base)
      echo "outputs/P3/A3/layout_input.json"
      ;;
    chain_skip)
      echo "outputs/P3/A3/instances/layout_input_chain_skip.json"
      ;;
    chain_skip_randw)
      echo "outputs/P3/A3/instances/layout_input_chain_skip_randw_s${SEED}.json"
      ;;
    cluster4)
      echo "outputs/P3/A3/instances/layout_input_cluster4_s${SEED}.json"
      ;;
    *)
      echo "[ERROR] Unknown INSTANCE=${INSTANCE}"
      exit 2
      ;;
  esac
}

# ---- ensure A3 middleware exists: outputs/P3/A3/layout_input.json ----
ensure_layout_input() {
  local LI_BASE="outputs/P3/A3/layout_input.json"
  local LI
  LI="$(resolve_layout_input)"

  if [[ ! -f "${LI_BASE}" ]]; then
    echo "[ensure_layout_input] MISSING: ${LI_BASE}"
    echo "[ensure_layout_input] Generating via Version-C Phase3 ..."
    python -m scripts.run_version_c --phase P3 --sub A3 || true
  fi

  if [[ "${INSTANCE}" != "base" && ! -f "${LI}" ]]; then
    echo "[ensure_layout_input] instance missing: ${LI}"
    echo "[ensure_layout_input] generating instances via scripts.make_layout_inputs ..."
    python -m scripts.make_layout_inputs --base "${LI_BASE}" --out_dir "outputs/P3/A3/instances" --seed "${SEED}" || true
  fi

  if [[ -f "${LI}" ]]; then
    echo "[ensure_layout_input] OK: ${LI}"
    return 0
  fi

  echo "[ensure_layout_input] STILL MISSING: ${LI}"
  exit 2
}

run_ast () {
  local cfg="$1"
  local out="$2"
  mkdir -p "$out"

  # Fresh-run guard: archive the whole output dir if it already has checkpoints
  # and AUTO_RESUME is disabled.
  if [[ "${FRESH_RUN}" == "1" && "${AUTO_RESUME}" == "0" ]]; then
    if [[ -d "$out/checkpoints" || -f "$out/metrics.json" || -f "$out/stdout.log" ]]; then
      local ts; ts="$(date '+%Y%m%d_%H%M%S')"
      local bak="${out}__bak__${ts}"
      echo "[FRESH_RUN] archiving existing out dir -> ${bak}"
      mv "$out" "$bak"
      mkdir -p "$out"
    fi
  fi

  # Create a stable symlink for this experiment's stdout.
  # You can tail: tail -f "$STDOUT_AGG_DIR/${EXP_ID}_seed${SEED}.log"
  : > "$out/stdout.log"
  ln -sf "$(realpath "$out/stdout.log")" "$STDOUT_AGG_DIR/${EXP_ID}${TAG_SUFFIX}_seed${SEED}.log"

  if [[ -t 1 && "${TEE_STDOUT:-1}" == "1" ]]; then
    SMOKE="${SMOKE}" python -m scripts.run_ast2_ucf101 --cfg "$cfg" --out_dir "$out" --seed "$SEED" 2>&1 | tee "$out/stdout.log"
  else
    # Non-interactive (nohup/setsid): avoid duplicating logs into the parent nohup log.
    SMOKE="${SMOKE}" python -m scripts.run_ast2_ucf101 --cfg "$cfg" --out_dir "$out" --seed "$SEED" > "$out/stdout.log" 2>&1
    echo "[INFO] logs -> $out/stdout.log"
  fi
}

run_vc () {
  local cfg="$1"
  local out="$2"
  mkdir -p "$out"

  if [[ "${FRESH_RUN}" == "1" && "${AUTO_RESUME}" == "0" ]]; then
    if [[ -d "$out/checkpoints" || -f "$out/metrics.json" || -f "$out/stdout.log" ]]; then
      local ts; ts="$(date '+%Y%m%d_%H%M%S')"
      local bak="${out}__bak__${ts}"
      echo "[FRESH_RUN] archiving existing out dir -> ${bak}"
      mv "$out" "$bak"
      mkdir -p "$out"
    fi
  fi

  : > "$out/stdout.log"
  ln -sf "$(realpath "$out/stdout.log")" "$STDOUT_AGG_DIR/${EXP_ID}${TAG_SUFFIX}_seed${SEED}.log"

  if [[ -t 1 && "${TEE_STDOUT:-1}" == "1" ]]; then
    SMOKE="${SMOKE}" python -m scripts.run_version_c --cfg "$cfg" --out_dir "$out" --seed "$SEED" 2>&1 | tee "$out/stdout.log"
  else
    SMOKE="${SMOKE}" python -m scripts.run_version_c --cfg "$cfg" --out_dir "$out" --seed "$SEED" > "$out/stdout.log" 2>&1
    echo "[INFO] logs -> $out/stdout.log"
  fi
}

run_layout () {
  local cfg="$1"
  local out="$2"
  # For B runs we default to "clean out_dir" to avoid historical accumulation.
  # Can be disabled per run: CLEAN_LAYOUT_OUTDIR=0 ...
  if [[ "${EXP_ID}" == EXP-B* ]]; then
    export CLEAN_LAYOUT_OUTDIR="${CLEAN_LAYOUT_OUTDIR:-1}"
  else
    export CLEAN_LAYOUT_OUTDIR="${CLEAN_LAYOUT_OUTDIR:-0}"
  fi

  if [[ "${CLEAN_LAYOUT_OUTDIR}" == "1" ]]; then
    # ultra-safe: only allow deletion under outputs/B or outputs/SMOKE/B
    case "$out" in
      outputs/B/*|outputs/SMOKE/B/*)
        rm -rf "$out"
        ;;
      *)
        echo "[WARN] CLEAN_LAYOUT_OUTDIR=1 but refusing to delete out_dir outside outputs/B*: $out"
        ;;
    esac
  fi
  mkdir -p "$out"

  ensure_layout_input
  # Avoid OMP threads = 0 crash
  if [[ "${OMP_NUM_THREADS:-0}" -le 0 ]]; then export OMP_NUM_THREADS=1; fi
  if [[ "${OPENBLAS_NUM_THREADS:-0}" -le 0 ]]; then export OPENBLAS_NUM_THREADS=1; fi
  if [[ "${MKL_NUM_THREADS:-0}" -le 0 ]]; then export MKL_NUM_THREADS=1; fi
  if [[ "${NUMEXPR_NUM_THREADS:-0}" -le 0 ]]; then export NUMEXPR_NUM_THREADS=1; fi
  export TOTAL_EVAL_BUDGET_OVERRIDE="${TOTAL_EVAL_BUDGET}"
  python -m scripts.run_layout_agent \
    --layout_input "$(resolve_layout_input)" \
    --cfg "$cfg" --out_dir "$out" --seed "$SEED"
}

run_layout_heuragenix () {
  local cfg="$1"
  local out="$2"
  if [[ "${EXP_ID}" == EXP-B* ]]; then
    export CLEAN_LAYOUT_OUTDIR="${CLEAN_LAYOUT_OUTDIR:-1}"
  else
    export CLEAN_LAYOUT_OUTDIR="${CLEAN_LAYOUT_OUTDIR:-0}"
  fi

  if [[ "${CLEAN_LAYOUT_OUTDIR}" == "1" ]]; then
    case "$out" in
      outputs/B/*|outputs/SMOKE/B/*)
        rm -rf "$out"
        ;;
      *)
        echo "[WARN] CLEAN_LAYOUT_OUTDIR=1 but refusing to delete out_dir outside outputs/B*: $out"
        ;;
    esac
  fi
  mkdir -p "$out"

  ensure_layout_input
  # Avoid OMP threads = 0 crash
  if [[ "${OMP_NUM_THREADS:-0}" -le 0 ]]; then export OMP_NUM_THREADS=1; fi
  if [[ "${OPENBLAS_NUM_THREADS:-0}" -le 0 ]]; then export OPENBLAS_NUM_THREADS=1; fi
  if [[ "${MKL_NUM_THREADS:-0}" -le 0 ]]; then export MKL_NUM_THREADS=1; fi
  if [[ "${NUMEXPR_NUM_THREADS:-0}" -le 0 ]]; then export NUMEXPR_NUM_THREADS=1; fi
  export TOTAL_EVAL_BUDGET_OVERRIDE="${TOTAL_EVAL_BUDGET}"
  python -m scripts.run_layout_heuragenix \
    --layout_input "$(resolve_layout_input)" \
    --cfg "$cfg" --out_dir "$out" --seed "$SEED"
}

odir () {
  # usage: odir EXP-B1  -> outputs/EXP-B1[-INSTANCE]/seed${SEED}
  local exp="$1"
  local exp2="${exp}${TAG_SUFFIX}"
  if [[ "${INSTANCE}" == "base" ]]; then
    echo "${OUT_PREFIX}/${exp2}/seed${SEED}"
  else
    echo "${OUT_PREFIX}/${exp2}-${INSTANCE}/seed${SEED}"
  fi
}

layout_outdir () {
  # usage: layout_outdir EXP-B2 chain_skip
  local exp="$1"
  local inst="$2"
  local exp2="${exp}${TAG_SUFFIX}"
  if [[ "${inst}" == "base" ]]; then
    echo "${OUT_PREFIX}/${exp2}/seed${SEED}"
  else
    echo "${OUT_PREFIX}/${exp2}-${inst}/seed${SEED}"
  fi
}

run_layout_multi () {
  # usage: run_layout_multi run_layout <cfg> <exp_name>
  local runner="$1"
  local cfg="$2"
  local exp="$3"
  if [[ "${INSTANCE}" == "all" ]]; then
    for inst in chain_skip chain_skip_randw cluster4; do
      INSTANCE="$inst" ${runner} "${cfg}" "$(layout_outdir "${exp}" "${inst}")"
    done
  else
    ${runner} "${cfg}" "$(layout_outdir "${exp}" "${INSTANCE}")"
  fi
}

case "$EXP_ID" in
  # -------------------------
  # Innovation A (Main/Core)
  # -------------------------
  EXP-A1)
    export BASELINE_STATS_EXPORT="outputs/dense_baseline/metrics.json"
    OUT="$(odir EXP-A1)"
    run_ast configs/ast2_ucf101_dense_A1.yaml "$OUT"
    python -m scripts.make_acc_ref_curve --stdout "$OUT/stdout.log" --out "$OUT/acc_ref_curve.json" --prefer full --ema-alpha 0.2 --curve-margin 0.0 || true
    ;;
  # A1-fast: dense baseline (FAST iteration: 20% data, 35 epochs)
  # NOTE: Still exports baseline stats for downstream stable_hw runs.
  EXP-A1-fast)
    export BASELINE_STATS_EXPORT="outputs/dense_baseline/metrics.json"
    OUT="$(odir EXP-A1-fast)"
    run_ast configs/ast2_ucf101_dense_A1_fast20.yaml "$OUT"
    python -m scripts.make_acc_ref_curve --stdout "$OUT/stdout.log" --out "$OUT/acc_ref_curve.json" --prefer fast --ema-alpha 0.2 --curve-margin 0.0 || true
    ;;
  # A1' (aligned): Dense baseline under Version-C trainer (avoid pipeline confounds).
  # Output -> outputs[/SMOKE]/NEW_A1/seed{SEED}
  EXP-A1p)
    # Keep legacy output path for backward compatibility.
    OUT="${OUT_PREFIX}/NEW_A1/seed${SEED}"
    run_vc configs/vc_phase3_denseonly_ucf101_A_aligned.yaml "$OUT"
    python -m scripts.make_acc_ref_curve --stdout "$OUT/stdout.log" --out "$OUT/acc_ref_curve.json" --prefer fast --ema-alpha 0.2 --curve-margin 0.0 || true
    ;;
  # A1' fast (aligned): Version-C dense baseline (FAST iteration: 20% data, 35 epochs)
  # This is the recommended reference for StableHW LockedAccRef in A4/ACHO/ROI.
  EXP-A1p-fast)
    export BASELINE_STATS_EXPORT="outputs/dense_baseline/metrics.json"
    OUT="$(odir EXP-A1p-fast)"
    run_vc configs/vc_phase3_denseonly_ucf101_A_aligned_fast20.yaml "$OUT"
    python -m scripts.make_acc_ref_curve --stdout "$OUT/stdout.log" --out "$OUT/acc_ref_curve.json" --prefer full --ema-alpha 0.2 --curve-margin 0.0 || true
    ;;
  # A2: Token-only (HW off, Window off, Token on)
  EXP-A2) run_ast configs/ast2_ucf101_ast_only_A2.yaml           "$(odir EXP-A2)" ;;
  # A2' (aligned): Token-pruning-only baseline under Version-C trainer.
  # Output -> outputs[/SMOKE]/NEW_B1/seed{SEED}
  EXP-A2p)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_aligned.yaml "${OUT_PREFIX}/NEW_B1/seed${SEED}"
    ;;

  # ---- FAST ITERATION (20% data, 35 epochs, token pruning OFF) ----
  # These are for rapid debugging/iteration; do NOT use for final paper numbers.
  EXP-A2p-fast)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_aligned_fast20.yaml "$(odir EXP-A2p-fast)"
    ;;
  EXP-A2p-fast-k90e02)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_aligned_fast20_k90_eps002.yaml "$(odir EXP-A2p-fast-k90e02)"
    ;;
  EXP-A2p-fast-k80)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_plain_fast20_k80.yaml "$(odir EXP-A2p-fast-k80)"
    ;;
  EXP-A2p-alloc-fast-k80)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_allocsearch_fast20_k80.yaml "$(odir EXP-A2p-alloc-fast-k80)"
    ;;

  # ---- FAST ITERATION (20% data) / HW-audit oriented keep_end=0.92 ----
  EXP-A2p-fast-k92)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_plain_fast20_k92.yaml "$(odir EXP-A2p-fast-k92)"
    ;;
  EXP-A2p-sensonly-fast-k92)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_sensonly_fast20_k92.yaml "$(odir EXP-A2p-sensonly-fast-k92)"
    ;;
  EXP-A2p-alloc-fast-k92)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_allocsearch_fast20_k92.yaml "$(odir EXP-A2p-alloc-fast-k92)"
    ;;
  EXP-A2p-alloc-nogate-fast-k92)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_allocsearch_nogate_fast20_k92.yaml "$(odir EXP-A2p-alloc-nogate-fast-k92)"
    ;;
  EXP-A2p-alloc-norisk-fast-k92)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_allocsearch_norisk_fast20_k92.yaml "$(odir EXP-A2p-alloc-norisk-fast-k92)"
    ;;
  EXP-A2p-alloc-nolookahead-fast-k92)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_allocsearch_nolookahead_fast20_k92.yaml "$(odir EXP-A2p-alloc-nolookahead-fast-k92)"
    ;;
  EXP-A2p-cem-fast-k92)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_cem_fast20_k92.yaml "$(odir EXP-A2p-cem-fast-k92)"
    ;;
  EXP-A2p-cem15-fast-k92)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_cem15_fast20_k92.yaml "$(odir EXP-A2p-cem15-fast-k92)"
    ;;
  EXP-A2p-cem20-fast-k92)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_cem20_fast20_k92.yaml "$(odir EXP-A2p-cem20-fast-k92)"
    ;;
  EXP-A2p-cem15-riskopen-fast-k92)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_cem15_riskopen_fast20_k92.yaml "$(odir EXP-A2p-cem15-riskopen-fast-k92)"
    ;;
  EXP-A2p-cem15-pool8-fast-k92)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_cem15_pool8_fast20_k92.yaml "$(odir EXP-A2p-cem15-pool8-fast-k92)"
    ;;
  EXP-A2p-hwloss-fast-k92)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_hwloss_fast20_k92.yaml "$(odir EXP-A2p-hwloss-fast-k92)"
    ;;
  EXP-A2p-hwsurr-fast-k92)
    run_vc configs/vc_phase3_pruningonly_ucf101_A_hwsurr_fast20_k92.yaml "$(odir EXP-A2p-hwsurr-fast-k92)"
    ;;
  EXP-A2p25-warm15-prep-k90)
    run_vc configs/vc_phase3_pruningonly_ucf101_A25_warm15_prep_k90.yaml "$(odir EXP-A2p25-warm15-prep-k90)"
    ;;
  EXP-A2p25-base-k90)
    run_vc configs/vc_phase3_pruningonly_ucf101_A25_base_k90.yaml "$(odir EXP-A2p25-base-k90)"
    ;;
  EXP-A2p25-hwloss-k90)
    run_vc configs/vc_phase3_pruningonly_ucf101_A25_hwloss_k90.yaml "$(odir EXP-A2p25-hwloss-k90)"
    ;;
  EXP-A2p25-cem60-k90)
    run_vc configs/vc_phase3_pruningonly_ucf101_A25_cem60_k90.yaml "$(odir EXP-A2p25-cem60-k90)"
    ;;
  EXP-A2p25-dsfixed-k90)
    run_vc configs/vc_phase3_pruningonly_ucf101_A25_dsfixed_k90.yaml "$(odir EXP-A2p25-dsfixed-k90)"
    ;;
  EXP-A2p25-halp-k90)
    run_vc configs/vc_phase3_pruningonly_ucf101_A25_halp_k90.yaml "$(odir EXP-A2p25-halp-k90)"
    ;;
  EXP-A2p25-newours-k90)
    run_vc configs/vc_phase3_pruningonly_ucf101_A25_newours_k90.yaml "$(odir EXP-A2p25-newours-k90)"
    ;;
  EXP-A2p25-ab-nomemory-k90)
    run_vc configs/vc_phase3_pruningonly_ucf101_A25_ab_nomemory_k90.yaml "$(odir EXP-A2p25-ab-nomemory-k90)"
    ;;
  EXP-A2p25-ab-nolookahead-k90)
    run_vc configs/vc_phase3_pruningonly_ucf101_A25_ab_nolookahead_k90.yaml "$(odir EXP-A2p25-ab-nolookahead-k90)"
    ;;
  EXP-A2p25-ab-nocandsel-k90)
    run_vc configs/vc_phase3_pruningonly_ucf101_A25_ab_nocandsel_k90.yaml "$(odir EXP-A2p25-ab-nocandsel-k90)"
    ;;
  EXP-A2p25-newours-tiebreak-k90)
    run_vc configs/vc_phase3_pruningonly_ucf101_A25_newours_tiebreak_k90.yaml "$(odir EXP-A2p25-newours-tiebreak-k90)"
    ;;
  EXP-A2p25-ab-notiebreak-k90)
    run_vc configs/vc_phase3_pruningonly_ucf101_A25_ab_notiebreak_k90.yaml "$(odir EXP-A2p25-ab-notiebreak-k90)"
    ;;
  EXP-A4-fast)
    run_vc configs/vc_phase3_full_ucf101_A_main_fast20.yaml "$(odir EXP-A4-fast)"
    ;;
  EXP-A4-fast-k90e02)
    run_vc configs/vc_phase3_full_ucf101_A_main_fast20_k90_eps002.yaml "$(odir EXP-A4-fast-k90e02)"
    ;;
  EXP-A4-roi-fast)
    run_vc configs/vc_phase3_full_ucf101_A_roi_fast20.yaml "$(odir EXP-A4-roi-fast)"
    ;;
  EXP-A4-roi-fast-k90e02)
    run_vc configs/vc_phase3_full_ucf101_A_roi_fast20_k90_eps002.yaml "$(odir EXP-A4-roi-fast-k90e02)"
    ;;
  EXP-A4-acho-fast)
    run_vc configs/vc_phase3_full_ucf101_A_acho_fast20.yaml "$(odir EXP-A4-acho-fast)"
    ;;
  EXP-A4-acho-fast-k90e02)
    run_vc configs/vc_phase3_full_ucf101_A_acho_fast20_k90_eps002.yaml "$(odir EXP-A4-acho-fast-k90e02)"
    ;;
  EXP-A4-acho-roi-fast)
    run_vc configs/vc_phase3_full_ucf101_A_acho_roi_fast20.yaml "$(odir EXP-A4-acho-roi-fast)"
    ;;
  EXP-A4-acho-roi-fast-k90e02)
    run_vc configs/vc_phase3_full_ucf101_A_acho_roi_fast20_k90_eps002.yaml "$(odir EXP-A4-acho-roi-fast-k90e02)"
    ;;
  EXP-A3) run_ast configs/ast2_ucf101_ast_hw_A_main.yaml         "$(odir EXP-A3)" ;;
  EXP-A4) run_vc  configs/vc_phase3_full_ucf101_A_main.yaml      "$(odir EXP-A4)" ;;
  EXP-A4-roi) run_vc configs/vc_phase3_full_ucf101_A_roi.yaml "$(odir EXP-A4-roi)" ;;
  # A4 + ACHO (same lambda_hw_max as A4 for fair comparison)
  EXP-A4-acho) run_vc configs/vc_phase3_full_ucf101_A_acho.yaml "$(odir EXP-A4-acho)" ;;
  # A4 + ACHO + ROI-Commit
  EXP-A4-acho-roi) run_vc configs/vc_phase3_full_ucf101_A_acho_roi.yaml "$(odir EXP-A4-acho-roi)" ;;
  EXP-A5) run_vc  configs/vc_phase3_twostage_ucf101_A_main.yaml  "$(odir EXP-A5_twostage)" ;;
  EXP-A6) run_vc  configs/vc_phase3_mapping_only_ucf101_A_main.yaml "$(odir EXP-A6_mappingonly)" ;;
  EXP-A7) run_vc  configs/vc_phase3_layout_only_ucf101_A_main.yaml  "$(odir EXP-A7_layoutonly)" ;;

  # A-G2 fairness (same rho_target)
  # G2 fairness: keep policies run under Token-only protocol (HW off, Window off)
  EXP-A-G2-uniform) run_ast configs/ablations/ast_uniform_keep.yaml "$(odir EXP-A-G2-uniform)" ;;
  EXP-A-G2-random)  run_ast configs/ablations/ast_random_keep.yaml  "$(odir EXP-A-G2-random)" ;;
  # ours should match A-main protocol (clip_len=16, no clip-window)
  EXP-A-G2-ours)    run_vc  configs/vc_phase3_full_ucf101_A_main.yaml "$(odir EXP-A-G2-ours)" ;;

  # A-G3 ablations
  EXP-Abl-time)     run_ast configs/ablations/ast_no_time.yaml      "$(odir EXP-Abl-time)" ;;
  EXP-Abl-space)    run_ast configs/ablations/ast_no_space.yaml     "$(odir EXP-Abl-space)" ;;
  EXP-Abl-vor)      run_ast configs/ablations/ast_no_voronoi.yaml   "$(odir EXP-Abl-vor)" ;;
  EXP-Abl-1lvl)     run_ast configs/ablations/ast_level1.yaml       "$(odir EXP-Abl-1lvl)" ;;
  EXP-Abl-nomodal)  run_ast configs/ablations/ast_no_modal.yaml     "$(odir EXP-Abl-nomodal)" ;;
  EXP-Abl-uniform)  run_ast configs/ablations/ast_uniform_keep.yaml "$(odir EXP-Abl-uniform)" ;;
  EXP-Abl-random)   run_ast configs/ablations/ast_random_keep.yaml  "$(odir EXP-Abl-random)" ;;

  # -------------------------
  # Innovation B (Layout)
  # -------------------------
  EXP-B0)        run_layout_multi run_layout_heuragenix configs/layout_agent/layout_B0_heuragenix_llm_hh_exp2.yaml       EXP-B0 ;;
  EXP-B0-strong) run_layout_multi run_layout_heuragenix configs/layout_agent/layout_B0_heuragenix_llm_hh_strong.yaml     EXP-B0-strong ;;
  EXP-B0-best1)  run_layout_multi run_layout_heuragenix configs/layout_agent/layout_B0_heuragenix_best_single.yaml        EXP-B0-best1 ;;
  EXP-B0-random) run_layout_multi run_layout_heuragenix configs/layout_agent/layout_B0_heuragenix_random_hh_exp2.yaml     EXP-B0-random ;;

  EXP-B1)        run_layout_multi run_layout configs/layout_agent/layout_L0_heuristic_exp.yaml                             EXP-B1 ;;
  EXP-B1-weak)   run_layout_multi run_layout configs/layout_agent/layout_L0_heuristic_weak_exp.yaml                        EXP-B1-weak ;;
  EXP-B2)        run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_exp.yaml                EXP-B2 ;;
  # --- Evidence suite: uncontrolled vs controlled, and controller ablations ---
  EXP-B2-uncontrolled)                 run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_uncontrolled_exp.yaml                 EXP-B2-uncontrolled ;;
  EXP-B2-ctl-ab-notrigger)             run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_ctl_ab_notrigger_exp.yaml             EXP-B2-ctl-ab-notrigger ;;
  EXP-B2-ctl-ab-nomacrostrict)         run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_ctl_ab_nomacrostrict_exp.yaml         EXP-B2-ctl-ab-nomacrostrict ;;
  EXP-B2-ctl-ab-nomemgate)             run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_ctl_ab_nomemgate_exp.yaml             EXP-B2-ctl-ab-nomemgate ;;
  # --- MPVS component-isolation variants (for paper ablations / sanity) ---
  EXP-B2-mpvs-only)            run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_ab_mpvs_only_exp.yaml            EXP-B2-mpvs-only ;;
  EXP-B2-memonly)              run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_ab_memonly_exp.yaml              EXP-B2-memonly ;;
  EXP-B2-macroonly)            run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_ab_macroonly_exp.yaml            EXP-B2-macroonly ;;
  EXP-B2-memonly-noverifier)   run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_ab_memonly_noverifier_exp.yaml   EXP-B2-memonly-noverifier ;;
  EXP-B2-macroonly-noverifier) run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_ab_macroonly_noverifier_exp.yaml EXP-B2-macroonly-noverifier ;;
  EXP-B3)        run_layout_multi run_layout configs/layout_agent/layout_L3_sa_baseline_exp.yaml                            EXP-B3 ;;
  EXP-B2-ab-nollm)      run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_ab_nollm_exp.yaml      EXP-B2-ab-nollm ;;
  EXP-B2-ab-noverifier) run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_ab_noverifier_exp.yaml EXP-B2-ab-noverifier ;;
  EXP-B2-ab-nomacro)    run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_ab_nomacro_exp.yaml    EXP-B2-ab-nomacro ;;
  EXP-B2-ab-nomem)      run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_ab_nomem_exp.yaml      EXP-B2-ab-nomem ;;

  # --- LEGACY (mixed_pick) ablations: kept only for historical comparison ---
  # These belong to the old "mixed_pick" planner family (pre-MPVS/controller).
  # They are NOT comparable to the MPVS/controller baselines, so we gate them
  # behind ALLOW_LEGACY_MIXED_PICK=1 and store outputs under EXP-B2-legacy-*.
  EXP-B2-legacy-mixedpick-ab-noqueue)
    run_layout_multi run_layout configs/layout_agent/_legacy_mixed_pick/layout_L4_region_pareto_llm_mixed_pick_ab_noqueue_exp.yaml EXP-B2-legacy-mixedpick-ab-noqueue
    ;;
  EXP-B2-legacy-mixedpick-ab-nofeas)
    run_layout_multi run_layout configs/layout_agent/_legacy_mixed_pick/layout_L4_region_pareto_llm_mixed_pick_ab_nofeas_exp.yaml  EXP-B2-legacy-mixedpick-ab-nofeas
    ;;
  EXP-B2-legacy-mixedpick-ab-nodiverse)
    run_layout_multi run_layout configs/layout_agent/_legacy_mixed_pick/layout_L4_region_pareto_llm_mixed_pick_ab_nodiverse_exp.yaml EXP-B2-legacy-mixedpick-ab-nodiverse
    ;;

  # Backward-compat aliases (deprecated). Refuse unless explicitly allowed.
  EXP-B2-ab-noqueue|EXP-B2-ab-nofeas|EXP-B2-ab-nodiverse)
    echo "[DEPRECATED] ${EXP_ID} is a LEGACY mixed_pick ablation (not MPVS/controller)."
    echo "             Use EXP-B2-legacy-mixedpick-ab-* instead."
    echo "             To run legacy on purpose: ALLOW_LEGACY_MIXED_PICK=1 ..."
    if [[ "${ALLOW_LEGACY_MIXED_PICK:-0}" != "1" ]]; then
      echo "[ERROR] Refusing to run legacy mixed_pick without ALLOW_LEGACY_MIXED_PICK=1"
      exit 2
    fi
    case "${EXP_ID}" in
      EXP-B2-ab-noqueue)   run_layout_multi run_layout configs/layout_agent/_legacy_mixed_pick/layout_L4_region_pareto_llm_mixed_pick_ab_noqueue_exp.yaml   EXP-B2-legacy-mixedpick-ab-noqueue ;;
      EXP-B2-ab-nofeas)    run_layout_multi run_layout configs/layout_agent/_legacy_mixed_pick/layout_L4_region_pareto_llm_mixed_pick_ab_nofeas_exp.yaml    EXP-B2-legacy-mixedpick-ab-nofeas ;;
      EXP-B2-ab-nodiverse) run_layout_multi run_layout configs/layout_agent/_legacy_mixed_pick/layout_L4_region_pareto_llm_mixed_pick_ab_nodiverse_exp.yaml EXP-B2-legacy-mixedpick-ab-nodiverse ;;
    esac
    ;;

  # --- New paper baselines/method (B1 & B3 unchanged) ---
  EXP-B2-std-budgetaware)
    _CFG="${STD_BUDGETAWARE_CFG:-configs/layout_agent/layout_L4_region_pareto_llm_mpvs_std_budgetaware_nollm_exp.yaml}"
    run_layout_multi run_layout "${_CFG}" EXP-B2-std-budgetaware
    ;;
  EXP-B2-taos-style)
    _CFG="${TAOS_CFG:-configs/layout_agent/layout_L4_region_pareto_llm_mpvs_taos_style_nollm_exp.yaml}"
    run_layout_multi run_layout "${_CFG}" EXP-B2-taos-style
    ;;
  # Main method (default): BC^2-CEC + stage probes + atomic counterfactual baseline
  # Allow overriding configs per sweep without editing code:
  #   BC2CEC_CFG=... BC2CEC_RAW_CFG=... BC2CEC_NOPROBE_CFG=... RUN_TAG_PREFIX=... bash scripts/launch_B_grid_parallel.sh
  EXP-B2-bc2cec)
    _CFG="${BC2CEC_CFG:-configs/layout_agent/layout_L4_region_pareto_llm_mpvs_bc2cec_best_nollm_exp.yaml}"
    run_layout_multi run_layout "${_CFG}" EXP-B2-bc2cec
    ;;
  EXP-B2-bc2cec-nolong)
    _CFG="${BC2CEC_NOLONG_CFG:-configs/layout_agent/layout_L4_region_pareto_llm_mpvs_bc2cec_best_nolong_nollm_exp.yaml}"
    run_layout_multi run_layout "${_CFG}" EXP-B2-bc2cec-nolong
    ;;
  # Ablations to keep story clean (run only when needed)
  EXP-B2-bc2cec-noprobe)
    _CFG="${BC2CEC_NOPROBE_CFG:-configs/layout_agent/layout_L4_region_pareto_llm_mpvs_bc2cec_noprobe_nollm_exp.yaml}"
    run_layout_multi run_layout "${_CFG}" EXP-B2-bc2cec-noprobe
    ;;
  EXP-B2-bc2cec-probe-raw)
    _CFG="${BC2CEC_RAW_CFG:-configs/layout_agent/layout_L4_region_pareto_llm_mpvs_bc2cec_best_probe_raw_nollm_exp.yaml}"
    run_layout_multi run_layout "${_CFG}" EXP-B2-bc2cec-probe-raw
    ;;

  # --- Headroom probes (no controller) ---
  EXP-B2-naive-mixed)      run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_naive_mixed_nollm_exp.yaml      EXP-B2-naive-mixed ;;
  EXP-B2-naive-atomiconly) run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_naive_atomiconly_nollm_exp.yaml EXP-B2-naive-atomiconly ;;
  EXP-B2-naive-macroonly) run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_naive_macroonly_nollm_exp.yaml EXP-B2-naive-macroonly ;;
  EXP-B2-naive-memonly)   run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_naive_memonly_nollm_exp.yaml   EXP-B2-naive-memonly ;;
  EXP-B2-naive-chainonly) run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_naive_chainonly_nollm_exp.yaml EXP-B2-naive-chainonly ;;
  EXP-B2-naive-ruinonly)  run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_naive_ruinonly_nollm_exp.yaml  EXP-B2-naive-ruinonly ;;
  EXP-B2-naive-blockonly) run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_naive_blockonly_nollm_exp.yaml EXP-B2-naive-blockonly ;;
  EXP-B2-naive-relinkonly) run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_naive_relinkonly_nollm_exp.yaml EXP-B2-naive-relinkonly ;;
  EXP-B2-naive-shakeonly)
    _CFG="${NAIVE_SHAKEONLY_CFG:-configs/layout_agent/layout_L4_region_pareto_llm_mpvs_naive_shakeonly_nollm_exp.yaml}"
    run_layout_multi run_layout "${_CFG}" EXP-B2-naive-shakeonly
    ;;
  EXP-B2-naive-tabuonly)   run_layout_multi run_layout configs/layout_agent/layout_L4_region_pareto_llm_mpvs_naive_tabuonly_nollm_exp.yaml   EXP-B2-naive-tabuonly ;;

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
