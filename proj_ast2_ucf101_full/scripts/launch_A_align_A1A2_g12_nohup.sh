#!/usr/bin/env bash
set -euo pipefail

# Launch "aligned" A1'/A2' baselines under Version-C trainer.
# Modeled after scripts/launch_A_stage2_only_g345_nohup.sh (known-good quoting),
# but runs on GPU 1/2 and writes to NEW output dirs:
#   - EXP-A1p -> outputs[/SMOKE]/NEW_A1/seed{SEED}
#   - EXP-A2p -> outputs[/SMOKE]/NEW_B1/seed{SEED}

# ========= User config =========
GPUS=(1 2)
SEED=0
PROJECT_DIR="${PROJECT_DIR:-$HOME/proj_ast2_ucf101_full}"
LOG_DIR="${LOG_DIR:-$HOME/runlogs_A_align}"
ENV_ACTIVATE="${ENV_ACTIVATE:-$HOME/envs/base_env/bin/activate}"

# Skip experiments that already have metrics (recommended).
SKIP_DONE="${SKIP_DONE:-1}"

# No baseline requirement for aligned baselines.
REQUIRE_BASELINE="${REQUIRE_BASELINE:-0}"
BASELINE_FILE="${BASELINE_FILE:-$PROJECT_DIR/outputs/dense_baseline/metrics.json}"

# Queues (2 GPUs)
Q_GPU0=("EXP-A1p")
Q_GPU1=("EXP-A2p")

# ========= Helpers =========
ensure_dirs() {
  mkdir -p "$LOG_DIR" "$LOG_DIR/pids"
}

is_running_pid() {
  local pid="$1"
  [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1
}

pid_of() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] && cat "$pidfile" 2>/dev/null || true
}

# Respect SMOKE routing used by scripts/experiments_version_c.sh
SMOKE="${SMOKE:-0}"
OUT_PREFIX_BASE="outputs"
if [[ "${SMOKE}" == "1" ]]; then
  OUT_PREFIX_BASE="outputs/SMOKE"
fi

# Map experiment ID to its actual output directory name used by experiments_version_c.sh
seed_outdir() {
  local exp="$1"
  case "$exp" in
    EXP-A1p) echo "$PROJECT_DIR/${OUT_PREFIX_BASE}/NEW_A1/seed${SEED}" ;;
    EXP-A2p) echo "$PROJECT_DIR/${OUT_PREFIX_BASE}/NEW_B1/seed${SEED}" ;;
    *)       echo "$PROJECT_DIR/${OUT_PREFIX_BASE}/${exp}/seed${SEED}" ;;
  esac
}

exp_done() {
  local exp="$1"
  local d
  d="$(seed_outdir "$exp")"
  [[ -f "$d/metrics.json" ]] && return 0
  [[ -f "$d/metrics/metrics.json" ]] && return 0
  return 1
}

launch_chain_bg() {
  local name="$1"; shift
  local gpu="$1"; shift
  local -a exps=("$@")

  ensure_dirs
  local logfile="$LOG_DIR/${name}.log"
  local pidfile="$LOG_DIR/pids/${name}.pid"

  # If already running, skip
  if [[ -f "$pidfile" ]]; then
    local oldpid
    oldpid="$(pid_of "$pidfile")"
    if is_running_pid "$oldpid"; then
      echo "[SKIP] ${name} already running (pid=$oldpid)"
      return 0
    fi
  fi

  # Build sequential chain on one GPU (with optional skip-done)
  local chain=""
  for exp in "${exps[@]}"; do
    chain+="echo \"==== \$(date '+%F %T') START ${exp} seed${SEED} GPU=${gpu} ====\"; "
    if [[ "$SKIP_DONE" == "1" ]]; then
      chain+="python - <<'PY'\n"
      chain+="import os\n"
      chain+="project=r'''${PROJECT_DIR}'''\n"
      chain+="seed=${SEED}\n"
      chain+="exp=r'''${exp}'''\n"
      chain+="smoke = os.environ.get('SMOKE','0')\n"
      chain+="out_base = 'outputs/SMOKE' if smoke == '1' else 'outputs'\n"
      chain+="def seed_outdir(exp):\n"
      chain+="  if exp=='EXP-A1p': return f\"{project}/{out_base}/NEW_A1/seed{seed}\"\n"
      chain+="  if exp=='EXP-A2p': return f\"{project}/{out_base}/NEW_B1/seed{seed}\"\n"
      chain+="  return f\"{project}/{out_base}/{exp}/seed{seed}\"\n"
      chain+="d=seed_outdir(exp)\n"
      chain+="done = os.path.isfile(os.path.join(d,'metrics.json')) or os.path.isfile(os.path.join(d,'metrics','metrics.json'))\n"
      chain+="print('[SKIP_DONE]' if done else '[RUN]', exp, 'out=', d)\n"
      chain+="exit(0 if done else 1)\n"
      chain+="PY\n"
      chain+="if [[ \$? -eq 0 ]]; then echo \"[SKIP] ${exp} already has metrics\"; "
      chain+="else CUDA_VISIBLE_DEVICES=${gpu} bash scripts/experiments_version_c.sh ${exp} ${SEED}; fi; "
    else
      chain+="CUDA_VISIBLE_DEVICES=${gpu} bash scripts/experiments_version_c.sh ${exp} ${SEED}; "
    fi
    chain+="echo \"==== \$(date '+%F %T') END   ${exp} seed${SEED} GPU=${gpu} ====\"; "
  done

  echo "[LAUNCH] ${name} on GPU ${gpu}"
  echo "         log -> ${logfile}"

  setsid bash -lc "
    set -euo pipefail
    cd '$PROJECT_DIR'

    if [[ -f '$ENV_ACTIVATE' ]]; then
      set +u
      source '$ENV_ACTIVATE'
      set -u
      conda-unpack >/dev/null 2>&1 || true
    fi
    export PYTHONPATH=.
    export PYTHONUNBUFFERED=1

    mkdir -p '$LOG_DIR'
    exec >> '$logfile' 2>&1

    echo \"==== \$(date '+%F %T') JOB ${name} BEGIN (GPU=${gpu}) ====\"
    echo \"[INFO] PROJECT_DIR=$PROJECT_DIR\"
    echo \"[INFO] LOG_DIR=$LOG_DIR\"
    echo \"[INFO] ENV_ACTIVATE=$ENV_ACTIVATE\"
    echo \"[INFO] CUDA_VISIBLE_DEVICES=${gpu}\"
    echo \"[INFO] SKIP_DONE=$SKIP_DONE\"
    echo \"[INFO] SMOKE=${SMOKE} OUT_PREFIX_BASE=${OUT_PREFIX_BASE}\"
    which python || true
    python -c \"import sys; print('[INFO] sys.executable=', sys.executable)\" || true

    ${chain}

    echo \"==== \$(date '+%F %T') JOB ${name} END (GPU=${gpu}) ====\"
  " < /dev/null &

  echo $! > "$pidfile"
  echo "         pid -> $(cat "$pidfile")"
}

check_baseline() {
  if [[ "$REQUIRE_BASELINE" == "1" ]]; then
    if [[ ! -f "$BASELINE_FILE" ]]; then
      echo "[ERR] Baseline file not found: $BASELINE_FILE"
      echo "      If you intentionally want to run without it, set REQUIRE_BASELINE=0."
      exit 1
    fi
    echo "[OK] Baseline exists: $BASELINE_FILE"
  else
    echo "[INFO] REQUIRE_BASELINE=0 (skipping baseline check)"
  fi
}

start_queues() {
  echo "[INFO] Aligned-baseline queues (parallel):"
  echo "  GPU${GPUS[0]}: ${Q_GPU0[*]:-(empty)}"
  echo "  GPU${GPUS[1]}: ${Q_GPU1[*]:-(empty)}"
  launch_chain_bg "A_align_g${GPUS[0]}" "${GPUS[0]}" "${Q_GPU0[@]}"
  launch_chain_bg "A_align_g${GPUS[1]}" "${GPUS[1]}" "${Q_GPU1[@]}"
}

# ========= Entry =========
ensure_dirs
echo "[INFO] PROJECT_DIR=$PROJECT_DIR"
echo "[INFO] LOG_DIR=$LOG_DIR"
echo "[INFO] Using GPUs: ${GPUS[*]}"
echo "[INFO] SEED=$SEED"
echo "[INFO] SKIP_DONE=$SKIP_DONE"
echo

check_baseline
start_queues

echo
echo "[DONE] Aligned baselines launched in background."
echo "Monitor:"
echo "  ls -lh $LOG_DIR"
echo "  tail -n 80 $LOG_DIR/A_align_g${GPUS[0]}.log"
echo "  tail -n 80 $LOG_DIR/A_align_g${GPUS[1]}.log"
echo "  nvidia-smi"
