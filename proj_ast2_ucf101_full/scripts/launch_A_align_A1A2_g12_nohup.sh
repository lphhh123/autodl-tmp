#!/usr/bin/env bash
set -euo pipefail

# Launch "aligned" A1'/A2' baselines under Version-C trainer.
# Modeled after launch_A_stage2_only_g345_nohup.sh,
# but runs on GPU 1/2 and writes to NEW output dirs
# (does NOT touch original EXP-A1/EXP-A2 folders).
#
# New experiments (defined in scripts/experiments_version_c.sh):
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
Q_GPU0=("EXP-A1p")   # dense aligned baseline
Q_GPU1=("EXP-A2p")   # pruning-only aligned baseline

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

launch_chain_bg() {
  local name="$1"; shift
  local gpu="$1"; shift
  local -a exps=("$@")

  if [[ ${#exps[@]} -eq 0 ]]; then
    echo "[SKIP] ${name}: empty queue"
    return 0
  fi

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

  # Flatten list for child shell
  local exp_list="${exps[*]}"

  echo "[LAUNCH] ${name} on GPU ${gpu}"
  echo "         log -> ${logfile}"

  setsid env \
    PROJECT_DIR="$PROJECT_DIR" \
    ENV_ACTIVATE="$ENV_ACTIVATE" \
    LOG_DIR="$LOG_DIR" \
    LOGFILE="$logfile" \
    GPU="$gpu" \
    SEED="$SEED" \
    SKIP_DONE="$SKIP_DONE" \
    EXP_LIST="$exp_list" \
    JOB_NAME="$name" \
    bash -lc '
      set -euo pipefail
      cd "$PROJECT_DIR"

      if [[ -f "$ENV_ACTIVATE" ]]; then
        set +u
        source "$ENV_ACTIVATE"
        set -u
        conda-unpack >/dev/null 2>&1 || true
      fi
      export PYTHONPATH=.
      export PYTHONUNBUFFERED=1

      mkdir -p "$LOG_DIR"
      exec >> "$LOGFILE" 2>&1

      # Match experiments_version_c.sh SMOKE routing for NEW_* dirs
      OUT_PREFIX_BASE="outputs"
      if [[ "${SMOKE:-0}" == "1" ]]; then
        OUT_PREFIX_BASE="outputs/SMOKE"
      fi

      seed_outdir() {
        local exp="$1"
        case "$exp" in
          EXP-A1p) echo "${PROJECT_DIR}/${OUT_PREFIX_BASE}/NEW_A1/seed${SEED}" ;;
          EXP-A2p) echo "${PROJECT_DIR}/${OUT_PREFIX_BASE}/NEW_B1/seed${SEED}" ;;
          *)       echo "${PROJECT_DIR}/${OUT_PREFIX_BASE}/${exp}/seed${SEED}" ;;
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

      echo "==== $(date "+%F %T") JOB ${JOB_NAME} BEGIN (GPU=${GPU}) ===="
      echo "[INFO] PROJECT_DIR=$PROJECT_DIR"
      echo "[INFO] LOG_DIR=$LOG_DIR"
      echo "[INFO] CUDA_VISIBLE_DEVICES=${GPU}"
      echo "[INFO] SKIP_DONE=${SKIP_DONE}"
      echo "[INFO] SMOKE=${SMOKE:-0} OUT_PREFIX_BASE=$OUT_PREFIX_BASE"
      which python || true
      python -c "import sys; print('[INFO] sys.executable=', sys.executable)" || true

      for exp in $EXP_LIST; do
        echo "==== $(date "+%F %T") START $exp seed${SEED} GPU=${GPU} ===="
        if [[ "$SKIP_DONE" == "1" ]] && exp_done "$exp"; then
          echo "[SKIP] $exp already has metrics -> $(seed_outdir "$exp")"
        else
          CUDA_VISIBLE_DEVICES="$GPU" bash scripts/experiments_version_c.sh "$exp" "$SEED"
        fi
        echo "==== $(date "+%F %T") END   $exp seed${SEED} GPU=${GPU} ===="
      done

      echo "==== $(date "+%F %T") JOB ${JOB_NAME} END (GPU=${GPU}) ===="
    ' < /dev/null &

  echo $! > "$pidfile"
  echo "         pid -> $(cat "$pidfile")"
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
