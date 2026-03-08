#!/usr/bin/env bash
set -euo pipefail

# Launch A4+ACHO and A4+ACHO+ROI with setsid (safe to close terminal).
# GPU4: EXP-A4-acho
# GPU0: EXP-A4-acho-roi

# ========= User config =========
GPUS=(0 4)   # order matters: GPUS[0]=GPU0, GPUS[1]=GPU4
SEED=0
PROJECT_DIR="${PROJECT_DIR:-$HOME/proj_ast2_ucf101_full}"
LOG_DIR="${LOG_DIR:-$HOME/runlogs_A}"
ENV_ACTIVATE="${ENV_ACTIVATE:-$HOME/envs/base_env/bin/activate}"

# Keep consistent with experiments_version_c.sh behavior
SMOKE="${SMOKE:-0}"
INSTANCE="${INSTANCE:-base}"
RUN_TAG="${RUN_TAG:-}"

# Skip experiments that already have metrics (recommended).
SKIP_DONE="${SKIP_DONE:-1}"

# Require baseline export before launching (StableHW may read this).
REQUIRE_BASELINE="${REQUIRE_BASELINE:-1}"
BASELINE_FILE="${BASELINE_FILE:-$PROJECT_DIR/outputs/dense_baseline/metrics.json}"

# Queues (2 GPUs)
Q_GPU0=("EXP-A4-acho-roi")  # GPU0
Q_GPU1=("EXP-A4-acho")      # GPU4

# ========= Helpers =========
ensure_dirs() {
  mkdir -p "$LOG_DIR" "$LOG_DIR/pids" "$LOG_DIR/stdout"
}

is_running_pid() {
  local pid="$1"
  [[ -n "${pid}" ]] && ps -p "${pid}" >/dev/null 2>&1
}

pid_of() {
  local pidfile="$1"
  [[ -f "$pidfile" ]] && cat "$pidfile" 2>/dev/null || true
}

sanitize_tag() {
  local s="${1:-}"
  if [[ -z "$s" ]]; then
    echo ""
    return 0
  fi
  LC_ALL=C echo "$s" | tr -c 'A-Za-z0-9_.+-' '_'
}

calc_out_prefix_base() {
  if [[ "${SMOKE}" == "1" ]]; then
    echo "outputs/SMOKE"
  else
    echo "outputs"
  fi
}

calc_tag_suffix() {
  local safe
  safe="$(sanitize_tag "${RUN_TAG}")"
  if [[ -n "${safe}" ]]; then
    echo "-${safe}"
  else
    echo ""
  fi
}

seed_outdir() {
  # mirror experiments_version_c.sh odir() logic for EXP-A* (non-B)
  local exp="$1"
  local out_prefix_base; out_prefix_base="$(calc_out_prefix_base)"
  local tag_suffix; tag_suffix="$(calc_tag_suffix)"
  local exp2="${exp}${tag_suffix}"
  if [[ "${INSTANCE}" == "base" ]]; then
    echo "$PROJECT_DIR/${out_prefix_base}/${exp2}/seed${SEED}"
  else
    echo "$PROJECT_DIR/${out_prefix_base}/${exp2}-${INSTANCE}/seed${SEED}"
  fi
}

exp_done() {
  local exp="$1"
  local d
  d="$(seed_outdir "$exp")"
  [[ -f "$d/metrics.json" ]] && return 0
  [[ -f "$d/metrics/metrics.json" ]] && return 0
  return 1
}

write_jobfile() {
  local jobfile="$1"
  local logfile="$2"
  local gpu="$3"
  shift 3
  local -a exps=("$@")

  cat > "$jobfile" <<EOF2
#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR}"
LOG_DIR="${LOG_DIR}"
ENV_ACTIVATE="${ENV_ACTIVATE}"
SEED="${SEED}"
GPU="${gpu}"
SKIP_DONE="${SKIP_DONE}"
SMOKE="${SMOKE}"
INSTANCE="${INSTANCE}"
RUN_TAG="${RUN_TAG}"

sanitize_tag() {
  local s="\${1:-}"
  if [[ -z "\$s" ]]; then echo ""; return 0; fi
  LC_ALL=C echo "\$s" | tr -c 'A-Za-z0-9_.+-' '_'
}

calc_out_prefix_base() {
  if [[ "\${SMOKE}" == "1" ]]; then echo "outputs/SMOKE"; else echo "outputs"; fi
}

calc_tag_suffix() {
  local safe; safe="\$(sanitize_tag "\${RUN_TAG}")"
  if [[ -n "\${safe}" ]]; then echo "-\${safe}"; else echo ""; fi
}

seed_outdir() {
  local exp="\$1"
  local out_prefix_base; out_prefix_base="\$(calc_out_prefix_base)"
  local tag_suffix; tag_suffix="\$(calc_tag_suffix)"
  local exp2="\${exp}\${tag_suffix}"
  if [[ "\${INSTANCE}" == "base" ]]; then
    echo "\${PROJECT_DIR}/\${out_prefix_base}/\${exp2}/seed\${SEED}"
  else
    echo "\${PROJECT_DIR}/\${out_prefix_base}/\${exp2}-\${INSTANCE}/seed\${SEED}"
  fi
}

exp_done() {
  local exp="\$1"
  local d; d="\$(seed_outdir "\$exp")"
  [[ -f "\$d/metrics.json" ]] && return 0
  [[ -f "\$d/metrics/metrics.json" ]] && return 0
  return 1
}

cd "\${PROJECT_DIR}"
if [[ -f "\${ENV_ACTIVATE}" ]]; then
  set +u
  source "\${ENV_ACTIVATE}"
  set -u
  conda-unpack >/dev/null 2>&1 || true
fi
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

mkdir -p "\${LOG_DIR}"
exec >> "${logfile}" 2>&1

echo "==== \$(date '+%F %T') JOB BEGIN (GPU=\${GPU}) ===="
echo "[INFO] PROJECT_DIR=\${PROJECT_DIR}"
echo "[INFO] LOG_DIR=\${LOG_DIR}"
echo "[INFO] CUDA_VISIBLE_DEVICES=\${GPU}"
echo "[INFO] SEED=\${SEED} SMOKE=\${SMOKE} INSTANCE=\${INSTANCE} RUN_TAG=\${RUN_TAG}"
echo "[INFO] SKIP_DONE=\${SKIP_DONE}"
which python || true
python -c "import sys; print('[INFO] sys.executable=', sys.executable)" || true

EOF2

  for exp in "${exps[@]}"; do
    cat >> "$jobfile" <<EOF2
echo "==== \$(date '+%F %T') START ${exp} seed\${SEED} GPU=\${GPU} ===="
if [[ "\${SKIP_DONE}" == "1" ]] && exp_done "${exp}"; then
  echo "[SKIP] ${exp} already has metrics -> \$(seed_outdir "${exp}")"
else
  CUDA_VISIBLE_DEVICES="\${GPU}" LOG_DIR="\${LOG_DIR}" SMOKE="\${SMOKE}" INSTANCE="\${INSTANCE}" RUN_TAG="\${RUN_TAG}" \\
    bash scripts/experiments_version_c.sh "${exp}" "\${SEED}"
fi
echo "==== \$(date '+%F %T') END   ${exp} seed\${SEED} GPU=\${GPU} ===="

EOF2
  done

  cat >> "$jobfile" <<'EOF2'
echo "==== $(date '+%F %T') JOB END ===="
EOF2

  chmod +x "$jobfile"
}

launch_chain_bg() {
  local name="$1"; shift
  local gpu="$1"; shift
  local -a exps=("$@")

  ensure_dirs
  local logfile="$LOG_DIR/${name}.log"
  local pidfile="$LOG_DIR/pids/${name}.pid"
  local jobfile="$LOG_DIR/pids/${name}.sh"

  if [[ -f "$pidfile" ]]; then
    local oldpid
    oldpid="$(pid_of "$pidfile")"
    if is_running_pid "$oldpid"; then
      echo "[SKIP] ${name} already running (pid=$oldpid)"
      return 0
    fi
  fi

  write_jobfile "$jobfile" "$logfile" "$gpu" "${exps[@]}"

  echo "[LAUNCH] ${name} on GPU ${gpu}"
  echo "         log -> ${logfile}"
  setsid bash "$jobfile" < /dev/null &
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
    echo "[WARN] REQUIRE_BASELINE=0 (skipping baseline check)"
  fi
}

start_runs() {
  echo "[INFO] ACHO ablation (parallel queues):"
  echo "  GPU${GPUS[0]}: ${Q_GPU0[*]}"
  echo "  GPU${GPUS[1]}: ${Q_GPU1[*]}"
  launch_chain_bg "A4_acho_roi_g${GPUS[0]}" "${GPUS[0]}" "${Q_GPU0[@]}"
  launch_chain_bg "A4_acho_g${GPUS[1]}"     "${GPUS[1]}" "${Q_GPU1[@]}"
}

# ========= Entry =========
ensure_dirs
echo "[INFO] PROJECT_DIR=$PROJECT_DIR"
echo "[INFO] LOG_DIR=$LOG_DIR"
echo "[INFO] Using GPUs: ${GPUS[*]}"
echo "[INFO] SEED=$SEED"
echo "[INFO] SKIP_DONE=$SKIP_DONE"
echo "[INFO] SMOKE=$SMOKE INSTANCE=$INSTANCE RUN_TAG=$RUN_TAG"
echo

check_baseline
start_runs

echo
echo "[DONE] A4+ACHO and A4+ACHO+ROI launched in background (setsid)."
echo "Monitor:"
echo "  ls -lh $LOG_DIR"
echo "  tail -n 80 $LOG_DIR/A4_acho_g${GPUS[1]}.log"
echo "  tail -n 80 $LOG_DIR/A4_acho_roi_g${GPUS[0]}.log"
echo "  # stdout symlinks created by experiments_version_c.sh:"
echo "  ls -lh $LOG_DIR/stdout | tail"
