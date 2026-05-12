#!/usr/bin/env bash
# Verification launcher: runs a slice of configs sequentially on a fixed GPU.
# Usage: verify_runner.sh <gpu_id> <queue_file> <slice_step> <slice_offset>
# (no `set -u` — conda activation hooks reference unset MKL vars.)
set -o pipefail

GPU=${1:?gpu id}
QUEUE=${2:?queue file}
STEP=${3:?slice step}
OFFSET=${4:?slice offset (0-indexed)}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sapef
export CUDA_VISIBLE_DEVICES="$GPU"

LOGDIR=logs/verify
mkdir -p "$LOGDIR"
TS=$(date +%Y%m%d_%H%M%S)
SUMMARY="$LOGDIR/summary_gpu${GPU}_${TS}.txt"
echo "[$(date)] launcher GPU=$GPU step=$STEP offset=$OFFSET" | tee -a "$SUMMARY"

i=0
while IFS= read -r cfg; do
  if (( i % STEP == OFFSET )); then
    tag=$(echo "$cfg" | sed 's|configs/||;s|/|_|g;s|\.toml||')
    log="$LOGDIR/${tag}.log"
    echo "[$(date)] start $cfg -> $log" | tee -a "$SUMMARY"
    if flwr run . gpu-simulation -c "$cfg" >"$log" 2>&1; then
      echo "[$(date)] PASS $cfg" | tee -a "$SUMMARY"
    else
      echo "[$(date)] FAIL $cfg (see $log)" | tee -a "$SUMMARY"
    fi
  fi
  i=$((i+1))
done < "$QUEUE"

echo "[$(date)] launcher GPU=$GPU done" | tee -a "$SUMMARY"
