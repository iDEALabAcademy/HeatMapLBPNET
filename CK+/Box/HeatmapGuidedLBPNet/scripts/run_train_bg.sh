#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_train_bg.sh <MODEL_PRESET> <RUN_DIR_NAME> <LOG_NAME>
# Example:
#   ./scripts/run_train_bg.sh paper_mnist_rp outputs_paper_rp_k8_soft2hard_v1 train_paper_rp_k8_soft2hard_v1.log

PRESET=${1:-paper_mnist_rp}
RUN=${2:-outputs_paper_rp_k8_soft2hard_v1}
LOGNAME=${3:-train_paper_rp_k8_soft2hard_v1.log}

ROOT_DIR=/home/hding22/binary
ARCHIVE_DIR=$ROOT_DIR/training_results_archive
OUTDIR=$ARCHIVE_DIR/$RUN
LOG=$ARCHIVE_DIR/$LOGNAME

mkdir -p "$OUTDIR"
cd "$ROOT_DIR"

# Stop previous trainings quietly
pkill -f 'train_original_model.py' || true

# Point outputs_mnist_original to this run
rm -f outputs_mnist_original
ln -s "$OUTDIR" outputs_mnist_original

# Truncate log
: > "$LOG"

echo "[LAUNCH] preset=$PRESET outdir=$OUTDIR log=$LOG"

# Launch in background
nohup env MODEL_PRESET="$PRESET" PYTHONUNBUFFERED=1 \
  python -u train_original_model.py >> "$LOG" 2>&1 &
PID=$!
echo $PID > "$OUTDIR/train.pid"
echo "[LAUNCHED] pid=$PID"

exit 0






