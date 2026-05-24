#!/bin/bash
# ============================================================
# LR Ablation Runner — MolPrecision / FlashOptim
#
# Runs Conditions A (FP32) and B (BF16) at a given learning rate.
# Saves results to GCS under a separate prefix per LR.
# Supports checkpoint resume — safe to rerun after preemption.
#
# Usage (inside tmux on TPU VM):
#   LR=5e-4 bash lr_ablation.sh
#   LR=5e-5 bash lr_ablation.sh
#
# Results land in:
#   gs://aegismind-tpu-results/aegis_flashoptim/lr_ablation_<LR>/
# ============================================================

set -euo pipefail

LR="${LR:-1e-4}"
PYTHON="${PYTHON:-python3}"
BATCH_SIZE=32
EPOCHS=100
DATA_DIR="/tmp/qm9"
OUTPUT_DIR="/tmp/lr_ablation_${LR}"
GCS_PREFIX="gs://aegismind-tpu-results/aegis_flashoptim/lr_ablation_${LR}"
CHECKPOINT_FILE="${GCS_PREFIX}/.checkpoint"

mkdir -p "$OUTPUT_DIR"

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${OUTPUT_DIR}/ablation.log"; }
checkpoint_set() { echo "$1" | gsutil cp - "$CHECKPOINT_FILE" 2>/dev/null; log "Checkpoint: $1"; }
checkpoint_get() { gsutil cat "$CHECKPOINT_FILE" 2>/dev/null || echo "NONE"; }
checkpoint_matches() { local c; c=$(checkpoint_get); echo "$c" | grep -qE "$1"; }

upload_results() {
    gsutil -m rsync -r "$OUTPUT_DIR" "$GCS_PREFIX/" 2>/dev/null || true
}

log "=== LR Ablation: lr=${LR} ==="
log "GCS prefix: $GCS_PREFIX"

# Restore any GCS results to local (for resume)
gsutil -m rsync -r "$GCS_PREFIX/" "$OUTPUT_DIR/" 2>/dev/null || true

# ── Condition A: FP32 baseline at this LR ──────────────────────────────────
if ! checkpoint_matches "A_DONE|B_DONE|COMPLETE"; then
    log ">>> Condition A: FP32, lr=${LR}"
    checkpoint_set "A_RUNNING"

    $PYTHON train.py \
        --condition A \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR"

    checkpoint_set "A_DONE"
    upload_results
    log "Condition A complete."
fi

# ── Condition B: BF16, same dim, at this LR ────────────────────────────────
if ! checkpoint_matches "B_DONE|COMPLETE"; then
    log ">>> Condition B: BF16, lr=${LR}"
    checkpoint_set "B_RUNNING"

    $PYTHON train.py \
        --condition B \
        --data-dir "$DATA_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR"

    checkpoint_set "B_DONE"
    upload_results
    log "Condition B complete."
fi

checkpoint_set "COMPLETE"
upload_results

log ""
log "=== LR=${LR} ablation complete. Results: ${GCS_PREFIX}/ ==="
