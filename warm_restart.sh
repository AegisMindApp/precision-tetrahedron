#!/bin/bash
# ============================================================
# Warm Restart Experiment — Condition A (FP32, 256-dim)
#
# Loads condition_A_epoch80.pt from GCS, strips the optimizer
# state (zeroes momentum), and trains from epoch 81 → 100.
#
# Purpose: controlled test of whether FP32 can escape its
# ~0.057 eV plateau when given the same optimizer reset that
# B received via preemption. Cleanly separates the restart
# effect from the precision effect.
#
# Expected runtime: ~2 hrs on v6e.
#
# Results land in:
#   gs://aegismind-tpu-results/aegis_flashoptim/condition_A_warmrestart/
# ============================================================

set -euo pipefail

PYTHON="${PYTHON:-python3}"
GCS_SRC="gs://aegismind-tpu-results/aegis_flashoptim/condition_A_epoch80.pt"
GCS_DEST="gs://aegismind-tpu-results/aegis_flashoptim/condition_A_warmrestart"
OUTPUT_DIR="/tmp/warmrestart_results"
CHECKPOINT_FILE="${GCS_DEST}/.checkpoint"

mkdir -p "$OUTPUT_DIR"

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${OUTPUT_DIR}/warmrestart.log"; }
checkpoint_get() { gsutil cat "$CHECKPOINT_FILE" 2>/dev/null || echo "NONE"; }
checkpoint_set() { echo "$1" | gsutil cp - "$CHECKPOINT_FILE" 2>/dev/null; log "Checkpoint: $1"; }
upload_results() { gsutil -m rsync -r "$OUTPUT_DIR" "${GCS_DEST}/" 2>/dev/null || true; }

log "=== Condition A Warm Restart ==="

# Resume if already started
CKP=$(checkpoint_get)
if [ "$CKP" = "COMPLETE" ]; then
    log "Already complete. Results at ${GCS_DEST}/"
    exit 0
fi

gsutil -m rsync -r "${GCS_DEST}/" "$OUTPUT_DIR/" 2>/dev/null || true

# If the stripped checkpoint isn't local yet, fetch + strip optimizer state
STRIPPED="${OUTPUT_DIR}/condition_A_epoch80.pt"
if [ ! -f "$STRIPPED" ]; then
    log "Downloading condition_A_epoch80.pt from GCS..."
    gsutil cp "$GCS_SRC" "$STRIPPED"

    log "Stripping optimizer state (preserving model weights + param_groups)..."
    $PYTHON - <<'PYEOF'
import torch, sys, os

path = "/tmp/warmrestart_results/condition_A_epoch80.pt"
ckpt = torch.load(path, map_location='cpu')

# Zero the optimizer momentum — keep param_groups (LR, weight_decay)
# so the scheduler resumes at the correct LR for epoch 81.
fresh_optim = {
    'state': {},
    'param_groups': ckpt['optimizer']['param_groups']
}
ckpt['optimizer'] = fresh_optim

torch.save(ckpt, path)
print(f"Stripped optimizer state. Model keys: {list(ckpt.keys())}")
PYEOF
    log "Checkpoint stripped."
fi

checkpoint_set "RUNNING"

log "Training Condition A from epoch 81 → 100 (warm restart)..."
$PYTHON train.py \
    --condition A \
    --data-dir /tmp/qm9 \
    --output-dir "$OUTPUT_DIR" \
    --epochs 100 \
    --batch-size 32 \
    --lr 1e-4

checkpoint_set "COMPLETE"
upload_results

log "=== Warm restart complete. Results: ${GCS_DEST}/ ==="
