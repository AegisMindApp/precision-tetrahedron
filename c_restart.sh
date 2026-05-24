#!/bin/bash
# ============================================================
# c_restart.sh — Condition C warm restart experiment
#
# Hypothesis: BF16 512-dim (Cond C) is stuck at 0.0564 eV not
# because of capacity, but because it never received a restart.
# Applying a fresh restart from ep40 (deep in plateau) with
# lr=5e-5 and a clean cosine schedule should allow it to escape,
# potentially beating both Cond B (0.0215) and A_ext (0.0169).
#
# Design:
#   Source checkpoint:  condition_C_epoch40.pt  (plateau region)
#   Restart LR:         5e-5  (half of 1e-4 — avoids Phase 4b overshoot)
#   Schedule:           CosineAnnealingLR(T_max=30, eta_min=1e-6)
#   Epochs after:       30  (ep41 → ep70)
#   Precision:          BF16
#   Model:              hidden_dim=512, 6 blocks (~6.6M params)
#
# Expected runtime: ~4 hrs on v6e-8.
#
# Results:
#   /tmp/c_restart_results/
#   gs://aegismind-tpu-results/aegis_flashoptim/c_restart/
#
# Usage (inside tmux on fresh TPU VM):
#   GCS_BUCKET=gs://aegismind-tpu-results bash c_restart.sh
# ============================================================

set -euo pipefail
cd /home/john/flashoptim

GCS_BUCKET="${GCS_BUCKET:-gs://aegismind-tpu-results}"
RUN_ID="${RUN_ID:-aegis_flashoptim}"
OUTPUT_DIR="/tmp/c_restart_results"
MAIN_RESULTS="/tmp/flashoptim_results"
CKPT_SRC="${MAIN_RESULTS}/condition_C_epoch40.pt"
GCS_SRC="${GCS_BUCKET}/${RUN_ID}/condition_C_epoch40.pt"
GCS_DEST="${GCS_BUCKET}/${RUN_ID}/c_restart"

PYTHON="python3 -u"
mkdir -p "$OUTPUT_DIR"

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "${OUTPUT_DIR}/c_restart.log"; }
upload() { gsutil -m rsync -r "${OUTPUT_DIR}/" "${GCS_DEST}/" 2>/dev/null || true; }

log "=== Condition C Warm Restart ==="
log "Source: condition_C_epoch40.pt  |  restart_lr=5e-5  |  30 epochs  |  BF16 512-dim"

# ── Fetch checkpoint ──────────────────────────────────────────────────────────
if [ ! -f "$CKPT_SRC" ]; then
    log "condition_C_epoch40.pt not local — downloading from GCS..."
    gsutil cp "$GCS_SRC" "$CKPT_SRC" \
        || { log "ERROR: could not fetch checkpoint from GCS"; exit 1; }
fi
cp "$CKPT_SRC" "${OUTPUT_DIR}/condition_C_epoch40.pt"
log "Checkpoint ready: ${OUTPUT_DIR}/condition_C_epoch40.pt"

# ── Run restart training ──────────────────────────────────────────────────────
log "Starting restart training (BF16, hidden_dim=512, lr=5e-5, 30 epochs)..."

$PYTHON restart_trainer.py \
    --condition      C \
    --checkpoint     "${OUTPUT_DIR}/condition_C_epoch40.pt" \
    --restart-lr     5e-5 \
    --epochs-after   30 \
    --data-dir       /tmp/qm9 \
    --output-dir     "$OUTPUT_DIR" \
    --batch-size     32 \
    --gcs-dest       "$GCS_DEST" \
    2>&1 | tee -a "${OUTPUT_DIR}/c_restart.log"

log "=== Condition C restart complete ==="
log "Results: ${OUTPUT_DIR}/condition_C_restart_results.json"
log "GCS:     ${GCS_DEST}/"

# ── Summary ───────────────────────────────────────────────────────────────────
python3 - <<'PYEOF'
import json
from pathlib import Path

r = json.loads(Path("/tmp/c_restart_results/condition_C_restart_results.json").read_text())
print("\n=== C_restart Summary ===")
print(f"  Source epoch:    {r['source_epoch']}")
print(f"  Best val_mae:    {r['best_val_mae_ev']:.4f} eV  (epoch {r['best_epoch']})")
print(f"  Reference:       B=0.0215 eV | A_ext=0.0169 eV | C_plateau=0.0564 eV")
print(f"  Beats Cond B:    {'YES' if r['best_val_mae_ev'] < 0.0215 else 'NO'}")
print(f"  Beats A_ext:     {'YES' if r['best_val_mae_ev'] < 0.0169 else 'NO'}")
print()
print("  Epoch trajectory:")
for e in r['epochs']:
    print(f"    ep{e['epoch']:>3d}  lr={e['lr']:.2e}  val_mae={e['val_mae_ev']:.4f} eV")
PYEOF
