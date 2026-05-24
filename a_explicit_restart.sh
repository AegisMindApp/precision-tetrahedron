#!/bin/bash
# ============================================================
# a_explicit_restart.sh — Condition A explicit warm restart
#
# Phase 4b showed FP32 can escape to 0.01842 eV at ep81 with a
# warm restart from ep80, but then degraded because the cosine
# schedule was inherited from the checkpoint's decayed state
# rather than being freshly anchored at the restart LR.
#
# This script reproduces Phase 4b with the fix applied:
#   - Fresh AdamW optimizer at restart_lr=5e-5 (not 1e-4)
#   - CosineAnnealingLR(T_max=20) anchored at 5e-5 exactly
#   - 20 epochs post-restart (ep81 → ep100)
#
# Expected: hold the ep81 improvement (~0.018 eV) and improve
# further as the cosine decays, rather than overshooting back.
#
# Design:
#   Source checkpoint:  condition_A_epoch80.pt  (plateau region)
#   Restart LR:         5e-5  (lower than Phase 4b's effective ~1e-5,
#                              but now correctly anchored in the schedule)
#   Schedule:           CosineAnnealingLR(T_max=20, eta_min=1e-6)
#   Epochs after:       20  (ep81 → ep100)
#   Precision:          FP32
#   Model:              hidden_dim=256, 6 blocks (~1.7M params)
#
# Expected runtime: ~2.5 hrs on v6e-8.
#
# Results:
#   /tmp/a_restart_results/
#   gs://aegismind-tpu-results/aegis_flashoptim/a_explicit_restart/
#
# Usage (inside tmux on fresh TPU VM):
#   GCS_BUCKET=gs://aegismind-tpu-results bash a_explicit_restart.sh
# ============================================================

set -euo pipefail
cd /home/john/flashoptim

GCS_BUCKET="${GCS_BUCKET:-gs://aegismind-tpu-results}"
RUN_ID="${RUN_ID:-aegis_flashoptim}"
OUTPUT_DIR="/tmp/a_restart_results"
MAIN_RESULTS="/tmp/flashoptim_results"
CKPT_SRC="${MAIN_RESULTS}/condition_A_epoch80.pt"
GCS_SRC="${GCS_BUCKET}/${RUN_ID}/condition_A_epoch80.pt"
GCS_DEST="${GCS_BUCKET}/${RUN_ID}/a_explicit_restart"

PYTHON="python3 -u"
mkdir -p "$OUTPUT_DIR"

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "${OUTPUT_DIR}/a_restart.log"; }
upload() { gsutil -m rsync -r "${OUTPUT_DIR}/" "${GCS_DEST}/" 2>/dev/null || true; }

log "=== Condition A Explicit Warm Restart (Phase 4b fix) ==="
log "Source: condition_A_epoch80.pt  |  restart_lr=5e-5  |  20 epochs  |  FP32 256-dim"
log "Fix vs Phase 4b: fresh optimizer anchored at 5e-5 (not inherited cosine decay)"

# ── Fetch checkpoint ──────────────────────────────────────────────────────────
if [ ! -f "$CKPT_SRC" ]; then
    log "condition_A_epoch80.pt not local — downloading from GCS..."
    gsutil cp "$GCS_SRC" "$CKPT_SRC" \
        || { log "ERROR: could not fetch checkpoint from GCS"; exit 1; }
fi
cp "$CKPT_SRC" "${OUTPUT_DIR}/condition_A_epoch80.pt"
log "Checkpoint ready: ${OUTPUT_DIR}/condition_A_epoch80.pt"

# ── Run restart training ──────────────────────────────────────────────────────
log "Starting restart training (FP32, hidden_dim=256, lr=5e-5, 20 epochs)..."

$PYTHON restart_trainer.py \
    --condition      A \
    --checkpoint     "${OUTPUT_DIR}/condition_A_epoch80.pt" \
    --restart-lr     5e-5 \
    --epochs-after   20 \
    --data-dir       /tmp/qm9 \
    --output-dir     "$OUTPUT_DIR" \
    --batch-size     32 \
    --gcs-dest       "$GCS_DEST" \
    2>&1 | tee -a "${OUTPUT_DIR}/a_restart.log"

log "=== Condition A explicit restart complete ==="
log "Results: ${OUTPUT_DIR}/condition_A_restart_results.json"
log "GCS:     ${GCS_DEST}/"

# ── Summary ───────────────────────────────────────────────────────────────────
python3 - <<'PYEOF'
import json
from pathlib import Path

r = json.loads(Path("/tmp/a_restart_results/condition_A_restart_results.json").read_text())
print("\n=== A_explicit_restart Summary ===")
print(f"  Source epoch:    {r['source_epoch']}")
print(f"  Best val_mae:    {r['best_val_mae_ev']:.4f} eV  (epoch {r['best_epoch']})")
print(f"  Reference:       B=0.0215 eV | A_ext=0.0169 eV | Phase4b=0.0184 eV (then degraded)")
print(f"  Beats Phase 4b:  {'YES' if r['best_val_mae_ev'] < 0.0184 else 'NO'}")
print(f"  Beats A_ext:     {'YES' if r['best_val_mae_ev'] < 0.0169 else 'NO'}")
print()
print("  Epoch trajectory (with LR):")
for e in r['epochs']:
    print(f"    ep{e['epoch']:>3d}  lr={e['lr']:.2e}  val_mae={e['val_mae_ev']:.4f} eV")
PYEOF
