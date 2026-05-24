#!/bin/bash
# ============================================================
# vina_phase5.sh — Phase 5: Protein-aware Vina Oracle BO
#
# Runs AFTER Phase 4 (or independently on a fresh VM).
# 1. Installs AutoDock Vina + OpenBabel
# 2. Prepares receptors for all 6 targets
# 3. Screens 2,639 FDA compounds × 6 targets with parallel Vina
# 4. Re-runs Phase 3 BO using Vina scores as oracle
# 5. Uploads all results to GCS
#
# Usage (inside tmux on TPU VM after Phase 4 completes):
#   GCS_BUCKET=gs://aegismind-tpu-results bash vina_phase5.sh
#
# Or run as a watcher that auto-starts when Phase 4 finishes:
#   bash vina_phase5.sh --watch
# ============================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

WATCH_MODE=0
[ "${1:-}" = "--watch" ] && WATCH_MODE=1

export GCS_BUCKET="${GCS_BUCKET:-gs://aegismind-tpu-results}"
export RUN_ID="${RUN_ID:-aegis_flashoptim}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/flashoptim_results}"
export PHASE2_DATA_DIR="${PHASE2_DATA_DIR:-/tmp/phase2_data}"
export RECEPTOR_DIR="${RECEPTOR_DIR:-/tmp/vina_receptors}"
export VINA_WORKERS="${VINA_WORKERS:-32}"
export VINA_EXHAUSTIVENESS="${VINA_EXHAUSTIVENESS:-8}"

CHECKPOINT_FILE="${OUTPUT_DIR}/.phase5_checkpoint"
PYTHON="python3 -u"

mkdir -p "$OUTPUT_DIR" "$RECEPTOR_DIR" "$PHASE2_DATA_DIR"

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "${OUTPUT_DIR}/phase5.log"; }
checkpoint_set() { echo "$1" > "$CHECKPOINT_FILE"; log "Checkpoint: $1"; }
checkpoint_get() { cat "$CHECKPOINT_FILE" 2>/dev/null || echo "NONE"; }
checkpoint_matches() { grep -qE "$1" "$CHECKPOINT_FILE" 2>/dev/null; }
upload() { gsutil -m rsync -r "${OUTPUT_DIR}/" "${GCS_BUCKET}/${RUN_ID}/" 2>/dev/null || true; }

# ── Optionally wait for Phase 4 to complete ────────────────────────────────────
if [ "$WATCH_MODE" -eq 1 ]; then
    log "Watch mode: waiting for Phase 4 to complete..."
    MAIN_CKPT="${OUTPUT_DIR}/.pipeline_checkpoint"
    while true; do
        STATE=$(cat "$MAIN_CKPT" 2>/dev/null || echo "NONE")
        if echo "$STATE" | grep -qE "PHASE4_DONE|PIPELINE_COMPLETE"; then
            log "Phase 4 complete. Starting Phase 5."
            break
        fi
        log "  Phase 4 still running (checkpoint: $STATE) — checking again in 5 min"
        sleep 300
    done
fi

log "=== Phase 5: Protein-aware Vina Oracle BO ==="
log "GCS bucket: $GCS_BUCKET  |  Workers: $VINA_WORKERS  |  Exhaustiveness: $VINA_EXHAUSTIVENESS"

# ── Step 1: Install dependencies ───────────────────────────────────────────────
if ! checkpoint_matches "DEPS_DONE"; then
    log "Installing AutoDock Vina and OpenBabel..."
    pip install vina --quiet 2>/dev/null || {
        log "pip install vina failed — trying conda/apt"
        # Try downloading the Vina binary directly
        if ! which vina 2>/dev/null; then
            wget -q https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64 \
                -O /usr/local/bin/vina \
                && chmod +x /usr/local/bin/vina \
                && log "Vina binary installed."
        fi
    }
    # If pip installed the vina Python package, it doesn't provide the binary
    # Try to find vina CLI or use the Python API
    if ! which vina 2>/dev/null; then
        log "WARNING: vina CLI not found — using vina Python API for docking"
    fi

    sudo apt-get install -y openbabel 2>/dev/null || apt-get install -y openbabel 2>/dev/null || \
        pip install openbabel-wheel --quiet 2>/dev/null || \
        log "WARNING: OpenBabel not installed — SMILES→PDBQT conversion may fail"

    pip install biopython --quiet 2>/dev/null || true

    checkpoint_set "DEPS_DONE"
    log "Dependencies ready."
fi

# ── Step 2: Restore FDA compound cache if needed ───────────────────────────────
if [ ! -f "${PHASE2_DATA_DIR}/pubchem_fda.json" ]; then
    log "Restoring FDA compound cache from GCS..."
    gsutil cp "${GCS_BUCKET}/phase2_setup/pubchem_fda.json" \
        "${PHASE2_DATA_DIR}/pubchem_fda.json" 2>/dev/null \
        || log "WARNING: FDA cache not found in GCS"
fi

# ── Step 3: Prepare receptor PDBQT files ──────────────────────────────────────
if ! checkpoint_matches "RECEPTORS_DONE"; then
    log "Preparing receptor PDBQT files for all 6 targets..."
    $PYTHON vina_receptor_prep.py \
        --repo-root "$(git rev-parse --show-toplevel 2>/dev/null || pwd)" \
        2>&1 | tee -a "${OUTPUT_DIR}/phase5.log"

    # Check how many receptors were prepared
    N_RECEPTORS=$(python3 -c "
import json
from pathlib import Path
m = Path('${RECEPTOR_DIR}/receptor_manifest.json')
if m.exists():
    d = json.loads(m.read_text())
    print(len(d))
else:
    print(0)
" 2>/dev/null || echo 0)

    log "Prepared ${N_RECEPTORS}/6 receptors."
    if [ "$N_RECEPTORS" -lt 1 ]; then
        log "ERROR: No receptors prepared. Check vina_receptor_prep.py output."
        exit 1
    fi
    checkpoint_set "RECEPTORS_DONE"
fi

# ── Step 4: Vina screening ─────────────────────────────────────────────────────
if ! checkpoint_matches "VINA_SCREEN_DONE"; then
    # Check if scores already exist in GCS (from parallel VM)
    if gsutil ls "${GCS_BUCKET}/${RUN_ID}/vina_scores.json" 2>/dev/null; then
        log "Vina scores already in GCS — downloading..."
        gsutil cp "${GCS_BUCKET}/${RUN_ID}/vina_scores.json" \
            "${OUTPUT_DIR}/vina_scores.json"
        checkpoint_set "VINA_SCREEN_DONE"
        log "Using existing Vina scores."
    else
        log "Running Vina screen (${VINA_WORKERS} parallel workers)..."
        $PYTHON vina_screen.py 2>&1 | tee -a "${OUTPUT_DIR}/phase5.log"
        checkpoint_set "VINA_SCREEN_DONE"
        upload
        log "Vina screen complete."
    fi
fi

# ── Step 5: Phase 3 re-run with Vina oracle ────────────────────────────────────
if ! checkpoint_matches "VINA_BO_DONE"; then
    log "Running Phase 3 Bayesian optimisation with Vina oracle..."
    $PYTHON phase3_vina_oracle.py \
        --targets LINGO1 PCSK9 KPC3 APEX1 MSH3 CREBBP \
        --rounds 30 \
        2>&1 | tee -a "${OUTPUT_DIR}/phase5.log"
    checkpoint_set "VINA_BO_DONE"
    upload
    log "Vina BO complete."
fi

checkpoint_set "PHASE5_DONE"
upload

log ""
log "=== Phase 5 complete ==="
log "Results:"
log "  Vina scores:   ${OUTPUT_DIR}/vina_scores.json"
log "  BO results:    ${OUTPUT_DIR}/phase3_vina_bo_results.json"
log "  GCS:           ${GCS_BUCKET}/${RUN_ID}/"
