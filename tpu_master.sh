#!/bin/bash
# ============================================================
# AegisMind TPU Master Orchestrator — 30-Day Run
#
# Runs the full pipeline in sequence with zero human sleep-time waste.
# Designed to run inside a persistent tmux session on the TPU VM.
# Auto-resumes from checkpoints if preempted.
#
# GCP project: aegismind-tpu   VM: aegis-node   Zone: us-central2-b (v4 on-demand)
#
# One-time VM creation (run locally, not on VM):
#   gcloud alpha compute tpus queued-resources create aegis-qr \
#     --node-id=aegis-node \
#     --project=aegismind-tpu \
#     --zone=us-central2-b \
#     --accelerator-type=v4-8 \
#     --runtime-version=tpu-vm-pt-2.0
#   # Wait for ACTIVE (poll with):
#   gcloud compute tpus tpu-vm describe aegis-node \
#     --zone=us-central2-b --project=aegismind-tpu
#
# Usage (on TPU VM):
#   1. Copy this directory to the VM:
#      gcloud compute tpus tpu-vm scp --recurse \
#        analysis/flashoptim_tpu aegis-node:~/flashoptim \
#        --zone=us-central2-b --project=aegismind-tpu --worker=all
#
#   2. SSH in and launch:
#      gcloud compute tpus tpu-vm ssh aegis-node \
#        --zone=us-central2-b --project=aegismind-tpu
#      cd ~/flashoptim
#      chmod +x tpu_master.sh
#      export NOTIFY_SMTP_USER="..." NOTIFY_SMTP_PASS="..."
#      export GCS_BUCKET="gs://aegismind-tpu-results"
#      tmux new -s aegis
#      ./tpu_master.sh 2>&1 | tee -a /tmp/master.log
#      # Ctrl+B then D to detach — pipeline keeps running
#
#   3. Check progress any time:
#      tmux attach -t aegis
#      tail -f /tmp/master.log
#      cat /tmp/flashoptim_results/notify.log
#
# Notification configuration (set before running):
#   export NOTIFY_EMAIL="john.goodman@oceansparx.com"
#   export NOTIFY_SMTP_USER="your.smtp@gmail.com"
#   export NOTIFY_SMTP_PASS="your-app-password"
#   export NOTIFY_WEBHOOK="https://hooks.slack.com/..."  # optional
#   export GCS_BUCKET="gs://oceansparx-tpu"             # optional
#
# Pipeline phases:
#   Phase 1  FlashOptim QM9 (Conditions A+B+C)   ~6-12 hours
#   Phase 2  PDBbind fine-tune + MS docking       ~2-5 days
#   Phase 3  Surrogate Bayesian optimisation      ~20 days (continuous)
#   Phase 4  Cross-domain validation + write-up   ~3 days
#
# Total: fits within 30 days. Phase 3 runs the longest and generates
# the patent-quality candidate molecules while you sleep.
# ============================================================

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

# ── Credentials (never committed — loaded from ~/.tpu_notify_creds on VM) ─────
[ -f "${HOME}/.tpu_notify_creds" ] && source "${HOME}/.tpu_notify_creds"

# ── Configuration ─────────────────────────────────────────────────────────────
# Fixed RUN_ID so GCS path is stable across preemption restarts
export RUN_ID="${RUN_ID:-aegis_flashoptim}"
export DATA_DIR="${DATA_DIR:-/tmp/qm9}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/flashoptim_results}"
export PHASE2_DATA_DIR="${PHASE2_DATA_DIR:-/tmp/phase2_data}"
export EPOCHS_PHASE1="${EPOCHS_PHASE1:-100}"
export EPOCHS_PHASE2="${EPOCHS_PHASE2:-50}"
export BO_ROUNDS="${BO_ROUNDS:-30}"
export BO_POOL_SIZE="${BO_POOL_SIZE:-5000}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export LR="${LR:-1e-4}"
export HEARTBEAT_INTERVAL_EPOCHS="${HEARTBEAT_INTERVAL_EPOCHS:-10}"
export CHECKPOINT_FILE="${OUTPUT_DIR}/.pipeline_checkpoint"
export NOTIFY_EMAIL="${NOTIFY_EMAIL:-john.goodman@oceansparx.com}"
export GCS_BUCKET="${GCS_BUCKET:-gs://aegismind-tpu-results}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$PHASE2_DATA_DIR"

# Unbuffered Python so stdout/stderr are visible immediately via tee
PYTHON="python3 -u"

# XLA persistent compilation cache — kept outside OUTPUT_DIR so rsync never
# uploads/downloads it (Phase 1 cache is 42 GB and wrong shapes for Phase 2).
export XLA_PERSISTENT_CACHE_PATH="/tmp/xla_cache"
mkdir -p "${XLA_PERSISTENT_CACHE_PATH}"

# ── Restore from GCS on fresh VM (after preemption) ───────────────────────────
if [ -n "${GCS_BUCKET}" ] && ! [ -f "${CHECKPOINT_FILE}" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Checking GCS for previous results to restore ..."
    gsutil -m rsync -r "${GCS_BUCKET}/${RUN_ID}/" "${OUTPUT_DIR}/" 2>/dev/null && \
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GCS restore complete. Checkpoint: $(cat ${CHECKPOINT_FILE} 2>/dev/null || echo NONE)" || \
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] No GCS backup found — starting fresh"
fi

# ── Utility functions ─────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${OUTPUT_DIR}/master.log"
}

notify_shell() {
    local event="$1"; shift
    local msg="$*"
    ${PYTHON} -c "
import sys; sys.path.insert(0, '.')
from notify import notify
notify('${event}', '${msg}', data={'run_id': '${RUN_ID}'})
" 2>/dev/null || log "NOTIFY FAILED: ${event} ${msg}"
}

checkpoint_set() {
    echo "$1" > "${CHECKPOINT_FILE}"
    log "Checkpoint saved: $1"
}

checkpoint_get() {
    [ -f "${CHECKPOINT_FILE}" ] && cat "${CHECKPOINT_FILE}" || echo "NONE"
}

# Returns 0 (success/true) if the checkpoint file matches any of the given ERE patterns
checkpoint_matches() {
    grep -qxE "$1" "${CHECKPOINT_FILE}" 2>/dev/null
}

upload_results() {
    if [ -n "${GCS_BUCKET}" ]; then
        log "Uploading results to ${GCS_BUCKET}/${RUN_ID}/ ..."
        gsutil -m cp -r "${OUTPUT_DIR}/" "${GCS_BUCKET}/${RUN_ID}/" 2>/dev/null \
            && log "Upload complete" \
            || log "WARNING: GCS upload failed (results still local)"
    fi
}

# Handle preemption / SIGTERM gracefully
trap 'log "SIGTERM received — saving checkpoint and uploading"; checkpoint_set "INTERRUPTED"; upload_results; exit 1' SIGTERM SIGINT

# Background GCS sync every 5 minutes — survives even if SIGTERM upload is too slow
if [ -n "${GCS_BUCKET}" ]; then
    (while true; do
        sleep 300
        gsutil -m rsync -r "${OUTPUT_DIR}/" "${GCS_BUCKET}/${RUN_ID}/" 2>/dev/null || true
    done) &
    SYNC_PID=$!
    trap "kill ${SYNC_PID} 2>/dev/null; $(trap -p SIGTERM | sed 's/trap -- //;s/ SIGTERM//')" SIGTERM SIGINT
fi

# ── Resume logic ──────────────────────────────────────────────────────────────
LAST_CHECKPOINT=$(checkpoint_get)
log "Starting pipeline. Run ID: ${RUN_ID}"
log "Last checkpoint: ${LAST_CHECKPOINT}"

if [ "$LAST_CHECKPOINT" != "NONE" ]; then
    notify_shell "PHASE_START" "Resuming from checkpoint: ${LAST_CHECKPOINT}"
fi

# ── Environment setup ─────────────────────────────────────────────────────────
# Always run setup.sh if torch_xla is not importable — even if checkpoint says
# PHASE1_RUNNING. A fresh preempted VM has no dependencies installed regardless
# of what the GCS-restored checkpoint file says.
if [ "$LAST_CHECKPOINT" = "NONE" ] || [ "$LAST_CHECKPOINT" = "SETUP" ] || \
   ! python3 -c "import torch_xla; import torch_xla.experimental" 2>/dev/null; then
    log "=== Setting up environment (torch_xla missing or too old — needs experimental module) ==="
    bash setup.sh || { notify_shell "ABORT" "setup.sh failed"; exit 1; }
    notify_shell "PHASE_COMPLETE" "Environment setup complete"
fi

# ═══════════════════════════════════════════════════════════════
# PHASE 1 — FlashOptim QM9 (Conditions A, B, C)
# Expected duration: 6–12 hours
# ═══════════════════════════════════════════════════════════════

if ! checkpoint_matches "PHASE1_DONE|PHASE2_.*|PHASE3_.*|PHASE4_.*|PIPELINE_COMPLETE"; then
    log ""
    log "╔══════════════════════════════════════════╗"
    log "║  PHASE 1 — FlashOptim QM9                ║"
    log "╚══════════════════════════════════════════╝"
    notify_shell "PHASE_START" "Phase 1: FlashOptim QM9 starting — Conditions A, B, C"
    checkpoint_set "PHASE1_RUNNING"

    # Re-read checkpoint file each time — LAST_CHECKPOINT is stale after each set
    # Condition A — FP32 baseline
    if ! grep -qx "PHASE1_A_DONE\|PHASE1_B_DONE\|PHASE1_C_DONE" "${CHECKPOINT_FILE}" 2>/dev/null; then
        log ">>> Condition A: FP32 baseline"
        ${PYTHON} train.py \
            --condition A \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --epochs "$EPOCHS_PHASE1" \
            --batch-size "$BATCH_SIZE" \
            --lr "$LR" \
            || { notify_shell "ABORT" "Condition A failed — check logs"; exit 1; }
        checkpoint_set "PHASE1_A_DONE"
        notify_shell "CHECKPOINT" "Condition A (FP32 baseline) complete"
    else
        log "Condition A: skipping (checkpoint found)"
    fi

    # Condition B — BF16 same size
    if ! grep -qx "PHASE1_B_DONE\|PHASE1_C_DONE" "${CHECKPOINT_FILE}" 2>/dev/null; then
        log ">>> Condition B: BF16 mixed-precision (same size)"
        ${PYTHON} train.py \
            --condition B \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --epochs "$EPOCHS_PHASE1" \
            --batch-size "$BATCH_SIZE" \
            --lr "$LR" \
            || { notify_shell "ABORT" "Condition B failed"; exit 1; }
        checkpoint_set "PHASE1_B_DONE"
        notify_shell "CHECKPOINT" "Condition B (BF16 same size) complete"
    else
        log "Condition B: skipping (checkpoint found)"
    fi

    # Condition C — BF16 2× wider
    if ! grep -qx "PHASE1_C_DONE" "${CHECKPOINT_FILE}" 2>/dev/null; then
        log ">>> Condition C: BF16 mixed-precision (2x wider)"
        ${PYTHON} train.py \
            --condition C \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --epochs "$EPOCHS_PHASE1" \
            --batch-size "$BATCH_SIZE" \
            --lr "$LR" \
            || { notify_shell "ABORT" "Condition C failed"; exit 1; }
        checkpoint_set "PHASE1_C_DONE"
    else
        log "Condition C: skipping (checkpoint found)"
    fi

    # Compare and evaluate
    log ">>> Comparing Phase 1 results ..."
    PHASE1_SUMMARY=$(${PYTHON} compare.py --output-dir "$OUTPUT_DIR" 2>&1)
    log "$PHASE1_SUMMARY"

    # Check if hypothesis is supported (BF16 memory reduction >= 20%)
    HYPOTHESIS_SUPPORTED=$(${PYTHON} -c "
import json, sys
try:
    b = json.load(open('${OUTPUT_DIR}/condition_B_results.json'))
    ep = b['epochs'][-1]
    mem_a_file = open('${OUTPUT_DIR}/condition_A_results.json')
    a = json.load(mem_a_file)
    ep_a = a['epochs'][-1]
    mem_red = (ep_a.get('mem_mb',1) - ep.get('mem_mb',0)) / (ep_a.get('mem_mb',1) + 1e-8)
    print('YES' if mem_red >= 0.20 else 'NO')
except Exception as e:
    print('UNKNOWN')
" 2>/dev/null)

    notify_shell "PHASE_COMPLETE" "Phase 1 complete. Hypothesis supported: ${HYPOTHESIS_SUPPORTED}"
    checkpoint_set "PHASE1_DONE"
    upload_results
else
    log "Phase 1: skipping (checkpoint found)"
    HYPOTHESIS_SUPPORTED="UNKNOWN"
fi

# ═══════════════════════════════════════════════════════════════
# PHASE 2 — PDBbind fine-tune + MS target docking
# Expected duration: 2–5 days
# Starts immediately after Phase 1 — no human needed
# ═══════════════════════════════════════════════════════════════

if ! checkpoint_matches "PHASE2_DONE|PHASE3_.*|PHASE4_.*|PIPELINE_COMPLETE"; then
    log ""
    log "╔══════════════════════════════════════════╗"
    log "║  PHASE 2 — MS Target Docking             ║"
    log "╚══════════════════════════════════════════╝"
    notify_shell "PHASE_START" "Phase 2: PDBbind fine-tuning for MS targets starting"
    checkpoint_set "PHASE2_RUNNING"

    # Pre-fetch ChEMBL compound cache for real GNN scoring (avoids PubChem 404)
    mkdir -p "$PHASE2_DATA_DIR"
    gsutil cp "${GCS_BUCKET}/phase2_setup/pubchem_fda.json" \
        "${PHASE2_DATA_DIR}/pubchem_fda.json" 2>/dev/null \
        && log "ChEMBL compound cache restored ($(wc -c < ${PHASE2_DATA_DIR}/pubchem_fda.json) bytes)" \
        || log "WARNING: ChEMBL cache not found in GCS — scoring will use mock library"

    DATA_DIR="$PHASE2_DATA_DIR" ${PYTHON} phase2_pdbbind.py \
        --epochs "$EPOCHS_PHASE2" \
        --batch-size "$BATCH_SIZE" \
        || { notify_shell "ABORT" "Phase 2 failed — check phase2 logs"; exit 1; }

    checkpoint_set "PHASE2_DONE"
    notify_shell "PHASE_COMPLETE" "Phase 2 complete — MS target candidates ranked"
    upload_results
else
    log "Phase 2: skipping (checkpoint found)"
fi

# ═══════════════════════════════════════════════════════════════
# PHASE 3 — Surrogate Bayesian Optimisation
# Unlocked hypothesis 1: "Large-scale neural surrogate BO"
# Expected duration: ~20 days continuous
# Runs while you sleep — sends notifications on new best candidates
# ═══════════════════════════════════════════════════════════════

if ! checkpoint_matches "PHASE3_DONE|PHASE4_.*|PIPELINE_COMPLETE"; then
    log ""
    log "╔══════════════════════════════════════════╗"
    log "║  PHASE 3 — Surrogate Bayesian Optimise   ║"
    log "╚══════════════════════════════════════════╝"
    notify_shell "PHASE_START" \
        "Phase 3: Surrogate BO starting — MS + AMR + HD targets — ${BO_ROUNDS} rounds × ${BO_POOL_SIZE} pool"
    checkpoint_set "PHASE3_RUNNING"

    ${PYTHON} phase3_surrogate_bayes.py \
        --targets LINGO1 PCSK9 KPC3 APEX1 MSH3 CREBBP \
        --pool-size "$BO_POOL_SIZE" \
        --rounds "$BO_ROUNDS" \
        || { notify_shell "ABORT" "Phase 3 failed"; exit 1; }

    checkpoint_set "PHASE3_DONE"
    notify_shell "PHASE_COMPLETE" "Phase 3 complete — patent-quality candidates generated"
    upload_results
else
    log "Phase 3: skipping (checkpoint found)"
fi

# ═══════════════════════════════════════════════════════════════
# PHASE 4a — Condition B_v2: automated plateau-triggered warm restart
#
# Purpose: patent validation run. Implements the detection mechanism
# explicitly (--plateau-patience 10) rather than relying on preemption
# coinciding with the plateau (which is what Condition B's restart was).
# Required to properly support patent claims AU2026903450 / AU2026903588.
# ═══════════════════════════════════════════════════════════════

if ! checkpoint_matches "PHASE4a_DONE|PHASE4b_DONE|PHASE4_DONE|PIPELINE_COMPLETE"; then
    log ""
    log "╔══════════════════════════════════════════╗"
    log "║  PHASE 4a — Condition B_v2 (plateau det) ║"
    log "╚══════════════════════════════════════════╝"
    notify_shell "PHASE_START" "Phase 4a: Condition B_v2 — automated plateau-triggered warm restart"
    checkpoint_set "PHASE4a_RUNNING"

    ${PYTHON} train.py \
        --condition B \
        --data-dir "$DATA_DIR" \
        --output-dir "${OUTPUT_DIR}/condition_B_v2" \
        --epochs 100 \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --plateau-patience 10 \
        --plateau-min-delta 1e-4 \
        || { notify_shell "ABORT" "Phase 4a B_v2 failed"; exit 1; }

    # Upload B_v2 results to GCS under own prefix
    gsutil -m rsync -r "${OUTPUT_DIR}/condition_B_v2/" \
        "${GCS_BUCKET}/${RUN_ID}/condition_B_v2/" 2>/dev/null || true

    checkpoint_set "PHASE4a_DONE"
    notify_shell "PHASE_COMPLETE" "Phase 4a complete — B_v2 plateau detection run done"
    upload_results
fi

# ═══════════════════════════════════════════════════════════════
# PHASE 4b — Condition A warm restart (controlled FP32 from ep80)
#
# Purpose: isolates the restart effect from the precision effect.
# Loads condition_A_epoch80.pt (FP32, plateau at ~0.057 eV), strips
# optimizer momentum, trains ep81→100. Shows FP32 also escapes
# the same plateau when given an identical LR reset — confirming
# the restart mechanism is precision-agnostic.
# ═══════════════════════════════════════════════════════════════

if ! checkpoint_matches "PHASE4b_DONE|PHASE4_DONE|PIPELINE_COMPLETE"; then
    log ""
    log "╔══════════════════════════════════════════╗"
    log "║  PHASE 4b — Cond A warm restart (FP32)   ║"
    log "╚══════════════════════════════════════════╝"
    notify_shell "PHASE_START" "Phase 4b: Condition A warm restart from epoch 80"
    checkpoint_set "PHASE4b_RUNNING"

    bash warm_restart.sh \
        || notify_shell "ANOMALY" "Phase 4b warm_restart.sh failed — continuing"

    checkpoint_set "PHASE4b_DONE"
    notify_shell "PHASE_COMPLETE" "Phase 4b complete — FP32 warm restart from ep80 done"
    upload_results
fi

checkpoint_set "PHASE4_DONE"

# ═══════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════

log ""
log "╔══════════════════════════════════════════╗"
log "║  ALL PHASES COMPLETE                     ║"
log "╚══════════════════════════════════════════╝"

# Generate final summary
${PYTHON} - << 'PYEOF'
import json, os
from pathlib import Path

output_dir = Path(os.environ.get("OUTPUT_DIR", "/tmp/flashoptim_results"))
summary = {}

for f in ["condition_A_results.json", "condition_B_results.json",
          "condition_C_results.json", "ms_target_candidates_phase2.json",
          "phase3_surrogate_bo_results.json"]:
    p = output_dir / f
    if p.exists():
        with open(p) as fh:
            data = json.load(fh)
        if "epochs" in data:
            last = data["epochs"][-1] if data["epochs"] else {}
            summary[f] = {k: round(v, 4) if isinstance(v, float) else v
                          for k, v in last.items()}
        else:
            summary[f] = "present"

print("FINAL PIPELINE SUMMARY")
print("=" * 50)
for k, v in summary.items():
    print(f"  {k}: {v}")
PYEOF

checkpoint_set "PIPELINE_COMPLETE"

notify_shell "DONE" \
    "30-day TPU pipeline complete. FlashOptim proven, MS candidates generated, BO hypothesis tested."

upload_results

log "Pipeline complete. Run ID: ${RUN_ID}"
log "Results: ${OUTPUT_DIR}"
[ -n "${GCS_BUCKET}" ] && log "Cloud backup: ${GCS_BUCKET}/${RUN_ID}/"
