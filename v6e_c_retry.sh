#!/bin/bash
# ============================================================
# v6e-c retry daemon — recreate preempted VM and run all
# pending experiments: LR ablations + C_restart + A_explicit_restart
#
# Run locally in a tmux/screen session:
#   tmux new -s v6e_c_retry
#   bash analysis/flashoptim_tpu/v6e_c_retry.sh
#
# Keeps trying VM creation every 5 min until capacity is available,
# then immediately sets up env and runs the full experiment queue.
#
# Queue (sequential in tmux, in priority order):
#   1. c_restart.sh       — BF16 512-dim + restart (highest scientific value)
#   2. a_explicit_restart.sh — FP32 256-dim + corrected restart (Phase 4b fix)
#   3. LR ablation A_lr1  — Condition A at LR=5e-4 (sensitivity)
#   4. LR ablation A_lr2  — Condition A at LR=5e-5 (sensitivity)
# ============================================================

set -euo pipefail

VM_NAME="aegis-node-v6e-c"
ZONE="us-east1-d"
PROJECT="aegismind-tpu"
ACCEL="v6e-8"
RUNTIME="v2-alpha-tpuv6e"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY_INTERVAL=300   # 5 minutes

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }

# ── Step 1: create VM (retry until capacity available) ─────────────────────
log "=== v6e-c retry daemon started ==="
log "Queue: c_restart → a_explicit_restart → lr_ablation_A_lr1 → lr_ablation_A_lr2"
log "Retrying VM creation every ${RETRY_INTERVAL}s until capacity available"

while true; do
    log "Attempting to create ${VM_NAME} in ${ZONE}..."
    if gcloud compute tpus tpu-vm create "$VM_NAME" \
        --zone "$ZONE" \
        --project "$PROJECT" \
        --accelerator-type "$ACCEL" \
        --version "$RUNTIME" \
        2>&1; then
        log "VM created successfully."
        break
    else
        log "Capacity not available — waiting ${RETRY_INTERVAL}s before retry"
        sleep "$RETRY_INTERVAL"
    fi
done

# ── Step 2: wait for VM to be READY ────────────────────────────────────────
log "Waiting for VM to reach READY state..."
for i in $(seq 1 30); do
    STATUS=$(gcloud compute tpus tpu-vm describe "$VM_NAME" \
        --zone "$ZONE" --project "$PROJECT" \
        --format 'value(state)' 2>/dev/null || echo "UNKNOWN")
    log "  VM state: $STATUS"
    [ "$STATUS" = "READY" ] && break
    sleep 20
done

# ── Step 3: copy flashoptim directory to VM ─────────────────────────────────
log "Copying flashoptim code to VM..."
gcloud compute tpus tpu-vm scp --recurse \
    "$LOCAL_DIR" "${VM_NAME}:~/flashoptim" \
    --zone "$ZONE" --project "$PROJECT" --worker all 2>&1
log "Code uploaded."

# ── Step 4: install environment and start experiment queue ──────────────────
log "Installing environment and launching experiment queue..."
gcloud compute tpus tpu-vm ssh "$VM_NAME" \
    --zone "$ZONE" --project "$PROJECT" \
    --command "
        set -e
        cd ~/flashoptim
        chmod +x setup.sh c_restart.sh a_explicit_restart.sh lr_ablation.sh

        # Install deps (idempotent)
        bash setup.sh 2>&1 | tail -5

        # Start full queue in tmux (SSH-safe)
        tmux new-session -d -s experiments 2>/dev/null || true

        # Build the sequential command chain
        # Each && ensures the next runs only if the previous succeeded.
        # If a script fails, the chain stops (prevents wasted compute).
        QUEUE='
            echo \"[QUEUE] Starting c_restart\" &&
            GCS_BUCKET=gs://aegismind-tpu-results bash ~/flashoptim/c_restart.sh &&
            echo \"[QUEUE] c_restart done. Starting a_explicit_restart\" &&
            GCS_BUCKET=gs://aegismind-tpu-results bash ~/flashoptim/a_explicit_restart.sh &&
            echo \"[QUEUE] a_explicit_restart done. Starting lr ablations\" &&
            GCS_BUCKET=gs://aegismind-tpu-results LR=5e-4 bash ~/flashoptim/lr_ablation.sh &&
            GCS_BUCKET=gs://aegismind-tpu-results LR=5e-5 bash ~/flashoptim/lr_ablation.sh &&
            echo \"[QUEUE] All experiments complete.\"
        '

        tmux send-keys -t experiments \"cd ~/flashoptim && eval \\\"\$QUEUE\\\" 2>&1 | tee /tmp/experiment_queue.log\" Enter

        echo 'Experiment queue started in tmux session: experiments'
        tmux list-sessions
    " 2>&1

log "=== v6e-c fully restored. Experiment queue running. ==="
log ""
log "Monitor progress:"
log "  gcloud compute tpus tpu-vm ssh ${VM_NAME} --zone ${ZONE} --project ${PROJECT} \\"
log "    --command 'tmux capture-pane -p -t experiments | tail -30'"
log ""
log "Check results:"
log "  gsutil ls gs://aegismind-tpu-results/aegis_flashoptim/c_restart/"
log "  gsutil ls gs://aegismind-tpu-results/aegis_flashoptim/a_explicit_restart/"
