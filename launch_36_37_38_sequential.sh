#!/bin/bash
# ============================================================
# launch_36_37_38_sequential.sh
#
# Retries VM creation every 5 min until any zone opens up,
# then runs phase36 → phase37 → phase38 sequentially on one VM.
# Only needs ONE zone to become available.
#
# Run in a persistent tmux session:
#   tmux new -s aegis36
#   bash analysis/flashoptim_tpu/launch_36_37_38_sequential.sh
# ============================================================

set -uo pipefail

PROJECT="aegismind-tpu"
ACCEL="v6e-8"
RUNTIME="v2-alpha-tpuv6e"
VM_NAME="aegis-exp-$(date -u +%m%d%H%M)"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY_INTERVAL=300

# Zones with confirmed v6e-8 support (showed "Insufficient" not "not found")
ZONES=("us-east5-b" "us-central1-b" "us-east1-d" "europe-west4-a")

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }

# ── Step 1: retry VM creation until a zone opens ─────────────────────────────
log "=== Sequential launcher for phase36 → phase37 → phase38 ==="
log "Retrying VM creation every ${RETRY_INTERVAL}s across zones: ${ZONES[*]}"

ZONE=""
while true; do
    for zone in "${ZONES[@]}"; do
        log "Trying $zone ..."
        if gcloud compute tpus tpu-vm create "$VM_NAME" \
            --zone "$zone" --project "$PROJECT" \
            --accelerator-type "$ACCEL" --version "$RUNTIME" 2>&1; then
            ZONE="$zone"
            break 2
        fi
    done
    log "All zones at capacity — waiting ${RETRY_INTERVAL}s"
    sleep "$RETRY_INTERVAL"
done

log "=== VM $VM_NAME created in $ZONE ==="

# ── Step 2: wait for READY ────────────────────────────────────────────────────
log "Waiting for READY state..."
for i in $(seq 1 30); do
    STATE=$(gcloud compute tpus tpu-vm describe "$VM_NAME" \
        --zone "$ZONE" --project "$PROJECT" \
        --format 'value(state)' 2>/dev/null || echo "UNKNOWN")
    [[ "$STATE" == "READY" ]] && break
    log "  state=$STATE (${i}/30)"; sleep 20
done
log "VM is READY"

# ── Step 3: copy code and install env ────────────────────────────────────────
log "Copying code..."
gcloud compute tpus tpu-vm scp --recurse \
    "$LOCAL_DIR" "${VM_NAME}:~/flashoptim" \
    --zone "$ZONE" --project "$PROJECT" --worker all 2>&1

log "Installing environment..."
gcloud compute tpus tpu-vm ssh "$VM_NAME" --zone "$ZONE" --project "$PROJECT" --command "
    set -e
    cd ~/flashoptim
    bash setup.sh 2>&1 | tail -5
    pip install rdkit-pypi vina biopython scipy -q 2>&1 | tail -3
    echo 'Env ready'
" 2>&1

# ── Step 4: launch all three experiments sequentially in tmux ────────────────
log "Launching phase36 → phase37 → phase38 in tmux..."
gcloud compute tpus tpu-vm ssh "$VM_NAME" --zone "$ZONE" --project "$PROJECT" --command "
    tmux new-session -d -s exps 2>/dev/null || true
    QUEUE='
        echo \"=== Phase 36 start ==\" &&
        GCS_BUCKET=gs://aegismind-tpu-results python3 ~/flashoptim/phase36_medium_scaling.py 2>&1 | tee /tmp/phase36.log &&
        echo \"=== Phase 37 start ==\" &&
        GCS_BUCKET=gs://aegismind-tpu-results python3 ~/flashoptim/phase37_int8_gpt.py 2>&1 | tee /tmp/phase37.log &&
        echo \"=== Phase 38 start ==\" &&
        GCS_BUCKET=gs://aegismind-tpu-results python3 ~/flashoptim/phase38_msh3_retry.py 2>&1 | tee /tmp/phase38.log &&
        echo \"=== All phases complete ==\" ||
        echo \"=== Queue stopped (check logs) ===\"
    '
    tmux send-keys -t exps \"eval '\$QUEUE'\" Enter
    echo 'Queue started in tmux session: exps'
    tmux list-sessions
" 2>&1

log "=== Experiments running on $VM_NAME in $ZONE ==="
log ""
log "Monitor: gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --project $PROJECT --command 'tmux capture-pane -p -t exps | tail -30'"
log "Phase 36 GCS: gsutil ls gs://aegismind-tpu-results/aegis_flashoptim/phase36_medium_scaling/"
log "Phase 37 GCS: gsutil ls gs://aegismind-tpu-results/aegis_flashoptim/phase37_int8_gpt/"
log "Phase 38 GCS: gsutil ls gs://aegismind-tpu-results/aegis_flashoptim/phase38_msh3_retry/"
