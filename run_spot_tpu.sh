#!/bin/bash
# ============================================================
# run_spot_tpu.sh
#
# End-to-end spot TPU launcher for TRC v6e-8 experiments.
# Creates spot queued-resource requests, waits for ACTIVE,
# copies code, installs env, and launches experiments in tmux.
#
# Usage:
#   bash run_spot_tpu.sh <phase> <zone> <script.py>
#
# Examples:
#   bash run_spot_tpu.sh phase36 us-east5-b   phase36_medium_scaling.py
#   bash run_spot_tpu.sh phase37 us-central1-b phase37_int8_gpt.py
#   bash run_spot_tpu.sh phase38 us-east1-d    phase38_msh3_retry.py
#
# To run all three in parallel:
#   bash run_spot_tpu.sh phase36 us-east5-b   phase36_medium_scaling.py &
#   bash run_spot_tpu.sh phase37 us-central1-b phase37_int8_gpt.py &
#   bash run_spot_tpu.sh phase38 us-east1-d    phase38_msh3_retry.py &
#   wait
#
# Environment:
#   PROJECT   GCP project (default: aegismind-tpu)
#   QR_TTL    valid-until-duration for the QR (default: 12h)
#   GCS_BUCKET GCS bucket for results (default: gs://aegismind-tpu-results)
# ============================================================

set -uo pipefail

PHASE="${1:?Usage: run_spot_tpu.sh <phase> <zone> <script.py>}"
ZONE="${2:?Usage: run_spot_tpu.sh <phase> <zone> <script.py>}"
SCRIPT="${3:?Usage: run_spot_tpu.sh <phase> <zone> <script.py>}"

PROJECT="${PROJECT:-aegismind-tpu}"
ACCEL="${ACCEL:-v6e-8}"
RUNTIME="${RUNTIME:-v2-alpha-tpuv6e}"
QR_TTL="${QR_TTL:-12h}"
GCS_BUCKET="${GCS_BUCKET:-gs://aegismind-tpu-results}"

VM_NAME="aegis-${PHASE}"
QR_NAME="aegis-${PHASE}-qr"
LOGFILE="/tmp/${PHASE}.log"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[$(date -u '+%H:%M:%S UTC')] [${PHASE}] $*"; }

# ── Step 1: clean up any stale QR / VM ──────────────────────────────────────
cleanup_stale() {
    local existing_state
    existing_state=$(gcloud alpha compute tpus queued-resources describe "$QR_NAME" \
        --zone "$ZONE" --project "$PROJECT" \
        --format 'value(state.state)' 2>/dev/null || echo "NONE")

    if [[ "$existing_state" != "NONE" ]]; then
        log "Deleting stale QR $QR_NAME (state=$existing_state)..."
        gcloud alpha compute tpus queued-resources delete "$QR_NAME" \
            --zone "$ZONE" --project "$PROJECT" --quiet 2>/dev/null || true
        sleep 5
    fi

    if gcloud compute tpus tpu-vm describe "$VM_NAME" \
           --zone "$ZONE" --project "$PROJECT" &>/dev/null 2>&1; then
        log "Deleting stale VM $VM_NAME..."
        gcloud compute tpus tpu-vm delete "$VM_NAME" \
            --zone "$ZONE" --project "$PROJECT" --quiet 2>/dev/null || true
    fi
}

# ── Step 2: submit spot queued-resource request ──────────────────────────────
submit_qr() {
    log "Submitting spot QR → $VM_NAME in $ZONE (TTL $QR_TTL)..."
    gcloud alpha compute tpus queued-resources create "$QR_NAME" \
        --node-id "$VM_NAME" \
        --zone "$ZONE" \
        --project "$PROJECT" \
        --accelerator-type "$ACCEL" \
        --runtime-version "$RUNTIME" \
        --provisioning-model=SPOT \
        --valid-until-duration "$QR_TTL" \
        --async 2>&1
    log "QR submitted. Waiting for capacity..."
}

# ── Step 3: poll until ACTIVE (handles WAITING_FOR_RESOURCES → PROVISIONING → ACTIVE)
wait_active() {
    for i in $(seq 1 2160); do   # 12 h max at 20 s intervals
        local state
        state=$(gcloud alpha compute tpus queued-resources describe "$QR_NAME" \
            --zone "$ZONE" --project "$PROJECT" \
            --format 'value(state.state)' 2>/dev/null || echo "UNKNOWN")
        case "$state" in
            ACTIVE)
                log "QR ACTIVE — VM $VM_NAME ready"; return 0 ;;
            FAILED|DELETING)
                log "ERROR: QR state=$state"; return 1 ;;
            SUSPENDED)
                log "SUSPENDED (preempted by GCP) — waiting for automatic resume" ;;
            WAITING_FOR_RESOURCES|PROVISIONING|ACCEPTED|UNKNOWN)
                [[ $((i % 15)) -eq 0 ]] && log "state=$state (~$((i * 20 / 60))min elapsed)" ;;
        esac
        sleep 20
    done
    log "ERROR: timed out after 12 hours"; return 1
}

# ── Step 4: copy code to VM ──────────────────────────────────────────────────
copy_code() {
    log "Copying code to $VM_NAME..."
    gcloud compute tpus tpu-vm scp --recurse \
        "$LOCAL_DIR" "${VM_NAME}:~/flashoptim" \
        --zone "$ZONE" --project "$PROJECT" --worker all 2>&1
}

# ── Step 5: install environment ───────────────────────────────────────────────
install_env() {
    log "Installing environment..."
    gcloud compute tpus tpu-vm ssh "$VM_NAME" \
        --zone "$ZONE" --project "$PROJECT" --command "
            set -e
            cd ~/flashoptim
            bash setup.sh 2>&1 | tail -5
            pip install rdkit-pypi vina biopython scipy -q 2>&1 | tail -3
            echo 'env ready'
        " 2>&1
}

# ── Step 6: launch experiment in tmux ─────────────────────────────────────────
launch_experiment() {
    log "Launching $SCRIPT in tmux..."
    gcloud compute tpus tpu-vm ssh "$VM_NAME" \
        --zone "$ZONE" --project "$PROJECT" --command "
            tmux new-session -d -s exp 2>/dev/null || true
            tmux send-keys -t exp \
                'GCS_BUCKET=${GCS_BUCKET} python3 ~/flashoptim/${SCRIPT} 2>&1 | tee ${LOGFILE}' \
                Enter
            echo '${SCRIPT} launched in tmux'
        " 2>&1
    log "Running. Monitor with:"
    log "  gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE --project $PROJECT --command 'tmux capture-pane -p -t exp | tail -30'"
}

# ── Main ─────────────────────────────────────────────────────────────────────
log "=== Starting $PHASE | zone=$ZONE | script=$SCRIPT ==="

cleanup_stale
submit_qr
wait_active || exit 1
copy_code
install_env
launch_experiment

log "=== $PHASE complete setup. Experiment running on $VM_NAME ==="
