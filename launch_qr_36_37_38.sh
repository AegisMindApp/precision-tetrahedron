#!/bin/bash
# ============================================================
# launch_qr_36_37_38.sh
#
# Uses GCP Queued Resource API to request v6e-8 TPUs for
# phase36/37/38. GCP fulfils the requests when capacity
# becomes available — no polling loop required.
#
# Each phase gets a QR in a different preferred zone to
# maximise the chance of parallel fulfilment.
#
# Usage: bash analysis/flashoptim_tpu/launch_qr_36_37_38.sh
# ============================================================

set -uo pipefail

PROJECT="aegismind-tpu"
ACCEL="v6e-8"
RUNTIME="v2-alpha-tpuv6e"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Zones with confirmed v6e-8 capacity (showed "Insufficient" not "not found")
# Assign a primary zone per phase to spread across zones
ZONE36="us-east5-b"
ZONE37="us-central1-b"
ZONE38="us-east1-d"

# QR names (must be unique per zone per project)
QR36="aegis-phase36-qr"
QR37="aegis-phase37-qr"
QR38="aegis-phase38-qr"

PHASES=("phase36" "phase37" "phase38")
QRS=("$QR36" "$QR37" "$QR38")
ZONES_ASSIGNED=("$ZONE36" "$ZONE37" "$ZONE38")
SCRIPTS=("phase36_medium_scaling.py" "phase37_int8_gpt.py" "phase38_msh3_retry.py")
LOGS=("/tmp/phase36.log" "/tmp/phase37.log" "/tmp/phase38.log")

log() { echo "[$(date -u '+%H:%M:%S UTC')] $*"; }

# ── Submit a queued resource request ─────────────────────────────────────────
submit_qr() {
    local qr="$1"; local vm="$2"; local zone="$3"

    # Delete any stale QR with same name
    local state
    state=$(gcloud compute tpus queued-resources describe "$qr" \
        --zone "$zone" --project "$PROJECT" \
        --format 'value(state.state)' 2>/dev/null || echo "NONE")

    if [[ "$state" != "NONE" ]]; then
        log "  Deleting stale QR $qr (state=$state)..."
        gcloud compute tpus queued-resources delete "$qr" \
            --zone "$zone" --project "$PROJECT" --quiet 2>/dev/null || true
    fi

    # Also delete stale VM if it exists
    if gcloud compute tpus tpu-vm describe "$vm" --zone "$zone" \
           --project "$PROJECT" &>/dev/null 2>&1; then
        log "  Deleting stale VM $vm..."
        gcloud compute tpus tpu-vm delete "$vm" --zone "$zone" \
            --project "$PROJECT" --quiet 2>/dev/null || true
    fi

    log "Submitting QR $qr → $vm in $zone"
    gcloud compute tpus queued-resources create "$qr" \
        --node-id "$vm" \
        --zone "$zone" \
        --project "$PROJECT" \
        --accelerator-type "$ACCEL" \
        --runtime-version "$RUNTIME" 2>&1
}

# ── Poll until QR is ACTIVE (or FAILED) ──────────────────────────────────────
wait_qr_active() {
    local qr="$1"; local zone="$2"; local phase="$3"
    log "[$phase] Waiting for QR $qr to become ACTIVE..."
    for i in $(seq 1 360); do   # max ~2 hours
        local state
        state=$(gcloud compute tpus queued-resources describe "$qr" \
            --zone "$zone" --project "$PROJECT" \
            --format 'value(state.state)' 2>/dev/null || echo "UNKNOWN")
        case "$state" in
            ACTIVE)      log "[$phase] QR $qr is ACTIVE"; return 0 ;;
            FAILED)      log "ERROR: [$phase] QR $qr FAILED"; return 1 ;;
            SUSPENDING|SUSPENDED) log "ERROR: [$phase] QR $qr $state"; return 1 ;;
        esac
        log "  [$phase] QR state=$state (${i}/360, ~$((i*20/60))min elapsed)"
        sleep 20
    done
    log "ERROR: [$phase] QR $qr timed out after 2 hours"
    return 1
}

# ── Setup VM and launch experiment ───────────────────────────────────────────
setup_and_run() {
    local vm="$1"; local zone="$2"; local script="$3"; local logfile="$4"

    log "[$vm] Copying code..."
    gcloud compute tpus tpu-vm scp --recurse \
        "$LOCAL_DIR" "${vm}:~/flashoptim" \
        --zone "$zone" --project "$PROJECT" --worker all 2>&1

    log "[$vm] Installing env and launching $script ..."
    gcloud compute tpus tpu-vm ssh "$vm" --zone "$zone" --project "$PROJECT" --command "
        set -e
        cd ~/flashoptim
        bash setup.sh 2>&1 | tail -5
        pip install rdkit-pypi vina biopython scipy -q 2>&1 | tail -3
        tmux new-session -d -s exp 2>/dev/null || true
        tmux send-keys -t exp \
            'GCS_BUCKET=gs://aegismind-tpu-results python3 ~/flashoptim/${script} 2>&1 | tee ${logfile}' \
            Enter
        echo '[${vm}] ${script} launched in tmux'
    " 2>&1

    log "[$vm] Running. Monitor:"
    log "  gcloud compute tpus tpu-vm ssh $vm --zone $zone --project $PROJECT --command 'tmux capture-pane -p -t exp | tail -20'"
}

# ── Per-phase worker (runs in background) ────────────────────────────────────
run_phase() {
    local i="$1"
    local phase="${PHASES[$i]}"
    local qr="${QRS[$i]}"
    local zone="${ZONES_ASSIGNED[$i]}"
    local script="${SCRIPTS[$i]}"
    local logfile="${LOGS[$i]}"
    local vm="aegis-${phase}"

    submit_qr "$qr" "$vm" "$zone" || { log "ERROR: failed to submit QR for $phase"; return 1; }
    wait_qr_active "$qr" "$zone" "$phase" || { log "ERROR: QR never became ACTIVE for $phase"; return 1; }
    setup_and_run "$vm" "$zone" "$script" "$logfile"
}

# ── Main ─────────────────────────────────────────────────────────────────────
log "=== Submitting Queued Resource requests for phase36/37/38 ==="
log "  phase36 → $ZONE36"
log "  phase37 → $ZONE37"
log "  phase38 → $ZONE38"

run_phase 0 &
run_phase 1 &
run_phase 2 &
wait

log "=== All phases launched. Monitor GCS: ==="
log "  gsutil ls gs://aegismind-tpu-results/aegis_flashoptim/phase36_medium_scaling/"
log "  gsutil ls gs://aegismind-tpu-results/aegis_flashoptim/phase37_int8_gpt/"
log "  gsutil ls gs://aegismind-tpu-results/aegis_flashoptim/phase38_msh3_retry/"
