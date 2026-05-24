#!/bin/bash
# ============================================================
# watch_qr_36_37_38.sh
#
# Watches the three spot QRs submitted for phase36/37/38.
# As each QR transitions to ACTIVE, copies code, installs env,
# and launches the experiment in tmux on that VM.
# ============================================================

set -uo pipefail

PROJECT="aegismind-tpu"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -A QR_ZONE=( [aegis-phase36-qr]="us-east5-b" [aegis-phase37-qr]="us-central1-b" [aegis-phase38-qr]="us-east1-d" )
declare -A QR_VM=(   [aegis-phase36-qr]="aegis-phase36" [aegis-phase37-qr]="aegis-phase37" [aegis-phase38-qr]="aegis-phase38" )
declare -A QR_SCRIPT=([aegis-phase36-qr]="phase36_medium_scaling.py" [aegis-phase37-qr]="phase37_int8_gpt.py" [aegis-phase38-qr]="phase38_msh3_retry.py")
declare -A QR_LOG=(  [aegis-phase36-qr]="/tmp/phase36.log" [aegis-phase37-qr]="/tmp/phase37.log" [aegis-phase38-qr]="/tmp/phase38.log")

log() { echo "[$(date -u '+%H:%M:%S UTC')] $*"; }

qr_state() {
    local qr="$1" zone="$2"
    gcloud alpha compute tpus queued-resources describe "$qr" \
        --zone "$zone" --project "$PROJECT" \
        --format 'value(state.state)' 2>/dev/null || echo "UNKNOWN"
}

setup_and_run() {
    local vm="$1" zone="$2" script="$3" logfile="$4"
    log "[$vm] Copying code..."
    gcloud compute tpus tpu-vm scp --recurse \
        "$LOCAL_DIR" "${vm}:~/flashoptim" \
        --zone "$zone" --project "$PROJECT" --worker all 2>&1

    log "[$vm] Installing env..."
    gcloud compute tpus tpu-vm ssh "$vm" --zone "$zone" --project "$PROJECT" --command "
        set -e
        cd ~/flashoptim
        bash setup.sh 2>&1 | tail -5
        pip install rdkit-pypi vina biopython scipy -q 2>&1 | tail -3
        echo 'env ready'
    " 2>&1

    log "[$vm] Launching $script in tmux..."
    gcloud compute tpus tpu-vm ssh "$vm" --zone "$zone" --project "$PROJECT" --command "
        tmux new-session -d -s exp 2>/dev/null || true
        tmux send-keys -t exp \
            'GCS_BUCKET=gs://aegismind-tpu-results python3 ~/flashoptim/${script} 2>&1 | tee ${logfile}' \
            Enter
        echo '${script} launched'
    " 2>&1
    log "[$vm] Running. Monitor: gcloud compute tpus tpu-vm ssh $vm --zone $zone --project $PROJECT --command 'tmux capture-pane -p -t exp | tail -20'"
}

watch_qr() {
    local qr="$1"
    local zone="${QR_ZONE[$qr]}"
    local vm="${QR_VM[$qr]}"
    local script="${QR_SCRIPT[$qr]}"
    local logfile="${QR_LOG[$qr]}"

    log "[$qr] Watching (zone=$zone vm=$vm)..."
    for i in $(seq 1 720); do   # up to 4 hours
        local state
        state=$(qr_state "$qr" "$zone")
        case "$state" in
            ACTIVE)
                log "[$qr] ACTIVE — setting up $vm"
                setup_and_run "$vm" "$zone" "$script" "$logfile"
                return 0
                ;;
            FAILED|DELETING)
                log "ERROR: [$qr] state=$state — giving up"; return 1 ;;
            SUSPENDED)
                log "[$qr] SUSPENDED (preempted) — waiting for GCP to resume" ;;
            *)
                log "  [$qr] state=$state (${i}/720)" ;;
        esac
        sleep 20
    done
    log "ERROR: [$qr] timed out after 4 hours"; return 1
}

log "=== Watching QRs for phase36/37/38 ==="
watch_qr aegis-phase36-qr &
watch_qr aegis-phase37-qr &
watch_qr aegis-phase38-qr &
wait
log "=== All watchers done ==="
