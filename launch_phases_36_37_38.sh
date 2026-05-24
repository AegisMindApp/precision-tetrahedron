#!/bin/bash
# ============================================================
# launch_phases_36_37_38.sh
#
# Launches three parallel v6e-8 TPU VMs, one per experiment:
#   VM 1 (phase36): MEDIUM GPT scaling-law fix (80ep, BLOCK=256)
#   VM 2 (phase37): INT8 QAT on GPT-6L/TinyShakespeare (tetrahedron)
#   VM 3 (phase38): MSH3 3THW docking retry (chain A, ADP box)
#
# Each VM gets a different zone from the probe list. The script
# tries zones in priority order; first zone where gcloud exits 0
# wins. Retries every 5 min until a zone becomes available.
#
# Usage: bash analysis/flashoptim_tpu/launch_phases_36_37_38.sh
# ============================================================

set -uo pipefail

PROJECT="aegismind-tpu"
ACCEL="v6e-8"
RUNTIME="v2-alpha-tpuv6e"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY_INTERVAL=300

ZONES=("us-east5-b" "us-central1-b" "us-east1-d" "europe-west4-a" "us-central1-a" "europe-west4-b" "us-south1-a")

log() { echo "[$(date -u '+%H:%M:%S UTC')] $*"; }

# ── Provision a VM: try each zone once, retry the whole list every 5 min ──────
provision_vm() {
    local vm_name="$1"
    shift
    local used_zones=("$@")

    while true; do
        for zone in "${ZONES[@]}"; do
            # Skip zones already taken by sibling VMs
            local skip=0
            for uz in "${used_zones[@]}"; do [[ "$zone" == "$uz" ]] && skip=1 && break; done
            [[ $skip -eq 1 ]] && continue

            log "[$vm_name] Trying zone $zone ..."
            if gcloud compute tpus tpu-vm create "$vm_name" \
                --zone "$zone" --project "$PROJECT" \
                --accelerator-type "$ACCEL" --version "$RUNTIME" \
                2>&1; then
                log "[$vm_name] VM created in $zone"
                echo "$zone"
                return 0
            else
                # Clean up any partial VM
                gcloud compute tpus tpu-vm delete "$vm_name" --zone "$zone" \
                    --project "$PROJECT" --quiet 2>/dev/null || true
            fi
        done
        log "[$vm_name] All zones exhausted — waiting ${RETRY_INTERVAL}s before retry"
        sleep "$RETRY_INTERVAL"
    done
}

wait_ready() {
    local vm="$1"; local zone="$2"
    log "[$vm] Waiting for READY state..."
    for i in $(seq 1 30); do
        local s
        s=$(gcloud compute tpus tpu-vm describe "$vm" --zone "$zone" --project "$PROJECT" \
            --format 'value(state)' 2>/dev/null || echo "UNKNOWN")
        [[ "$s" == "READY" ]] && return 0
        log "  [$vm] state=$s (${i}/30)"; sleep 20
    done
    return 1
}

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

run_phase() {
    local phase="$1"; local script="$2"; local logfile="$3"
    shift 3
    local used_zones=("$@")

    local vm_name="aegis-${phase}"

    # Delete stale VM if it exists from a previous failed attempt
    for zone in "${ZONES[@]}"; do
        if gcloud compute tpus tpu-vm describe "$vm_name" --zone "$zone" \
               --project "$PROJECT" &>/dev/null 2>&1; then
            log "[$phase] Deleting stale VM in $zone"
            gcloud compute tpus tpu-vm delete "$vm_name" --zone "$zone" \
                --project "$PROJECT" --quiet 2>/dev/null || true
        fi
    done

    local zone
    zone=$(provision_vm "$vm_name" "${used_zones[@]}")

    wait_ready "$vm_name" "$zone" || { log "ERROR: $vm_name not READY"; return 1; }
    setup_and_run "$vm_name" "$zone" "$script" "$logfile"
    echo "$zone"
}

# ── Main: provision 3 VMs sequentially to track zone assignments ──────────────
# We need zone assignment order to avoid reuse, so provision sequentially
# but run the actual experiment setup in parallel.

log "=== Provisioning VM for phase36 ==="
ZONE36=$(run_phase "phase36" "phase36_medium_scaling.py" "/tmp/phase36.log")
log "=== phase36 assigned to $ZONE36 ==="

log "=== Provisioning VM for phase37 (avoiding $ZONE36) ==="
ZONE37=$(run_phase "phase37" "phase37_int8_gpt.py" "/tmp/phase37.log" "$ZONE36")
log "=== phase37 assigned to $ZONE37 ==="

log "=== Provisioning VM for phase38 (avoiding $ZONE36 $ZONE37) ==="
ZONE38=$(run_phase "phase38" "phase38_msh3_retry.py" "/tmp/phase38.log" "$ZONE36" "$ZONE37")
log "=== phase38 assigned to $ZONE38 ==="

log "=== All VMs launched. Monitor GCS for results: ==="
log "  gsutil ls gs://aegismind-tpu-results/aegis_flashoptim/phase36_medium_scaling/"
log "  gsutil ls gs://aegismind-tpu-results/aegis_flashoptim/phase37_int8_gpt/"
log "  gsutil ls gs://aegismind-tpu-results/aegis_flashoptim/phase38_msh3_retry/"
