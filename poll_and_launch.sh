#!/bin/bash
# poll_and_launch.sh
#
# Polls for TPU allocation and automatically deploys + launches tpu_master.sh.
# Run locally and leave it running — it will fire when a node becomes ACTIVE.
#
# Usage:
#   export NOTIFY_SMTP_USER="your@gmail.com"
#   export NOTIFY_SMTP_PASS="your-app-password"
#   nohup bash poll_and_launch.sh > /tmp/poll.log 2>&1 &
#   echo "Poller PID: $!"
#
# To watch progress:
#   tail -f /tmp/poll.log
#
# To cancel:
#   kill <PID>
#
# QR recreation (IMPORTANT: use --provisioning-model=SPOT, not --spot or --best-effort):
#   gcloud alpha compute tpus queued-resources create <name> \
#     --node-id=<node> --accelerator-type=v6e-8 --runtime-version=v2-alpha-tpuv6e \
#     --zone=<zone> --project=aegismind-tpu --provisioning-model=SPOT --quiet

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT=aegismind-tpu
GCS_BUCKET="${GCS_BUCKET:-gs://aegismind-tpu-results}"
SMTP_USER="${NOTIFY_SMTP_USER:-}"
SMTP_PASS="${NOTIFY_SMTP_PASS:-}"
POLL_INTERVAL=120   # seconds between checks

# Pipeline source directory (directory containing this script)
PIPELINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Queued resources to watch — v6e-8 SPOT across 4 quota zones
declare -A QR_ZONES=(
    [hd-east5]=us-east5-b
    [hd-east5-b2]=us-east5-b
    [hd-east5-p34]=us-east5-b
    [hd-useast1d]=us-east1-d
    [hd-useast1d-b2]=us-east1-d
    [hd-europewest4a-b2]=europe-west4-a
    [hd-uscentral1b-b2]=us-central1-b
    [hd-uscentral1b-p34]=us-central1-b
)

declare -A QR_NODES=(
    [hd-east5]=aegis-node-v6e-e5
    [hd-east5-b2]=aegis-node-v6e-e5-b2
    [hd-east5-p34]=aegis-node-v6e-e5-p34
    [hd-useast1d]=aegis-node-v6e-useast1d
    [hd-useast1d-b2]=aegis-node-v6e-useast1d-b2
    [hd-europewest4a-b2]=aegis-node-v6e-europewest4a-b2
    [hd-uscentral1b-b2]=aegis-node-v6e-uscentral1b-b2
    [hd-uscentral1b-p34]=aegis-node-v6e-uscentral1b-p34
)

# ── Utilities ──────────────────────────────────────────────────────────────────
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

get_qr_state() {
    local qr=$1 zone=$2
    gcloud alpha compute tpus queued-resources describe "$qr" \
        --zone="$zone" --project="$PROJECT" 2>/dev/null \
        | grep -E "^\s+state:" | head -1 | awk '{print $2}' || echo "UNKNOWN"
}

# ── Deploy and launch ──────────────────────────────────────────────────────────
deploy_and_launch() {
    local node=$1 zone=$2
    log "=========================================="
    log "TPU ACTIVE: $node ($zone) — deploying pipeline"
    log "=========================================="

    # Write run_queue.sh locally based on node name, then SCP it
    local queue_file
    queue_file=$(mktemp /tmp/run_queue_XXXXXX.sh)

    if echo "$node" | grep -qE "e5|east5"; then
        cat > "$queue_file" << 'QEOF'
#!/bin/bash
set -uo pipefail
cd ~/flashoptim
log() { echo "[$(date -u +%Y-%m-%d\ %H:%M:%S\ UTC)] $*"; }
log "=== EAST5 QUEUE: phases 24 / 25 ==="
log "[1/2] Phase 24 INT8 tetrahedron (§4.21)"
python3 phase24_retry.py 2>&1 | tee /tmp/phase24_retry.log
echo "${PIPESTATUS[0]}" > /tmp/.phase24_exit
log "Phase 24 done (exit $(cat /tmp/.phase24_exit))"
log "[2/2] Phase 25 PTLE arm_b (§4.22)"
python3 phase25_ptle_retry.py 2>&1 | tee /tmp/phase25_ptle.log
echo "${PIPESTATUS[0]}" > /tmp/.phase25_exit
log "Phase 25 done (exit $(cat /tmp/.phase25_exit))"
log "=== EAST5 ALL DONE ==="
touch /tmp/.all_phases_done
QEOF
    elif echo "$node" | grep -qE "useast1d|east1d"; then
        cat > "$queue_file" << 'QEOF'
#!/bin/bash
set -uo pipefail
cd ~/flashoptim
log() { echo "[$(date -u +%Y-%m-%d\ %H:%M:%S\ UTC)] $*"; }
log "=== EAST1D QUEUE: phases 31 / 33 / 34 ==="
log "[1/3] Phase 31 MEDIUM 38M GPT LMC (§4.28)"
python3 phase31_medium_retry.py 2>&1 | tee /tmp/phase31_medium.log
echo "${PIPESTATUS[0]}" > /tmp/.phase31_exit
log "Phase 31 done (exit $(cat /tmp/.phase31_exit))"
log "[2/3] Phase 33 GPT-2 finetune LMC (§4.30)"
python3 phase33_finetune_lmc.py 2>&1 | tee /tmp/phase33_finetune.log
echo "${PIPESTATUS[0]}" > /tmp/.phase33_exit
log "Phase 33 done (exit $(cat /tmp/.phase33_exit))"
log "[3/3] Phase 34 Nash AMR drug combination (§4.31)"
python3 phase34_nash_amr.py 2>&1 | tee /tmp/phase34_nash_amr.log
echo "${PIPESTATUS[0]}" > /tmp/.phase34_exit
log "Phase 34 done (exit $(cat /tmp/.phase34_exit))"
log "=== EAST1D ALL DONE ==="
touch /tmp/.all_phases_done
QEOF
    elif echo "$node" | grep -qE "p34"; then
        # Dedicated phase 34 nodes — skip directly to Nash AMR run
        cat > "$queue_file" << 'QEOF'
#!/bin/bash
set -uo pipefail
cd ~/flashoptim
log() { echo "[$(date -u +%Y-%m-%d\ %H:%M:%S\ UTC)] $*"; }
log "=== P34 QUEUE: phase 34 only ==="
log "[1/1] Phase 34 Nash AMR drug combination (§4.31)"
python3 phase34_nash_amr.py 2>&1 | tee /tmp/phase34_nash_amr.log
echo "${PIPESTATUS[0]}" > /tmp/.phase34_exit
log "Phase 34 done (exit $(cat /tmp/.phase34_exit))"
log "=== P34 ALL DONE ==="
touch /tmp/.all_phases_done
QEOF
    elif echo "$node" | grep -qE "europewest4a|europe"; then
        # europewest4a: dedicated phase 29 node — fp32_clean already in GCS, resumes INT8-optim
        cat > "$queue_file" << 'QEOF'
#!/bin/bash
set -uo pipefail
cd ~/flashoptim
log() { echo "[$(date -u +%Y-%m-%d\ %H:%M:%S\ UTC)] $*"; }
log "=== EUROPEWEST4A QUEUE: phase 29 ==="
log "[1/1] Phase 29 INT8 optimizer LMC (§4.26)"
python3 phase29_optim_int8_lmc.py 2>&1 | tee /tmp/phase29_optim_int8.log
echo "${PIPESTATUS[0]}" > /tmp/.phase29_exit
log "Phase 29 done (exit $(cat /tmp/.phase29_exit))"
log "=== EUROPEWEST4A ALL DONE ==="
touch /tmp/.all_phases_done
QEOF
    else
        # b2 spare nodes: run phases 24/25/31/33/34 — GCS checkpoints prevent duplication
        cat > "$queue_file" << 'QEOF'
#!/bin/bash
set -uo pipefail
cd ~/flashoptim
log() { echo "[$(date -u +%Y-%m-%d\ %H:%M:%S\ UTC)] $*"; }
log "=== B2 SPARE QUEUE: phases 24 / 25 / 31 / 33 / 34 ==="
log "[1/5] Phase 24 INT8 tetrahedron"
python3 phase24_retry.py 2>&1 | tee /tmp/phase24_retry.log
log "Phase 24 done"
log "[2/5] Phase 25 PTLE arm_b"
python3 phase25_ptle_retry.py 2>&1 | tee /tmp/phase25_ptle.log
log "Phase 25 done"
log "[3/5] Phase 31 MEDIUM LMC"
python3 phase31_medium_retry.py 2>&1 | tee /tmp/phase31_medium.log
log "Phase 31 done"
log "[4/5] Phase 33 GPT-2 finetune LMC"
python3 phase33_finetune_lmc.py 2>&1 | tee /tmp/phase33_finetune.log
log "Phase 33 done"
log "[5/5] Phase 34 Nash AMR drug combination"
python3 phase34_nash_amr.py 2>&1 | tee /tmp/phase34_nash_amr.log
log "Phase 34 done"
log "=== B2 ALL DONE ==="
touch /tmp/.all_phases_done
QEOF
    fi
    chmod +x "$queue_file"

    # Copy pipeline files to VM
    log "Copying pipeline files to $node ..."
    gcloud compute tpus tpu-vm scp --recurse \
        "$PIPELINE_DIR" "$node":~/flashoptim \
        --zone="$zone" --project="$PROJECT" --worker=all

    # Copy queue script to VM
    gcloud compute tpus tpu-vm scp \
        "$queue_file" "$node":/tmp/run_queue.sh \
        --zone="$zone" --project="$PROJECT" --worker=all

    rm -f "$queue_file"
    log "Files copied."

    # Install env and launch queue in tmux
    log "Running setup.sh and launching experiment queue in tmux ..."
    gcloud compute tpus tpu-vm ssh "$node" \
        --zone="$zone" --project="$PROJECT" \
        --command='
            set -e
            cd ~/flashoptim
            bash setup.sh 2>&1 | tail -5
            pip install biopython scipy transformers accelerate --quiet
            pip uninstall -y torchvision 2>/dev/null || true
            chmod +x /tmp/run_queue.sh
            tmux new-session -d -s exp_queue 2>/dev/null || true
            tmux send-keys -t exp_queue "bash /tmp/run_queue.sh 2>&1 | tee /tmp/exp_queue.log" Enter
        '

    log "=========================================="
    log "Pipeline running in tmux session 'aegis'"
    log ""
    log "To attach:"
    log "  gcloud compute tpus tpu-vm ssh $node --zone=$zone --project=$PROJECT"
    log "  tmux attach -t aegis"
    log ""
    log "To tail logs from here:"
    log "  gcloud compute tpus tpu-vm ssh $node --zone=$zone --project=$PROJECT --command='tail -f /tmp/master.log'"
    log "=========================================="
}

# ── Main poll loop ─────────────────────────────────────────────────────────────
if [ -z "$SMTP_USER" ] || [ -z "$SMTP_PASS" ]; then
    log "WARNING: NOTIFY_SMTP_USER or NOTIFY_SMTP_PASS not set — email notifications will be skipped"
fi

log "Polling every ${POLL_INTERVAL}s for queued resources:"
for qr in "${!QR_ZONES[@]}"; do
    log "  $qr → node=${QR_NODES[$qr]} zone=${QR_ZONES[$qr]}"
done
log ""

# Track which nodes have been deployed so we don't redeploy on every poll
declare -A DEPLOYED=()

while true; do
    for qr in "${!QR_ZONES[@]}"; do
        zone="${QR_ZONES[$qr]}"
        node="${QR_NODES[$qr]}"
        state=$(get_qr_state "$qr" "$zone")
        log "[$qr] state=$state"

        if [ "$state" = "ACTIVE" ] && [ -z "${DEPLOYED[$qr]+x}" ]; then
            deploy_and_launch "$node" "$zone"
            DEPLOYED[$qr]=1
        elif [ "$state" != "ACTIVE" ]; then
            # Reset deploy flag if node was preempted — allow redeploy on next ACTIVE
            unset "DEPLOYED[$qr]"
        fi
    done

    log "Sleeping ${POLL_INTERVAL}s ..."
    sleep "$POLL_INTERVAL"
done
