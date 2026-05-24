#!/bin/bash
# ============================================================
# v6e_b_restart.sh — recreate aegis-node-v6e-b and resume
# Phase 18 (cyclic LR FP32).
#
# Phases 16 and 17 are complete. Phase 18 was interrupted by
# preemption with fp32_baseline already saved to GCS.
# phase18_cyclic_lr_fp32.py has resume logic that will download
# fp32_baseline from GCS and skip retraining it.
#
# Run locally in tmux:
#   tmux new -s v6e_b_restart
#   bash analysis/flashoptim_tpu/v6e_b_restart.sh
# ============================================================

set -euo pipefail

VM_NAME="aegis-node-v6e-b"
ZONE="us-east5-b"
PROJECT="aegismind-tpu"
ACCEL="v6e-8"
RUNTIME="v2-alpha-tpuv6e"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY_INTERVAL=300

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }

log "=== v6e-b Phase 18 restart daemon ==="
log "Phases 16+17 complete. Resuming Phase 18 (fp32_baseline from GCS, fp32_cyclic from scratch)."

# ── Step 1: Create VM ───────────────────────────────────────────────────────
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
        log "Capacity not available — retrying in ${RETRY_INTERVAL}s"
        sleep "$RETRY_INTERVAL"
    fi
done

# ── Step 2: Wait for READY ──────────────────────────────────────────────────
log "Waiting for VM to reach READY state..."
for i in $(seq 1 30); do
    STATUS=$(gcloud compute tpus tpu-vm describe "$VM_NAME" \
        --zone "$ZONE" --project "$PROJECT" \
        --format 'value(state)' 2>/dev/null || echo "UNKNOWN")
    log "  VM state: $STATUS"
    [ "$STATUS" = "READY" ] && break
    sleep 20
done

# ── Step 3: Upload code ─────────────────────────────────────────────────────
log "Uploading flashoptim code..."
gcloud compute tpus tpu-vm scp --recurse \
    "$LOCAL_DIR" "${VM_NAME}:~/flashoptim" \
    --zone "$ZONE" --project "$PROJECT" --worker all 2>&1
log "Code uploaded."

# ── Step 4: Install environment + start Phase 18 ───────────────────────────
log "Installing environment and launching Phase 18..."
gcloud compute tpus tpu-vm ssh "$VM_NAME" \
    --zone "$ZONE" --project "$PROJECT" \
    --command "
        set -e
        cd ~/flashoptim
        mkdir -p ~/pipeline_logs

        bash setup.sh 2>&1 | tail -10

        # Mark phases 16+17 done so any downstream checks are satisfied
        touch /tmp/.phase16_done /tmp/.phase17_done

        # Start Phase 18 in a persistent tmux session
        tmux new-session -d -s ml_18 2>/dev/null || true

        QUEUE='
            set -euo pipefail
            cd ~/flashoptim
            log() { echo \"[\$(date -u +%Y-%m-%d\ %H:%M:%S\ UTC)] \$*\"; }

            log \"=== Phase 18 resume (fp32_baseline from GCS, fp32_cyclic from scratch) ===\"
            python3 phase18_cyclic_lr_fp32.py 2>&1 | tee ~/pipeline_logs/phase18.log
            EXIT_18=\${PIPESTATUS[0]}
            echo \$EXIT_18 > /tmp/.phase18_exit
            if [ \"\$EXIT_18\" != \"0\" ]; then log \"ERROR: Phase 18 exit \$EXIT_18\"; exit 1; fi
            touch /tmp/.phase18_done
            log \"Phase 18 complete.\"

            log \"=== ALL PHASES COMPLETE ===\"
        '

        tmux send-keys -t ml_18 \"eval '\$QUEUE' 2>&1 | tee /tmp/pipeline_18.log\" Enter

        echo 'Phase 18 started in tmux session: ml_18'
        tmux list-sessions
    " 2>&1

log "=== v6e-b restored. Phase 18 running. ==="
log ""
log "Monitor:"
log "  gcloud compute tpus tpu-vm ssh ${VM_NAME} --zone ${ZONE} --project ${PROJECT} \\"
log "    --command 'tail -40 ~/pipeline_logs/phase18.log'"
log ""
log "Done flag: /tmp/.phase18_done"
log "GCS output: gs://aegismind-tpu-results/aegis_flashoptim/phase18_cyclic_lr_fp32/results.json"
