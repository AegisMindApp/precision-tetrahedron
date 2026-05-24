#!/bin/bash
# ============================================================
# v6e_east5_restart.sh — recover from east5 preemption
#
# Recreates a v6e-8 VM and runs the full pending experiment queue:
#   phase26_resume_frac08   → §4.23 ρ-degradation frac=0.8+1.0
#   phase28_enrich_smiles   → ChEMBL SMILES enrichment (CPU)
#   phase28_kpc3_lmc        → §4.25 KPC-3 MLP surrogate LMC
#   phase29_optim_int8_lmc  → §4.26 INT8 optimizer-state LMC
#   phase30_msh3_3thw       → §4.27 MSH3 3THW docking (fixed -xr)
#   phase31_medium_retry    → §4.28 38M-param GPT LMC (MEDIUM)
#   phase33_finetune_lmc    → §4.30 GPT-2 fine-tune LMC
#   phase24_retry           → §4.21 INT8 tetrahedron
#   phase25_ptle_retry      → §4.22 PTLE arm_b
#
# Run locally in a persistent tmux session:
#   tmux new -s v6e_east5_restart
#   bash analysis/flashoptim_tpu/v6e_east5_restart.sh
# ============================================================

# No set -e: gcloud create returns non-zero on quota/capacity errors,
# which are expected and handled explicitly in the retry loop.
set -uo pipefail

VM_NAME="aegis-node-v6e-e5"
PROJECT="aegismind-tpu"
ACCEL="v6e-8"
RUNTIME="v2-alpha-tpuv6e"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RETRY_INTERVAL=300   # 5 min between capacity retries
LOG_FILE="/tmp/v6e_east5_restart.log"

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "$LOG_FILE"; }

log "=== v6e east5 restart daemon ==="
log "Local scripts dir: $LOCAL_DIR"

# Try zones in order: east5-b first (lowest latency to GCS bucket), then fallbacks
# Zones with confirmed quota (non-zero TPUV6EPerProjectPerZoneForTPUAPI):
# us-east5-b: "Insufficient capacity" — retry
# us-east1-d: "Insufficient capacity" — retry
# europe-west4-a: "Insufficient capacity" — retry
# All others: quota=0 or v6e-8 not found — skip
ZONES=(us-east5-b us-east1-d europe-west4-a us-central1-a us-central1-b us-central1-c us-east4-a us-east4-b us-south1-a)

# ── Step 1: Delete any stale VM with this name ───────────────────────────────
for ZONE in "${ZONES[@]}"; do
    if gcloud compute tpus tpu-vm describe "$VM_NAME" \
        --zone "$ZONE" --project "$PROJECT" &>/dev/null; then
        log "Found stale VM $VM_NAME in $ZONE — deleting..."
        gcloud compute tpus tpu-vm delete "$VM_NAME" \
            --zone "$ZONE" --project "$PROJECT" --quiet || true
        break
    fi
done

# ── Step 2: Create VM (retry until capacity) ─────────────────────────────────
CREATED_ZONE=""
while [ -z "$CREATED_ZONE" ]; do
    for ZONE in "${ZONES[@]}"; do
        log "Trying to create $VM_NAME in $ZONE (ACCEL=$ACCEL, RUNTIME=$RUNTIME)..."
        if gcloud compute tpus tpu-vm create "$VM_NAME" \
            --zone "$ZONE" \
            --project "$PROJECT" \
            --accelerator-type "$ACCEL" \
            --version "$RUNTIME" \
            2>&1 | tee -a "$LOG_FILE"; then
            CREATED_ZONE="$ZONE"
            log "VM created in $ZONE"
            break
        else
            log "No capacity in $ZONE"
        fi
    done
    if [ -z "$CREATED_ZONE" ]; then
        log "No capacity in any zone — retrying in ${RETRY_INTERVAL}s"
        sleep "$RETRY_INTERVAL"
    fi
done

ZONE="$CREATED_ZONE"

# ── Step 3: Wait for READY ───────────────────────────────────────────────────
log "Waiting for VM to reach READY state..."
for i in $(seq 1 40); do
    STATUS=$(gcloud compute tpus tpu-vm describe "$VM_NAME" \
        --zone "$ZONE" --project "$PROJECT" \
        --format 'value(state)' 2>/dev/null || echo "UNKNOWN")
    log "  VM state: $STATUS (attempt $i)"
    [ "$STATUS" = "READY" ] && break
    sleep 15
done

# ── Step 4: Upload code ──────────────────────────────────────────────────────
log "Uploading flashoptim code to VM..."
gcloud compute tpus tpu-vm scp --recurse \
    "$LOCAL_DIR" "${VM_NAME}:~/flashoptim" \
    --zone "$ZONE" --project "$PROJECT" --worker all 2>&1 | tee -a "$LOG_FILE"
log "Code uploaded."

# ── Step 5: Install environment and start queue ──────────────────────────────
log "Installing environment and launching experiment queue..."
gcloud compute tpus tpu-vm ssh "$VM_NAME" \
    --zone "$ZONE" --project "$PROJECT" \
    --command "
        set -e
        cd ~/flashoptim

        echo '=== Installing base environment ==='
        bash setup.sh 2>&1 | tail -20

        echo '=== Installing extra dependencies ==='
        pip install biopython scipy transformers accelerate --quiet
        pip uninstall -y torchvision 2>/dev/null || true

        echo '=== Environment ready ==='

        # Start experiment queue in persistent tmux session
        tmux new-session -d -s exp_queue 2>/dev/null || true

        QUEUE='
            set -euo pipefail
            cd ~/flashoptim
            log() { echo \"[\$(date -u +%Y-%m-%d\\ %H:%M:%S\\ UTC)] \$*\"; }
            gcs_cp() { gsutil -q cp \"\$1\" \"\$2\" 2>/dev/null || true; }

            log \"=============================\"
            log \" EAST5 EXPERIMENT QUEUE\"
            log \"=============================\"

            log \"[1/9] Phase 26 resume frac=0.8+1.0 (§4.23)\"
            python3 phase26_resume_frac08.py 2>&1 | tee /tmp/phase26_resume.log
            echo \"\${PIPESTATUS[0]}\" > /tmp/.phase26_resume_exit
            log \"Phase 26 resume done (exit \$(cat /tmp/.phase26_resume_exit))\"

            log \"[2/9] Phase 28 SMILES enrichment\"
            python3 phase28_enrich_smiles.py 2>&1 | tee /tmp/phase28_enrich.log
            log \"Phase 28 enrichment done\"

            log \"[3/9] Phase 28 KPC-3 surrogate LMC (§4.25)\"
            python3 phase28_kpc3_surrogate_lmc.py 2>&1 | tee /tmp/phase28_kpc3.log
            echo \"\${PIPESTATUS[0]}\" > /tmp/.phase28_exit
            log \"Phase 28 done (exit \$(cat /tmp/.phase28_exit))\"

            log \"[4/9] Phase 29 INT8 optimizer LMC (§4.26)\"
            python3 phase29_optim_int8_lmc.py 2>&1 | tee /tmp/phase29_int8.log
            echo \"\${PIPESTATUS[0]}\" > /tmp/.phase29_exit
            log \"Phase 29 done (exit \$(cat /tmp/.phase29_exit))\"

            log \"[5/9] Phase 30 MSH3 3THW docking (§4.27)\"
            python3 phase30_msh3_3thw.py 2>&1 | tee /tmp/phase30_msh3.log
            echo \"\${PIPESTATUS[0]}\" > /tmp/.phase30_exit
            log \"Phase 30 done (exit \$(cat /tmp/.phase30_exit))\"

            log \"[6/9] Phase 31 MEDIUM retry (§4.28)\"
            python3 phase31_medium_retry.py 2>&1 | tee /tmp/phase31_medium.log
            echo \"\${PIPESTATUS[0]}\" > /tmp/.phase31_medium_exit
            log \"Phase 31 MEDIUM done (exit \$(cat /tmp/.phase31_medium_exit))\"

            log \"[7/9] Phase 33 GPT-2 finetune LMC (§4.30)\"
            python3 phase33_finetune_lmc.py 2>&1 | tee /tmp/phase33_finetune.log
            echo \"\${PIPESTATUS[0]}\" > /tmp/.phase33_exit
            log \"Phase 33 done (exit \$(cat /tmp/.phase33_exit))\"

            log \"[8/9] Phase 24 INT8 tetrahedron (§4.21)\"
            python3 phase24_retry.py 2>&1 | tee /tmp/phase24_retry.log
            echo \"\${PIPESTATUS[0]}\" > /tmp/.phase24_exit
            log \"Phase 24 done (exit \$(cat /tmp/.phase24_exit))\"

            log \"[9/9] Phase 25 PTLE arm_b (§4.22)\"
            python3 phase25_ptle_retry.py 2>&1 | tee /tmp/phase25_ptle.log
            echo \"\${PIPESTATUS[0]}\" > /tmp/.phase25_exit
            log \"Phase 25 done (exit \$(cat /tmp/.phase25_exit))\"

            log \"=============================\"
            log \" ALL EXPERIMENTS COMPLETE\"
            log \"=============================\"
            touch /tmp/.all_phases_done
        '

        tmux send-keys -t exp_queue \
            \"eval '\$QUEUE' 2>&1 | tee /tmp/exp_queue.log\" Enter

        echo 'Experiment queue started in tmux session: exp_queue'
        tmux list-sessions
    " 2>&1 | tee -a "$LOG_FILE"

log "=== v6e-e5 fully restored. Queue running in tmux exp_queue ==="
log ""
log "Monitor:"
log "  gcloud compute tpus tpu-vm ssh ${VM_NAME} --zone ${ZONE} --project ${PROJECT} \\"
log "    --command 'tail -50 /tmp/exp_queue.log'"
log ""
log "Watch phase logs:"
log "  gcloud compute tpus tpu-vm ssh ${VM_NAME} --zone ${ZONE} --project ${PROJECT} \\"
log "    --command 'tail -f /tmp/phase26_resume.log'"
log ""
log "GCS results:"
log "  gsutil ls gs://aegismind-tpu-results/aegis_flashoptim/phase26_retry/"
