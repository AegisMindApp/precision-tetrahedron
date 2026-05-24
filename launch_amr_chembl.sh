#!/bin/bash
# ============================================================
# launch_amr_chembl.sh — provision a second v6e-8 VM via
# queued-resources and run phase_amr_chembl.py.
#
# Tries us-east1-d first (64-chip quota, never attempted),
# falls back to europe-west4-a.
#
# Uploads: phase_amr_chembl.py, 3RXX_receptor.pdbqt
# Installs: vina (pip), obabel (apt) if not already present
#
# Usage (local tmux):
#   bash analysis/flashoptim_tpu/launch_amr_chembl.sh
# ============================================================
set -euo pipefail

VM_NAME="aegis-node-amr-1"
QR_NAME="aegis-qr-amr-1"
ZONE="us-east1-d"       # 64-chip v6e quota — never attempted
FALLBACK_ZONE="europe-west4-a"
PROJECT="aegismind-tpu"
ACCEL="v6e-8"
RUNTIME="v2-alpha-tpuv6e"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LOCAL_DIR}/../.." && pwd)"
RECEPTOR_SRC="${REPO_ROOT}/analysis/amr_glass/docking/results/3RXX_receptor.pdbqt"
RETRY_INTERVAL=300

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }

log "=== launch_amr_chembl ==="

# ── Step 1: Submit QR (try us-east1-d, fall back to europe-west4-a) ──────────
for ATTEMPT_ZONE in "${ZONE}" "${FALLBACK_ZONE}"; do
    log "Submitting queued-resource ${QR_NAME} in ${ATTEMPT_ZONE}..."
    if gcloud compute tpus queued-resources create "${QR_NAME}" \
        --node-id "${VM_NAME}" \
        --zone "${ATTEMPT_ZONE}" \
        --project "${PROJECT}" \
        --accelerator-type "${ACCEL}" \
        --runtime-version "${RUNTIME}" \
        --spot \
        2>&1; then
        ZONE="${ATTEMPT_ZONE}"
        log "QR submitted in ${ZONE}."
        break
    else
        log "QR creation failed in ${ATTEMPT_ZONE} — trying fallback..."
    fi
done

# ── Step 2: Poll until ACTIVE ─────────────────────────────────────────────────
log "Waiting for QR to become ACTIVE..."
while true; do
    STATE=$(gcloud compute tpus queued-resources describe "${QR_NAME}" \
        --zone "${ZONE}" --project "${PROJECT}" \
        --format 'value(state.state)' 2>/dev/null || echo "UNKNOWN")
    log "  QR state: ${STATE}"
    [ "${STATE}" = "ACTIVE" ] && break
    if [ "${STATE}" = "SUSPENDED" ] || [ "${STATE}" = "FAILED" ]; then
        log "ERROR: QR ${STATE}. Deleting and exiting."
        gcloud compute tpus queued-resources delete "${QR_NAME}" \
            --zone "${ZONE}" --project "${PROJECT}" --quiet 2>/dev/null || true
        exit 1
    fi
    sleep "${RETRY_INTERVAL}"
done
log "VM ${VM_NAME} is ACTIVE."

# ── Step 3: Wait for READY ────────────────────────────────────────────────────
log "Waiting for VM to reach READY state..."
for i in $(seq 1 30); do
    STATUS=$(gcloud compute tpus tpu-vm describe "${VM_NAME}" \
        --zone "${ZONE}" --project "${PROJECT}" \
        --format 'value(state)' 2>/dev/null || echo "UNKNOWN")
    log "  VM state: ${STATUS}"
    [ "${STATUS}" = "READY" ] && break
    sleep 20
done

# ── Step 4: Upload code and receptor ─────────────────────────────────────────
log "Uploading flashoptim code..."
gcloud compute tpus tpu-vm scp --recurse \
    "${LOCAL_DIR}" "${VM_NAME}:~/flashoptim" \
    --zone "${ZONE}" --project "${PROJECT}" --worker all

log "Uploading KPC-3 receptor PDBQT..."
gcloud compute tpus tpu-vm ssh "${VM_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" \
    --command "mkdir -p /tmp/vina_receptors_amr"
gcloud compute tpus tpu-vm scp \
    "${RECEPTOR_SRC}" "${VM_NAME}:/tmp/vina_receptors_amr/3RXX_receptor.pdbqt" \
    --zone "${ZONE}" --project "${PROJECT}" --worker all

log "Files uploaded."

# ── Step 5: Install deps + launch phase_amr_chembl ───────────────────────────
log "Installing dependencies and launching AMR screen..."
gcloud compute tpus tpu-vm ssh "${VM_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" \
    --command '
        set -e
        cd ~/flashoptim
        mkdir -p ~/pipeline_logs

        bash setup.sh 2>&1 | tail -5

        # Extra deps for AMR screen
        pip install vina 2>&1 | tail -2 || true
        sudo apt-get install -y openbabel 2>&1 | tail -3 || true

        tmux new-session -d -s amr_p 2>/dev/null || true

        CMD="
set -euo pipefail
cd ~/flashoptim
log() { echo \"[\$(date -u +%Y-%m-%d\\ %H:%M:%S\\ UTC)] \$*\"; }
log \"=== phase_amr_chembl start ===\"
python3 phase_amr_chembl.py 2>&1 | tee ~/pipeline_logs/amr_chembl.log
EXIT=\${PIPESTATUS[0]}
echo \$EXIT > /tmp/.amr_chembl_exit
[ \"\$EXIT\" = \"0\" ] && touch /tmp/.amr_chembl_done && log \"COMPLETE\" || log \"ERROR exit \$EXIT\"
"
        tmux send-keys -t amr_p "eval '\''${CMD}'\''" Enter

        echo "phase_amr_chembl launched in tmux session amr_p"
        tmux list-sessions
    '

log "=== AMR ChEMBL screen launched ==="
log ""
log "Monitor:"
log "  gcloud compute tpus tpu-vm ssh ${VM_NAME} --zone ${ZONE} --project ${PROJECT} \\"
log "    --command 'tail -40 ~/pipeline_logs/amr_chembl.log'"
log ""
log "Results on GCS when done:"
log "  gs://aegismind-tpu-results/aegis_flashoptim/phase_amr_chembl/results.json"
