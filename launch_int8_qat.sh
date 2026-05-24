#!/bin/bash
# ============================================================
# launch_int8_qat.sh — upload phase_int8_qat.py to the
# existing aegis-node-v6e-b5 VM and run it in tmux int8q.
#
# Requires Phase 17 checkpoints already on GCS:
#   gs://aegismind-tpu-results/aegis_flashoptim/phase17_precision_dial/{fp32,bf16,fp16}_ep80.pt
#
# Runtime: ~3h  (80 epochs INT8-QAT on QM9) + ~1h LMC (CPU)
#
# Usage:
#   bash analysis/flashoptim_tpu/launch_int8_qat.sh
# ============================================================
set -euo pipefail

VM_NAME="aegis-node-v6e-b5"
ZONE="us-central1-b"
PROJECT="aegismind-tpu"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }

log "=== launch_int8_qat ==="
log "VM: ${VM_NAME}  zone: ${ZONE}"

# ── Verify Phase 17 checkpoints exist on GCS ─────────────────────────────────
log "Verifying Phase 17 GCS checkpoints..."
for prec in fp32 bf16 fp16; do
    GCS_PATH="gs://aegismind-tpu-results/aegis_flashoptim/phase17_precision_dial/${prec}_ep80.pt"
    if ! gsutil -q stat "${GCS_PATH}" 2>/dev/null; then
        log "ERROR: Missing Phase 17 checkpoint: ${GCS_PATH}"
        log "Run phase17_precision_dial.py first or check GCS bucket."
        exit 1
    fi
    log "  ✓ ${prec}_ep80.pt"
done

# ── Upload script ─────────────────────────────────────────────────────────────
log "Uploading phase_int8_qat.py..."
gcloud compute tpus tpu-vm scp \
    "${LOCAL_DIR}/phase_int8_qat.py" \
    "${VM_NAME}:~/flashoptim/phase_int8_qat.py" \
    --zone "${ZONE}" --project "${PROJECT}" --worker all

log "Upload complete."

# ── Launch in tmux int8q ──────────────────────────────────────────────────────
gcloud compute tpus tpu-vm ssh "${VM_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" \
    --command '
        set -e
        mkdir -p ~/pipeline_logs

        tmux new-session -d -s int8q 2>/dev/null || true

        CMD="
set -euo pipefail
cd ~/flashoptim
log() { echo \"[\$(date -u +%Y-%m-%d\\ %H:%M:%S\\ UTC)] \$*\"; }
log \"=== phase_int8_qat start ===\"
python3 phase_int8_qat.py 2>&1 | tee ~/pipeline_logs/int8_qat.log
EXIT=\${PIPESTATUS[0]}
echo \$EXIT > /tmp/.int8_qat_exit
[ \"\$EXIT\" = \"0\" ] && touch /tmp/.int8_qat_done && log \"COMPLETE\" || log \"ERROR exit \$EXIT\"
"
        tmux send-keys -t int8q "eval '\''${CMD}'\''" Enter

        echo "phase_int8_qat launched in tmux session int8q"
        tmux list-sessions
    '

log ""
log "Monitor:"
log "  gcloud compute tpus tpu-vm ssh ${VM_NAME} --zone ${ZONE} --project ${PROJECT} \\"
log "    --command 'tail -40 ~/pipeline_logs/int8_qat.log'"
log ""
log "Results on GCS when done:"
log "  gs://aegismind-tpu-results/aegis_flashoptim/phase_int8_qat/results.json"
