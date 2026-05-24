#!/bin/bash
# ============================================================
# launch_resnet_restart.sh — upload phase_resnet_restart.py
# to aegis-node-v6e-b5 and run it in tmux resnet_r.
#
# Architecture: SmallResNet (GroupNorm, GELU, ~1.2M params)
# Dataset:      CIFAR-10 (downloaded from cs.toronto.edu)
# Conditions:   A (FP32 baseline), B (BF16+restart), C (FP32+restart)
# Runtime:      ~6-8h  (3 × 100-epoch runs + LMC)
#
# Launch AFTER phase_int8_qat completes (they share the TPU device).
#
# Usage:
#   bash analysis/flashoptim_tpu/launch_resnet_restart.sh
# ============================================================
set -euo pipefail

VM_NAME="aegis-node-v6e-b5"
ZONE="us-central1-b"
PROJECT="aegismind-tpu"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }

log "=== launch_resnet_restart ==="
log "VM: ${VM_NAME}  zone: ${ZONE}"

# ── Check INT8 QAT is not still running ──────────────────────────────────────
log "Checking INT8 QAT status..."
INT8_RUNNING=$(gcloud compute tpus tpu-vm ssh "${VM_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" \
    --command 'pgrep -f phase_int8_qat.py && echo running || echo done' 2>/dev/null | tail -1)

if [ "${INT8_RUNNING}" = "running" ]; then
    log "WARNING: phase_int8_qat.py is still running."
    log "Wait for it to finish before launching ResNet (they share the TPU device)."
    log "Check: gcloud compute tpus tpu-vm ssh ${VM_NAME} --zone ${ZONE} --project ${PROJECT} \\"
    log "         --command 'tail -10 ~/pipeline_logs/int8_qat.log'"
    log "Re-run this script once INT8 QAT is done."
    exit 1
fi
log "INT8 QAT done (or not started) — safe to launch ResNet."

# ── Upload script ─────────────────────────────────────────────────────────────
log "Uploading phase_resnet_restart.py..."
gcloud compute tpus tpu-vm scp \
    "${LOCAL_DIR}/phase_resnet_restart.py" \
    "${VM_NAME}:~/flashoptim/phase_resnet_restart.py" \
    --zone "${ZONE}" --project "${PROJECT}" --worker all

log "Upload complete."

# ── Launch in tmux resnet_r ───────────────────────────────────────────────────
gcloud compute tpus tpu-vm ssh "${VM_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" \
    --command '
        set -e
        mkdir -p ~/pipeline_logs

        tmux new-session -d -s resnet_r 2>/dev/null || true

        CMD="
set -euo pipefail
cd ~/flashoptim
log() { echo \"[\$(date -u +%Y-%m-%d\\ %H:%M:%S\\ UTC)] \$*\"; }
log \"=== phase_resnet_restart start ===\"
python3 phase_resnet_restart.py 2>&1 | tee ~/pipeline_logs/resnet_restart.log
EXIT=\${PIPESTATUS[0]}
echo \$EXIT > /tmp/.resnet_restart_exit
[ \"\$EXIT\" = \"0\" ] && touch /tmp/.resnet_restart_done && log \"COMPLETE\" || log \"ERROR exit \$EXIT\"
"
        tmux send-keys -t resnet_r "eval '\''${CMD}'\''" Enter

        echo "phase_resnet_restart launched in tmux session resnet_r"
        tmux list-sessions
    '

log ""
log "Monitor:"
log "  gcloud compute tpus tpu-vm ssh ${VM_NAME} --zone ${ZONE} --project ${PROJECT} \\"
log "    --command 'tail -40 ~/pipeline_logs/resnet_restart.log'"
log ""
log "Results on GCS when done:"
log "  gs://aegismind-tpu-results/aegis_flashoptim/phase_resnet_restart/results.json"
