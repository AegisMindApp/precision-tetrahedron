#!/bin/bash
# ============================================================
# launch_xarch_lmc.sh — upload phase_xarch_lmc.py to the
# existing aegis-node-v6e-b5 VM and run it in tmux xl_22.
#
# Runs sequentially: FP32 → BF16 → FP16 (each ~60-80 min on
# v6e-8), then LMC on CPU (~15 min). Total: ~4-5 hours.
#
# Usage (local tmux):
#   bash analysis/flashoptim_tpu/launch_xarch_lmc.sh
# ============================================================
set -euo pipefail

VM_NAME="aegis-node-v6e-b5"
ZONE="us-central1-b"
PROJECT="aegismind-tpu"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }

log "=== launch_xarch_lmc ==="
log "VM: ${VM_NAME}  zone: ${ZONE}"

# ── Upload script ─────────────────────────────────────────────────────────────
log "Uploading phase_xarch_lmc.py..."
gcloud compute tpus tpu-vm scp \
    "${LOCAL_DIR}/phase_xarch_lmc.py" \
    "${VM_NAME}:~/flashoptim/phase_xarch_lmc.py" \
    --zone "${ZONE}" --project "${PROJECT}" --worker all

log "Upload complete."

# ── Launch in tmux xl_22 ──────────────────────────────────────────────────────
gcloud compute tpus tpu-vm ssh "${VM_NAME}" \
    --zone "${ZONE}" --project "${PROJECT}" \
    --command '
        set -e
        mkdir -p ~/pipeline_logs

        tmux new-session -d -s xl_22 2>/dev/null || true

        CMD="
set -euo pipefail
cd ~/flashoptim
log() { echo \"[\$(date -u +%Y-%m-%d\\ %H:%M:%S\\ UTC)] \$*\"; }

log \"=== phase_xarch_lmc start ===\"
python3 phase_xarch_lmc.py 2>&1 | tee ~/pipeline_logs/xarch_lmc.log
EXIT=\${PIPESTATUS[0]}
echo \$EXIT > /tmp/.xarch_lmc_exit
[ \"\$EXIT\" = \"0\" ] && touch /tmp/.xarch_lmc_done && log \"COMPLETE\" || log \"ERROR exit \$EXIT\"
"
        tmux send-keys -t xl_22 "eval '\''${CMD}'\''" Enter

        echo "phase_xarch_lmc launched in tmux session xl_22"
        tmux list-sessions
    '

log ""
log "Monitor:"
log "  gcloud compute tpus tpu-vm ssh ${VM_NAME} --zone ${ZONE} --project ${PROJECT} \\"
log "    --command 'tail -40 ~/pipeline_logs/xarch_lmc.log'"
log ""
log "Results on GCS when done:"
log "  gs://aegismind-tpu-results/aegis_flashoptim/phase_xarch_lmc/results.json"
