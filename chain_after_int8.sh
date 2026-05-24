#!/bin/bash
# chain_after_int8.sh
#
# Waits for int8_qat to finish on aegis-node-v6e-b5 (us-east5-b),
# then chains: xarch_lmc → resnet_restart on the same VM.
#
# Idempotent: restart-safe — already-completed stages are skipped.
# Preemption: 3 consecutive SSH failures → exits with recovery instructions.
#
# Usage (run locally in a tmux session so it survives terminal close):
#   tmux new-session -d -s chain
#   tmux send-keys -t chain "bash analysis/flashoptim_tpu/chain_after_int8.sh 2>&1 | tee /tmp/chain.log" Enter
#   tmux attach -t chain

set -uo pipefail

VM="aegis-node-v6e-b5"
ZONE="us-east5-b"
PROJECT="aegismind-tpu"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLL=30

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }

vm_ssh() {
    gcloud compute tpus tpu-vm ssh "$VM" --zone "$ZONE" --project "$PROJECT" \
        --command "$1" 2>/dev/null
}

vm_scp() {
    gcloud compute tpus tpu-vm scp "$1" "${VM}:$2" \
        --zone "$ZONE" --project "$PROJECT" --worker all 2>/dev/null
}

vm_alive() {
    vm_ssh "echo ok" | grep -q ok
}

flag_exists() {
    vm_ssh "[ -f '$1' ] && echo yes" | grep -q yes
}

# Write a local shell script to a remote path via base64 (safe for any content).
write_remote_script() {
    local remote_path="$1" content="$2"
    local b64
    b64=$(printf '%s' "$content" | base64 -w0)
    vm_ssh "echo '$b64' | base64 -d > '$remote_path' && chmod +x '$remote_path'"
}

# Wait for a completion flag with preemption detection (3-strike rule).
# Returns 0 on success, exits 1 on failure or preemption.
wait_done() {
    local done_flag="$1" exit_flag="$2" log_file="$3" label="$4"
    local missed=0
    while true; do
        if flag_exists "$done_flag"; then
            log "$label: COMPLETE ✓"
            return 0
        fi
        if flag_exists "$exit_flag"; then
            local code
            code=$(vm_ssh "cat '$exit_flag'" || echo "?")
            [ "$code" = "0" ] && { log "$label: COMPLETE ✓"; return 0; }
            log "FATAL: $label failed (exit=$code)"
            log "  gcloud compute tpus tpu-vm ssh $VM --zone $ZONE --project $PROJECT --command 'tail -40 ~/pipeline_logs/$log_file'"
            exit 1
        fi
        if ! vm_alive; then
            missed=$((missed + 1))
            if [ "$missed" -ge 3 ]; then
                log "FATAL: VM unreachable for 3 polls — likely preempted during $label"
                log "  1. Restore VM via infinite_race.sh (will auto-resume from GCS checkpoint)"
                log "  2. Re-run this script — completed stages are skipped automatically"
                exit 1
            fi
            log "  WARNING: VM ping failed ($missed/3) — retrying..."
        else
            missed=0
            local tail_line
            tail_line=$(vm_ssh "tail -1 ~/pipeline_logs/$log_file 2>/dev/null" || echo "…")
            log "  [$label] $tail_line"
        fi
        sleep "$POLL"
    done
}

# Upload, launch in tmux (idempotent), then wait for completion.
launch_and_wait() {
    local label="$1" local_script="$2" remote_script="$3" \
          tmux_session="$4" runner_path="$5" runner_content="$6" \
          done_flag="$7" exit_flag="$8" log_file="$9"

    if flag_exists "$done_flag"; then
        log "$label: already complete — skipping"
        return 0
    fi

    log "$label: uploading ${local_script##*/}..."
    vm_scp "$local_script" "$remote_script"

    log "$label: writing runner script..."
    write_remote_script "$runner_path" "$runner_content"

    if vm_ssh "tmux has-session -t '$tmux_session' 2>/dev/null && echo running" | grep -q running; then
        log "$label: already running in tmux $tmux_session"
    else
        log "$label: launching in tmux $tmux_session..."
        vm_ssh "mkdir -p ~/pipeline_logs; tmux new-session -d -s '$tmux_session' 2>/dev/null || true; tmux send-keys -t '$tmux_session' 'bash $runner_path' Enter"
        sleep 3
        log "$label: launched"
    fi

    wait_done "$done_flag" "$exit_flag" "$log_file" "$label"
}

# ─────────────────────────────────────────────────────────────────────────────
log "=== CHAIN: int8_qat → xarch_lmc → resnet_restart ==="
log "VM: $VM  ZONE: $ZONE"
log "Poll interval: ${POLL}s"

# ── Stage 0: wait for int8_qat ────────────────────────────────────────────────
if flag_exists /tmp/.int8_qat_done; then
    log "int8_qat: already complete — proceeding"
else
    log "Waiting for int8_qat..."
    wait_done /tmp/.int8_qat_done /tmp/.int8_qat_exit int8_qat.log "int8_qat"
fi

# ── Stage 1: xarch_lmc ───────────────────────────────────────────────────────
XARCH_RUNNER='#!/bin/bash
set -euo pipefail
cd ~/flashoptim
log() { echo "[$(date -u +%Y-%m-%d\ %H:%M:%S\ UTC)] $*"; }
log "=== phase_xarch_lmc start ==="
python3 -u phase_xarch_lmc.py 2>&1 | tee ~/pipeline_logs/xarch_lmc.log
EXIT=${PIPESTATUS[0]}
echo $EXIT > /tmp/.xarch_lmc_exit
[ "$EXIT" = "0" ] && touch /tmp/.xarch_lmc_done && log "COMPLETE" || log "ERROR exit $EXIT"
'

launch_and_wait \
    "xarch_lmc" \
    "$LOCAL_DIR/phase_xarch_lmc.py" "~/flashoptim/phase_xarch_lmc.py" \
    "xl_22" "/tmp/run_xarch_lmc.sh" "$XARCH_RUNNER" \
    "/tmp/.xarch_lmc_done" "/tmp/.xarch_lmc_exit" "xarch_lmc.log"

# ── Stage 2: resnet_restart ───────────────────────────────────────────────────
RESNET_RUNNER='#!/bin/bash
set -euo pipefail
cd ~/flashoptim
log() { echo "[$(date -u +%Y-%m-%d\ %H:%M:%S\ UTC)] $*"; }
log "=== phase_resnet_restart start ==="
python3 -u phase_resnet_restart.py 2>&1 | tee ~/pipeline_logs/resnet_restart.log
EXIT=${PIPESTATUS[0]}
echo $EXIT > /tmp/.resnet_restart_exit
[ "$EXIT" = "0" ] && touch /tmp/.resnet_restart_done && log "COMPLETE" || log "ERROR exit $EXIT"
'

launch_and_wait \
    "resnet_restart" \
    "$LOCAL_DIR/phase_resnet_restart.py" "~/flashoptim/phase_resnet_restart.py" \
    "resnet_r" "/tmp/run_resnet_restart.sh" "$RESNET_RUNNER" \
    "/tmp/.resnet_restart_done" "/tmp/.resnet_restart_exit" "resnet_restart.log"

# ─────────────────────────────────────────────────────────────────────────────
log ""
log "=== ALL STAGES COMPLETE ==="
log "  int8_qat     ✓  → write §4.16 (INT8 QAT — 4th precision vertex)"
log "  xarch_lmc    ✓  → write §4.15 (transformer precision triangle)"
log "  resnet_restart ✓ → write §4.17 (ResNet cross-arch warm restart)"
log "  Commit + push after each section."
