#!/bin/bash
# chain_hd_pipeline.sh
#
# Races a new TPU VM, then chains: phase21 (HD Vina) → phase22 (HD BO) → phase23 (ADMET)
# Idempotent: completed stages are skipped. 3-strike preemption detection.
#
# Usage:
#   tmux new-session -d -s hd_chain
#   tmux send-keys -t hd_chain "bash analysis/flashoptim_tpu/chain_hd_pipeline.sh 2>&1 | tee /tmp/hd_chain.log" Enter

set -uo pipefail

PROJECT="aegismind-tpu"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLL=30
RACE_ZONES=("europe-west4-a" "us-east1-d" "us-central1-b" "us-east5-b")
RACE_VMS=("aegis-node-v6e-eu1" "aegis-node-v6e-b6" "aegis-node-v6e-b7" "aegis-node-v6e-b5")
RACE_QRS=("hd-eu" "hd-east1" "hd-central" "hd-east5")
ACCEL="v6e-8"
VERSION="v2-alpha-tpuv6e"

WIN_VM=""; WIN_ZONE=""; WIN_QR=""

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }

vm_ssh()  { gcloud compute tpus tpu-vm ssh "$WIN_VM" --zone "$WIN_ZONE" --project "$PROJECT" --command "$1" 2>/dev/null; }
vm_scp()  { gcloud compute tpus tpu-vm scp "$1" "${WIN_VM}:$2" --zone "$WIN_ZONE" --project "$PROJECT" --worker all 2>/dev/null; }
vm_alive(){ vm_ssh "echo ok" | grep -q ok; }

flag_exists() { vm_ssh "[ -f '$1' ] && echo yes" | grep -q yes; }

write_remote_script() {
    local remote_path="$1" content="$2"
    local b64; b64=$(printf '%s' "$content" | base64 -w0)
    vm_ssh "echo '$b64' | base64 -d > '$remote_path' && chmod +x '$remote_path'"
}

submit_qr() {
    local qr=$1 zone=$2 vm=$3
    gcloud compute tpus queued-resources create "$qr" \
        --node-id "$vm" --zone "$zone" --project "$PROJECT" \
        --accelerator-type "$ACCEL" --runtime-version "$VERSION" \
        --spot 2>&1 | tail -2 || true
}

delete_qr() {
    local qr=$1 zone=$2
    gcloud compute tpus queued-resources delete "$qr" \
        --zone "$zone" --project "$PROJECT" --quiet 2>/dev/null || true
}

wait_done() {
    local done_flag="$1" exit_flag="$2" log_file="$3" label="$4"
    local missed=0
    while true; do
        if flag_exists "$done_flag"; then log "$label: COMPLETE ✓"; return 0; fi
        if flag_exists "$exit_flag"; then
            local code; code=$(vm_ssh "cat '$exit_flag'" || echo "?")
            [ "$code" = "0" ] && { log "$label: COMPLETE ✓"; return 0; }
            log "FATAL: $label failed (exit=$code)"; exit 1
        fi
        if ! vm_alive; then
            missed=$((missed + 1))
            if [ "$missed" -ge 3 ]; then
                log "FATAL: VM unreachable for 3 polls — likely preempted during $label"
                log "  Re-run this script — completed stages are skipped automatically"
                exit 1
            fi
            log "  WARNING: VM ping failed ($missed/3)..."
        else
            missed=0
            local tail_line; tail_line=$(vm_ssh "tail -1 ~/pipeline_logs/$log_file 2>/dev/null" || echo "…")
            log "  [$label] $tail_line"
        fi
        sleep "$POLL"
    done
}

launch_and_wait() {
    local label="$1" local_script="$2" remote_script="$3" \
          tmux_session="$4" runner_path="$5" runner_content="$6" \
          done_flag="$7" exit_flag="$8" log_file="$9"

    if flag_exists "$done_flag"; then log "$label: already complete — skipping"; return 0; fi

    log "$label: uploading..."
    vm_scp "$local_script" "$remote_script"
    write_remote_script "$runner_path" "$runner_content"

    if vm_ssh "tmux has-session -t '$tmux_session' 2>/dev/null && echo running" | grep -q running; then
        log "$label: already running in tmux $tmux_session"
    else
        log "$label: launching in tmux $tmux_session..."
        vm_ssh "mkdir -p ~/pipeline_logs; tmux new-session -d -s '$tmux_session' 2>/dev/null || true; tmux send-keys -t '$tmux_session' 'bash $runner_path' Enter"
        sleep 3
    fi
    wait_done "$done_flag" "$exit_flag" "$log_file" "$label"
}

# ── Race for a VM ─────────────────────────────────────────────────────────────
log "=== HD PIPELINE: phase21 → phase22 → phase23 ==="

# Submit all QRs
for i in "${!RACE_QRS[@]}"; do
    qr="${RACE_QRS[$i]}"; zone="${RACE_ZONES[$i]}"; vm="${RACE_VMS[$i]}"
    state=$(gcloud compute tpus queued-resources describe "$qr" \
        --zone "$zone" --project "$PROJECT" \
        --format 'value(state.state)' 2>/dev/null || echo "GONE")
    if [ "$state" = "GONE" ] || [ "$state" = "FAILED" ] || [ "$state" = "SUSPENDED" ]; then
        log "  Submitting $qr in $zone..."
        submit_qr "$qr" "$zone" "$vm"
    else
        log "  $qr: $state"
    fi
done

# Poll until a winner
while true; do
    for i in "${!RACE_QRS[@]}"; do
        qr="${RACE_QRS[$i]}"; zone="${RACE_ZONES[$i]}"; vm="${RACE_VMS[$i]}"
        state=$(gcloud compute tpus queued-resources describe "$qr" \
            --zone "$zone" --project "$PROJECT" \
            --format 'value(state.state)' 2>/dev/null || echo "GONE")
        log "  $qr ($zone): $state"
        if [ "$state" = "ACTIVE" ]; then
            WIN_QR="$qr"; WIN_ZONE="$zone"; WIN_VM="$vm"
            log "WINNER: $WIN_QR -> $WIN_VM in $WIN_ZONE"
            break 2
        fi
        if [ "$state" = "SUSPENDED" ] || [ "$state" = "FAILED" ] || [ "$state" = "GONE" ]; then
            delete_qr "$qr" "$zone"; submit_qr "$qr" "$zone" "$vm"
        fi
    done
    sleep "$POLL"
done

# Cancel losers
for i in "${!RACE_QRS[@]}"; do
    qr="${RACE_QRS[$i]}"; [ "$qr" = "$WIN_QR" ] && continue
    delete_qr "$qr" "${RACE_ZONES[$i]}"
done

# Wait for READY
log "Waiting for $WIN_VM READY..."
for _ in $(seq 1 40); do
    STATUS=$(gcloud compute tpus tpu-vm describe "$WIN_VM" \
        --zone "$WIN_ZONE" --project "$PROJECT" \
        --format 'value(state)' 2>/dev/null || echo "UNKNOWN")
    log "  VM: $STATUS"
    [ "$STATUS" = "READY" ] && break
    sleep 15
done
sleep 15

# Upload and setup
log "Uploading code..."
gcloud compute tpus tpu-vm scp --recurse "$LOCAL_DIR" "${WIN_VM}:~/flashoptim" \
    --zone "$WIN_ZONE" --project "$PROJECT" --worker all
gcloud compute tpus tpu-vm ssh "$WIN_VM" --zone "$WIN_ZONE" --project "$PROJECT" \
    --command 'cd ~/flashoptim && bash setup.sh 2>&1 | tail -10'
gcloud compute tpus tpu-vm ssh "$WIN_VM" --zone "$WIN_ZONE" --project "$PROJECT" \
    --command '
        mkdir -p /tmp/qm9/raw
        printf "0\n\n" > /tmp/qm9/raw/uncharacterized.txt
        cp /tmp/qm9/raw/uncharacterized.txt /tmp/qm9/raw/3195404
    ' || true
log "Setup done."

# ── Stage 0: resnet_lmc (CPU, runs in parallel with phase21) ─────────────────
RLMC_RUNNER='#!/bin/bash
set -euo pipefail
cd ~/flashoptim
log() { echo "[$(date -u +%Y-%m-%d\ %H:%M:%S\ UTC)] $*"; }
log "=== phase_resnet_lmc start (CPU — FP32↔BF16 CIFAR-10 basin test) ==="
python3 -u phase_resnet_lmc.py 2>&1 | tee ~/pipeline_logs/resnet_lmc.log
EXIT=${PIPESTATUS[0]}
echo $EXIT > /tmp/.resnet_lmc_exit
[ "$EXIT" = "0" ] && touch /tmp/.resnet_lmc_done && log "COMPLETE" || log "ERROR exit $EXIT"
'
if flag_exists /tmp/.resnet_lmc_done; then
    log "resnet_lmc: already complete — skipping"
else
    log "resnet_lmc: launching in background (CPU)..."
    vm_scp "$LOCAL_DIR/phase_resnet_lmc.py" "~/flashoptim/phase_resnet_lmc.py"
    write_remote_script "/tmp/run_resnet_lmc.sh" "$RLMC_RUNNER"
    vm_ssh "mkdir -p ~/pipeline_logs; tmux new-session -d -s rlmc 2>/dev/null || true; tmux send-keys -t rlmc 'bash /tmp/run_resnet_lmc.sh' Enter"
    log "resnet_lmc: running in tmux rlmc (non-blocking)"
fi

# ── Stage 1: phase21 HD Vina screen ──────────────────────────────────────────
P21_RUNNER='#!/bin/bash
set -euo pipefail
cd ~/flashoptim
log() { echo "[$(date -u +%Y-%m-%d\ %H:%M:%S\ UTC)] $*"; }
log "=== installing openbabel ==="
sudo apt-get install -y openbabel 2>&1 | tail -2 || pip install openbabel-wheel --quiet || true
log "=== phase21_hd_vina_screen start ==="
python3 -u phase21_hd_vina_screen.py 2>&1 | tee ~/pipeline_logs/phase21.log
EXIT=${PIPESTATUS[0]}
echo $EXIT > /tmp/.phase21_exit
[ "$EXIT" = "0" ] && touch /tmp/.phase21_done && log "COMPLETE" || log "ERROR exit $EXIT"
'
launch_and_wait "phase21" "$LOCAL_DIR/phase21_hd_vina_screen.py" "~/flashoptim/phase21_hd_vina_screen.py" \
    "hd21" "/tmp/run_phase21.sh" "$P21_RUNNER" \
    "/tmp/.phase21_done" "/tmp/.phase21_exit" "phase21.log"

# ── Stage 2: phase22 HD surrogate BO ─────────────────────────────────────────
P22_RUNNER='#!/bin/bash
set -euo pipefail
cd ~/flashoptim
log() { echo "[$(date -u +%Y-%m-%d\ %H:%M:%S\ UTC)] $*"; }
log "=== phase22_hd_surrogate_bo start ==="
python3 -u phase22_hd_surrogate_bo.py 2>&1 | tee ~/pipeline_logs/phase22.log
EXIT=${PIPESTATUS[0]}
echo $EXIT > /tmp/.phase22_exit
[ "$EXIT" = "0" ] && touch /tmp/.phase22_done && log "COMPLETE" || log "ERROR exit $EXIT"
'
launch_and_wait "phase22" "$LOCAL_DIR/phase22_hd_surrogate_bo.py" "~/flashoptim/phase22_hd_surrogate_bo.py" \
    "hd22" "/tmp/run_phase22.sh" "$P22_RUNNER" \
    "/tmp/.phase22_done" "/tmp/.phase22_exit" "phase22.log"

# ── Stage 3: phase23 ADMET filter ────────────────────────────────────────────
P23_RUNNER='#!/bin/bash
set -euo pipefail
cd ~/flashoptim
log() { echo "[$(date -u +%Y-%m-%d\ %H:%M:%S\ UTC)] $*"; }
log "=== phase23_admet_filter start ==="
python3 -u phase23_admet_filter.py 2>&1 | tee ~/pipeline_logs/phase23.log
EXIT=${PIPESTATUS[0]}
echo $EXIT > /tmp/.phase23_exit
[ "$EXIT" = "0" ] && touch /tmp/.phase23_done && log "COMPLETE" || log "ERROR exit $EXIT"
'
launch_and_wait "phase23" "$LOCAL_DIR/phase23_admet_filter.py" "~/flashoptim/phase23_admet_filter.py" \
    "hd23" "/tmp/run_phase23.sh" "$P23_RUNNER" \
    "/tmp/.phase23_done" "/tmp/.phase23_exit" "phase23.log"

# ── Done ──────────────────────────────────────────────────────────────────────
log ""
log "=== HD PIPELINE COMPLETE ==="
log "  phase21 (HD Vina)     ✓  → write §4.18 (HD Vina screen)"
log "  phase22 (HD BO)       ✓  → write §4.19 (HD surrogate BO)"
log "  phase23 (ADMET)       ✓  → write §4.20 (ADMET/CNS filter)"
log "  Commit + push after each section."
