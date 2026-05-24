#!/bin/bash
# phase161718_orchestrator.sh
# Waits for HD pipeline (phases 21/22/23) to complete then runs:
#   Phase 20 → Phase 16 → Phase 17 → Phase 18 sequentially.
#
# Start in its own tmux session:
#   tmux new-session -d -s ml_orchestrator 'bash ~/flashoptim/phase161718_orchestrator.sh'

set -euo pipefail

LOG() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*"; }
cd ~/flashoptim

DONE_FLAG=/tmp/.phase23_complete

LOG "=== Phase 20/16/17/18 orchestrator started ==="
LOG "Waiting for Phase 23 (ADMET filter) to finish..."

# Wait for HD pipeline completion
while true; do
    if [ -f "$DONE_FLAG" ]; then
        LOG "Phase 23 done flag detected — proceeding"
        break
    fi
    # Also check if phase23 process has exited cleanly
    if ! pgrep -f "phase22_hd_surrogate_bo.py" > /dev/null 2>&1 && \
       ! pgrep -f "phase23_admet_filter.py"    > /dev/null 2>&1; then
        # Check GCS for phase23 completion marker
        if gsutil -q stat gs://aegismind-tpu-results/aegis_flashoptim/phase23_admet/admet_results.json 2>/dev/null; then
            LOG "Phase 23 GCS output detected — proceeding"
            break
        fi
    fi
    sleep 300
done

# ── Phase 20: Uncertainty-gated BO ───────────────────────────────────────────
LOG "Launching Phase 20 (uncertainty-gated BO)..."
tmux new-window -t ml_orchestrator -n phase20
tmux send-keys -t ml_orchestrator:phase20 \
    "cd ~/flashoptim && python3 phase20_uncertainty_bo.py 2>&1 | tee ~/pipeline_logs/phase20.log; echo \$? > /tmp/.phase20_exit" \
    Enter

LOG "Waiting for Phase 20..."
while true; do
    if [ -f /tmp/.phase20_exit ]; then
        EXIT=$(cat /tmp/.phase20_exit)
        if [ "$EXIT" = "0" ]; then
            LOG "Phase 20 complete (exit 0)"
            break
        else
            LOG "ERROR: Phase 20 exited with code $EXIT"
            exit 1
        fi
    fi
    sleep 120
done

# ── Phase 16: Longitudinal LMC ────────────────────────────────────────────────
LOG "Launching Phase 16 (longitudinal LMC)..."
tmux new-window -t ml_orchestrator -n phase16
tmux send-keys -t ml_orchestrator:phase16 \
    "cd ~/flashoptim && python3 phase16_longitudinal_lmc.py 2>&1 | tee ~/pipeline_logs/phase16.log; echo \$? > /tmp/.phase16_exit" \
    Enter

LOG "Waiting for Phase 16..."
while true; do
    if [ -f /tmp/.phase16_exit ]; then
        EXIT=$(cat /tmp/.phase16_exit)
        if [ "$EXIT" = "0" ]; then
            LOG "Phase 16 complete (exit 0)"
            break
        else
            LOG "ERROR: Phase 16 exited with code $EXIT"
            exit 1
        fi
    fi
    sleep 120
done

# ── Phase 17: Precision dial ──────────────────────────────────────────────────
LOG "Launching Phase 17 (precision dial)..."
tmux new-window -t ml_orchestrator -n phase17
tmux send-keys -t ml_orchestrator:phase17 \
    "cd ~/flashoptim && python3 phase17_precision_dial.py 2>&1 | tee ~/pipeline_logs/phase17.log; echo \$? > /tmp/.phase17_exit" \
    Enter

LOG "Waiting for Phase 17..."
while true; do
    if [ -f /tmp/.phase17_exit ]; then
        EXIT=$(cat /tmp/.phase17_exit)
        if [ "$EXIT" = "0" ]; then
            LOG "Phase 17 complete (exit 0)"
            break
        else
            LOG "ERROR: Phase 17 exited with code $EXIT"
            exit 1
        fi
    fi
    sleep 120
done

# ── Phase 18: Cyclic LR FP32 control ─────────────────────────────────────────
LOG "Launching Phase 18 (cyclic LR FP32)..."
tmux new-window -t ml_orchestrator -n phase18
tmux send-keys -t ml_orchestrator:phase18 \
    "cd ~/flashoptim && python3 phase18_cyclic_lr_fp32.py 2>&1 | tee ~/pipeline_logs/phase18.log; echo \$? > /tmp/.phase18_exit" \
    Enter

LOG "Waiting for Phase 18..."
while true; do
    if [ -f /tmp/.phase18_exit ]; then
        EXIT=$(cat /tmp/.phase18_exit)
        if [ "$EXIT" = "0" ]; then
            LOG "Phase 18 complete (exit 0)"
            break
        else
            LOG "ERROR: Phase 18 exited with code $EXIT"
            exit 1
        fi
    fi
    sleep 120
done

LOG "=== ALL ML MECHANISM PHASES (20/16/17/18) COMPLETE ==="
