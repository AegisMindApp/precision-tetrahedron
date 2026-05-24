#!/bin/bash
# run_pipeline.sh — Full AegisMind TPU pipeline orchestrator
# ============================================================
# Run once on the TPU VM; it chains all phases as tmux sessions
# and monitors each one before launching the next.
#
# Usage:
#   bash run_pipeline.sh
#
# Monitor progress:
#   tmux ls                          — list active sessions
#   tmux attach -t <name>            — view live output
#   tail -f ~/pipeline_logs/<name>.log
#
# Assumes:
#   - phase6 is ALREADY running as tmux session "phase6"
#   - phase7 is ALREADY queued as tmux session "phase7"
#   - ~/flashoptim/ contains all phase*.py scripts

set -euo pipefail

FLASHDIR=~/flashoptim
LOG_DIR=~/pipeline_logs
GCS_BASE="gs://aegismind-tpu-results/aegis_flashoptim"
PYTHON=python3

mkdir -p "$LOG_DIR"

ts() { date -u "+[%Y-%m-%d %H:%M:%S UTC]"; }

log() { echo "$(ts) $*"; }

# ── Helpers ───────────────────────────────────────────────────────────────────

wait_for_session() {
    local sess=$1
    local poll_secs=${2:-30}
    log "Waiting for tmux session: $sess"
    while tmux has-session -t "$sess" 2>/dev/null; do
        sleep "$poll_secs"
    done
    log "Session '$sess' has exited"
}

kill_if_exists() {
    local sess=$1
    if tmux has-session -t "$sess" 2>/dev/null; then
        log "WARNING: session '$sess' already exists — killing before relaunch"
        tmux kill-session -t "$sess" || true
        sleep 2
    fi
}

launch_session() {
    local name=$1
    local cmd=$2
    local log_file="$LOG_DIR/${name}.log"
    kill_if_exists "$name"
    tmux new-session -d -s "$name" \
        "cd $FLASHDIR && $cmd 2>&1 | tee $log_file; echo \"\$(date -u '+[%Y-%m-%d %H:%M:%S UTC]') SESSION_EXIT $name\" >> $log_file"
    log "Launched session: $name"
    log "  cmd : $cmd"
    log "  log : $log_file"
}

check_log_complete() {
    # Returns 0 if PHASE_COMPLETE found in log, 1 otherwise
    local log_file="$LOG_DIR/${1}.log"
    grep -q "PHASE_COMPLETE" "$log_file" 2>/dev/null && return 0 || return 1
}

# ── Phase 6 BO conditional launch ────────────────────────────────────────────

maybe_launch_phase6_bo() {
    log "Checking phase6 bf16 fidelity for BO launch ..."
    local results_json=/tmp/p6_results.json
    gsutil -q cp "${GCS_BASE}/phase6/phase6_surrogate_results.json" \
        "$results_json" 2>/dev/null || true

    local BF16_PASS="False"
    if [ -f "$results_json" ]; then
        BF16_PASS=$($PYTHON -c "
import json, sys
try:
    d = json.load(open('$results_json'))
    print(d.get('bf16', {}).get('fidelity_pass', False))
except Exception as e:
    print('False')
" 2>/dev/null || echo "False")
    fi

    log "Phase6 bf16 fidelity_pass = $BF16_PASS"
    if [ "$BF16_PASS" = "True" ]; then
        log "Fidelity criterion met — launching phase6_vina_bo.py"
        launch_session "phase6bo" "$PYTHON phase6_vina_bo.py"
        wait_for_session "phase6bo" 60
    else
        log "Fidelity criterion NOT met (or results unavailable) — skipping phase6bo"
    fi
}

# ── Summary at end ────────────────────────────────────────────────────────────

print_summary() {
    log "================================================================"
    log "PIPELINE SUMMARY"
    log "================================================================"
    local phases=(
        phase6 phase6bo phase7 phase7b phase7c phase7d
        phase8 phase9 phase10 phase11 phase12
        phase13 phase14 phase15
    )
    for sess in "${phases[@]}"; do
        local log_file="$LOG_DIR/${sess}.log"
        if [ ! -f "$log_file" ]; then
            log "  $sess : NOT RUN (no log)"
            continue
        fi
        if check_log_complete "$sess"; then
            log "  $sess : PHASE_COMPLETE"
        else
            local last_line
            last_line=$(tail -1 "$log_file" 2>/dev/null || echo "(empty log)")
            log "  $sess : INCOMPLETE — last line: $last_line"
        fi
    done
    log "================================================================"
}

# ── Main pipeline ─────────────────────────────────────────────────────────────

log "================================================================"
log "AegisMind TPU Pipeline — starting orchestrator"
log "  FLASHDIR : $FLASHDIR"
log "  LOG_DIR  : $LOG_DIR"
log "  GCS_BASE : $GCS_BASE"
log "================================================================"

# Phase 6 — surrogate training (assumed already running)
log "Waiting for phase6 ..."
if ! tmux has-session -t "phase6" 2>/dev/null; then
    log "WARNING: phase6 session not found — either already done or not started"
    log "Checking for phase6 completion marker in GCS ..."
    gsutil -q cp "${GCS_BASE}/phase6/phase6_surrogate_results.json" \
        /tmp/p6_check.json 2>/dev/null \
        && log "  phase6 results found on GCS — treating as complete" \
        || log "  phase6 results NOT found — launching phase6 now"
    if [ ! -f /tmp/p6_check.json ]; then
        launch_session "phase6" "$PYTHON phase6_vina_surrogate.py"
        wait_for_session "phase6" 60
    fi
else
    wait_for_session "phase6" 60
fi

# Phase 6 BO (conditional on fidelity)
maybe_launch_phase6_bo

# Phase 7 — FP32 LMC (assumed already queued; wait or launch)
if tmux has-session -t "phase7" 2>/dev/null; then
    log "phase7 already running — waiting ..."
    wait_for_session "phase7" 30
else
    log "phase7 session not found — launching ..."
    launch_session "phase7" "$PYTHON phase7_fp32_lmc.py"
    wait_for_session "phase7" 30
fi

# Phase 7b — condition C LMC
launch_session "phase7b" "$PYTHON phase7b_condition_c_lmc.py"
wait_for_session "phase7b" 30

# Phase 7c — Hessian sharpness
launch_session "phase7c" "$PYTHON phase7c_hessian_sharpness.py"
wait_for_session "phase7c" 30

# Phase 7d — extend restart
launch_session "phase7d" "$PYTHON phase7d_extend_restart.py"
wait_for_session "phase7d" 30

# Phase 8 — GAT on QM9
launch_session "phase8" "$PYTHON phase8_gat_qm9.py"
wait_for_session "phase8" 30

# Phase 9 — ESOL solubility
launch_session "phase9" "$PYTHON phase9_esol.py"
wait_for_session "phase9" 30

# Phase 10 — width scaling
launch_session "phase10" "$PYTHON phase10_width_scaling.py"
wait_for_session "phase10" 30

# Phase 11 — gradient noise
launch_session "phase11" "$PYTHON phase11_gradient_noise.py"
wait_for_session "phase11" 30

# Phase 12 — diffusion model on ZINC-250K (~5 days)
log "Launching phase12 (E(3)-diffusion on ZINC-250K, ~5 days) ..."
launch_session "phase12" "$PYTHON phase12_diffusion_zinc.py"
wait_for_session "phase12" 120

# Phase 13 — screen generated molecules
launch_session "phase13" "$PYTHON phase13_screen_generated.py"
wait_for_session "phase13" 30

# Phase 14 — foundation pre-training (~7 days)
log "Launching phase14 (foundation pre-training, ~7 days) ..."
launch_session "phase14" "$PYTHON phase14_foundation_pretrain.py"
wait_for_session "phase14" 120

# Phase 15 — AMR fine-tuning + virtual screen
launch_session "phase15" "$PYTHON phase15_finetune_amr.py"
wait_for_session "phase15" 30

# ── Final summary ─────────────────────────────────────────────────────────────
log "All sessions completed."
print_summary
log "Pipeline orchestrator exiting."
