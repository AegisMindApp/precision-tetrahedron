#!/bin/bash
# ============================================================
# lr_range_test.sh — Empirical maximum restart LR from ep80
#
# Loads Condition A epoch-80 checkpoint and trains for 1 epoch
# at each LR in a geometric sweep. The LR where val_mae first
# rises above the ep80 baseline (0.020 eV) identifies the
# practical cliff — the maximum restart LR that doesn't eject
# the model from its current basin.
#
# This bounds the LR-state hypothesis quantitatively:
#   lr < cliff    → stays in basin (or finds adjacent better basin)
#   lr >= cliff   → ejected (higher val_mae than starting point)
#
# Sweep: 1e-6, 3e-6, 1e-5, 3e-5, 5e-5, 1e-4, 3e-4
# (7 conditions × ~8 min = ~56 min total)
#
# Each condition gets a FRESH optimizer (zero momentum) and
# trains exactly 1 epoch from the ep80 checkpoint.
#
# Usage (inside tmux on TPU VM, run CONCURRENTLY with Phase 5):
#   GCS_BUCKET=gs://aegismind-tpu-results bash lr_range_test.sh
# ============================================================

set -euo pipefail
cd /home/john/flashoptim

GCS_BUCKET="${GCS_BUCKET:-gs://aegismind-tpu-results}"
RUN_ID="${RUN_ID:-aegis_flashoptim}"
OUTPUT_DIR="/tmp/lr_range_results"
CKPT="${OUTPUT_DIR}/condition_A_epoch80.pt"
GCS_DEST="${GCS_BUCKET}/${RUN_ID}/lr_range_test"
PYTHON="python3 -u"

mkdir -p "$OUTPUT_DIR"
log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] $*" | tee -a "${OUTPUT_DIR}/lr_range.log"; }

log "=== LR Range Test from ep80 checkpoint ==="
log "Sweep: 1e-6 → 3e-4  (7 LRs × 1 epoch each)"

# Fetch checkpoint
if [ ! -f "$CKPT" ]; then
    log "Fetching condition_A_epoch80.pt from GCS..."
    gsutil cp "${GCS_BUCKET}/${RUN_ID}/condition_A_epoch80.pt" "$CKPT" \
        || cp /tmp/flashoptim_results/condition_A_epoch80.pt "$CKPT" 2>/dev/null \
        || { log "ERROR: cannot find checkpoint"; exit 1; }
fi
log "Checkpoint ready."

# Run the sweep — each LR trains 1 epoch with fresh optimizer
$PYTHON - <<'PYEOF'
import os, sys, json, math, time
import torch
import torch.nn.functional as F

try:
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
    try:
        import torch_xla.experimental as _x
        _x.eager_mode(True)
    except Exception:
        pass
except ImportError:
    XLA_AVAILABLE = False

sys.path.insert(0, '/home/john/flashoptim')
from model import build_model
from data import get_dataloaders, batch_to_graph

OUTPUT_DIR = '/tmp/lr_range_results'
CKPT_PATH  = f'{OUTPUT_DIR}/condition_A_epoch80.pt'
LOG_PATH   = f'{OUTPUT_DIR}/lr_range.log'
EP80_BASELINE = 0.0200   # approximate val_mae at ep80 plateau

def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, 'a') as f: f.write(line + '\n')

if XLA_AVAILABLE:
    device = xm.xla_device()
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

train_loader, val_loader, _ = get_dataloaders('/tmp/qm9', batch_size=32, num_workers=4)
std = val_loader.dataset.std

sd_orig = torch.load(CKPT_PATH, map_location='cpu')

LR_SWEEP = [1e-6, 3e-6, 1e-5, 3e-5, 5e-5, 1e-4, 3e-4]
results = []

log(f"{'LR':>10}  {'val_mae':>10}  {'vs_baseline':>14}  {'verdict'}")
log(f"{'─'*10}  {'─'*10}  {'─'*14}  {'─'*20}")

for lr in LR_SWEEP:
    # Fresh model from ep80 weights
    model = build_model('A', device)
    model.load_state_dict(sd_orig['model'])
    model.train()

    # Fresh optimizer at this LR (no momentum)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Train exactly 1 epoch
    total_loss_t = torch.zeros(1, device=device)
    n = 0
    optimizer.zero_grad()
    for batch in train_loader:
        z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid = \
            batch_to_graph(batch, device)
        target = batch['target'].to(device)
        pred = model(z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid)
        loss = F.mse_loss(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if XLA_AVAILABLE:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        total_loss_t = total_loss_t + loss.detach()
        n += 1

    # Evaluate
    model.eval()
    mae_sum, n_val = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid = \
                batch_to_graph(batch, device)
            target = batch['target'].to(device)
            pred = model(z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid)
            mae_sum += ((pred - target).abs() * std).sum().item()
            n_val += num_graphs
            if XLA_AVAILABLE:
                xm.mark_step()
    val_mae = mae_sum / max(n_val, 1)
    delta = val_mae - EP80_BASELINE
    verdict = 'CLIFF EXCEEDED' if val_mae > EP80_BASELINE else ('IMPROVEMENT' if delta < -0.001 else 'NEUTRAL')

    log(f"{lr:>10.1e}  {val_mae:>10.4f}  {delta:>+14.4f}  {verdict}")
    results.append({'lr': lr, 'val_mae_ev': val_mae, 'delta_vs_baseline': delta, 'verdict': verdict})

# Find cliff
cliff_lr = None
for r in results:
    if r['val_mae_ev'] > EP80_BASELINE:
        cliff_lr = r['lr']
        break

best = min(results, key=lambda r: r['val_mae_ev'])
log(f"\nCliff LR (first LR where val_mae > ep80 baseline {EP80_BASELINE}): {cliff_lr}")
log(f"Best single-epoch LR: {best['lr']:.1e}  →  val_mae={best['val_mae_ev']:.4f} eV")

summary = {
    'sweep': results,
    'ep80_baseline_ev': EP80_BASELINE,
    'cliff_lr': cliff_lr,
    'best_lr': best['lr'],
    'best_val_mae_ev': best['val_mae_ev'],
}
with open(f'{OUTPUT_DIR}/lr_range_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\nResults: {OUTPUT_DIR}/lr_range_results.json")
PYEOF

log "=== LR range test complete ==="
gsutil -m rsync -r "${OUTPUT_DIR}/" "${GCS_DEST}/" 2>/dev/null || true
log "GCS upload: ${GCS_DEST}/"
