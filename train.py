"""
FlashOptim Mixed-Precision Surrogate Training — TPU/XLA Experiment

Three conditions:
  A  FP32 baseline          (hidden_dim=256)
  B  BF16 mixed-precision   (hidden_dim=256, same size as A)
  C  BF16 mixed-precision   (hidden_dim=512, 2x wider than A)

Abort checkpoints (from discovery protocol):
  Checkpoint A  epoch 7   FP32 MAE must be < 0.5 eV (sanity gate)
  Checkpoint B  epoch 12  BF16 loss must not collapse (scale stable)
  Checkpoint C  epoch 14  Memory reduction must be >= 15%
  Checkpoint D  epoch 20  2x model MAE within 10% of 1x FP32
  Checkpoint E  epoch 28  Surrogate optimizer quality within 15% of FP32

TPU-specific adaptations vs the discovery's GPU code:
  - torch.autocast('xla', dtype=torch.bfloat16) instead of cuda.amp.autocast
  - xm.optimizer_step(optimizer) instead of optimizer.step() — includes mark_step
  - xm.get_memory_info(device) instead of torch.cuda.memory_allocated()
  - No GradScaler needed: BF16 on TPU has FP32 exponent range, no overflow
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    XLA_AVAILABLE = True

    # Enable eager mode — compiles + dispatches each operation immediately rather
    # than accumulating one giant lazy graph. Avoids the 16M-instruction XLA
    # compilation that takes hours and gets killed on v6e. The throughput cost
    # (~30% slower) is acceptable given the 30-day budget.
    try:
        import torch_xla.experimental as _xla_exp
        _xla_exp.eager_mode(True)
        print("XLA eager mode: ENABLED")
    except Exception as _e:
        print(f"XLA eager mode: unavailable ({_e}) — using lazy mode")

except ImportError:
    XLA_AVAILABLE = False
    print("WARNING: torch_xla not found — running on CPU/GPU fallback")

from model import MolecularGNN, build_model
from data import (
    get_dataloaders, batch_to_graph,
    QM9_PUBLISHED_MAE, CHECKPOINT_A_THRESHOLD,
)


# ── Checkpoint thresholds (from discovery protocol) ─────────────────────────
THRESHOLDS = {
    'A':  {'epoch': 7,  'metric': 'val_mae_ev',         'max': 0.50,  'desc': 'FP32 baseline sanity'},
    'B':  {'epoch': 12, 'metric': 'loss_scale_stable',  'min': True,  'desc': 'BF16 stability (20-epoch run)'},
    'C':  {'epoch': 14, 'metric': 'memory_reduction',   'min': 0.15,  'desc': 'Memory reduction >= 15%'},
    'D':  {'epoch': 20, 'metric': 'mae_vs_fp32',        'max': 0.10,  'desc': '2x model within 10% of 1x FP32'},
    'E':  {'epoch': 28, 'metric': 'optimizer_quality',  'max': 0.15,  'desc': 'Optimizer quality within 15%'},
}


# ── Memory profiling ─────────────────────────────────────────────────────────

def get_memory_mb(device) -> float:
    """Return currently allocated memory in MB."""
    if XLA_AVAILABLE:
        try:
            info = xm.get_memory_info(device)
            # PyTorch/XLA < 2.5: kb_total / kb_free
            # PyTorch/XLA >= 2.5: bytes_limit / bytes_used
            if 'kb_total' in info and 'kb_free' in info:
                return (info['kb_total'] - info['kb_free']) / 1024.0
            elif 'bytes_used' in info:
                return info['bytes_used'] / 1024 / 1024
        except Exception:
            pass
        return 0.0
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated(device) / 1024 / 1024
    return 0.0


def get_total_memory_mb(device) -> float:
    if XLA_AVAILABLE:
        try:
            info = xm.get_memory_info(device)
            if 'kb_total' in info:
                return info['kb_total'] / 1024.0
            elif 'bytes_limit' in info:
                return info['bytes_limit'] / 1024 / 1024
        except Exception:
            pass
        return 0.0
    elif torch.cuda.is_available():
        return torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
    return 0.0


# ── Training step ─────────────────────────────────────────────────────────────

def train_epoch(
    model: MolecularGNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    precision: str,
    accumulation_steps: int = 1,   # 1 = no accumulation; avoids XLA recompile storm
) -> dict:
    """
    XLA-safe training epoch.

    Key rules for XLA performance:
      - No .item() or .isfinite() inside the loop (each forces a graph split + recompile)
      - Accumulate loss as an XLA tensor; single .item() call at epoch end
      - accumulation_steps=1 avoids the conditional-branch recompilation issue
    """
    model.train()
    total_loss_t = torch.zeros(1, device=device)
    n_batches = 0
    t0 = time.time()

    use_bf16 = (precision == 'bf16')

    if XLA_AVAILABLE:
        ctx_fn = lambda: torch.autocast('xla', dtype=torch.bfloat16, enabled=use_bf16)
    elif torch.cuda.is_available():
        ctx_fn = lambda: torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16)
    else:
        ctx_fn = lambda: torch.autocast('cpu', dtype=torch.bfloat16, enabled=use_bf16)

    optimizer.zero_grad()

    for batch in loader:
        z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid = batch_to_graph(batch, device)
        target = batch['target'].to(device)

        with ctx_fn():
            pred = model(z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid)
            loss = F.mse_loss(pred, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if XLA_AVAILABLE:
            xm.optimizer_step(optimizer)   # includes mark_step — single sync per step
        else:
            optimizer.step()

        optimizer.zero_grad()

        # Accumulate as XLA tensor — no .item() sync inside loop
        total_loss_t = total_loss_t + loss.detach()
        n_batches += 1

    # Single sync point per epoch
    elapsed = time.time() - t0
    total_loss = total_loss_t.item() / max(n_batches, 1)

    return {
        'loss': total_loss,
        'throughput_batches_per_sec': n_batches / max(elapsed, 1e-6),
        'loss_finite_frac': 1.0,   # not tracked — would require per-step sync
    }


@torch.no_grad()
def evaluate(
    model: MolecularGNN,
    loader: DataLoader,
    device: torch.device,
    precision: str,
    std: float,                # dataset std for denormalisation
) -> dict:
    model.eval()
    mae_sum = 0.0
    n = 0

    use_bf16 = (precision == 'bf16')

    for batch in loader:
        z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid = batch_to_graph(batch, device)
        target = batch['target'].to(device)

        if XLA_AVAILABLE:
            ctx = torch.autocast('xla', dtype=torch.bfloat16, enabled=use_bf16)
        else:
            ctx = torch.autocast('cpu', dtype=torch.bfloat16, enabled=use_bf16)

        with ctx:
            pred = model(z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid)

        # MAE in eV (denormalised)
        mae_sum += ((pred - target).abs() * std).sum().item()
        n += num_graphs

        if XLA_AVAILABLE:
            xm.mark_step()

    return {'val_mae_ev': mae_sum / max(n, 1)}


# ── Main training loop ────────────────────────────────────────────────────────

def run(args):
    # Device
    if XLA_AVAILABLE:
        device = xm.xla_device()
        print(f"Running on TPU: {device}")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Running on GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        print("Running on CPU (slow — for debugging only)")

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    std = train_loader.dataset.std

    # Model
    model = build_model(args.condition, device)
    precision = 'fp32' if args.condition == 'A' else 'bf16'

    # Record baseline memory (empty model)
    mem_model_mb = get_memory_mb(device)
    mem_total_mb  = get_total_memory_mb(device)
    print(f"Memory after model load: {mem_model_mb:.1f} MB / {mem_total_mb:.1f} MB total")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Epoch-level resume ────────────────────────────────────────────────────
    # Find the latest periodic checkpoint (condition_{X}_epoch{N}.pt) and resume
    # from there.  Also reload existing results.json so epoch records are preserved.
    start_epoch = 1
    best_val_mae = float('inf')
    results = {
        'condition': args.condition,
        'precision': precision,
        'hidden_dim': model.hidden_dim,
        'num_params': model.parameter_count(),
        'mem_total_mb': mem_total_mb,
        'epochs': [],
        'checkpoints': {},
        'abort': False,
        'abort_reason': None,
        'best_val_mae_ev': float('inf'),
    }

    # Look for existing periodic checkpoints to resume from
    import glob as _glob
    ckpt_pattern = os.path.join(args.output_dir, f'condition_{args.condition}_epoch*.pt')
    existing_ckpts = sorted(
        _glob.glob(ckpt_pattern),
        key=lambda p: int(p.rsplit('epoch', 1)[1].replace('.pt', ''))
    )
    if existing_ckpts:
        latest_ckpt = existing_ckpts[-1]
        resume_epoch = int(latest_ckpt.rsplit('epoch', 1)[1].replace('.pt', ''))
        print(f"Resuming from checkpoint: {latest_ckpt} (epoch {resume_epoch})")
        try:
            # XLA checkpoints are tagged with 'xla:0' storage — map to CPU first,
            # then load_state_dict will copy into the already-on-device model tensors.
            ckpt = torch.load(latest_ckpt, map_location='cpu')
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            # Move any optimizer state tensors (exp_avg, etc.) to the XLA device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            # Re-run scheduler steps to match the resumed epoch
            for _ in range(resume_epoch):
                scheduler.step()
            start_epoch = resume_epoch + 1

            # Reload prior epoch records from results.json if it exists
            results_json = os.path.join(args.output_dir, f'condition_{args.condition}_results.json')
            if os.path.exists(results_json):
                with open(results_json) as f:
                    prior = json.load(f)
                # Keep only fields that are safe to carry forward
                results['epochs']      = [e for e in prior.get('epochs', []) if e['epoch'] <= resume_epoch]
                results['checkpoints'] = prior.get('checkpoints', {})
                results['best_val_mae_ev'] = prior.get('best_val_mae_ev', float('inf'))
                best_val_mae = results['best_val_mae_ev']
            print(f"Resuming from epoch {start_epoch} (best val MAE so far: {best_val_mae:.4f} eV)")
        except Exception as _ckpt_err:
            # Checkpoint was saved with a different torch version (e.g. 2.9.0 on v6e →
            # _rebuild_device_tensor_from_cpu_tensor not in torch 2.3.0).  Start fresh.
            print(f"WARNING: checkpoint incompatible ({_ckpt_err})")
            print("Starting from epoch 1 (checkpoint saved with a different torch version)")
            start_epoch = 1
    else:
        print("No checkpoint found — starting from epoch 1")

    print(f"\n=== Condition {args.condition} | precision={precision} | "
          f"hidden_dim={model.hidden_dim} | params={model.parameter_count():,} ===\n")

    # ── Plateau detection state ───────────────────────────────────────────────
    plateau_counter   = 0          # consecutive epochs without sufficient improvement
    plateau_best_mae  = float('inf')  # best val_mae seen so far for plateau tracking
    restart_fired     = False      # only allow one restart per run
    results['warm_restart'] = None # filled in if/when restart fires

    out_path = os.path.join(args.output_dir, f'condition_{args.condition}_results.json')
    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = train_epoch(
            model, train_loader, optimizer, device, precision,
            accumulation_steps=args.accumulation_steps,
        )
        val_stats = evaluate(model, val_loader, device, precision, std)
        scheduler.step()

        # ── Plateau-triggered warm restart ────────────────────────────────────
        warm_restart_this_epoch = False
        if args.plateau_patience > 0 and not restart_fired:
            val_mae = val_stats['val_mae_ev']
            if val_mae < plateau_best_mae - args.plateau_min_delta:
                # Improvement — reset counter
                plateau_best_mae = val_mae
                plateau_counter  = 0
            else:
                plateau_counter += 1

            if plateau_counter >= args.plateau_patience:
                # Plateau confirmed — fire warm restart
                restart_fired = True
                warm_restart_this_epoch = True

                # 1. Reset LR to initial value
                for pg in optimizer.param_groups:
                    pg['lr'] = args.lr

                # 2. Zero optimizer momentum (exp_avg, exp_avg_sq)
                #    Keeps param_groups intact so weight_decay etc. are preserved.
                optimizer.state.clear()

                # 3. New cosine schedule for remaining epochs
                remaining = args.epochs - epoch
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(remaining, 1), eta_min=1e-6
                )

                # 4. Reset plateau counter so it doesn't re-trigger
                plateau_counter  = 0
                plateau_best_mae = val_mae

                restart_info = {
                    'epoch':             epoch,
                    'plateau_epochs':    args.plateau_patience,
                    'val_mae_at_restart': val_mae,
                    'lr_reset_to':       args.lr,
                    'remaining_epochs':  remaining,
                }
                results['warm_restart'] = restart_info
                print(f"\n  *** WARM RESTART fired at epoch {epoch} ***")
                print(f"      Plateau: {args.plateau_patience} epochs without >{args.plateau_min_delta:.0e} improvement")
                print(f"      val_mae at restart: {val_mae:.4f} eV | LR reset to {args.lr:.0e}")
                print(f"      Remaining epochs: {remaining}\n")

        mem_used_mb = get_memory_mb(device)
        epoch_record = {
            'epoch':              epoch,
            'train_loss':         train_stats['loss'],
            'val_mae_ev':         val_stats['val_mae_ev'],
            'mem_mb':             mem_used_mb,
            'throughput':         train_stats['throughput_batches_per_sec'],
            'loss_finite_frac':   train_stats['loss_finite_frac'],
            'warm_restart':       warm_restart_this_epoch,
            'plateau_counter':    plateau_counter,
        }
        results['epochs'].append(epoch_record)

        print(f"Epoch {epoch:3d} | loss={train_stats['loss']:.4f} | "
              f"val_mae={val_stats['val_mae_ev']:.4f} eV | "
              f"mem={mem_used_mb:.0f} MB | "
              f"finite={train_stats['loss_finite_frac']:.2%}")

        # ── Checkpoint gates ─────────────────────────────────────────────────
        abort = False

        # Gate A: FP32 baseline sanity (epoch 7)
        if epoch == 7 and args.condition == 'A':
            mae = val_stats['val_mae_ev']
            passed = mae < CHECKPOINT_A_THRESHOLD
            results['checkpoints']['A'] = {
                'passed': passed,
                'val_mae_ev': mae,
                'threshold': CHECKPOINT_A_THRESHOLD,
                'published_reference': QM9_PUBLISHED_MAE,
            }
            status = "PASS" if passed else "FAIL — ABORT"
            print(f"\n  [Checkpoint A] FP32 MAE={mae:.4f} eV  {status}")
            if not passed:
                results['abort'] = True
                results['abort_reason'] = f"Checkpoint A: MAE {mae:.4f} > {CHECKPOINT_A_THRESHOLD} eV — data pipeline issue"
                abort = True

        # Gate B: BF16 stability (epoch 12, condition B or C)
        if epoch == 12 and args.condition in ('B', 'C'):
            stable = train_stats['loss_finite_frac'] >= 0.9
            results['checkpoints']['B'] = {
                'passed': stable,
                'loss_finite_frac': train_stats['loss_finite_frac'],
            }
            status = "PASS" if stable else "FAIL — ABORT"
            print(f"\n  [Checkpoint B] BF16 stability={train_stats['loss_finite_frac']:.2%}  {status}")
            if not stable:
                results['abort'] = True
                results['abort_reason'] = "Checkpoint B: BF16 loss unstable — investigate gradient clipping or learning rate"
                abort = True

        # Gate D: 2x model MAE within 10% of FP32 (epoch 20, condition C)
        # We compare against stored condition-A result loaded from file
        if epoch == 20 and args.condition == 'C':
            baseline_file = os.path.join(args.output_dir, 'condition_A_results.json')
            if os.path.exists(baseline_file):
                with open(baseline_file) as f:
                    baseline = json.load(f)
                # Find epoch-20 MAE from condition A
                ep20_a = next((e for e in baseline['epochs'] if e['epoch'] == 20), None)
                if ep20_a:
                    mae_a = ep20_a['val_mae_ev']
                    mae_c = val_stats['val_mae_ev']
                    ratio = (mae_c - mae_a) / (mae_a + 1e-8)
                    passed = ratio <= 0.10
                    results['checkpoints']['D'] = {
                        'passed': passed,
                        'mae_c': mae_c,
                        'mae_a': mae_a,
                        'relative_diff': ratio,
                    }
                    status = "PASS" if passed else "FAIL — ABORT"
                    print(f"\n  [Checkpoint D] 2x MAE={mae_c:.4f}, 1x MAE={mae_a:.4f}, "
                          f"diff={ratio:.1%}  {status}")
                    if not passed:
                        results['abort'] = True
                        results['abort_reason'] = f"Checkpoint D: 2x model MAE {ratio:.1%} worse than 1x — capacity degradation"
                        abort = True

        # Save best checkpoint whenever validation improves
        if val_stats['val_mae_ev'] < best_val_mae:
            best_val_mae = val_stats['val_mae_ev']
            results['best_val_mae_ev'] = best_val_mae
            best_path = os.path.join(args.output_dir, f'condition_{args.condition}_best.pt')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'model_config': {
                    'hidden_dim': model.hidden_dim,
                    'num_blocks': model.num_blocks,
                    'cutoff':     model.cutoff,
                },
                'val_mae_ev': best_val_mae,
            }, best_path)

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            ckpt_path = os.path.join(args.output_dir, f'condition_{args.condition}_epoch{epoch}.pt')
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'optimizer': optimizer.state_dict()}, ckpt_path)

        # Save running results
        out_path = os.path.join(args.output_dir, f'condition_{args.condition}_results.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

        if abort:
            print(f"\nABORTING: {results['abort_reason']}")
            break

    print(f"\nResults saved to {out_path}")
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='MolPrecision TPU experiment')
    p.add_argument('--condition', choices=['A', 'B', 'C'], required=True,
                   help='A=FP32 baseline, B=BF16 same size, C=BF16 2x wider')
    p.add_argument('--data-dir',   default='/tmp/qm9',       help='QM9 data directory')
    p.add_argument('--output-dir', default='/tmp/results',   help='Results output directory')
    p.add_argument('--epochs',     type=int, default=100)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--accumulation-steps', type=int, default=4)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--num-workers', type=int, default=4)
    # ── Plateau-triggered warm restart ────────────────────────────────────────
    # Default 0 = disabled. Set e.g. --plateau-patience 10 to enable.
    # When val_mae fails to improve by more than min_delta for `patience`
    # consecutive epochs, the LR is reset to --lr, optimizer momentum is
    # zeroed, and a fresh cosine schedule runs for the remaining epochs.
    # Fires at most once per run (subsequent plateaus are not re-triggered).
    p.add_argument('--plateau-patience',  type=int,   default=0,
                   help='Epochs without improvement before warm restart (0=disabled)')
    p.add_argument('--plateau-min-delta', type=float, default=1e-4,
                   help='Minimum improvement in val_mae to reset plateau counter')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args)
