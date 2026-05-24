#!/usr/bin/env python3
"""
restart_trainer.py
------------------
Warm restart training from an existing checkpoint, with a clean optimizer
and explicitly anchored cosine schedule.

Key difference from warm_restart.sh / train.py resume:
  - Builds a FRESH optimizer at the specified restart LR (not inherited
    from checkpoint param_groups, which may be decayed or mismatched).
  - CosineAnnealingLR is constructed AFTER the fresh optimizer, so
    base_lrs = [restart_lr] exactly, and T_max = epochs_after_restart.
  - This avoids the scheduler / load_state_dict interaction that caused
    Phase 4b's ep82-100 degradation.

Usage examples:

  # Condition C — BF16 512-dim, restart from ep40 → train 30 more epochs
  python3 restart_trainer.py \
      --condition C \
      --checkpoint /tmp/flashoptim_results/condition_C_epoch40.pt \
      --restart-lr 5e-5 \
      --epochs-after 30 \
      --output-dir /tmp/c_restart_results

  # Condition A — FP32 256-dim, restart from ep80 → train 20 more epochs
  python3 restart_trainer.py \
      --condition A \
      --checkpoint /tmp/flashoptim_results/condition_A_epoch80.pt \
      --restart-lr 5e-5 \
      --epochs-after 20 \
      --output-dir /tmp/a_restart_results
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn.functional as F

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    XLA_AVAILABLE = True
    try:
        import torch_xla.experimental as _xla_exp
        _xla_exp.eager_mode(True)
        print("XLA eager mode: ENABLED")
    except Exception as _e:
        print(f"XLA eager mode: unavailable ({_e})")
except ImportError:
    XLA_AVAILABLE = False
    print("WARNING: torch_xla not found — CPU/GPU fallback")

# Import shared model + data from existing flashoptim codebase
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model
from data import get_dataloaders, batch_to_graph


def log(msg, log_path):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}] {msg}"
    print(line, flush=True)
    with open(log_path, 'a') as f:
        f.write(line + '\n')


def get_memory_mb(device) -> float:
    if XLA_AVAILABLE:
        try:
            info = xm.get_memory_info(device)
            if 'kb_total' in info and 'kb_free' in info:
                return (info['kb_total'] - info['kb_free']) / 1024.0
            elif 'bytes_used' in info:
                return info['bytes_used'] / 1024 / 1024
        except Exception:
            pass
    elif torch.cuda.is_available():
        return torch.cuda.memory_allocated(device) / 1024 / 1024
    return 0.0


def train_epoch(model, loader, optimizer, device, precision):
    model.train()
    total_loss_t = torch.zeros(1, device=device)
    n_batches = 0
    t0 = time.time()
    use_bf16 = (precision == 'bf16')

    if XLA_AVAILABLE:
        ctx_fn = lambda: torch.autocast('xla', dtype=torch.bfloat16, enabled=use_bf16)
    else:
        ctx_fn = lambda: torch.autocast('cpu', dtype=torch.bfloat16, enabled=use_bf16)

    optimizer.zero_grad()
    for batch in loader:
        z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid = \
            batch_to_graph(batch, device)
        target = batch['target'].to(device)
        with ctx_fn():
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
        n_batches += 1

    elapsed = time.time() - t0
    return {
        'loss': total_loss_t.item() / max(n_batches, 1),
        'throughput': n_batches / max(elapsed, 1e-6),
    }


@torch.no_grad()
def evaluate(model, loader, device, precision, std):
    model.eval()
    mae_sum = 0.0
    n = 0
    use_bf16 = (precision == 'bf16')
    for batch in loader:
        z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid = \
            batch_to_graph(batch, device)
        target = batch['target'].to(device)
        if XLA_AVAILABLE:
            ctx = torch.autocast('xla', dtype=torch.bfloat16, enabled=use_bf16)
        else:
            ctx = torch.autocast('cpu', dtype=torch.bfloat16, enabled=use_bf16)
        with ctx:
            pred = model(z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid)
        mae_sum += ((pred - target).abs() * std).sum().item()
        n += num_graphs
        if XLA_AVAILABLE:
            xm.mark_step()
    return mae_sum / max(n, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--condition',    choices=['A', 'B', 'C'], required=True)
    p.add_argument('--checkpoint',   required=True,
                   help='Path to the .pt checkpoint file to restart from')
    p.add_argument('--restart-lr',   type=float, default=5e-5,
                   help='Learning rate at the restart point (fresh, not inherited from checkpoint)')
    p.add_argument('--epochs-after', type=int, required=True,
                   help='Number of epochs to train after the restart')
    p.add_argument('--data-dir',     default='/tmp/qm9')
    p.add_argument('--output-dir',   default='/tmp/restart_results')
    p.add_argument('--batch-size',   type=int, default=32)
    p.add_argument('--num-workers',  type=int, default=4)
    p.add_argument('--gcs-dest',     default='',
                   help='Optional GCS path for result upload')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, 'restart.log')
    precision = 'fp32' if args.condition == 'A' else 'bf16'

    log(f"=== Restart Trainer: Condition {args.condition} ===", log_path)
    log(f"checkpoint={args.checkpoint}  restart_lr={args.restart_lr}  "
        f"epochs_after={args.epochs_after}  precision={precision}", log_path)

    # ── Device ────────────────────────────────────────────────────────────────
    if XLA_AVAILABLE:
        device = xm.xla_device()
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    log(f"Device: {device}", log_path)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    std = train_loader.dataset.std

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.condition, device)
    log(f"Model: hidden_dim={model.hidden_dim}  params={model.parameter_count():,}", log_path)

    # Load weights from checkpoint (model only — ignore optimizer completely)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    source_epoch = ckpt.get('epoch', '?')
    log(f"Loaded model weights from epoch {source_epoch} checkpoint", log_path)

    # ── Fresh optimizer — anchored at restart_lr ──────────────────────────────
    # Critically: we do NOT call optimizer.load_state_dict(). This gives us:
    #   1. Zero momentum buffers (clean gradient direction on first step)
    #   2. LR exactly at restart_lr (not inherited from checkpoint decay)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.restart_lr, weight_decay=1e-4)

    # ── Fresh cosine schedule from restart_lr → eta_min over epochs_after ────
    # T_max = epochs_after so the schedule covers exactly the post-restart window.
    # base_lrs = [restart_lr] exactly (scheduler constructed before any step).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs_after, eta_min=1e-6
    )

    log(f"Optimizer: AdamW(lr={args.restart_lr})  "
        f"Scheduler: CosineAnnealingLR(T_max={args.epochs_after}, eta_min=1e-6)", log_path)
    log("Optimizer state: FRESH (momentum=0, LR not inherited from checkpoint)", log_path)

    # ── Training loop ─────────────────────────────────────────────────────────
    results = {
        'experiment':    f'condition_{args.condition}_restart',
        'condition':     args.condition,
        'precision':     precision,
        'source_epoch':  source_epoch,
        'restart_lr':    args.restart_lr,
        'epochs_after':  args.epochs_after,
        'hidden_dim':    model.hidden_dim,
        'num_params':    model.parameter_count(),
        'epochs':        [],
        'best_val_mae_ev': float('inf'),
        'best_epoch':    None,
    }

    best_val_mae = float('inf')
    label_epoch = int(source_epoch) if isinstance(source_epoch, (int, str)) and str(source_epoch).isdigit() else 0

    for step in range(1, args.epochs_after + 1):
        current_lr = optimizer.param_groups[0]['lr']
        train_stats = train_epoch(model, train_loader, optimizer, device, precision)
        val_mae = evaluate(model, val_loader, device, precision, std)
        scheduler.step()

        epoch_label = label_epoch + step   # display as absolute epoch number
        mem_mb = get_memory_mb(device)

        record = {
            'step':          step,
            'epoch':         epoch_label,
            'lr':            current_lr,
            'train_loss':    train_stats['loss'],
            'val_mae_ev':    val_mae,
            'mem_mb':        mem_mb,
            'throughput':    train_stats['throughput'],
        }
        results['epochs'].append(record)

        log(f"Step {step:3d} (ep{epoch_label}) | "
            f"lr={current_lr:.2e} | "
            f"loss={train_stats['loss']:.4f} | "
            f"val_mae={val_mae:.4f} eV | "
            f"mem={mem_mb:.0f} MB", log_path)

        # Save best checkpoint
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            results['best_val_mae_ev'] = best_val_mae
            results['best_epoch'] = epoch_label
            best_path = os.path.join(
                args.output_dir,
                f'condition_{args.condition}_restart_best.pt'
            )
            torch.save({
                'epoch':       epoch_label,
                'model':       model.state_dict(),
                'val_mae_ev':  best_val_mae,
                'restart_lr':  args.restart_lr,
                'source_epoch': source_epoch,
            }, best_path)

        # Periodic checkpoint every 10 steps
        if step % 10 == 0:
            ckpt_path = os.path.join(
                args.output_dir,
                f'condition_{args.condition}_restart_ep{epoch_label}.pt'
            )
            torch.save({
                'epoch':     epoch_label,
                'step':      step,
                'model':     model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, ckpt_path)

        # Save running results
        out_json = os.path.join(args.output_dir, f'condition_{args.condition}_restart_results.json')
        with open(out_json, 'w') as f:
            json.dump(results, f, indent=2)

    log(f"\nBest val_mae: {best_val_mae:.4f} eV at epoch {results['best_epoch']}", log_path)
    log(f"Results: {out_json}", log_path)

    # ── GCS upload ────────────────────────────────────────────────────────────
    if args.gcs_dest:
        import subprocess
        log(f"Uploading to {args.gcs_dest} ...", log_path)
        subprocess.run(
            ['gsutil', '-m', 'rsync', '-r', args.output_dir, args.gcs_dest],
            capture_output=True
        )
        log("GCS upload complete.", log_path)


if __name__ == '__main__':
    main()
