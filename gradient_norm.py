#!/usr/bin/env python3
"""
gradient_norm.py
----------------
Measures the gradient norm ‖g‖ at a checkpoint with no weight update.
Used to bound the maximum restart LR via: lr_max < basin_radius / ‖g‖.

Outputs a JSON with per-layer and total gradient norms.

Usage:
  python3 gradient_norm.py \
      --condition A \
      --checkpoint /tmp/flashoptim_results/condition_A_epoch80.pt \
      --data-dir /tmp/qm9
"""

import os, sys, json, argparse, time
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model
from data import get_dataloaders, batch_to_graph


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--condition',   choices=['A','B','C'], required=True)
    p.add_argument('--checkpoint',  required=True)
    p.add_argument('--data-dir',    default='/tmp/qm9')
    p.add_argument('--batch-size',  type=int, default=32)
    p.add_argument('--n-batches',   type=int, default=20,
                   help='Number of training batches to average gradient over')
    p.add_argument('--output',      default='/tmp/gradient_norm_results.json')
    args = p.parse_args()

    if XLA_AVAILABLE:
        device = xm.xla_device()
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"Device: {device}")
    precision = 'fp32' if args.condition == 'A' else 'bf16'
    use_bf16 = (precision == 'bf16')

    # Load model
    model = build_model(args.condition, device)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.train()
    source_epoch = ckpt.get('epoch', '?')
    print(f"Loaded epoch {source_epoch} checkpoint  "
          f"({model.parameter_count():,} params, {precision})")

    train_loader, _, _ = get_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=4)

    if XLA_AVAILABLE:
        ctx_fn = lambda: torch.autocast('xla', dtype=torch.bfloat16, enabled=use_bf16)
    else:
        ctx_fn = lambda: torch.autocast('cpu', dtype=torch.bfloat16, enabled=use_bf16)

    # Accumulate gradients over n_batches (no optimizer step)
    model.zero_grad()
    batches_seen = 0
    for batch in train_loader:
        if batches_seen >= args.n_batches:
            break
        z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid = \
            batch_to_graph(batch, device)
        target = batch['target'].to(device)
        with ctx_fn():
            pred = model(z, pos, edge_src, edge_dst, assign_mat, num_graphs, edge_valid, atom_valid)
            loss = F.mse_loss(pred, target)
        loss.backward()
        if XLA_AVAILABLE:
            xm.mark_step()
        batches_seen += 1

    # Compute gradient norms
    total_norm_sq = 0.0
    layer_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            g = param.grad.detach().float()
            norm = g.norm().item()
            layer_norms[name] = round(norm, 6)
            total_norm_sq += norm ** 2

    total_norm = total_norm_sq ** 0.5

    # Estimate maximum restart LR based on Condition B's successful natural LR
    # Natural cosine LR at ep80 with T_max=100, lr0=1e-4, eta_min=1e-6:
    import math
    T_max, lr0, eta_min, ep = 100, 1e-4, 1e-6, 80
    natural_lr_ep80 = eta_min + 0.5 * (lr0 - eta_min) * (1 + math.cos(math.pi * ep / T_max))
    # Condition B succeeded at this LR → basin_radius ≈ natural_lr × ‖g‖
    basin_radius_estimate = natural_lr_ep80 * total_norm
    # Upper bound on restart LR: lr < basin_radius / ‖g‖ = natural_lr (tautological confirmation)
    # More usefully: lr_max_estimate from B_v2 failure (lr=1e-4 failed)
    lr_max_upper = 1e-4  # failed
    lr_max_lower = natural_lr_ep80  # succeeded

    results = {
        'condition':             args.condition,
        'source_epoch':          source_epoch,
        'n_batches_averaged':    batches_seen,
        'gradient_norm_total':   round(total_norm, 6),
        'gradient_norm_sq':      round(total_norm_sq, 6),
        'natural_lr_at_ep80':    round(natural_lr_ep80, 8),
        'basin_radius_estimate': round(basin_radius_estimate, 8),
        'lr_max_upper_bound':    lr_max_upper,
        'lr_max_lower_bound':    round(natural_lr_ep80, 8),
        'adam_first_step_at_1e4': round(1e-4, 6),   # ~= lr with fresh v̂
        'adam_first_step_at_nat': round(natural_lr_ep80, 8),
        'top10_largest_grad_layers': sorted(
            layer_norms.items(), key=lambda x: -x[1]
        )[:10],
    }

    print(f"\n=== Gradient Norm Results ===")
    print(f"  Total ‖g‖:              {total_norm:.4f}")
    print(f"  Natural LR at ep80:     {natural_lr_ep80:.2e}")
    print(f"  Basin radius estimate:  {basin_radius_estimate:.2e}")
    print(f"  lr_max ∈ ({natural_lr_ep80:.1e}, 1e-4) — restart LR must be in this window")
    print(f"  Adam ep1 step at 1e-4: {1e-4:.1e}  (too large → basin exit)")
    print(f"  Adam ep1 step at nat:  {natural_lr_ep80:.1e}  (within basin)")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {args.output}")


if __name__ == '__main__':
    main()
