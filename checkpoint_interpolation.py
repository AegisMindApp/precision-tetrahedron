#!/usr/bin/env python3
"""
checkpoint_interpolation.py
---------------------------
Linear Mode Connectivity test: interpolates between two checkpoints
(α=0 → θ₀, α=1 → θ₁) and evaluates val_mae at 11 points.

A smooth monotone valley confirms the two checkpoints lie in the same
loss basin (linearly connected). A barrier peak means they are in
distinct basins — the restart must "jump the barrier."

Usage:
  python3 checkpoint_interpolation.py \
      --condition A \
      --ckpt0 /tmp/flashoptim_results/condition_A_epoch80.pt \
      --ckpt1 /tmp/flashoptim_results/condition_B_best.pt \
      --label0 "A_ep80 (plateau)" \
      --label1 "B_ep83 (post-escape)" \
      --data-dir /tmp/qm9

Condition A (FP32) and Condition B (BF16 same-dim) share identical
architecture — interpolating between them tests whether the plateau
and escape points are in the same loss basin.
"""

import os, sys, json, argparse
import torch

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


@torch.no_grad()
def evaluate(model, loader, device, precision, std):
    model.eval()
    use_bf16 = (precision == 'bf16')
    mae_sum, n = 0.0, 0
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


def interpolate_state_dicts(sd0, sd1, alpha):
    """Return θ = (1-α)θ₀ + αθ₁ for matching keys."""
    result = {}
    for key in sd0:
        if key in sd1:
            v0 = sd0[key].float()
            v1 = sd1[key].float()
            result[key] = ((1 - alpha) * v0 + alpha * v1)
        else:
            result[key] = sd0[key]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--condition',  choices=['A','B','C'], default='A')
    p.add_argument('--ckpt0',      required=True, help='Checkpoint at α=0')
    p.add_argument('--ckpt1',      required=True, help='Checkpoint at α=1')
    p.add_argument('--label0',     default='ckpt0')
    p.add_argument('--label1',     default='ckpt1')
    p.add_argument('--n-steps',    type=int, default=11,
                   help='Number of interpolation points including endpoints')
    p.add_argument('--data-dir',   default='/tmp/qm9')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--output',     default='/tmp/interpolation_results.json')
    args = p.parse_args()

    if XLA_AVAILABLE:
        device = xm.xla_device()
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    precision = 'fp32' if args.condition == 'A' else 'bf16'
    print(f"Device: {device}  |  Condition: {args.condition}  |  Precision: {precision}")

    _, val_loader, _ = get_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=4)
    std = val_loader.dataset.std

    sd0 = torch.load(args.ckpt0, map_location='cpu')['model']
    sd1 = torch.load(args.ckpt1, map_location='cpu')['model']
    ep0 = torch.load(args.ckpt0, map_location='cpu').get('epoch', '?')
    ep1 = torch.load(args.ckpt1, map_location='cpu').get('epoch', '?')

    print(f"\n  α=0: {args.label0} (ep{ep0})")
    print(f"  α=1: {args.label1} (ep{ep1})")
    print(f"\n  {'α':>6}  {'val_mae (eV)':>14}  {'notes'}")
    print(f"  {'─'*6}  {'─'*14}  {'─'*20}")

    alphas = [i / (args.n_steps - 1) for i in range(args.n_steps)]
    records = []

    model = build_model(args.condition, device)

    for alpha in alphas:
        interpolated = interpolate_state_dicts(sd0, sd1, alpha)
        # Cast back to model dtype (BF16 if condition B/C)
        if precision == 'bf16':
            interpolated = {k: v.bfloat16() for k, v in interpolated.items()}
        model.load_state_dict(interpolated)
        mae = evaluate(model, val_loader, device, precision, std)

        note = ''
        if alpha == 0.0:
            note = f'← {args.label0}'
        elif alpha == 1.0:
            note = f'← {args.label1}'

        print(f"  {alpha:>6.2f}  {mae:>14.4f}  {note}")
        records.append({'alpha': alpha, 'val_mae_ev': mae})

    # Detect barrier
    mid_maes = [r['val_mae_ev'] for r in records[1:-1]]
    end_maes = [records[0]['val_mae_ev'], records[-1]['val_mae_ev']]
    barrier_height = max(mid_maes) - min(end_maes)
    linearly_connected = barrier_height < 0.002   # <2 meV threshold

    print(f"\n  Barrier height: {barrier_height:.4f} eV")
    print(f"  Linearly connected: {'YES' if linearly_connected else 'NO'}")
    if linearly_connected:
        print("  → Plateau and escape points are in the same loss basin.")
        print("    Restart moves continuously through this basin — lower LR stays closer to start.")
    else:
        print("  → Distinct basins. Restart must jump the barrier.")
        print(f"    Peak α: {alphas[mid_maes.index(max(mid_maes))+1]:.2f}")

    results = {
        'condition':          args.condition,
        'ckpt0':              {'path': args.ckpt0, 'label': args.label0, 'epoch': str(ep0)},
        'ckpt1':              {'path': args.ckpt1, 'label': args.label1, 'epoch': str(ep1)},
        'interpolation':      records,
        'barrier_height_ev':  round(barrier_height, 6),
        'linearly_connected': linearly_connected,
    }
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults: {args.output}")


if __name__ == '__main__':
    main()
