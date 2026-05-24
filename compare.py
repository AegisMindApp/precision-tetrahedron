"""
Compare results across conditions A, B, C and report against
success/failure criteria from the FlashOptim discovery protocol.

Run after all three conditions complete:
    python compare.py --output-dir /tmp/results
"""

import os
import json
import argparse


SUCCESS_CRITERIA = {
    'memory_reduction':   ('>=', 0.30, '30% memory reduction (same model)'),
    'mae_relative_error': ('<=', 0.05, 'MAE within 5% of FP32 baseline'),
    'loss_stability':     ('>=', 0.90, '90% of steps with finite loss'),
    'throughput_ratio':   ('>=', 0.90, 'Throughput >= 90% of FP32 speed'),
}

FAILURE_CRITERIA = {
    'accuracy_disproof':  ('>', 0.05, 'MAE degradation > 5% across 3 benchmarks'),
    'memory_disproof':    ('<', 0.20, 'Memory reduction < 20%'),
    'stability_disproof': ('<', 0.70, 'Loss NaN/Inf in > 30% of steps'),
    'overhead_disproof':  ('<', 0.85, 'Wall-clock slowdown > 15%'),
}


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def final_epoch(results: dict) -> dict:
    return results['epochs'][-1] if results['epochs'] else {}


def report(output_dir: str):
    files = {
        cond: os.path.join(output_dir, f'condition_{cond}_results.json')
        for cond in ('A', 'B', 'C')
    }

    loaded = {}
    for cond, path in files.items():
        if os.path.exists(path):
            loaded[cond] = load(path)
        else:
            print(f"Missing: {path}")

    if 'A' not in loaded:
        print("Need at least condition A (FP32 baseline) to compare.")
        return

    ep_a = final_epoch(loaded['A'])
    mae_a = ep_a.get('val_mae_ev', float('inf'))
    mem_a = ep_a.get('mem_mb', 0)
    thr_a = ep_a.get('throughput', 1)

    print("=" * 65)
    print("FlashOptim Mixed-Precision Experiment — Results Summary")
    print("=" * 65)
    print(f"\n{'Metric':<35} {'A (FP32)':>10} {'B (BF16)':>10} {'C (BF16 2x)':>12}")
    print("-" * 65)

    results_table = {}

    for cond, label in [('B', 'BF16 same'), ('C', 'BF16 2x')]:
        if cond not in loaded:
            continue
        ep = final_epoch(loaded[cond])
        mae = ep.get('val_mae_ev', float('inf'))
        mem = ep.get('mem_mb', 0)
        thr = ep.get('throughput', 0)
        fin = ep.get('loss_finite_frac', 0)

        mae_rel  = (mae - mae_a) / (mae_a + 1e-8)
        mem_red  = (mem_a - mem) / (mem_a + 1e-8) if mem_a > 0 else 0
        thr_rat  = thr / (thr_a + 1e-8)

        results_table[cond] = {
            'mae_ev':       mae,
            'mae_relative': mae_rel,
            'mem_mb':       mem,
            'mem_reduction': mem_red,
            'throughput':   thr,
            'throughput_ratio': thr_rat,
            'loss_stability': fin,
        }

    # Print table
    for metric, label in [
        ('val_mae_ev',        'MAE (eV)'),
        ('mem_mb',            'Memory (MB)'),
        ('throughput',        'Throughput (batch/s)'),
        ('loss_finite_frac',  'Loss stability'),
    ]:
        row = f"  {label:<33} {ep_a.get(metric, 0):>10.4f}"
        for cond in ('B', 'C'):
            if cond in results_table:
                ep = final_epoch(loaded[cond])
                row += f" {ep.get(metric, 0):>10.4f}" if cond == 'B' else f" {ep.get(metric, 0):>12.4f}"
        print(row)

    print("-" * 65)

    print("\n=== Success Criteria Evaluation ===\n")
    for cond in ('B', 'C'):
        if cond not in results_table:
            continue
        r = results_table[cond]
        label = 'BF16 same size' if cond == 'B' else 'BF16 2x wider'
        print(f"Condition {cond} ({label}):")

        # Memory
        mem_red = r['mem_reduction']
        status = "PASS" if mem_red >= 0.30 else ("MARGINAL" if mem_red >= 0.20 else "FAIL")
        print(f"  Memory reduction:  {mem_red:+.1%}  [{status}]  (target: >=30%)")

        # MAE
        mae_rel = r['mae_relative']
        status = "PASS" if mae_rel <= 0.05 else ("MARGINAL" if mae_rel <= 0.10 else "FAIL")
        print(f"  MAE relative diff: {mae_rel:+.1%}  [{status}]  (target: <=5%)")

        # Stability
        fin = r['loss_stability']
        status = "PASS" if fin >= 0.90 else "FAIL"
        print(f"  Loss stability:    {fin:.1%}    [{status}]  (target: >=90%)")

        # Throughput
        thr = r['throughput_ratio']
        status = "PASS" if thr >= 0.90 else "FAIL"
        print(f"  Throughput ratio:  {thr:.2f}x     [{status}]  (target: >=0.90x)")

        # Abort flag
        if loaded[cond].get('abort'):
            print(f"  !! ABORTED: {loaded[cond].get('abort_reason')}")

        print()

    # Checkpoints summary
    print("=== Protocol Checkpoints ===\n")
    for cond, data in loaded.items():
        cps = data.get('checkpoints', {})
        for cp_name, cp_data in cps.items():
            passed = cp_data.get('passed', False)
            print(f"  [{cond}] Checkpoint {cp_name}: {'PASS' if passed else 'FAIL'}  {cp_data}")

    print("\n=== Verdict ===\n")
    if 'B' in results_table and 'C' in results_table:
        r_b = results_table['B']
        r_c = results_table['C']
        hypothesis_supported = (
            r_b['mem_reduction'] >= 0.20 and
            r_b['mae_relative'] <= 0.10 and
            r_b['loss_stability'] >= 0.70
        )
        print("Hypothesis SUPPORTED" if hypothesis_supported else "Hypothesis REFUTED")
        print(f"  B memory reduction: {r_b['mem_reduction']:.1%}  (need >=20% for partial support)")
        print(f"  B MAE degradation:  {r_b['mae_relative']:+.1%}  (need <=10%)")
        print(f"  B loss stability:   {r_b['loss_stability']:.1%}  (need >=70%)")
    else:
        print("Insufficient data — run all three conditions.")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output-dir', default='/tmp/results')
    args = p.parse_args()
    report(args.output_dir)
