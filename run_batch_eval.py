#!/usr/bin/env python3
"""
Batch evaluation script for decentralized RL foraging.

Runs samples sequentially — Webots opens for each sample in fast-simulation
mode, the supervisor runs for DURATION_SIM_MIN simulation minutes, then both
exit automatically. Results are appended to a CSV for statistical analysis.

Usage:
    python3 run_batch_eval.py                          # defaults below
    python3 run_batch_eval.py --arena_size 7x7 --samples 1-10 --duration 10

After all samples finish, mean/std are printed and a boxplot is saved.
"""

import argparse
import csv
import os
import subprocess
import sys
import time

# ── Default configuration ──────────────────────────────────────────────────────
ARENA_SIZE       = '5x5'
RUN_NAME         = 'decentralized_indep_v9'
SAMPLES          = list(range(1, 11))     # 1–10
DURATION_SIM_MIN = 10.0                   # sim-minutes per sample
WEBOTS_BIN       = 'webots'               # set full path if webots not on PATH
WEBOTS_STARTUP_S = 15                     # seconds to wait for Webots to be ready
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
SUPERVISOR_DIR = os.path.join(PROJECT_ROOT, 'controllers', 'eval_decentralized')
SUPERVISOR_PY  = os.path.join(SUPERVISOR_DIR, 'eval_decentralized.py')
WORLDS_DIR     = os.path.join(PROJECT_ROOT, 'worlds')


def write_config(arena_size, run_name):
    for fname, content in [
        ('current_eval_run.txt',   run_name),
        ('current_eval_arena.txt', arena_size),
    ]:
        with open(os.path.join(PROJECT_ROOT, fname), 'w') as f:
            f.write(content)


def run_sample(sample_num, arena_size, run_name, duration_sim_min, results_csv,
               webots_bin=WEBOTS_BIN):
    world = os.path.join(WORLDS_DIR, f'eval_sample{sample_num}_{arena_size}.wbt')
    if not os.path.exists(world):
        print(f'[BATCH] World not found, skipping: {world}')
        return False

    print(f'\n{"="*60}')
    print(f'[BATCH] Sample {sample_num} | Arena: {arena_size} | '
          f'Duration: {duration_sim_min} sim-min')
    print(f'{"="*60}')

    write_config(arena_size, run_name)

    # Launch Webots (background)
    webots_proc = subprocess.Popen(
        [webots_bin, '--batch', world],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f'[BATCH] Webots started (PID {webots_proc.pid}) — '
          f'waiting {WEBOTS_STARTUP_S}s for ready...')
    time.sleep(WEBOTS_STARTUP_S)

    # Run supervisor (blocks until simulationQuit is called inside)
    sup_cmd = [
        sys.executable,
        SUPERVISOR_PY,
        '--arena_size',       arena_size,
        '--run_name',         run_name,
        '--duration_sim_min', str(duration_sim_min),
        '--results_csv',      results_csv,
        '--sample_id',        str(sample_num),
    ]
    t0 = time.time()
    result = subprocess.run(sup_cmd, cwd=SUPERVISOR_DIR)
    wall_sec = time.time() - t0
    print(f'[BATCH] Supervisor finished in {wall_sec:.1f}s '
          f'(exit code {result.returncode})')

    # Webots exits via simulationQuit; terminate it anyway in case it's still up
    if webots_proc.poll() is None:
        webots_proc.terminate()
        try:
            webots_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            webots_proc.kill()

    time.sleep(2)   # brief pause before next sample
    return result.returncode == 0


def print_summary(results_csv):
    if not os.path.exists(results_csv):
        print('[BATCH] No results CSV found.')
        return

    with open(results_csv, newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print('[BATCH] CSV is empty.')
        return

    rates = [float(r['sim_rate']) for r in rows]
    deps  = [int(r['deposits'])   for r in rows]
    mean  = sum(rates) / len(rates)
    std   = (sum((x - mean) ** 2 for x in rates) / len(rates)) ** 0.5

    print(f'\n{"="*60}')
    print('BATCH EVAL RESULTS')
    print(f'{"="*60}')
    print(f'{"Sample":>8} {"Deposits":>10} {"SimTime(min)":>13} '
          f'{"SimRate":>10} {"WallTime(min)":>14}')
    print('-' * 60)
    for r in rows:
        print(f'{r["sample"]:>8} {r["deposits"]:>10} {r["sim_time_min"]:>13} '
              f'{r["sim_rate"]:>10} {r["elapsed_min"]:>14}')
    print('-' * 60)
    print(f'{"mean":>8} {sum(deps)/len(deps):>10.1f} {"":>13} {mean:>10.4f}')
    print(f'{"std":>8} {"":>10} {"":>13} {std:>10.4f}')
    print(f'{"min":>8} {"":>10} {"":>13} {min(rates):>10.4f}')
    print(f'{"max":>8} {"":>10} {"":>13} {max(rates):>10.4f}')
    print(f'\nCSV saved to: {results_csv}')

    # Boxplot (optional — only if matplotlib is available)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.boxplot(rates, patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.7))
        ax.set_ylabel('SimRate (tags / sim-min)')
        ax.set_title(f'Decentralized RL  |  {rows[0]["arena"]}  |  '
                     f'{len(rows)} samples')
        ax.set_xticks([1])
        ax.set_xticklabels([rows[0]["arena"]])
        plot_path = results_csv.replace('.csv', '_boxplot.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f'Boxplot saved to: {plot_path}')
    except ImportError:
        print('(matplotlib not available — skipping boxplot)')


def parse_sample_range(s):
    """Parse '1-10' or '1,3,5' or '1' into a list of ints."""
    samples = []
    for part in s.split(','):
        if '-' in part:
            a, b = part.split('-')
            samples.extend(range(int(a), int(b) + 1))
        else:
            samples.append(int(part))
    return samples


def main():
    parser = argparse.ArgumentParser(description='Batch eval for decentralized RL')
    parser.add_argument('--arena_size',  default=ARENA_SIZE,
                        choices=['5x5', '7x7', '9x9', '12x12'])
    parser.add_argument('--run_name',    default=RUN_NAME)
    parser.add_argument('--samples',     default='1-10',
                        help='Range or list: "1-10", "1,3,5", "1-5"')
    parser.add_argument('--duration',    type=float, default=DURATION_SIM_MIN,
                        help='Sim-minutes per sample (default: 10)')
    parser.add_argument('--webots_bin',  default=WEBOTS_BIN,
                        help='Path to webots executable')
    args = parser.parse_args()

    samples     = parse_sample_range(args.samples)
    results_csv = os.path.join(
        PROJECT_ROOT,
        f'batch_results_{args.arena_size}_{args.run_name}.csv'
    )

    # Remove old CSV so header is written fresh
    if os.path.exists(results_csv):
        os.remove(results_csv)
        print(f'[BATCH] Removed existing: {os.path.basename(results_csv)}')

    print(f'\n{"="*60}')
    print(f'BATCH EVAL  |  Arena: {args.arena_size}  |  '
          f'Samples: {samples}  |  Duration: {args.duration} sim-min')
    print(f'Model: {args.run_name}')
    print(f'Results: {results_csv}')
    print(f'{"="*60}\n')

    ok = 0
    for sample in samples:
        success = run_sample(sample, args.arena_size, args.run_name,
                             args.duration, results_csv,
                             webots_bin=args.webots_bin)
        if success:
            ok += 1

    print(f'\n[BATCH] Done: {ok}/{len(samples)} samples completed successfully.')
    print_summary(results_csv)


if __name__ == '__main__':
    main()
