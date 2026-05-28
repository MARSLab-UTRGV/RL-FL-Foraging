#!/usr/bin/env python3
"""
Batch evaluation script for decentralized RL foraging.

Runs samples in parallel by default — each sample gets its own Webots instance
on a dedicated port.  Webots prints the supervisor controller URL via
--extern-urls; the batch runner captures that URL and passes it to the
supervisor subprocess via WEBOTS_CONTROLLER_URL.

Per-sample CSVs are written to temp files then merged at the end.

Usage:
    python3 run_batch_eval.py                               # defaults below
    python3 run_batch_eval.py --arena_size 7x7 --samples 1-10 --duration 10

    # Multi-robot scalability experiment
    python3 run_batch_eval.py --arena_size 7x7 --num_robots 4  --num_tags 32
    python3 run_batch_eval.py --arena_size 7x7 --num_robots 8  --num_tags 64
    python3 run_batch_eval.py --arena_size 7x7 --num_robots 12 --num_tags 128
    python3 run_batch_eval.py --arena_size 7x7 --num_robots 16 --num_tags 208

    # Limit concurrency (e.g. 4 at a time to avoid RAM pressure)
    python3 run_batch_eval.py --max_parallel 4

    # Fall back to fully sequential (one Webots at a time)
    python3 run_batch_eval.py --max_parallel 1
"""

import argparse
import csv
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Default configuration ──────────────────────────────────────────────────────
ARENA_SIZE       = '5x5'
RUN_NAME         = 'decentralized_indep_v9'
DURATION_SIM_MIN = 10.0
NUM_ROBOTS       = 4
NUM_TAGS         = None          # None = use arena default
BASE_PORT        = 5100          # port for sample index 0; each sample adds its index
WEBOTS_BIN       = 'webots'
WEBOTS_STARTUP_S = 45            # seconds to wait for Webots supervisor URL
MAX_PARALLEL     = 10            # run all 10 samples simultaneously by default
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
SUPERVISOR_DIR = os.path.join(PROJECT_ROOT, 'controllers', 'eval_decentralized')
SUPERVISOR_PY  = os.path.join(SUPERVISOR_DIR, 'eval_decentralized.py')
WORLDS_DIR     = os.path.join(PROJECT_ROOT, 'worlds')

EXTERN_URL_PREFIXES     = ('ipc://', 'tcp://')
SUPERVISOR_CTRL_NAME    = 'supervisor'


# ── Webots environment helpers ────────────────────────────────────────────────

def _infer_webots_home(webots_bin):
    if os.environ.get('WEBOTS_HOME'):
        return os.environ['WEBOTS_HOME']
    resolved = shutil.which(webots_bin)
    if resolved is None:
        return None
    resolved = os.path.realpath(resolved)
    # webots wrapper → .../bin/webots-bin → .../<WEBOTS_HOME>/bin/
    if os.path.basename(resolved) == 'webots-bin':
        return os.path.dirname(os.path.dirname(resolved))
    return os.path.dirname(resolved)


def _add_webots_controller_env(env, webots_bin):
    home = _infer_webots_home(webots_bin)
    if home is None:
        return
    env['WEBOTS_HOME'] = home
    py_lib = os.path.join(home, 'lib', 'controller', 'python')
    ld_lib = os.path.join(home, 'lib', 'controller')
    env['PYTHONPATH'] = py_lib + os.pathsep + env.get('PYTHONPATH', '')
    env['LD_LIBRARY_PATH'] = ld_lib + os.pathsep + env.get('LD_LIBRARY_PATH', '')


# ── Webots stdout pump (captures extern controller URLs) ──────────────────────

def _pump_webots_output(stdout, url_queue):
    """Read Webots stdout; push lines starting with ipc:// or tcp:// to queue."""
    try:
        for line in stdout:
            line = line.strip()
            if line.startswith(EXTERN_URL_PREFIXES):
                url_queue.put(line)
    finally:
        url_queue.put(None)   # sentinel


def _wait_for_supervisor_url(port, timeout_s, webots_proc, url_queue):
    """
    Block until Webots prints the supervisor controller URL (via --extern-urls),
    then return it.  Raises if Webots exits early or the timeout is exceeded.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if webots_proc.poll() is not None:
            raise RuntimeError(
                f'Webots (port {port}) exited before supervisor URL appeared')
        remaining = max(0.0, deadline - time.time())
        try:
            url = url_queue.get(timeout=min(0.5, remaining))
        except queue.Empty:
            continue
        if url is None:
            continue
        # URL path ends in the controller name, e.g. .../supervisor
        if url.rstrip('/').split('/')[-1] == SUPERVISOR_CTRL_NAME:
            return url
    raise TimeoutError(
        f'Timed out waiting for supervisor URL on port {port} '
        f'after {timeout_s:.0f}s')


# ── Process cleanup ───────────────────────────────────────────────────────────

def _terminate(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


# ── Config files (written once before parallel launch) ───────────────────────

def write_config(arena_size, run_name):
    for fname, content in [
        ('current_eval_run.txt',   run_name),
        ('current_eval_arena.txt', arena_size),
    ]:
        with open(os.path.join(PROJECT_ROOT, fname), 'w') as f:
            f.write(content)


# ── Per-sample runner (called from thread pool) ───────────────────────────────

def run_sample(sample_num, arena_size, run_name, duration_sim_min,
               sample_csv, num_robots, num_tags,
               port, webots_bin, startup_s):
    """
    Launch one Webots instance + one supervisor process for a single sample.
    Returns True on success.
    """
    suffix = f'_{num_robots}r' if num_robots > 4 else ''
    world  = os.path.join(WORLDS_DIR,
                          f'eval_sample{sample_num}_{arena_size}{suffix}.wbt')
    if not os.path.exists(world):
        print(f'[s{sample_num}] World not found, skipping: {world}')
        return False

    print(f'[s{sample_num}] Starting | port={port} | {os.path.basename(world)}')

    webots_cmd = [
        webots_bin,
        '--batch',
        f'--port={port}',
        '--extern-urls',
        '--mode=fast',
        '--minimize',
        '--no-rendering',
        world,
    ]

    webots_env = os.environ.copy()
    webots_env['WEBOTS_PORT'] = str(port)

    webots_proc = None
    pump_thread = None
    try:
        webots_proc = subprocess.Popen(
            webots_cmd,
            cwd=PROJECT_ROOT,
            env=webots_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )

        url_queue = queue.Queue()
        pump_thread = threading.Thread(
            target=_pump_webots_output,
            args=(webots_proc.stdout, url_queue),
            daemon=True,
        )
        pump_thread.start()

        ctrl_url = _wait_for_supervisor_url(port, startup_s, webots_proc, url_queue)
        print(f'[s{sample_num}] Supervisor URL: {ctrl_url}')

        # Build supervisor command
        sup_cmd = [
            sys.executable, SUPERVISOR_PY,
            '--arena_size',       arena_size,
            '--run_name',         run_name,
            '--duration_sim_min', str(duration_sim_min),
            '--results_csv',      sample_csv,
            '--sample_id',        str(sample_num),
            '--num_robots',       str(num_robots),
        ]
        if num_tags is not None:
            sup_cmd += ['--num_tags', str(num_tags)]

        ctrl_env = os.environ.copy()
        ctrl_env['WEBOTS_CONTROLLER_URL'] = ctrl_url
        ctrl_env['WEBOTS_PORT']           = str(port)
        ctrl_env['PYTHONUNBUFFERED']      = '1'
        _add_webots_controller_env(ctrl_env, webots_bin)

        t0 = time.time()
        result = subprocess.run(sup_cmd, cwd=SUPERVISOR_DIR, env=ctrl_env)
        wall_s = time.time() - t0
        ok = result.returncode == 0
        print(f'[s{sample_num}] Done in {wall_s:.1f}s '
              f'(code {result.returncode}) {"✓" if ok else "✗"}')
        return ok

    except Exception as exc:
        print(f'[s{sample_num}] ERROR: {exc}')
        return False

    finally:
        _terminate(webots_proc)
        if pump_thread is not None:
            pump_thread.join(timeout=5)


# ── CSV merge ─────────────────────────────────────────────────────────────────

def merge_sample_csvs(samples, results_csv):
    """Collect per-sample temp CSVs, sort by (sample, time_min), write final CSV."""
    all_rows = []
    for s in samples:
        tmp = results_csv + f'.s{s}.tmp'
        if not os.path.exists(tmp):
            continue
        with open(tmp, newline='') as f:
            all_rows.extend(list(csv.DictReader(f)))
        os.remove(tmp)

    if not all_rows:
        return

    def sort_key(r):
        try:
            return (int(r.get('sample', 0)), float(r.get('time_min', 0)))
        except (ValueError, TypeError):
            return (0, 0.0)

    all_rows.sort(key=sort_key)

    with open(results_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print(f'[BATCH] Merged {len(all_rows)} rows → {results_csv}')


# ── Summary / boxplot ─────────────────────────────────────────────────────────

def print_summary(results_csv, arena_size, num_robots, duration):
    if not os.path.exists(results_csv):
        print('[BATCH] No results CSV found.')
        return

    with open(results_csv, newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print('[BATCH] CSV is empty.')
        return

    print(f'\n{"="*60}')
    print('BATCH EVAL RESULTS')
    print(f'Arena: {arena_size} | Robots: {num_robots} | Duration: {duration} sim-min')
    print(f'{"="*60}')
    print(f'{"Sample":>8} {"Deposits":>10} {"SimRate":>10} {"Wall(min)":>10}')
    print('-' * 42)

    deposits = []
    for r in sorted(rows, key=lambda r: (len(r['sample']), r['sample'])):
        d = int(r['deposits'])
        deposits.append(d)
        print(f'{r["sample"]:>8} {d:>10} {r["sim_rate"]:>10} {r["wall_time_min"]:>10}')

    if not deposits:
        return
    mean = sum(deposits) / len(deposits)
    std  = (sum((x - mean) ** 2 for x in deposits) / len(deposits)) ** 0.5
    print('-' * 42)
    print(f'{"mean":>8} {mean:>10.1f}')
    print(f'{"std":>8} {std:>10.1f}')
    print(f'{"min":>8} {min(deposits):>10}')
    print(f'{"max":>8} {max(deposits):>10}')
    print(f'\nCSV: {results_csv}')

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.boxplot(deposits, patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.7))
        ax.set_ylabel(f'Deposits in {duration} sim-min')
        ax.set_title(f'Decentralized RL  |  {arena_size}  |  '
                     f'{num_robots} robots  |  {len(deposits)} samples')
        ax.set_xticks([1])
        ax.set_xticklabels([f'{num_robots}r'])
        plot_path = results_csv.replace('.csv', '_boxplot.png')
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f'Boxplot: {plot_path}')
    except ImportError:
        pass


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_sample_range(s):
    samples = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            samples.extend(range(int(a), int(b) + 1))
        else:
            samples.append(int(part))
    return samples


def main():
    parser = argparse.ArgumentParser(description='Parallel batch eval for decentralized RL')
    parser.add_argument('--arena_size',   default=ARENA_SIZE,
                        choices=['5x5', '7x7', '9x9', '12x12'])
    parser.add_argument('--run_name',     default=RUN_NAME)
    parser.add_argument('--samples',      default='1-10',
                        help='Range or list: "1-10", "1,3,5"')
    parser.add_argument('--duration',     type=float, default=DURATION_SIM_MIN,
                        help='Sim-minutes per sample (default: 10)')
    parser.add_argument('--num_robots',   type=int, default=NUM_ROBOTS)
    parser.add_argument('--num_tags',     type=int, default=NUM_TAGS,
                        help='Active tag count override')
    parser.add_argument('--max_parallel', type=int, default=MAX_PARALLEL,
                        help='Max simultaneous Webots instances (1 = sequential)')
    parser.add_argument('--base_port',    type=int, default=BASE_PORT,
                        help='Base Webots port; sample i uses base_port + i')
    parser.add_argument('--webots_bin',   default=WEBOTS_BIN)
    parser.add_argument('--startup_s',    type=float, default=WEBOTS_STARTUP_S,
                        help='Seconds to wait for Webots supervisor URL')
    args = parser.parse_args()

    samples     = parse_sample_range(args.samples)
    max_workers = min(args.max_parallel, len(samples))

    tag_str = f'_t{args.num_tags}' if args.num_tags else ''
    r_str   = f'_{args.num_robots}r' if args.num_robots != 4 else ''
    results_csv = os.path.join(
        PROJECT_ROOT,
        f'batch_results_{args.arena_size}{r_str}{tag_str}_{args.run_name}.csv'
    )

    # Clean old CSV and temp files
    if os.path.exists(results_csv):
        os.remove(results_csv)
    for s in samples:
        tmp = results_csv + f'.s{s}.tmp'
        if os.path.exists(tmp):
            os.remove(tmp)

    # Write shared config once (all samples use same run/arena)
    write_config(args.arena_size, args.run_name)

    print(f'\n{"="*60}')
    print(f'BATCH EVAL  |  Arena: {args.arena_size}  |  Robots: {args.num_robots}  |  '
          f'Tags: {args.num_tags or "default"}')
    print(f'Samples: {samples}  |  Duration: {args.duration} sim-min')
    print(f'Parallel: {max_workers}  |  Base port: {args.base_port}')
    print(f'Model: {args.run_name}')
    print(f'Results: {results_csv}')
    print(f'{"="*60}\n')

    ok = 0
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {}
        for idx, sample in enumerate(samples):
            port       = args.base_port + idx
            sample_csv = results_csv + f'.s{sample}.tmp'
            future = pool.submit(
                run_sample,
                sample, args.arena_size, args.run_name,
                args.duration, sample_csv,
                args.num_robots, args.num_tags,
                port, args.webots_bin, args.startup_s,
            )
            future_map[future] = sample

        for future in as_completed(future_map):
            sample = future_map[future]
            try:
                success = future.result()
                if success:
                    ok += 1
                else:
                    failed.append(sample)
            except Exception as exc:
                print(f'[s{sample}] EXCEPTION: {exc}')
                failed.append(sample)

    print(f'\n[BATCH] Done: {ok}/{len(samples)} succeeded'
          + (f', failed: {sorted(failed)}' if failed else ''))

    merge_sample_csvs(samples, results_csv)
    print_summary(results_csv, args.arena_size, args.num_robots, args.duration)


if __name__ == '__main__':
    main()
