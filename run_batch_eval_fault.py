#!/usr/bin/env python3
"""
EX4b — Fault tolerance batch evaluation (1-fault and 2-fault modes).

Usage:
    # 1-robot fault (default)
    python3 run_batch_eval_fault.py --run_name decentralized_indep_v10 --samples 1-20

    # 2-robot fault
    python3 run_batch_eval_fault.py --schedule failure_schedule_exp4b_2fault.csv --samples 1-20

    # Limit parallelism
    python3 run_batch_eval_fault.py --max_parallel 4
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

# ── Defaults ──────────────────────────────────────────────────────────────────
ARENA_SIZE       = '5x5'
RUN_NAME         = 'decentralized_indep_v10'
DURATION_SIM_MIN = 10.0
BASE_PORT        = 5200          # different port range from run_batch_eval.py
WEBOTS_BIN       = 'webots'
WEBOTS_STARTUP_S = 90
MAX_PARALLEL     = 10
# ──────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
SUPERVISOR_DIR = os.path.join(PROJECT_ROOT, 'controllers', 'eval_decentralized_fault')
SUPERVISOR_PY  = os.path.join(SUPERVISOR_DIR, 'eval_decentralized_fault.py')
WORLDS_DIR     = os.path.join(PROJECT_ROOT, 'worlds')
DEFAULT_SCHEDULE = os.path.join(PROJECT_ROOT, 'failure_schedule_exp4b.csv')

EXTERN_URL_PREFIXES  = ('ipc://', 'tcp://')
SUPERVISOR_CTRL_NAME = 'supervisor'


def _infer_webots_home(webots_bin):
    if os.environ.get('WEBOTS_HOME'):
        return os.environ['WEBOTS_HOME']
    resolved = shutil.which(webots_bin)
    if resolved is None:
        return None
    resolved = os.path.realpath(resolved)
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


def _pump_webots_output(stdout, url_queue):
    try:
        for line in stdout:
            line = line.strip()
            if line.startswith(EXTERN_URL_PREFIXES):
                url_queue.put(line)
    finally:
        url_queue.put(None)


def _wait_for_supervisor_url(port, timeout_s, webots_proc, url_queue):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if webots_proc.poll() is not None:
            raise RuntimeError(f'Webots (port {port}) exited before supervisor URL appeared')
        remaining = max(0.0, deadline - time.time())
        try:
            url = url_queue.get(timeout=min(0.5, remaining))
        except queue.Empty:
            continue
        if url is None:
            continue
        if url.rstrip('/').split('/')[-1] == SUPERVISOR_CTRL_NAME:
            return url
    raise TimeoutError(
        f'Timed out waiting for supervisor URL on port {port} after {timeout_s:.0f}s')


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


def run_sample(sample_num, run_name, duration_sim_min, sample_csv,
               port, webots_bin, startup_s, schedule_csv=None):
    world = os.path.join(WORLDS_DIR, f'eval_sample{sample_num}_{ARENA_SIZE}.wbt')
    if not os.path.exists(world):
        print(f'[s{sample_num}] World not found: {world}')
        return False

    print(f'[s{sample_num}] Starting | port={port}')

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

        sup_cmd = [
            sys.executable, SUPERVISOR_PY,
            '--arena_size',       ARENA_SIZE,
            '--run_name',         run_name,
            '--duration_sim_min', str(duration_sim_min),
            '--results_csv',      sample_csv,
            '--sample_id',        str(sample_num),
            '--failure_schedule', schedule_csv or DEFAULT_SCHEDULE,
        ]

        ctrl_env = os.environ.copy()
        ctrl_env['WEBOTS_CONTROLLER_URL'] = ctrl_url
        ctrl_env['WEBOTS_PORT']           = str(port)
        ctrl_env['PYTHONUNBUFFERED']      = '1'
        _add_webots_controller_env(ctrl_env, webots_bin)

        t0 = time.time()
        result = subprocess.run(sup_cmd, cwd=SUPERVISOR_DIR, env=ctrl_env)
        wall_s = time.time() - t0
        ok = result.returncode == 0
        print(f'[s{sample_num}] Done in {wall_s:.1f}s (code {result.returncode}) {"✓" if ok else "✗"}')
        return ok

    except Exception as exc:
        print(f'[s{sample_num}] ERROR: {exc}')
        return False

    finally:
        _terminate(webots_proc)
        if pump_thread is not None:
            pump_thread.join(timeout=5)


def merge_sample_csvs(samples, results_csv):
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

    all_rows.sort(key=lambda r: int(r.get('sample', 0)))

    with open(results_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print(f'[BATCH] Merged {len(all_rows)} rows → {results_csv}')


def print_summary(results_csv):
    if not os.path.exists(results_csv):
        print('[BATCH] No results CSV found.')
        return

    with open(results_csv, newline='') as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print('[BATCH] CSV is empty.')
        return

    print(f'\n{"="*65}')
    n_faults = int(rows[0].get('num_faults', 1)) if rows else 1
    print(f'EX4b — FAULT TOLERANCE RESULTS  ({n_faults} robot(s) halted)')
    print(f'Arena: {ARENA_SIZE} | Duration: {DURATION_SIM_MIN} sim-min | n={len(rows)}')
    print(f'{"="*65}')
    print(f'{"Sample":>8} {"Deposits":>10} {"Rate":>8}')
    print('-' * 30)

    rates = []
    for r in sorted(rows, key=lambda r: int(r.get('sample', 0))):
        rate = float(r.get('sim_rate', 0))
        rates.append(rate)
        print(f'{r["sample"]:>8} {r["deposits"]:>10} {rate:>8.3f}')

    if rates:
        mean = sum(rates) / len(rates)
        std  = (sum((x - mean)**2 for x in rates) / len(rates))**0.5
        print('-' * 50)
        print(f'{"mean":>8} {"":>10} {mean:>8.3f}')
        print(f'{"std":>8} {"":>10} {std:>8.3f}')
        print(f'{"min":>8} {"":>10} {min(rates):>8.3f}')
        print(f'{"max":>8} {"":>10} {max(rates):>8.3f}')

    print(f'\nCSV: {results_csv}')


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
    parser = argparse.ArgumentParser(description='EX4b fault tolerance batch eval')
    parser.add_argument('--run_name',     default=RUN_NAME)
    parser.add_argument('--samples',      default='1-20')
    parser.add_argument('--duration',     type=float, default=DURATION_SIM_MIN)
    parser.add_argument('--max_parallel', type=int,   default=MAX_PARALLEL)
    parser.add_argument('--base_port',    type=int,   default=BASE_PORT)
    parser.add_argument('--webots_bin',   default=WEBOTS_BIN)
    parser.add_argument('--startup_s',    type=float, default=WEBOTS_STARTUP_S)
    parser.add_argument('--schedule',     type=str,   default=None,
                        help='Failure schedule CSV (default: failure_schedule_exp4b.csv). '
                             'Pass failure_schedule_exp4b_2fault.csv for 2-robot fault.')
    args = parser.parse_args()

    schedule_csv = os.path.join(PROJECT_ROOT, args.schedule) if args.schedule \
                   else DEFAULT_SCHEDULE
    if not os.path.exists(schedule_csv):
        print(f'ERROR: failure schedule not found: {schedule_csv}')
        sys.exit(1)

    # Derive fault count from schedule filename for the output CSV name
    fault_tag = '_2fault' if '2fault' in os.path.basename(schedule_csv) else ''

    samples     = parse_sample_range(args.samples)
    max_workers = min(args.max_parallel, len(samples))

    results_csv = os.path.join(PROJECT_ROOT,
                               f'batch_results_fault{fault_tag}_{ARENA_SIZE}_{args.run_name}.csv')

    if os.path.exists(results_csv):
        os.remove(results_csv)
    for s in samples:
        tmp = results_csv + f'.s{s}.tmp'
        if os.path.exists(tmp):
            os.remove(tmp)

    # Write config so robot controllers load the right model
    for fname, val in [('current_eval_run.txt', args.run_name),
                       ('current_eval_arena.txt', ARENA_SIZE)]:
        with open(os.path.join(PROJECT_ROOT, fname), 'w') as f:
            f.write(val)

    print(f'\n{"="*65}')
    print(f'EX4b BATCH EVAL  |  Arena: {ARENA_SIZE}  |  Model: {args.run_name}')
    print(f'Samples: {samples}  |  Duration: {args.duration} sim-min')
    print(f'Parallel: {max_workers}  |  Schedule: {os.path.basename(schedule_csv)}')
    print(f'Results: {results_csv}')
    print(f'{"="*65}\n')

    ok = 0
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {}
        for idx, sample in enumerate(samples):
            port       = args.base_port + idx
            sample_csv = results_csv + f'.s{sample}.tmp'
            future = pool.submit(
                run_sample,
                sample, args.run_name, args.duration, sample_csv,
                port, args.webots_bin, args.startup_s, schedule_csv,
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
    print_summary(results_csv)


if __name__ == '__main__':
    main()
