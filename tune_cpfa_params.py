#!/usr/bin/env python3
"""
Bayesian hyperparameter search for CPFA baseline parameters using Optuna.

Each trial:
  - Samples 7 CPFA parameters
  - Writes a temporary params yaml
  - Runs N_SAMPLES parallel Webots simulations (5x5, 10 min, clustered)
  - Returns mean deposits (maximize)

Usage:
    python3 tune_cpfa_params.py
    python3 tune_cpfa_params.py --n-trials 100 --n-samples 5 --base-port 2000
"""

import argparse
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
WORLDS_DIR   = PROJECT_ROOT / "worlds"
CONTROLLER   = PROJECT_ROOT / "controllers/cpfa_baseline/cpfa_baseline.py"

ARENA           = "5x5"
FORAGING_MIN    = 10.0
N_SAMPLES       = 5       # worlds per trial (parallel)
N_TRIALS        = 50      # Optuna budget
BASE_PORT       = 2100    # avoid clashing with other scripts
STARTUP_SEC     = 30.0
WEBOTS_BIN      = "webots"

BATCH_RE              = re.compile(r"BATCH_RESULT pickups=(\d+) deposits=(\d+)")
EXTERN_URL_PREFIXES   = ("ipc://", "tcp://")
SUPERVISOR_NAME       = "supervisor"

# Search bounds for each parameter
PARAM_SPACE = {
    "rate_of_laying_pheromone":        (1.0,  12.0),
    "rate_of_site_fidelity":           (0.5,   4.0),
    "rate_of_pheromone_decay":         (0.02,  0.20),
    "probability_of_returning_to_nest":(0.005, 0.15),
    "probability_of_switching_to_searching": (0.3, 0.95),
    "uninformed_search_variation_rad":  (0.05,  4.0),
    "rate_of_informed_search_decay":    (0.05,  2.5),
}

# =============================================================================
# Helpers (adapted from run_foraging_experiments.py)
# =============================================================================

def write_params_yaml(path, params):
    with open(path, "w") as f:
        for k, v in params.items():
            f.write(f"{k}: {v}\n")


def infer_webots_home():
    if os.environ.get("WEBOTS_HOME"):
        return Path(os.environ["WEBOTS_HOME"])
    resolved = shutil.which(WEBOTS_BIN)
    if resolved is None:
        return None
    p = Path(resolved).resolve()
    if p.name == "webots-bin" and p.parent.name == "bin":
        return p.parent.parent
    return p.parent


def add_webots_env(env):
    home = infer_webots_home()
    if home is None:
        return
    env["WEBOTS_HOME"] = str(home)
    def prepend(key, val):
        cur = env.get(key)
        env[key] = f"{val}{os.pathsep}{cur}" if cur else str(val)
    prepend("PYTHONPATH",      home / "lib/controller/python")
    prepend("LD_LIBRARY_PATH", home / "lib/controller")


def pump_output(stdout, log_file, url_queue):
    try:
        for line in stdout:
            log_file.write(line)
            log_file.flush()
            s = line.strip()
            if s.startswith(EXTERN_URL_PREFIXES):
                url_queue.put(s)
    finally:
        url_queue.put(None)


def wait_for_url(port, timeout, proc, log_path, url_queue):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Webots exited early on port {port}")
        try:
            url = url_queue.get(timeout=min(0.5, deadline - time.time()))
        except queue.Empty:
            continue
        if url is None:
            continue
        if url.rstrip("/").split("/")[-1] == SUPERVISOR_NAME:
            return url
    raise TimeoutError(f"Timed out waiting for supervisor on port {port}")


def kill_proc(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
        except Exception:
            pass


def run_one_sample(sample, port, params_yaml_path, logs_dir):
    """Run a single Webots simulation and return deposits, or None on failure."""
    world = WORLDS_DIR / f"eval_sample{sample}_{ARENA}.wbt"
    if not world.exists():
        raise FileNotFoundError(f"World not found: {world}")

    prefix       = f"tune_sample{sample}_port{port}"
    webots_log   = logs_dir / f"{prefix}_webots.log"
    ctrl_log     = logs_dir / f"{prefix}_ctrl.log"

    webots_env = os.environ.copy()
    webots_env.update({
        "WEBOTS_PORT":                  str(port),
        "__NV_PRIME_RENDER_OFFLOAD":    "1",
        "__GLX_VENDOR_LIBRARY_NAME":    "nvidia",
    })

    ctrl_env = os.environ.copy()
    ctrl_env["WEBOTS_PORT"] = str(port)
    ctrl_env["PYTHONUNBUFFERED"] = "1"
    add_webots_env(ctrl_env)

    webots_cmd = [
        WEBOTS_BIN, "--batch", f"--port={port}",
        "--extern-urls", "--mode=fast", "--minimize", "--no-rendering",
        str(world),
    ]
    ctrl_cmd = [
        sys.executable, str(CONTROLLER),
        "--params", str(params_yaml_path),
        "--duration-sim-min", str(FORAGING_MIN),
    ]

    webots_proc = None
    deposits = None
    with open(webots_log, "w") as wf:
        url_q = queue.Queue()
        webots_proc = subprocess.Popen(
            webots_cmd, cwd=PROJECT_ROOT, env=webots_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        try:
            t = threading.Thread(
                target=pump_output,
                args=(webots_proc.stdout, wf, url_q),
                daemon=True,
            )
            t.start()
            ctrl_env["WEBOTS_CONTROLLER_URL"] = wait_for_url(
                port, STARTUP_SEC, webots_proc, webots_log, url_q)

            with open(ctrl_log, "w") as cf:
                ctrl_proc = subprocess.Popen(
                    ctrl_cmd, cwd=PROJECT_ROOT, env=ctrl_env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, start_new_session=True,
                )
                try:
                    for line in ctrl_proc.stdout:
                        cf.write(line)
                        cf.flush()
                        m = BATCH_RE.search(line)
                        if m:
                            deposits = int(m.group(2))
                    ctrl_proc.wait()
                finally:
                    kill_proc(ctrl_proc)
            t.join(timeout=5)
        finally:
            kill_proc(webots_proc)

    return deposits


def evaluate(params, trial_num, logs_dir):
    """Run N_SAMPLES in parallel and return mean deposits."""
    params_file = logs_dir / "params.yaml"
    write_params_yaml(params_file, params)

    results = {}
    with ThreadPoolExecutor(max_workers=N_SAMPLES) as pool:
        futures = {
            pool.submit(run_one_sample, s, BASE_PORT + trial_num * N_SAMPLES + s,
                        params_file, logs_dir): s
            for s in range(1, N_SAMPLES + 1)
        }
        for fut in as_completed(futures):
            s = futures[fut]
            try:
                dep = fut.result()
                results[s] = dep if dep is not None else 0
            except Exception as e:
                print(f"  [WARN] sample {s} failed: {e}")
                results[s] = 0

    deposits = list(results.values())
    mean_dep = sum(deposits) / len(deposits) if deposits else 0.0
    return mean_dep, deposits


# =============================================================================
# Optuna objective
# =============================================================================

def make_objective(study_logs_dir):
    def objective(trial):
        params = {
            k: trial.suggest_float(k, lo, hi)
            for k, (lo, hi) in PARAM_SPACE.items()
        }

        trial_dir = study_logs_dir / f"trial_{trial.number:03d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        mean_dep, deposits = evaluate(params, trial.number, trial_dir)

        # Log to file
        with open(study_logs_dir / "search_log.txt", "a") as f:
            f.write(
                f"Trial {trial.number:3d} | mean={mean_dep:.2f} | "
                f"samples={deposits} | "
                f"params={{{', '.join(f'{k}={v:.4f}' for k,v in params.items())}}}\n"
            )

        print(
            f"Trial {trial.number:3d} | mean_deposits={mean_dep:.2f} | "
            f"samples={[f'{d}' for d in deposits]}"
        )
        return mean_dep

    return objective


# =============================================================================
# Main
# =============================================================================

def main():
    global N_SAMPLES, BASE_PORT

    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials",  type=int, default=N_TRIALS)
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES,
                        help="Worlds per trial (parallel Webots instances)")
    parser.add_argument("--base-port", type=int, default=BASE_PORT)
    args = parser.parse_args()

    N_SAMPLES = args.n_samples
    BASE_PORT = args.base_port

    stamp     = time.strftime("%Y%m%d_%H%M%S")
    logs_dir  = PROJECT_ROOT / f"cpfa_tuning_{stamp}"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"CPFA Parameter Search")
    print(f"  Trials: {args.n_trials}  |  Samples/trial: {N_SAMPLES}  |  Arena: {ARENA} {FORAGING_MIN}min")
    print(f"  Logs: {logs_dir}")
    print(f"  Parameter ranges:")
    for k, (lo, hi) in PARAM_SPACE.items():
        print(f"    {k}: [{lo}, {hi}]")
    print()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="cpfa_param_search",
    )

    # Seed with known good configurations (warm start)
    study.enqueue_trial({
        "rate_of_laying_pheromone":              9.0,
        "rate_of_site_fidelity":                 2.29,
        "rate_of_pheromone_decay":               0.1,
        "probability_of_returning_to_nest":      0.08,
        "probability_of_switching_to_searching": 0.55,
        "uninformed_search_variation_rad":        0.13,
        "rate_of_informed_search_decay":          1.26,
    })
    study.enqueue_trial({
        "rate_of_laying_pheromone":              3.0,
        "rate_of_site_fidelity":                 1.376,
        "rate_of_pheromone_decay":               0.05,
        "probability_of_returning_to_nest":      0.0189,
        "probability_of_switching_to_searching": 0.765,
        "uninformed_search_variation_rad":        3.67,
        "rate_of_informed_search_decay":          0.346,
    })

    study.optimize(make_objective(logs_dir), n_trials=args.n_trials)

    best = study.best_trial
    print(f"\n{'='*65}")
    print(f"BEST TRIAL: {best.number}  |  mean_deposits={best.value:.2f}")
    print(f"{'='*65}")
    best_params = best.params
    for k, v in best_params.items():
        print(f"  {k}: {v:.4f}")

    # Write best params to cpfa_params_best.yaml
    best_yaml = PROJECT_ROOT / "controllers/cpfa_baseline/cpfa_params_best.yaml"
    write_params_yaml(best_yaml, best_params)
    print(f"\nBest params saved to: {best_yaml}")
    print(f"Full search log:       {logs_dir / 'search_log.txt'}")
    print(f"\nTo use best params:")
    print(f"  cp {best_yaml} controllers/cpfa_baseline/cpfa_params.yaml")
    print(f"  python3 run_foraging_experiments.py --method cpfa_baseline --arena 5x5 --samples 1-20")


if __name__ == "__main__":
    main()
