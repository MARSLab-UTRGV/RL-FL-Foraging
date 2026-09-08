#!/usr/bin/env python3
"""Exp 2 — vary robot swarm size in 7×7 arena.

Centralized PPO: same v9 model loaded once per team of 4 (team replication).
  4  robots → 1 team  → 64  tags  (1.31/m²)
  8  robots → 2 teams → 128 tags  (2.61/m²)
  12 robots → 3 teams → 192 tags  (3.92/m²)
  16 robots → 4 teams → 256 tags  (5.22/m²)

Usage:
    python run_exp2_robots.py --method centralized_ppo --num-robots 8
    python run_exp2_robots.py --method cpfa_baseline   --num-robots 16 --samples 1-10
    python run_exp2_robots.py --method centralized_ppo --num-robots 4  --dry-run
"""

import argparse
import csv
import os
import queue
import re
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
WORLDS_DIR   = PROJECT_ROOT / "worlds"
RESULTS_DIR  = PROJECT_ROOT / "results"

ARENA                 = "7x7"
DEFAULT_FORAGING_TIME = 10.0

# Exp 2 fixed mapping: num_robots → num_tags  (density = n_tags/49 m²)
ROBOT_CONFIGS = {
    4:  64,    # 1.31 /m²
    8:  128,   # 2.61 /m²
    12: 192,   # 3.92 /m²
    16: 256,   # 5.22 /m²
}

CONTROLLERS = {
    "centralized_ppo": "controllers/eval_best_model/eval_best_model_7x7.py",
    "cpfa_baseline":   "controllers/cpfa_baseline/cpfa_baseline_7x7.py",
}

CSV_FIELDS = [
    "method",
    "arena",
    "num_robots",
    "num_tags",
    "sample",
    "foraging_time_min",
    "pickups",
    "deposits",
]

BATCH_RESULT_RE           = re.compile(r"BATCH_RESULT pickups=(\d+) deposits=(\d+)")
EXTERN_URL_PREFIXES       = ("ipc://", "tcp://")
SUPERVISOR_CONTROLLER_NAME = "supervisor"


# =============================================================================
# Helpers
# =============================================================================

def parse_sample_range(value):
    samples = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            samples.extend(range(int(start), int(end) + 1))
        else:
            samples.append(int(part))
    if not samples:
        raise argparse.ArgumentTypeError("samples cannot be empty")
    return samples


def format_float(value):
    return f"{value:g}"


def default_results_csv(method, num_robots):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"exp2_{method}_{ARENA}_{num_robots}r_{stamp}.csv"


def model_exists(model):
    path = Path(model)
    candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path]
    if not str(path).endswith(".zip"):
        candidates.append(Path(str(candidates[0]) + ".zip"))
    return any(c.exists() for c in candidates)


def prepend_env_path(env, key, value):
    value = str(value)
    current = env.get(key)
    env[key] = f"{value}{os.pathsep}{current}" if current else value


def infer_webots_home(webots_bin):
    if os.environ.get("WEBOTS_HOME"):
        return Path(os.environ["WEBOTS_HOME"])
    resolved = shutil.which(webots_bin)
    if resolved is None:
        return None
    path = Path(resolved).resolve()
    if path.name == "webots-bin" and path.parent.name == "bin":
        return path.parent.parent
    return path.parent


def add_webots_controller_env(env, webots_bin):
    webots_home = infer_webots_home(webots_bin)
    if webots_home is None:
        return
    env["WEBOTS_HOME"] = str(webots_home)
    prepend_env_path(env, "PYTHONPATH", webots_home / "lib" / "controller" / "python")
    prepend_env_path(env, "LD_LIBRARY_PATH", webots_home / "lib" / "controller")


def world_path(num_robots, sample):
    """4-robot runs reuse existing eval_sampleN_7x7.wbt; others use _Nr suffix."""
    if num_robots == 4:
        return WORLDS_DIR / f"eval_sample{sample}_{ARENA}.wbt"
    return WORLDS_DIR / f"eval_sample{sample}_{ARENA}_{num_robots}r.wbt"


def build_commands(args, sample, port):
    num_tags   = ROBOT_CONFIGS[args.num_robots]
    world      = world_path(args.num_robots, sample)
    controller = PROJECT_ROOT / CONTROLLERS[args.method]

    webots_cmd = [
        args.webots_bin,
        "--batch",
        f"--port={port}",
        "--extern-urls",
        "--mode=fast",
        "--minimize",
        "--no-rendering",
        str(world),
    ]

    controller_cmd = [sys.executable, str(controller)]
    if args.method == "centralized_ppo":
        controller_cmd.append(args.model)
    controller_cmd.extend([
        "--num-robots",        str(args.num_robots),
        "--num-tags",          str(num_tags),
        "--duration-sim-min",  format_float(args.foraging_time),
    ])

    return world, controller, webots_cmd, controller_cmd


# =============================================================================
# Process management (identical pattern to run_foraging_experiments.py)
# =============================================================================

def tail_file(path, max_bytes=4000):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes), os.SEEK_SET)
            return f.read().decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def pump_webots_output(stdout, log_file, extern_urls):
    try:
        for line in stdout:
            log_file.write(line)
            log_file.flush()
            stripped = line.strip()
            if stripped.startswith(EXTERN_URL_PREFIXES):
                extern_urls.put(stripped)
    finally:
        extern_urls.put(None)


def extern_controller_name(url):
    return url.rstrip("/").split("/")[-1]


def wait_for_extern_url(controller_name, port, timeout, proc, log_path, extern_urls):
    deadline = time.time() + timeout
    seen_urls = []
    while time.time() < deadline:
        if proc.poll() is not None:
            log_tail = tail_file(log_path)
            detail = f"\nLast Webots log lines:\n{log_tail}" if log_tail else ""
            raise RuntimeError(
                f"webots exited before extern controller '{controller_name}' "
                f"became ready on port {port}; see {log_path}{detail}"
            )
        wait_time = max(0.0, min(0.5, deadline - time.time()))
        try:
            url = extern_urls.get(timeout=wait_time)
        except queue.Empty:
            continue
        if url is None:
            continue
        seen_urls.append(url)
        if extern_controller_name(url) == controller_name:
            return url

    log_tail = tail_file(log_path)
    detail = f"\nLast Webots log lines:\n{log_tail}" if log_tail else ""
    seen_detail = f" Saw extern URL(s): {', '.join(seen_urls)}." if seen_urls else ""
    raise TimeoutError(
        f"timed out waiting for extern controller '{controller_name}' "
        f"on Webots port {port}.{seen_detail} See {log_path}{detail}"
    )


def terminate_process(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


# =============================================================================
# Sample runner
# =============================================================================

def run_sample(args, sample, port, logs_dir):
    world, controller, webots_cmd, controller_cmd = build_commands(args, sample, port)

    if not world.exists():
        raise FileNotFoundError(
            f"World not found: {world}\n"
            f"  4-robot runs reuse existing 7x7 worlds.\n"
            f"  For {args.num_robots} robots you need: {world.name}"
        )
    if not controller.exists():
        raise FileNotFoundError(f"Controller not found: {controller}")

    num_tags = ROBOT_CONFIGS[args.num_robots]
    prefix   = (f"{args.method}_{ARENA}_{args.num_robots}r_{num_tags}t"
                f"_sample{sample}_port{port}")
    webots_log     = logs_dir / f"{prefix}_webots.log"
    controller_log = logs_dir / f"{prefix}_controller.log"

    webots_env = os.environ.copy()
    webots_env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
    webots_env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    webots_env["WEBOTS_PORT"] = str(port)

    controller_env = os.environ.copy()
    controller_env["WEBOTS_PORT"] = str(port)
    controller_env["PYTHONUNBUFFERED"] = "1"
    add_webots_controller_env(controller_env, args.webots_bin)
    if args.method == "centralized_ppo":
        controller_env["CUDA_VISIBLE_DEVICES"] = ""

    webots_proc         = None
    webots_output_thread = None
    result_match        = None
    return_code         = None

    with open(webots_log, "w") as webots_out:
        extern_urls = queue.Queue()
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
        try:
            assert webots_proc.stdout is not None
            webots_output_thread = threading.Thread(
                target=pump_webots_output,
                args=(webots_proc.stdout, webots_out, extern_urls),
                daemon=True,
            )
            webots_output_thread.start()
            controller_env["WEBOTS_CONTROLLER_URL"] = wait_for_extern_url(
                SUPERVISOR_CONTROLLER_NAME,
                port,
                args.startup_seconds,
                webots_proc,
                webots_log,
                extern_urls,
            )
            with open(controller_log, "w") as controller_out:
                for attempt in range(1, args.controller_start_retries + 1):
                    if attempt > 1:
                        controller_out.write(
                            f"\n[BATCH] Restarting controller startup attempt {attempt} "
                            f"of {args.controller_start_retries}\n"
                        )
                        controller_out.flush()

                    controller_proc = subprocess.Popen(
                        controller_cmd,
                        cwd=PROJECT_ROOT,
                        env=controller_env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        start_new_session=True,
                    )
                    try:
                        assert controller_proc.stdout is not None
                        for line in controller_proc.stdout:
                            controller_out.write(line)
                            controller_out.flush()
                            match = BATCH_RESULT_RE.search(line)
                            if match:
                                result_match = match
                        return_code = controller_proc.wait()
                    finally:
                        terminate_process(controller_proc)

                    if return_code == 0 or result_match is not None:
                        break

                    log_tail = tail_file(controller_log)
                    if "Loading model:" in log_tail or "Traceback" in log_tail:
                        break
                    time.sleep(2.0)
        finally:
            terminate_process(webots_proc)
            if webots_output_thread is not None:
                webots_output_thread.join(timeout=5)

    if result_match is None:
        log_tail = tail_file(controller_log)
        detail = f"\nLast controller log lines:\n{log_tail}" if log_tail else ""
        if return_code != 0:
            raise RuntimeError(
                f"Controller failed for sample {sample} (exit {return_code}); "
                f"see {controller_log}{detail}"
            )
        raise RuntimeError(
            f"Missing BATCH_RESULT for sample {sample}; see {controller_log}{detail}"
        )

    num_tags = ROBOT_CONFIGS[args.num_robots]
    return {
        "method":           args.method,
        "arena":            ARENA,
        "num_robots":       args.num_robots,
        "num_tags":         num_tags,
        "sample":           sample,
        "foraging_time_min": format_float(args.foraging_time),
        "pickups":          int(result_match.group(1)),
        "deposits":         int(result_match.group(2)),
    }


# =============================================================================
# CSV
# =============================================================================

def write_rows(results_csv, rows):
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not results_csv.exists() or results_csv.stat().st_size == 0
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


# =============================================================================
# Dry-run
# =============================================================================

def print_dry_run(args, samples, results_csv):
    num_tags = ROBOT_CONFIGS[args.num_robots]
    print(f"Results CSV:   {results_csv}")
    print(f"Arena:         {ARENA}")
    print(f"Robots:        {args.num_robots}  (teams of 4: {args.num_robots // 4})")
    print(f"Tags:          {num_tags}")
    print(f"Foraging time: {format_float(args.foraging_time)} sim min")
    for index, sample in enumerate(samples):
        port = args.base_port + index
        world, _, webots_cmd, controller_cmd = build_commands(args, sample, port)
        env_prefix = (
            f"__NV_PRIME_RENDER_OFFLOAD=1 "
            f"__GLX_VENDOR_LIBRARY_NAME=nvidia "
            f"WEBOTS_PORT={port}"
        )
        print(f"\nSample {sample} | port {port} | world {world.name}")
        print(f"{env_prefix} {shlex.join(webots_cmd)}")
        print(
            f"WEBOTS_CONTROLLER_URL=<from --extern-urls> "
            f"WEBOTS_PORT={port} {shlex.join(controller_cmd)}"
        )


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Exp 2: vary robot swarm size in 7×7 arena."
    )
    parser.add_argument("--method",      required=True, choices=sorted(CONTROLLERS))
    parser.add_argument("--num-robots",  required=True, type=int,
                        choices=sorted(ROBOT_CONFIGS),
                        help="Swarm size. Tags are fixed per config: "
                             "4→64, 8→128, 12→192, 16→256.")
    parser.add_argument("--samples",     type=parse_sample_range,
                        default=parse_sample_range("1-10"))
    parser.add_argument("--foraging-time", type=float, default=DEFAULT_FORAGING_TIME,
                        help="Simulated minutes per run (default 10).")
    parser.add_argument("--model",       default="ppo_cpfa_v9.zip",
                        help="Model path for centralized_ppo (default ppo_cpfa_v9.zip).")
    parser.add_argument("--base-port",   type=int, default=1438)
    parser.add_argument("--max-parallel", type=int, default=10,
                        help="Max parallel Webots instances (default = number of samples).")
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--webots-bin",  default="webots")
    parser.add_argument("--startup-seconds", type=float, default=60.0,
                        help="Seconds to wait for Webots extern URL (default 60).")
    parser.add_argument("--controller-start-retries", type=int, default=3)
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    if args.foraging_time <= 0:
        parser.error("--foraging-time must be greater than 0")
    if args.max_parallel is None:
        args.max_parallel = 10
    if args.max_parallel <= 0:
        parser.error("--max-parallel must be greater than 0")
    if args.method == "centralized_ppo" and not model_exists(args.model):
        parser.error(f"--model not found: {args.model}")

    results_csv = args.results_csv or default_results_csv(args.method, args.num_robots)
    if not results_csv.is_absolute():
        results_csv = PROJECT_ROOT / results_csv

    if args.dry_run:
        print_dry_run(args, args.samples, results_csv)
        return 0

    num_tags = ROBOT_CONFIGS[args.num_robots]
    run_dir  = results_csv.parent / f"{results_csv.stem}_logs"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method:        {args.method}")
    print(f"Arena:         {ARENA}")
    print(f"Robots:        {args.num_robots}  (teams of 4: {args.num_robots // 4})")
    print(f"Tags:          {num_tags}")
    print(f"Samples:       {args.samples}")
    print(f"Foraging time: {format_float(args.foraging_time)} sim min")
    print(f"Results CSV:   {results_csv}")
    print(f"Logs:          {logs_dir}")
    print(f"Parallel jobs: {min(args.max_parallel, len(args.samples))}")

    rows     = []
    failures = []
    with ThreadPoolExecutor(
        max_workers=min(args.max_parallel, len(args.samples))
    ) as pool:
        future_to_sample = {}
        for index, sample in enumerate(args.samples):
            port   = args.base_port + index
            future = pool.submit(run_sample, args, sample, port, logs_dir)
            future_to_sample[future] = sample

        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            try:
                row = future.result()
                rows.append(row)
                print(
                    f"[sample {sample}] pickups={row['pickups']} "
                    f"deposits={row['deposits']}"
                )
            except Exception as exc:
                failures.append((sample, exc))
                print(f"[sample {sample}] FAILED: {exc}", file=sys.stderr)

    rows.sort(key=lambda r: int(r["sample"]))
    if rows:
        write_rows(results_csv, rows)

    if failures:
        print(f"\nCompleted with {len(failures)} failure(s).", file=sys.stderr)
        return 1

    print(f"\nWrote {len(rows)} row(s) to {results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
