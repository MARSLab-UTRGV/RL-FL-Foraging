#!/usr/bin/env python3
"""Run one arena/method foraging evaluation batch."""

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
WORLDS_DIR = PROJECT_ROOT / "worlds"
RESULTS_DIR = PROJECT_ROOT / "results"

DEFAULT_FORAGING_TIMES = {
    "5x5": 10.0,
    "7x7": 25.0,
    "9x9": 50.0,
    "12x12": 95.0,
}

CONTROLLERS = {
    "centralized_ppo": {
        "5x5": "controllers/eval_best_model/eval_best_model_5x5.py",
        "7x7": "controllers/eval_best_model/eval_best_model_7x7.py",
        "9x9": "controllers/eval_best_model/eval_best_model_9x9.py",
        "12x12": "controllers/eval_best_model/eval_best_model_12x12.py",
    },
    "cpfa_baseline": {
        "5x5": "controllers/cpfa_baseline/cpfa_baseline.py",
        "7x7": "controllers/cpfa_baseline/cpfa_baseline_7x7.py",
        "9x9": "controllers/cpfa_baseline/cpfa_baseline_9x9.py",
        "12x12": "controllers/cpfa_baseline/cpfa_baseline_12x12.py",
    },
}

CSV_FIELDS = [
    "method",
    "arena",
    "sample",
    "foraging_time_min",
    "pickups",
    "deposits",
]

BATCH_RESULT_RE = re.compile(r"BATCH_RESULT pickups=(\d+) deposits=(\d+)")
EXTERN_URL_PREFIXES = ("ipc://", "tcp://")
SUPERVISOR_CONTROLLER_NAME = "supervisor"


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


def default_results_csv(method, arena):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"foraging_{method}_{arena}_{stamp}.csv"


def model_exists(model):
    path = Path(model)
    candidates = [path] if path.is_absolute() else [PROJECT_ROOT / path]
    if not str(path).endswith(".zip"):
        candidates.append(Path(str(candidates[0]) + ".zip"))
    return any(candidate.exists() for candidate in candidates)


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


def build_commands(args, sample, port, duration):
    world = WORLDS_DIR / f"eval_sample{sample}_{args.arena}.wbt"
    controller = PROJECT_ROOT / CONTROLLERS[args.method][args.arena]

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

    controller_cmd = [
        sys.executable,
        str(controller),
    ]
    if args.method == "centralized_ppo":
        controller_cmd.append(args.model)
    controller_cmd.extend(["--duration-sim-min", format_float(duration)])

    return world, controller, webots_cmd, controller_cmd


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


def run_sample(args, sample, port, duration, logs_dir):
    world, controller, webots_cmd, controller_cmd = build_commands(
        args, sample, port, duration
    )

    if not world.exists():
        raise FileNotFoundError(f"world not found: {world}")
    if not controller.exists():
        raise FileNotFoundError(f"controller not found: {controller}")

    prefix = f"{args.method}_{args.arena}_sample{sample}_port{port}"
    webots_log = logs_dir / f"{prefix}_webots.log"
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

    webots_proc = None
    webots_output_thread = None
    result_match = None
    return_code = None
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

    if return_code != 0:
        log_tail = tail_file(controller_log)
        detail = f"\nLast controller log lines:\n{log_tail}" if log_tail else ""
        raise RuntimeError(
            f"controller failed for sample {sample} with exit code {return_code}; "
            f"see {controller_log}{detail}"
        )
    if result_match is None:
        log_tail = tail_file(controller_log)
        detail = f"\nLast controller log lines:\n{log_tail}" if log_tail else ""
        raise RuntimeError(
            f"missing BATCH_RESULT for sample {sample}; see {controller_log}{detail}"
        )

    pickups = int(result_match.group(1))
    deposits = int(result_match.group(2))
    return {
        "method": args.method,
        "arena": args.arena,
        "sample": sample,
        "foraging_time_min": format_float(duration),
        "pickups": pickups,
        "deposits": deposits,
    }


def write_rows(results_csv, rows):
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not results_csv.exists() or results_csv.stat().st_size == 0
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_dry_run(args, samples, duration, results_csv):
    print(f"Results CSV: {results_csv}")
    print(f"Foraging time: {format_float(duration)} sim min")
    for index, sample in enumerate(samples):
        port = args.base_port + index
        _, _, webots_cmd, controller_cmd = build_commands(args, sample, port, duration)
        env_prefix = (
            f"__NV_PRIME_RENDER_OFFLOAD=1 "
            f"__GLX_VENDOR_LIBRARY_NAME=nvidia "
            f"WEBOTS_PORT={port}"
        )
        print(f"\nSample {sample} | port {port}")
        print(f"{env_prefix} {shlex.join(webots_cmd)}")
        print(
            f"WEBOTS_CONTROLLER_URL=<from --extern-urls> "
            f"WEBOTS_PORT={port} {shlex.join(controller_cmd)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run one foraging evaluation method and arena."
    )
    parser.add_argument("--method", required=True, choices=sorted(CONTROLLERS))
    parser.add_argument("--arena", required=True, choices=sorted(DEFAULT_FORAGING_TIMES))
    parser.add_argument("--samples", type=parse_sample_range, default=parse_sample_range("1-10"))
    parser.add_argument("--foraging-time", type=float, default=None,
                        help="Simulated minutes. Defaults by arena.")
    parser.add_argument("--model", default="ppo_cpfa_v9.zip",
                        help="Model path for centralized_ppo.")
    parser.add_argument("--base-port", type=int, default=1438)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--webots-bin", default="webots")
    parser.add_argument("--startup-seconds", type=float, default=30.0)
    parser.add_argument("--controller-start-retries", type=int, default=3,
                        help="Retry extern controller startup if it exits before model load.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    duration = args.foraging_time
    if duration is None:
        duration = DEFAULT_FORAGING_TIMES[args.arena]
    if duration <= 0:
        parser.error("--foraging-time must be greater than 0")

    if args.max_parallel is None:
        args.max_parallel = len(args.samples)
    if args.max_parallel <= 0:
        parser.error("--max-parallel must be greater than 0")

    if args.method == "centralized_ppo" and not model_exists(args.model):
        parser.error(f"--model not found: {args.model}")

    results_csv = args.results_csv or default_results_csv(args.method, args.arena)
    if not results_csv.is_absolute():
        results_csv = PROJECT_ROOT / results_csv

    if args.dry_run:
        print_dry_run(args, args.samples, duration, results_csv)
        return 0

    run_dir = results_csv.parent / f"{results_csv.stem}_logs"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method: {args.method}")
    print(f"Arena: {args.arena}")
    print(f"Samples: {args.samples}")
    print(f"Foraging time: {format_float(duration)} sim min")
    print(f"Results CSV: {results_csv}")
    print(f"Logs: {logs_dir}")
    print(f"Parallel jobs: {min(args.max_parallel, len(args.samples))}")

    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=min(args.max_parallel, len(args.samples))) as pool:
        future_to_sample = {}
        for index, sample in enumerate(args.samples):
            port = args.base_port + index
            future = pool.submit(run_sample, args, sample, port, duration, logs_dir)
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

    rows.sort(key=lambda row: int(row["sample"]))
    if rows:
        write_rows(results_csv, rows)

    if failures:
        print(f"\nCompleted with {len(failures)} failure(s).", file=sys.stderr)
        return 1

    print(f"\nWrote {len(rows)} row(s) to {results_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
