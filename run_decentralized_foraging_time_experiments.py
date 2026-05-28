#!/usr/bin/env python3
"""Run parallel 5x5 decentralized fixed-time foraging evaluation batches."""

import argparse
import csv
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import run_decentralized_completion_experiments as completion


PROJECT_ROOT = completion.PROJECT_ROOT
RESULTS_DIR = completion.RESULTS_DIR
CONTROLLER = completion.CONTROLLER
ARENA = completion.ARENA
METHOD = completion.METHOD
DEFAULT_RUN_NAME = completion.DEFAULT_RUN_NAME
DEFAULT_FORAGING_TIME_MIN = 30.0
EXPERIMENT = "foraging_time"

CSV_FIELDS = [
    "method",
    "experiment",
    "distribution",
    "arena",
    "sample",
    "foraging_time_min",
    "completed_all_tags",
    "status",
    "elapsed_sim_min",
    "steps",
    "pickups",
    "deposits",
]


def time_token(value):
    return completion.format_float(value).replace(".", "p")


def default_results_csv(run_name, distribution, foraging_time_min):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / (
        f"decentralized_{EXPERIMENT}_{run_name}_{distribution}_{ARENA}_"
        f"{time_token(foraging_time_min)}min_{stamp}.csv"
    )


def default_results_csv_in(directory, run_name, distribution, foraging_time_min):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return directory / (
        f"decentralized_{EXPERIMENT}_{run_name}_{distribution}_{ARENA}_"
        f"{time_token(foraging_time_min)}min_{stamp}.csv"
    )


def resolve_results_csv(results_csv, run_name, distribution, foraging_time_min):
    if results_csv is None:
        return default_results_csv(run_name, distribution, foraging_time_min)
    if not results_csv.is_absolute():
        results_csv = PROJECT_ROOT / results_csv
    if results_csv.exists() and results_csv.is_dir():
        return default_results_csv_in(
            results_csv, run_name, distribution, foraging_time_min
        )
    if not results_csv.exists() and results_csv.suffix == "":
        return default_results_csv_in(
            results_csv, run_name, distribution, foraging_time_min
        )
    return results_csv


def build_commands(args, sample, port):
    world = completion.world_path(args.distribution, sample)

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
        str(CONTROLLER),
        "--arena_size",
        ARENA,
        "--run_name",
        args.run_name,
        "--duration_sim_min",
        completion.format_float(args.foraging_time_min),
        "--sample_id",
        str(sample),
    ]

    return world, webots_cmd, controller_cmd


def row_from_result(args, sample, result_match):
    label = result_match.group(1)
    pickups = int(result_match.group(2))
    deposits = int(result_match.group(3))
    steps = int(result_match.group(4))
    sim_time_min = float(result_match.group(5))
    completed = label == "COMPLETION_RESULT"

    return {
        "method": METHOD,
        "experiment": EXPERIMENT,
        "distribution": args.distribution,
        "arena": ARENA,
        "sample": sample,
        "foraging_time_min": completion.format_float(args.foraging_time_min),
        "completed_all_tags": "true" if completed else "false",
        "status": "completed_all_tags" if completed else "duration_elapsed",
        "elapsed_sim_min": completion.format_float(sim_time_min),
        "steps": steps,
        "pickups": pickups,
        "deposits": deposits,
    }


def run_sample(args, sample, port, logs_dir):
    world, webots_cmd, controller_cmd = build_commands(args, sample, port)

    if not world.exists():
        raise FileNotFoundError(f"world not found: {world}")
    if not CONTROLLER.exists():
        raise FileNotFoundError(f"controller not found: {CONTROLLER}")

    prefix = (
        f"{METHOD}_{EXPERIMENT}_{args.run_name}_{args.distribution}_{ARENA}_"
        f"sample{sample}_port{port}"
    )
    webots_log = logs_dir / f"{prefix}_webots.log"
    controller_log = logs_dir / f"{prefix}_controller.log"

    webots_env = os.environ.copy()
    webots_env["__NV_PRIME_RENDER_OFFLOAD"] = "1"
    webots_env["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
    webots_env["WEBOTS_PORT"] = str(port)

    controller_env = os.environ.copy()
    controller_env["WEBOTS_PORT"] = str(port)
    controller_env["PYTHONUNBUFFERED"] = "1"
    completion.add_webots_controller_env(controller_env, args.webots_bin)

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
                target=completion.pump_webots_output,
                args=(webots_proc.stdout, webots_out, extern_urls),
                daemon=True,
            )
            webots_output_thread.start()
            controller_env["WEBOTS_CONTROLLER_URL"] = completion.wait_for_extern_url(
                completion.SUPERVISOR_CONTROLLER_NAME,
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
                            match = completion.RESULT_RE.search(line)
                            if match:
                                result_match = match
                        return_code = controller_proc.wait()
                    finally:
                        completion.terminate_process(controller_proc)

                    if return_code == 0 or result_match is not None:
                        break

                    log_tail = completion.tail_file(controller_log)
                    if "Loading" in log_tail or "Traceback" in log_tail:
                        break
                    time.sleep(2.0)
        finally:
            completion.terminate_process(webots_proc)
            if webots_output_thread is not None:
                webots_output_thread.join(timeout=5)

    if return_code != 0:
        log_tail = completion.tail_file(controller_log)
        detail = f"\nLast controller log lines:\n{log_tail}" if log_tail else ""
        raise RuntimeError(
            f"controller failed for sample {sample} with exit code {return_code}; "
            f"see {controller_log}{detail}"
        )
    if result_match is None:
        log_tail = completion.tail_file(controller_log)
        detail = f"\nLast controller log lines:\n{log_tail}" if log_tail else ""
        raise RuntimeError(
            f"missing COMPLETION_RESULT/TIMEOUT_RESULT for sample {sample}; "
            f"see {controller_log}{detail}"
        )

    return row_from_result(args, sample, result_match)


def write_rows(results_csv, rows):
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not results_csv.exists() or results_csv.stat().st_size == 0
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_dry_run(args, samples, results_csv):
    print(f"Results CSV: {results_csv}")
    print(f"Run name: {args.run_name}")
    print(f"Distribution: {args.distribution}")
    print(f"Foraging time: {completion.format_float(args.foraging_time_min)} sim-min")
    print(f"Config: current_eval_run.txt={args.run_name}, current_eval_arena.txt={ARENA}")
    print("Completion stop: disabled")
    for index, sample in enumerate(samples):
        port = args.base_port + index
        world, webots_cmd, controller_cmd = build_commands(args, sample, port)
        env_prefix = (
            f"__NV_PRIME_RENDER_OFFLOAD=1 "
            f"__GLX_VENDOR_LIBRARY_NAME=nvidia "
            f"WEBOTS_PORT={port}"
        )
        print(f"\nSample {sample} | port {port} | world {world}")
        print(f"{env_prefix} {shlex.join(webots_cmd)}")
        print(
            f"WEBOTS_CONTROLLER_URL=<from --extern-urls> "
            f"WEBOTS_PORT={port} {shlex.join(controller_cmd)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel 5x5 decentralized fixed-time foraging experiments."
    )
    parser.add_argument(
        "--distribution",
        required=True,
        choices=sorted(completion.WORLD_PREFIXES),
    )
    parser.add_argument(
        "--samples",
        type=completion.parse_sample_range,
        default=completion.parse_sample_range("1-10"),
        help="Sample ids to run, e.g. 1-10 or 1,3,7. Default: 1-10.",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help="Per-robot run name, e.g. decentralized_indep_v9.",
    )
    parser.add_argument(
        "--foraging-time-min",
        "--duration-sim-min",
        "--duration",
        dest="foraging_time_min",
        type=float,
        default=DEFAULT_FORAGING_TIME_MIN,
        help="Fixed foraging horizon in simulated minutes. Default: 30.",
    )
    parser.add_argument("--base-port", type=int, default=1438)
    parser.add_argument("--max-parallel", type=int, default=None)
    parser.add_argument("--results-csv", type=Path, default=None)
    parser.add_argument("--webots-bin", default="webots")
    parser.add_argument("--startup-seconds", type=float, default=30.0)
    parser.add_argument(
        "--controller-start-retries",
        type=int,
        default=3,
        help="Retry extern controller startup if it exits before model load.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.foraging_time_min <= 0:
        parser.error("--foraging-time-min must be greater than 0")

    if args.max_parallel is None:
        args.max_parallel = len(args.samples)
    if args.max_parallel <= 0:
        parser.error("--max-parallel must be greater than 0")

    missing_models = completion.missing_run_models(args.run_name)
    if missing_models:
        models = ", ".join(str(path) for path in missing_models)
        parser.error(f"missing per-robot model file(s): {models}")

    results_csv = resolve_results_csv(
        args.results_csv,
        args.run_name,
        args.distribution,
        args.foraging_time_min,
    )

    if args.dry_run:
        print_dry_run(args, args.samples, results_csv)
        return 0

    completion.write_eval_config(args.run_name)

    run_dir = results_csv.parent / f"{results_csv.stem}_logs"
    logs_dir = run_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Method: {METHOD}")
    print(f"Experiment: {EXPERIMENT}")
    print(f"Run name: {args.run_name}")
    print(f"Distribution: {args.distribution}")
    print(f"Arena: {ARENA}")
    print(f"Samples: {args.samples}")
    print(f"Foraging time: {completion.format_float(args.foraging_time_min)} sim-min")
    print("Completion stop: disabled")
    print(f"Results CSV: {results_csv}")
    print(f"Logs: {logs_dir}")
    print(f"Parallel jobs: {min(args.max_parallel, len(args.samples))}")

    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=min(args.max_parallel, len(args.samples))) as pool:
        future_to_sample = {}
        for index, sample in enumerate(args.samples):
            port = args.base_port + index
            future = pool.submit(run_sample, args, sample, port, logs_dir)
            future_to_sample[future] = sample

        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            try:
                row = future.result()
                rows.append(row)
                print(
                    f"[sample {sample}] {row['status']} "
                    f"sim={row['elapsed_sim_min']} min "
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
