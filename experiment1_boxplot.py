#!/usr/bin/env python3
"""Plot grouped deposit box plots for experiment 1 foraging results."""

import argparse
import csv
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache")
)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


METHODS = ["cpfa_baseline", "centralized_ppo", "decentralized_ppo"]

METHOD_LABELS = {
    "cpfa_baseline": "Baseline",
    "centralized_ppo": "Centralized PPO",
    "decentralized_ppo": "Decentralized PPO",
}

METHOD_COLORS = {
    "cpfa_baseline": "#d62728",  # red
    "centralized_ppo": "#1f77b4",  # blue
    "decentralized_ppo": "#2ca02c",  # green
}


def arena_sort_key(arena):
    match = re.match(r"^(\d+)x(\d+)$", arena)
    if match:
        return int(match.group(1)), int(match.group(2))
    return arena


def format_time(minutes):
    value = float(minutes)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def load_deposits(input_dir):
    data = defaultdict(list)
    times_by_arena = {}
    csv_paths = sorted(input_dir.glob("foraging_*.csv"))

    if not csv_paths:
        raise FileNotFoundError(f"No foraging_*.csv files found in {input_dir}")

    for csv_path in csv_paths:
        with csv_path.open(newline="") as file:
            reader = csv.DictReader(file)
            required = {"method", "arena", "foraging_time_min", "deposits"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{csv_path} is missing required columns: {sorted(missing)}"
                )

            for row in reader:
                method = row["method"].strip()
                if method not in METHODS:
                    continue

                arena = row["arena"].strip()
                time_min = format_time(row["foraging_time_min"])
                deposit_count = float(row["deposits"])

                data[(arena, method)].append(deposit_count)
                times_by_arena.setdefault(arena, time_min)

    return data, times_by_arena


def plot_boxplots(data, times_by_arena, output_path):
    arenas = sorted(times_by_arena, key=arena_sort_key)
    if not arenas:
        raise ValueError("No experiment 1 rows found for the configured methods.")

    missing = [
        (arena, method)
        for arena in arenas
        for method in METHODS
        if not data.get((arena, method))
    ]
    if missing:
        missing_text = ", ".join(
            f"{arena} / {METHOD_LABELS[method]}" for arena, method in missing
        )
        raise ValueError(f"No deposit data found for: {missing_text}")

    fig, ax = plt.subplots(figsize=(11, 6))

    group_centers = list(range(1, len(arenas) + 1))
    offsets = [-0.25, 0.0, 0.25]
    width = 0.2

    for method_index, method in enumerate(METHODS):
        series = [data[(arena, method)] for arena in arenas]
        positions = [
            center + offsets[method_index]
            for center in group_centers
        ]

        box = ax.boxplot(
            series,
            positions=positions,
            widths=width,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": 4,
            },
            medianprops={"color": "black", "linewidth": 1.3},
            boxprops={"linewidth": 1.1},
            whiskerprops={"linewidth": 1.1},
            capprops={"linewidth": 1.1},
            flierprops={
                "marker": "x",
                "markeredgecolor": METHOD_COLORS[method],
                "markersize": 5,
            },
        )

        for patch in box["boxes"]:
            patch.set_facecolor(METHOD_COLORS[method])
            patch.set_alpha(0.65)

    ax.set_title("Experiment 1 Deposits by Arena and Method")
    ax.set_xlabel("Arena Size and Foraging Time")
    ax.set_ylabel("Deposits")
    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [f"({arena}, {times_by_arena[arena]} min)" for arena in arenas]
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    handles = [
        Patch(facecolor=METHOD_COLORS[method], alpha=0.65, label=METHOD_LABELS[method])
        for method in METHODS
    ]
    ax.legend(handles=handles, title="Method")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create side-by-side deposit box plots for experiment 1."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/experiment1"),
        help="Directory containing experiment 1 foraging CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/experiment1/experiment1_deposits_boxplot.png"),
        help="Output image path.",
    )
    args = parser.parse_args()

    data, times_by_arena = load_deposits(args.input_dir)
    plot_boxplots(data, times_by_arena, args.output)


if __name__ == "__main__":
    main()
