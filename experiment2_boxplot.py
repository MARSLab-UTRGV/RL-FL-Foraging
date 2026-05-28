#!/usr/bin/env python3
"""Plot grouped deposit box plots for experiment 2 multirobot results."""

import argparse
import csv
import os
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
    "cpfa_baseline": "CPFA Baseline",
    "centralized_ppo": "Centralized PPO",
    "decentralized_ppo": "Decentralized PPO",
}

METHOD_COLORS = {
    "cpfa_baseline": "#d62728",  # red
    "centralized_ppo": "#1f77b4",  # blue
    "decentralized_ppo": "#2ca02c",  # green
}


def numeric_value(value):
    return int(float(value))


def load_deposits(input_dir):
    data = defaultdict(list)
    groups = set()
    csv_paths = sorted(input_dir.glob("multirobot_*_10min.csv"))

    if not csv_paths:
        raise FileNotFoundError(f"No multirobot_*_10min.csv files found in {input_dir}")

    for csv_path in csv_paths:
        with csv_path.open(newline="") as file:
            reader = csv.DictReader(file)
            required = {"method", "robots", "active_tags", "deposits"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{csv_path} is missing required columns: {sorted(missing)}"
                )

            for row in reader:
                method = row["method"].strip()
                if method not in METHODS:
                    continue

                robots = numeric_value(row["robots"])
                active_tags = numeric_value(row["active_tags"])
                deposits = float(row["deposits"])
                group = (robots, active_tags)

                groups.add(group)
                data[(group, method)].append(deposits)

    return data, sorted(groups)


def plot_boxplots(data, groups, output_path):
    if not groups:
        raise ValueError("No experiment 2 rows found for the configured methods.")

    missing = [
        (group, method)
        for group in groups
        for method in METHODS
        if not data.get((group, method))
    ]
    if missing:
        missing_text = ", ".join(
            f"{robots} robots / {active_tags} tags / {METHOD_LABELS[method]}"
            for (robots, active_tags), method in missing
        )
        raise ValueError(f"No deposit data found for: {missing_text}")

    fig, ax = plt.subplots(figsize=(11, 6))

    group_centers = list(range(1, len(groups) + 1))
    offsets = [-0.25, 0.0, 0.25]
    width = 0.2

    for method_index, method in enumerate(METHODS):
        series = [data[(group, method)] for group in groups]
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

    ax.set_title("Experiment 2", fontsize=16)
    ax.set_xlabel("Robot Count and Number of Resources", fontsize=13)
    ax.set_ylabel("Deposits", fontsize=13)
    ax.set_xticks(group_centers)
    ax.set_xticklabels(
        [f"{robots} robots\n{active_tags} resources" for robots, active_tags in groups]
    )
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    handles = [
        Patch(facecolor=METHOD_COLORS[method], alpha=0.65, label=METHOD_LABELS[method])
        for method in METHODS
    ]
    ax.legend(handles=handles)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create side-by-side deposit box plots for experiment 2."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/experiment2"),
        help="Directory containing experiment 2 multirobot CSV files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/experiment2/experiment2_deposits_boxplot.png"),
        help="Output image path.",
    )
    args = parser.parse_args()

    data, groups = load_deposits(args.input_dir)
    plot_boxplots(data, groups, args.output)


if __name__ == "__main__":
    main()
