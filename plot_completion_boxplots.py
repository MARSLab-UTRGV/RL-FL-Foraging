#!/usr/bin/env python3
"""Plot grouped box plots for foraging completion times."""

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


DISTRIBUTIONS = ["powerlaw", "clustered", "random"]
METHODS = ["cpfa_baseline", "centralized_ppo", "decentralized"]

DISTRIBUTION_LABELS = {
    "clustered": "Clustered",
    "random": "Random",
    "powerlaw": "Powerlaw",
}

METHOD_LABELS = {
    "cpfa_baseline": "CPFA Baseline",
    "centralized_ppo": "Centralized PPO",
    "decentralized": "Decentralized PPO",
}

METHOD_COLORS = {
    "cpfa_baseline": "#d62728",  # red
    "centralized_ppo": "#1f77b4",  # blue
    "decentralized": "#2ca02c",  # green
}

METHOD_ALIASES = {
    "cpfa": "cpfa_baseline",
    "cpfa_baseline": "cpfa_baseline",
    "centralized": "centralized_ppo",
    "centralized_ppo": "centralized_ppo",
    "decentralized": "decentralized",
    "decentralized_ppo": "decentralized",
    "decentralized_indep_v9": "decentralized",
}

DISTRIBUTION_ALIASES = {
    "clustered": "clustered",
    "clustured": "clustered",
    "random": "random",
    "powerlaw": "powerlaw",
    "power_law": "powerlaw",
    "power-law": "powerlaw",
}


def normalize(value, aliases):
    return aliases.get(value.strip().lower(), value.strip().lower())


def load_completion_times(input_dir, time_column, include_incomplete):
    data = defaultdict(list)
    csv_paths = sorted(input_dir.glob("foraging_completion_*.csv"))

    if not csv_paths:
        raise FileNotFoundError(f"No foraging_completion_*.csv files found in {input_dir}")

    for csv_path in csv_paths:
        with csv_path.open(newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if not include_incomplete and row.get("completed", "").lower() != "true":
                    continue

                method = normalize(row.get("method", ""), METHOD_ALIASES)
                distribution = normalize(row.get("distribution", ""), DISTRIBUTION_ALIASES)

                if method not in METHODS or distribution not in DISTRIBUTIONS:
                    continue

                raw_time = row.get(time_column, "")
                if raw_time == "":
                    continue

                data[(distribution, method)].append(float(raw_time))

    return data


def plot_boxplots(data, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    group_centers = list(range(1, len(DISTRIBUTIONS) + 1))
    offsets = [-0.25, 0.0, 0.25]
    width = 0.2

    for method_index, method in enumerate(METHODS):
        series = [data.get((distribution, method), []) for distribution in DISTRIBUTIONS]
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

    ax.set_title("Foraging Completion Time by Distribution and Method")
    ax.set_xlabel("Food Distribution")
    ax.set_ylabel("Completion Time (minutes)")
    ax.set_xticks(group_centers)
    ax.set_xticklabels([DISTRIBUTION_LABELS[item] for item in DISTRIBUTIONS])
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
        description="Create side-by-side box plots from foraging completion CSVs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/completion_results"),
        help="Directory containing foraging_completion_*.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/completion_results/completion_time_boxplot.png"),
        help="Output image path.",
    )
    parser.add_argument(
        "--time-column",
        default="completion_time_min",
        help="CSV column to plot on the y-axis.",
    )
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Include rows where completed is not true.",
    )
    args = parser.parse_args()

    data = load_completion_times(
        args.input_dir,
        time_column=args.time_column,
        include_incomplete=args.include_incomplete,
    )

    missing = [
        (distribution, method)
        for distribution in DISTRIBUTIONS
        for method in METHODS
        if not data.get((distribution, method))
    ]
    if missing:
        missing_text = ", ".join(
            f"{DISTRIBUTION_LABELS[distribution]} / {METHOD_LABELS[method]}"
            for distribution, method in missing
        )
        raise ValueError(f"No data found for: {missing_text}")

    plot_boxplots(data, args.output)


if __name__ == "__main__":
    main()
