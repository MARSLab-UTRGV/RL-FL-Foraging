import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parent

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

DISTRIBUTIONS = [
    ("clustered", "Clustered"),
    ("powerlaw", "Power-law"),
    ("random", "Random"),
]

MODELS = [
    (
        "CPFA Baseline",
        "#D55E00",
        "completion_time_min",
        {
            "clustered": "foraging_completion_cpfa_baseline_clustered_5x5_20260708_123650.csv",
            "powerlaw": "foraging_completion_cpfa_baseline_powerlaw_5x5_20260708_125708.csv",
            "random": "foraging_completion_cpfa_baseline_random_5x5_20260708_134027.csv",
        },
    ),
    (
        "Centralized RL",
        "#56B4E9",
        "completion_time_min",
        {
            "clustered": "foraging_completion_centralized_ppo_clustered_5x5_20260707_153249.csv",
            "powerlaw": "foraging_completion_centralized_ppo_powerlaw_5x5_20260707_154207.csv",
            "random": "foraging_completion_centralized_ppo_random_5x5_20260707_160521.csv",
        },
    ),
    (
        "Decentralized RL",
        "#B3DE69",
        "sim_time_min",
        {
            "clustered": "batch_results_5x5_t64_clustered_complete_decentralized_indep_v10.csv",
            "powerlaw": "batch_results_5x5_t64_powerlaw_complete_decentralized_indep_v10.csv",
            "random": "batch_results_5x5_t64_random_complete_decentralized_indep_v10.csv",
        },
    ),
]


def read_completion_times(csv_name: str, column_name: str) -> list[float]:
    frame = pd.read_csv(ROOT / csv_name)
    if column_name not in frame.columns:
        raise ValueError(f"{csv_name} does not contain a {column_name} column")

    times = frame[column_name].dropna().tolist()
    if len(times) != 10:
        raise ValueError(f"{csv_name} has {len(times)} samples, expected 10")

    if "deposits" in frame.columns and not (frame["deposits"] == 64).all():
        raise ValueError(f"{csv_name} contains rows that did not collect all 64 tags")

    if "completed" in frame.columns and not frame["completed"].astype(bool).all():
        raise ValueError(f"{csv_name} contains incomplete runs")

    return times


def main() -> None:
    data = []
    positions = []
    colors = []

    group_centers = [1.0, 3.2, 5.4]
    offsets = [-0.46, 0.0, 0.46]

    for distribution_index, (distribution_key, _) in enumerate(DISTRIBUTIONS):
        center = group_centers[distribution_index]
        for model_index, (_, color, column_name, files_by_distribution) in enumerate(MODELS):
            data.append(
                read_completion_times(
                    files_by_distribution[distribution_key],
                    column_name,
                )
            )
            positions.append(center + offsets[model_index])
            colors.append(color)

    fig, ax = plt.subplots(figsize=(7.16, 4.2), dpi=600)

    plot = ax.boxplot(
        data,
        positions=positions,
        widths=0.30,
        patch_artist=True,
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "white",
            "markeredgecolor": "#333333",
            "markersize": 2.5,
            "markeredgewidth": 0.7,
        },
        medianprops={"color": "#222222", "linewidth": 1.0},
        whiskerprops={"color": "#222222", "linewidth": 0.8},
        capprops={"color": "#222222", "linewidth": 0.8},
        flierprops={
            "marker": "x",
            "markeredgecolor": "#333333",
            "markerfacecolor": "#333333",
            "markersize": 2.8,
            "markeredgewidth": 0.7,
        },
    )

    for patch, color in zip(plot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#222222")
        patch.set_linewidth(0.8)

    for flier, color in zip(plot["fliers"], colors):
        flier.set_markeredgecolor(color)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([label for _, label in DISTRIBUTIONS])
    ax.set_xlabel("Resource distribution")
    ax.set_ylabel("Time to collect all resources (min)")

    ax.yaxis.grid(True, linestyle="-", color="#d9d9d9", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.set_xlim(group_centers[0] - 0.8, group_centers[-1] + 0.8)

    max_value = max(max(group) for group in data)
    upper_limit = ((int(max_value) // 10) + 2) * 10
    ax.set_ylim(0, upper_limit)
    ax.set_yticks(range(0, upper_limit + 1, 10))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    legend_handles = [
        Patch(facecolor=color, edgecolor="#222222", label=name)
        for name, color, _, _ in MODELS
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.6,
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(ROOT / "exp3_completion_time_boxplot.png", bbox_inches="tight", dpi=600)
    fig.savefig(ROOT / "exp3_completion_time_boxplot.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "exp3_completion_time_boxplot.eps", bbox_inches="tight")


if __name__ == "__main__":
    main()
