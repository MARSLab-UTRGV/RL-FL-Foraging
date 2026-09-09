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

SWARMS = [
    ("4 robots", "15 min"),
    ("8 robots", "15 min"),
    ("12 robots", "15 min"),
    ("16 robots", "15 min"),
]

MODELS = [
    (
        "CPFA Baseline",
        "#D55E00",
        {
            "4 robots": "exp2_cpfa_baseline_7x7_4r_20260811_120422.csv",
            "8 robots": "exp2_cpfa_baseline_7x7_8r_20260811_121426.csv",
            "12 robots": "exp2_cpfa_baseline_7x7_12r_20260811_124333.csv",
            "16 robots": "exp2_cpfa_baseline_7x7_16r_20260811_132418.csv",
        },
    ),
    (
        "Centralized RL",
        "#56B4E9",
        {
            "4 robots": "exp2_centralized_ppo_7x7_4r_20260810_145305.csv",
            "8 robots": "exp2_centralized_ppo_7x7_8r_20260810_150637.csv",
            "12 robots": "exp2_centralized_ppo_7x7_12r_20260810_153317.csv",
            "16 robots": "exp2_centralized_ppo_7x7_16r_20260811_025602.csv",
        },
    ),
    (
        "Decentralized RL",
        "#B3DE69",
        {
            "4 robots": "batch_results_7x7_t64_decentralized_indep_v10.csv",
            "8 robots": "batch_results_7x7_8r_t128_decentralized_indep_v10.csv",
            "12 robots": "batch_results_7x7_12r_t192_decentralized_indep_v10.csv",
            "16 robots": "batch_results_7x7_16r_t256_decentralized_indep_v10.csv",
        },
    ),
]


def read_deposits(csv_name: str) -> list[float]:
    frame = pd.read_csv(ROOT / csv_name)
    if "deposits" not in frame.columns:
        raise ValueError(f"{csv_name} does not contain a deposits column")

    deposits = frame["deposits"].dropna().tolist()
    if len(deposits) != 20:
        raise ValueError(f"{csv_name} has {len(deposits)} deposit samples, expected 20")

    return deposits


def main() -> None:
    data = []
    positions = []
    colors = []

    group_centers = [1.0, 2.15, 3.30, 4.45]
    offsets = [-0.30, 0.0, 0.30]

    for swarm_index, (swarm, _) in enumerate(SWARMS):
        center = group_centers[swarm_index]
        for model_index, (_, color, files_by_swarm) in enumerate(MODELS):
            data.append(read_deposits(files_by_swarm[swarm]))
            positions.append(center + offsets[model_index])
            colors.append(color)

    fig, ax = plt.subplots(figsize=(7.16, 4.2), dpi=600)

    plot = ax.boxplot(
        data,
        positions=positions,
        widths=0.24,
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
    ax.set_xticklabels([f"{swarm}, {minutes}" for swarm, minutes in SWARMS])
    ax.set_xlabel("Swarm size and foraging time")
    ax.set_ylabel("Resources collected")

    ax.yaxis.grid(True, linestyle="-", color="#d9d9d9", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.set_xlim(group_centers[0] - 0.45, group_centers[-1] + 0.45)

    ax.set_ylim(0, 300)
    ax.set_yticks(range(0, 301, 50))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    legend_handles = [
        Patch(facecolor=color, edgecolor="#222222", label=name)
        for name, color, _ in MODELS
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=1,
        frameon=False,
        handlelength=1.5,
        borderpad=0.0,
        labelspacing=0.25,
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(ROOT / "exp2_deposits_boxplot_standard.png", bbox_inches="tight", dpi=600)
    fig.savefig(ROOT / "exp2_deposits_boxplot_standard.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "exp2_deposits_boxplot_standard.eps", bbox_inches="tight")


if __name__ == "__main__":
    main()
