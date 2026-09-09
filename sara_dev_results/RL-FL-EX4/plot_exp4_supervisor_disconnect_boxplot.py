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

CONDITIONS = [
    (
        "Decentralized RL\nreference",
        "Decentralized RL",
        "#B3DE69",
        "batch_results_5x5_decentralized_indep_v10.csv",
    ),
    (
        "Centralized RL\n1 min disconnect",
        "Centralized RL",
        "#56B4E9",
        "exp4_disconnect_1min_5x5.csv",
    ),
    (
        "Decentralized RL\nreference",
        "Decentralized RL",
        "#B3DE69",
        "batch_results_5x5_decentralized_indep_v10.csv",
    ),
    (
        "Centralized RL\n2 min disconnect",
        "Centralized RL",
        "#56B4E9",
        "exp4_disconnect_2min_5x5.csv",
    ),
    (
        "Decentralized RL\nreference",
        "Decentralized RL",
        "#B3DE69",
        "batch_results_5x5_decentralized_indep_v10.csv",
    ),
    (
        "Centralized RL\n3 min disconnect",
        "Centralized RL",
        "#56B4E9",
        "exp4_disconnect_3min_5x5.csv",
    ),
    (
        "Decentralized RL\nreference",
        "Decentralized RL",
        "#B3DE69",
        "batch_results_5x5_decentralized_indep_v10.csv",
    ),
    (
        "Centralized RL\n5 min disconnect",
        "Centralized RL",
        "#56B4E9",
        "exp4_disconnect_5min_5x5.csv",
    ),
    (
        "Decentralized RL\nreference",
        "Decentralized RL",
        "#B3DE69",
        "batch_results_5x5_decentralized_indep_v10.csv",
    ),
    (
        "Centralized RL\n10 min disconnect",
        "Centralized RL",
        "#56B4E9",
        "exp4_disconnect_10min_5x5.csv",
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
    colors = []
    labels = []

    for label, _, color, csv_name in CONDITIONS:
        data.append(read_deposits(csv_name))
        colors.append(color)
        labels.append(label)

    positions = [0.80, 1.20, 2.30, 2.70, 3.80, 4.20, 5.30, 5.70, 6.80, 7.20]
    group_centers = [1.0, 2.5, 4.0, 5.5, 7.0]
    fig, ax = plt.subplots(figsize=(7.16, 4.2), dpi=600)

    plot = ax.boxplot(
        data,
        positions=positions,
        widths=0.42,
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
    ax.set_xticklabels(["1 min", "2 min", "3 min", "5 min", "10 min"])
    ax.set_xlabel("Supervisor disconnection duration")
    ax.set_ylabel("Resources collected")

    ax.yaxis.grid(True, linestyle="-", color="#d9d9d9", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.set_xlim(0.45, 7.55)

    max_value = max(max(group) for group in data)
    upper_limit = ((int(max_value) // 10) + 2) * 10
    ax.set_ylim(0, upper_limit)
    ax.set_yticks(range(0, upper_limit + 1, 10))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    legend_handles = [
        Patch(facecolor="#B3DE69", edgecolor="#222222", label="Decentralized RL"),
        Patch(facecolor="#56B4E9", edgecolor="#222222", label="Centralized RL"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.8,
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(ROOT / "exp4_supervisor_disconnect_boxplot.png", bbox_inches="tight", dpi=600)
    fig.savefig(ROOT / "exp4_supervisor_disconnect_boxplot.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "exp4_supervisor_disconnect_boxplot.eps", bbox_inches="tight")


if __name__ == "__main__":
    main()
