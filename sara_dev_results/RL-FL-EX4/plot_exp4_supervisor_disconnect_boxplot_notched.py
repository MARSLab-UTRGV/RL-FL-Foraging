import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import ttest_ind


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

DECENTRALIZED_FILE = "batch_results_5x5_decentralized_indep_v10.csv"
CONDITIONS = [
    ("1 min", DECENTRALIZED_FILE, "exp4_disconnect_1min_5x5.csv"),
    ("2 min", DECENTRALIZED_FILE, "exp4_disconnect_2min_5x5.csv"),
    ("3 min", DECENTRALIZED_FILE, "exp4_disconnect_3min_5x5.csv"),
    ("5 min", DECENTRALIZED_FILE, "exp4_disconnect_5min_5x5.csv"),
    ("10 min", DECENTRALIZED_FILE, "exp4_disconnect_10min_5x5.csv"),
]

MODEL_COLORS = {
    "Decentralized RL": "#B3DE69",
    "Centralized RL": "#56B4E9",
}


def read_deposits(csv_name: str) -> list[float]:
    frame = pd.read_csv(ROOT / csv_name)
    if "deposits" not in frame.columns:
        raise ValueError(f"{csv_name} does not contain a deposits column")

    deposits = frame["deposits"].dropna().tolist()
    if len(deposits) != 20:
        raise ValueError(f"{csv_name} has {len(deposits)} deposit samples, expected 20")

    return deposits


def format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def add_pvalue_bracket(ax, x1: float, x2: float, y: float, p_value: float) -> None:
    bracket_height = 1.2
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + bracket_height, y + bracket_height, y],
        color="#222222",
        linewidth=0.65,
        clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2,
        y + bracket_height + 0.5,
        format_p_value(p_value),
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#222222",
        clip_on=False,
    )


def main() -> None:
    data = []
    colors = []
    positions = []

    group_centers = [1.0, 2.15, 3.30, 4.45, 5.60]
    offsets = [-0.18, 0.18]

    for group_index, (_, dec_file, cen_file) in enumerate(CONDITIONS):
        center = group_centers[group_index]
        data.append(read_deposits(dec_file))
        colors.append(MODEL_COLORS["Decentralized RL"])
        positions.append(center + offsets[0])

        data.append(read_deposits(cen_file))
        colors.append(MODEL_COLORS["Centralized RL"])
        positions.append(center + offsets[1])

    fig, ax = plt.subplots(figsize=(7.16, 4.2), dpi=600)

    plot = ax.boxplot(
        data,
        positions=positions,
        widths=0.28,
        notch=True,
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

    for group_index, _ in enumerate(CONDITIONS):
        left = group_index * 2
        right = left + 1
        local_max = max(max(data[left]), max(data[right]))
        p_value = ttest_ind(data[left], data[right], equal_var=False).pvalue
        add_pvalue_bracket(ax, positions[left], positions[right], local_max + 4.0, p_value)

    ax.set_xticks(group_centers)
    ax.set_xticklabels([duration for duration, _, _ in CONDITIONS])
    ax.set_xlabel("Supervisor disconnection duration")
    ax.set_ylabel("Resources collected")

    ax.yaxis.grid(True, linestyle="-", color="#d9d9d9", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.set_xlim(group_centers[0] - 0.45, group_centers[-1] + 0.45)

    max_value = max(max(group) for group in data)
    upper_limit = max(((int(max_value + 25) // 10) + 1) * 10, 90)
    ax.set_ylim(0, upper_limit)
    ax.set_yticks(range(0, upper_limit + 1, 10))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    legend_handles = [
        Patch(facecolor=MODEL_COLORS["Decentralized RL"], edgecolor="#222222", label="Decentralized RL"),
        Patch(facecolor=MODEL_COLORS["Centralized RL"], edgecolor="#222222", label="Centralized RL"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        ncol=1,
        frameon=True,
        framealpha=1.0,
        edgecolor="#cccccc",
        handlelength=1.5,
        borderpad=0.35,
        labelspacing=0.35,
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(ROOT / "exp4_supervisor_disconnect_boxplot_notched.png", bbox_inches="tight", dpi=600)
    fig.savefig(ROOT / "exp4_supervisor_disconnect_boxplot_notched.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "exp4_supervisor_disconnect_boxplot_notched.eps", bbox_inches="tight")


if __name__ == "__main__":
    main()
