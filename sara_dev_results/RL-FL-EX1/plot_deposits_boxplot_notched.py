import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_ind
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

ARENAS = [
    ("5×5", "10 min"),
    ("7×7", "25 min"),
    ("9×9", "50 min"),
    ("12×12", "95 min"),
]

MODELS = [
    (
        "CPFA Baseline",
        "#D55E00",
        "",
        {
            "5×5": "foraging_cpfa_baseline_5x5_20260706_151602.csv",
            "7×7": "foraging_cpfa_baseline_7x7_20260706_154946.csv",
            "9×9": "foraging_cpfa_baseline_9x9_20260706_173344.csv",
            "12×12": "foraging_cpfa_baseline_12x12_20260706_192528.csv",
        },
    ),
    (
        "Centralized RL",
        "#56B4E9",
        "",
        {
            "5×5": "foraging_centralized_ppo_5x5_20260704_114002.csv",
            "7×7": "foraging_centralized_ppo_7x7_20260704_121112.csv",
            "9×9": "foraging_centralized_ppo_9x9_20260704_123938.csv",
            "12×12": "foraging_centralized_ppo_12x12_20260704_131920.csv",
        },
    ),
    (
        "Decentralized RL",
        "#B3DE69",
        "",
        {
            "5×5": "batch_results_5x5_decentralized_indep_v10.csv",
            "7×7": "batch_results_7x7_t128_decentralized_indep_v10.csv",
            "9×9": "batch_results_9x9_t208_decentralized_indep_v10.csv",
            "12×12": "batch_results_12x12_t368_decentralized_indep_v10.csv",
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


def format_p_value(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def add_pvalue_bracket(ax, x1: float, x2: float, y: float, p_value: float, bracket_height: float) -> None:
    ax.plot(
        [x1, x1, x2, x2],
        [y, y + bracket_height, y + bracket_height, y],
        color="#222222",
        linewidth=0.65,
        clip_on=False,
    )
    ax.text(
        (x1 + x2) / 2,
        y + bracket_height + bracket_height * 0.5,
        format_p_value(p_value),
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#222222",
        clip_on=False,
    )


def main() -> None:
    data = []
    positions = []
    colors = []
    hatches = []

    group_centers = [1.0, 2.15, 3.30, 4.45]
    offsets = [-0.30, 0.0, 0.30]

    for arena_index, (arena, _) in enumerate(ARENAS):
        center = group_centers[arena_index]
        for model_index, (_, color, hatch, files_by_arena) in enumerate(MODELS):
            data.append(read_deposits(files_by_arena[arena]))
            positions.append(center + offsets[model_index])
            colors.append(color)
            hatches.append(hatch)

    fig, ax = plt.subplots(figsize=(7.16, 4.2), dpi=600)

    plot = ax.boxplot(
        data,
        positions=positions,
        widths=0.24,
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

    for patch, color, hatch in zip(plot["boxes"], colors, hatches):
        patch.set_facecolor(color)
        patch.set_edgecolor("#222222")
        patch.set_linewidth(0.8)
        patch.set_hatch(hatch)

    for flier, color in zip(plot["fliers"], colors):
        flier.set_markeredgecolor(color)

    for group_index in range(len(ARENAS)):
        base = group_index * len(MODELS)
        group_values = data[base : base + len(MODELS)]
        group_positions = positions[base : base + len(MODELS)]
        local_max = max(max(values) for values in group_values)
        y_start = min(local_max + 18, 340)
        comparisons = [(0, 2), (1, 2)]
        for comparison_index, (left, right) in enumerate(comparisons):
            p_value = ttest_ind(
                group_values[left],
                group_values[right],
                equal_var=False,
            ).pvalue
            add_pvalue_bracket(
                ax,
                group_positions[left],
                group_positions[right],
                y_start + comparison_index * 12,
                p_value,
                bracket_height=2.0,
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels([f"{arena}, {minutes}" for arena, minutes in ARENAS])
    ax.set_xlabel("Arena size and foraging time")
    ax.set_ylabel("Resources collected")

    ax.yaxis.grid(True, linestyle="-", color="#d9d9d9", linewidth=0.45)
    ax.set_axisbelow(True)
    ax.set_xlim(group_centers[0] - 0.45, group_centers[-1] + 0.45)
    ax.set_ylim(0, 370)
    ax.set_yticks(range(0, 351, 50))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    legend_handles = [
        Patch(facecolor=color, edgecolor="#222222", hatch=hatch, label=name)
        for name, color, hatch, _ in MODELS
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.94),
        ncol=1,
        frameon=True,
        framealpha=1.0,
        edgecolor="#cccccc",
        handlelength=1.5,
        borderpad=0.35,
        labelspacing=0.35,
    )

    fig.tight_layout(pad=0.4)
    fig.savefig(ROOT / "deposits_boxplot_notched.png", bbox_inches="tight", dpi=600)
    fig.savefig(ROOT / "deposits_boxplot_notched.pdf", bbox_inches="tight")
    fig.savefig(ROOT / "deposits_boxplot_notched.eps", bbox_inches="tight")


if __name__ == "__main__":
    main()
