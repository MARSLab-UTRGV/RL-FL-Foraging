import os
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np

# ── Style (matches EX1–EX4 boxplots) ────────────────────────────────────────
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

ROOT = Path(__file__).resolve().parent
STEPS_PER_EP = 16384
EP_DUR_MIN = (STEPS_PER_EP * 32 / 1000) / 60  # 8.738 min per episode

COLOR_DECEN = "#B3DE69"
COLOR_DECEN_DARK = "#6aaa1f"   # darker shade for raw faint line
COLOR_CNTRL = "#56B4E9"
COLOR_CNTRL_DARK = "#1a7ab8"
COLOR_THRESHOLD = "#888888"
COLOR_ANNOT = "#333333"

SMOOTH_WINDOW = 15

# ── Parse logs ───────────────────────────────────────────────────────────────
DECEN_LOG_DIR = Path("/home/sara/Documents/Centralized Learning/RL-FL-Foraging/logs")
DECEN_RUN = "decentralized_indep_v11"
CNTRL_LOG = Path(
    "/home/sara/Downloads/RL-FL-Foraging-sara_dev_centralizedRL"
    "/logs/ppo_cpfa_v6/training_log.txt"
)

EP_RE_DECEN = re.compile(
    r"\[EP (\d+)\] robot\d \| Picks: \d+ \| Deps: \d+ \| Rate: ([\d.]+) tags/min"
)
EP_RE_CNTRL = re.compile(
    r"\[EP (\d+)\] Picks: \d+ \| Deps: \d+ \| Rate: ([\d.]+) tags/min"
)


def parse_decen() -> list[tuple[int, float]]:
    robot_eps: dict[int, dict[int, float]] = {r: {} for r in range(1, 5)}
    for r in range(1, 5):
        path = DECEN_LOG_DIR / f"robot{r}_{DECEN_RUN}" / "training_log.txt"
        with open(path) as f:
            for line in f:
                m = EP_RE_DECEN.search(line)
                if m:
                    robot_eps[r][int(m.group(1))] = float(m.group(2))
    common = sorted(
        set(robot_eps[1]) & set(robot_eps[2]) & set(robot_eps[3]) & set(robot_eps[4])
    )
    return [(ep, sum(robot_eps[r].get(ep, 0.0) for r in range(1, 5))) for ep in common]


def parse_cntrl() -> list[tuple[int, float]]:
    eps = {}
    with open(CNTRL_LOG) as f:
        for line in f:
            m = EP_RE_CNTRL.search(line)
            if m:
                eps[int(m.group(1))] = float(m.group(2))
    return sorted(eps.items())


def rolling_avg(values: list[float], window: int) -> list[float]:
    out = []
    for i, v in enumerate(values):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


def find_crossover(eps, smoothed, threshold=2.0):
    for ep, val in zip(eps, smoothed):
        if val >= threshold:
            return ep
    return None


# ── Build data series ────────────────────────────────────────────────────────
decen_raw = parse_decen()
cntrl_raw = parse_cntrl()

decen_eps = [ep for ep, _ in decen_raw]
decen_rates = [r for _, r in decen_raw]
decen_steps = [ep * STEPS_PER_EP / 1e6 for ep in decen_eps]  # in millions
decen_smooth = rolling_avg(decen_rates, SMOOTH_WINDOW)

cntrl_eps = [ep for ep, _ in cntrl_raw]
cntrl_rates = [r for _, r in cntrl_raw]
cntrl_steps = [ep * STEPS_PER_EP / 1e6 for ep in cntrl_eps]
cntrl_smooth = rolling_avg(cntrl_rates, SMOOTH_WINDOW)

decen_cross_ep = find_crossover(decen_eps, decen_smooth)
cntrl_cross_ep = find_crossover(cntrl_eps, cntrl_smooth)
decen_cross_Msteps = decen_cross_ep * STEPS_PER_EP / 1e6
cntrl_cross_Msteps = cntrl_cross_ep * STEPS_PER_EP / 1e6
speedup = cntrl_cross_Msteps / decen_cross_Msteps

decen_final = sum(decen_smooth[-10:]) / 10
cntrl_final = sum(cntrl_smooth[-10:]) / 10

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7.16, 4.0), dpi=600)

# Raw faint background traces
ax.plot(
    decen_steps, decen_rates,
    color=COLOR_DECEN_DARK, alpha=0.35, linewidth=0.6, zorder=1,
)
ax.plot(
    cntrl_steps, cntrl_rates,
    color=COLOR_CNTRL_DARK, alpha=0.25, linewidth=0.6, zorder=1,
)

# Smoothed main lines
ax.plot(
    decen_steps, decen_smooth,
    color=COLOR_DECEN_DARK, linewidth=1.5, zorder=3, solid_capstyle="round",
    label="Decentralized RL (15-ep avg)",
)
ax.plot(
    cntrl_steps, cntrl_smooth,
    color=COLOR_CNTRL_DARK, linewidth=1.5, zorder=3, solid_capstyle="round",
    label="Centralized RL (15-ep avg)",
)

# 2.0 tags/min threshold dashed line
x_max = max(decen_steps[-1], cntrl_steps[-1])
ax.axhline(2.0, color=COLOR_THRESHOLD, linewidth=0.7, linestyle="--", zorder=2, alpha=0.8)
ax.text(
    x_max + 0.05, 2.0, "2.0", ha="left", va="center",
    fontsize=6.5, color=COLOR_THRESHOLD,
)

# Vertical crossover markers
ax.axvline(
    decen_cross_Msteps, color=COLOR_DECEN_DARK,
    linewidth=0.65, linestyle=":", zorder=2, alpha=0.7,
)
ax.axvline(
    cntrl_cross_Msteps, color=COLOR_CNTRL_DARK,
    linewidth=0.65, linestyle=":", zorder=2, alpha=0.7,
)

# ── Speedup bracket annotation ───────────────────────────────────────────────
bracket_y = 2.55
tick_h = 0.08
# horizontal bar
ax.annotate(
    "",
    xy=(cntrl_cross_Msteps, bracket_y),
    xytext=(decen_cross_Msteps, bracket_y),
    arrowprops=dict(
        arrowstyle="<->",
        color=COLOR_ANNOT,
        lw=0.8,
    ),
    zorder=5,
)
ax.text(
    (decen_cross_Msteps + cntrl_cross_Msteps) / 2,
    bracket_y + 0.12,
    f"{speedup:.1f}× faster",
    ha="center", va="bottom",
    fontsize=6.8, color=COLOR_ANNOT, fontweight="bold",
    zorder=5,
)

# Crossover x-tick labels
ax.annotate(
    f"{decen_cross_Msteps:.2f}M",
    xy=(decen_cross_Msteps, 2.0),
    xytext=(decen_cross_Msteps - 0.25, 1.52),
    fontsize=6.2, color=COLOR_DECEN_DARK, ha="center",
    arrowprops=dict(arrowstyle="-", color=COLOR_DECEN_DARK, lw=0.5),
    zorder=5,
)
ax.annotate(
    f"{cntrl_cross_Msteps:.2f}M",
    xy=(cntrl_cross_Msteps, 2.0),
    xytext=(cntrl_cross_Msteps + 0.28, 1.52),
    fontsize=6.2, color=COLOR_CNTRL_DARK, ha="center",
    arrowprops=dict(arrowstyle="-", color=COLOR_CNTRL_DARK, lw=0.5),
    zorder=5,
)

# End-of-line value labels
ax.text(
    decen_steps[-1] + 0.06, decen_smooth[-1],
    f"{decen_final:.2f}",
    ha="left", va="center", fontsize=6.5, color=COLOR_DECEN_DARK,
)
ax.text(
    cntrl_steps[-1] + 0.06, cntrl_smooth[-1],
    f"{cntrl_final:.2f}",
    ha="left", va="center", fontsize=6.5, color=COLOR_CNTRL_DARK,
)

# ── Legend ───────────────────────────────────────────────────────────────────
solid_decen = mlines.Line2D([], [], color=COLOR_DECEN_DARK, linewidth=1.5,
                             label="Decentralized RL (smoothed)")
solid_cntrl = mlines.Line2D([], [], color=COLOR_CNTRL_DARK, linewidth=1.5,
                             label="Centralized RL (smoothed)")
faint_patch = mlines.Line2D([], [], color="#888888", linewidth=0.6, alpha=0.5,
                              label="Per-episode rate (raw)")

ax.legend(
    handles=[solid_decen, solid_cntrl, faint_patch],
    loc="upper left",
    bbox_to_anchor=(0.015, 0.985),
    ncol=1,
    frameon=True,
    framealpha=1.0,
    edgecolor="#cccccc",
    handlelength=1.8,
    borderpad=0.4,
    labelspacing=0.35,
)

# ── Axes formatting ──────────────────────────────────────────────────────────
ax.yaxis.grid(True, linestyle="-", color="#d9d9d9", linewidth=0.45)
ax.set_axisbelow(True)

ax.set_xlim(-0.05, x_max + 0.35)
ax.set_ylim(0, 5.8)
ax.set_yticks([0, 1, 2, 3, 4, 5])

ax.set_xlabel("Training steps per robot (millions)")
ax.set_ylabel("Swarm foraging rate (resources / min)")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

fig.tight_layout(pad=0.4)

out = ROOT / "training_efficiency_curves"
fig.savefig(str(out) + ".png", bbox_inches="tight", dpi=600)
fig.savefig(str(out) + ".pdf", bbox_inches="tight")
fig.savefig(str(out) + ".eps", bbox_inches="tight", format="eps")
print(f"Saved → {out}.png / .pdf / .eps")
print(f"Decen crosses 2.0 at ep {decen_cross_ep} ({decen_cross_Msteps:.3f}M steps)")
print(f"Cntrl crosses 2.0 at ep {cntrl_cross_ep} ({cntrl_cross_Msteps:.3f}M steps)")
print(f"Speedup: {speedup:.1f}×")
print(f"Decen final avg: {decen_final:.2f}  Cntrl final avg: {cntrl_final:.2f}")
