# Experiment 2 — Robot Count Scalability Analysis
## RL-FL-Foraging · ICRA 2027

---

## 1. What This Experiment Measures

Experiment 2 evaluates how all three methods scale as the swarm size increases from
4 to 16 robots in a fixed 7×7 m arena. The key research questions are:
- Does decentralized RL maintain its advantage over centralized RL as more robots
  are added?
- Does either learned approach degrade relative to CPFA with more robots (crowding)?
- At what swarm size, if any, does centralized coordination outperform decentralized?

The number of tags scales proportionally with robots (16 tags per robot) so that each
robot faces a comparable individual workload across conditions.

---

## 2. Experimental Configuration

| Parameter             | Value                                       |
|-----------------------|---------------------------------------------|
| Arena                 | 7×7 m (fixed)                               |
| Robot counts          | 4, 8, 12, 16                                |
| Tag counts            | 64, 128, 192, 256 (16 per robot)            |
| Tag distribution      | Clustered (11 clusters)                     |
| Evaluation duration   | 15 minutes (simulated)                      |
| Samples per condition | 20 per method per robot count               |
| Metric                | Total deposits in 15 min (tags collected)   |
| Methods compared      | Decentralized RL, Centralized RL, CPFA      |

**Models used:**
- Decentralized: `decentralized_indep_v10` (per-robot independent PPO, 19D obs)
- Centralized: `ppo_cpfa_v16` (joint PPO, 72D obs → 8D action)
- CPFA: hand-coded Central Place Foraging Algorithm (baseline)

---

## 3. Results — Full Statistics

### 3.1 Total Deposits in 15 Minutes

| Robots | Method        | Mean   | Std   | Min | Max |
|--------|---------------|--------|-------|-----|-----|
| 4      | Decentralized | **49.25** | 8.75 | 29 | 64 |
| 4      | Centralized   | 45.60  | 6.68  | 32  | 59  |
| 4      | CPFA          | 26.80  | 8.35  | 12  | 42  |
| 8      | Decentralized | **105.10** | 11.53 | 85 | 128 |
| 8      | Centralized   | 98.60  | 8.27  | 82  | 114 |
| 8      | CPFA          | 63.70  | 6.89  | 46  | 77  |
| 12     | Decentralized | **156.80** | 11.20 | 142 | 182 |
| 12     | Centralized   | 153.75 | 9.54  | 134 | 168 |
| 12     | CPFA          | 109.25 | 13.30 | 86  | 132 |
| 16     | Centralized   | **214.25** | 14.19 | 188 | 246 |
| 16     | Decentralized | 209.50 | 17.60 | 167 | 253 |
| 16     | CPFA          | 161.00 | 12.38 | 143 | 181 |

### 3.2 Percentage Differences

| Robots | Decen vs Cntrl | Decen vs CPFA | Cntrl vs CPFA |
|--------|----------------|---------------|---------------|
| 4      | **+8.0%**      | +83.8%        | +70.1%        |
| 8      | **+6.6%**      | +65.0%        | +54.8%        |
| 12     | **+2.0%**      | +43.5%        | +40.7%        |
| 16     | **-2.2%**      | +30.1%        | +33.1%        |

### 3.3 Throughput per Robot (deposits / robot / 15 min)

| Robots | Decen per robot | Cntrl per robot | CPFA per robot |
|--------|-----------------|-----------------|----------------|
| 4      | 12.31           | 11.40           | 6.70           |
| 8      | 13.14           | 12.33           | 7.96           |
| 12     | 13.07           | 12.81           | 9.10           |
| 16     | 13.09           | 13.39           | 10.06          |

Note: throughput per robot is roughly constant across swarm sizes for learned methods
(~12–13 deposits/robot/15 min), while CPFA's per-robot efficiency increases slightly
with more robots (crowd-assisted discovery). This shows that the learned policies
scale with near-linear throughput.

---

## 4. Key Findings

**Finding 1 — Decentralized wins at 4, 8, and 12 robots.**
At the three smaller swarm sizes (4, 8, 12 robots), decentralized outperforms
centralized by +8.0%, +6.6%, and +2.0%, respectively. The gap narrows as swarm
size increases.

**Finding 2 — Centralized wins at 16 robots by a small margin (+2.2%).**
At 16 robots in a 7×7 arena (density ≈ 0.33 robots/m²), the centralized joint
controller slightly outperforms decentralized. The centralized joint policy can
avoid scheduling all robots to the same cluster simultaneously; decentralized robots
independently choose the highest-density pheromone target, creating temporary crowding.

**Finding 3 — The crossover is at ~14–15 robots.**
The performance curves intersect between 12r (decen +2.0%) and 16r (decen -2.2%),
suggesting the crossover occurs around 14–15 robots. Below this density threshold,
decentralized coordination via P2P pheromone is more effective than central control.

**Finding 4 — Both learned approaches substantially outperform CPFA at all swarm sizes.**
Decentralized exceeds CPFA by 30–84% and centralized exceeds CPFA by 33–70% across
all conditions. The CPFA gap shrinks at higher swarm sizes (crowds of robots help
CPFA find clusters via collision), but remains large.

**Finding 5 — Decentralized variance increases at 16 robots.**
Decentralized std at 16r (17.60) is higher than centralized (14.19), reflecting the
stochastic nature of independent decisions leading to more variable crowding at clusters.
In the best runs (max=253), decentralized still beats centralized (max=246); in the
worst runs (min=167), crowding causes more variance.

**Finding 6 — Near-linear throughput per robot for learned approaches.**
Both learned methods maintain approximately 12–13 deposits/robot/15 min across all
swarm sizes, confirming that neither approach hits a coordination bottleneck up to
16 robots. CPFA's per-robot efficiency also improves with swarm size due to
crowd-assisted cluster discovery.

---

## 5. Why Centralized Edges Ahead at 16 Robots

**Crowding at clusters**: With 16 independent robots all receiving pheromone about
the same high-density cluster, many arrive simultaneously. Independent robots have
no mechanism to spread themselves — each picks the highest-density target independently.
The centralized controller sees all robots' current positions jointly and can
implicitly spread assignments through its joint action.

**Competition-aware spreading (mitigation)**: The decentralized v10 model includes
a competition-aware penalty in the obs space that signals when a cluster is being
visited by multiple robots. However, this is learned behavior, not a hard constraint.
The joint centralized policy can enforce complementary actions directly.

**Recommendation for paper**: The 16r result is the honest finding. It defines the
operational sweet spot: decentralized coordination is superior for swarm sizes that
are practical for real robot deployments (4–12 robots per 7×7 arena), while centralized
coordination has a slight edge in extremely dense swarms.

---

## 6. Suggested Paper Section (Draft)

> **Experiment 2: Swarm Scalability**
>
> We fix the arena at 7×7 m and vary swarm size from 4 to 16 robots, scaling the
> tag count proportionally (16 tags/robot) to maintain a consistent per-robot workload.
> All methods are evaluated for 15 minutes per trial (n=20).
>
> Decentralized RL outperforms Centralized RL at 4 robots (+8.0%), 8 robots (+6.6%),
> and 12 robots (+2.0%). At 16 robots (swarm density 0.33 robots/m²), centralized
> gains a small advantage (+2.2%), as the joint policy implicitly avoids assigning
> multiple robots to the same cluster simultaneously. This represents the boundary
> condition where independent pheromone-guided decisions lead to transient crowding.
> Both learned approaches exceed CPFA by 30–84% across all swarm sizes, confirming
> their robustness to scaling. Per-robot throughput remains approximately constant
> (12–13 deposits/robot/15 min) for both learned methods, indicating near-linear
> scaling without coordination bottlenecks up to 12 robots.

---

## 7. Figure Description

**Files in this folder**: `exp2_deposits_boxplot.png`, `exp2_deposits_boxplot_notched.png`,
`exp2_percentage_boxplot.png`, `exp2_percentage_boxplot_notched.png`

**Recommended primary figure**: `exp2_deposits_boxplot_notched.png`
- X-axis: Robot count (4, 8, 12, 16)
- Y-axis: Total deposits in 15 minutes
- Three grouped boxes per count: Decen (orange), Cntrl (blue), CPFA (grey)
- Annotate the crossover: small arrow at 16r pointing to Cntrl winning

**Draft caption**:
> Total deposits collected in a fixed 15-minute evaluation window, across swarm sizes
> of 4–16 robots in a 7×7 m arena (n=20 samples per condition, tags scaled at 16/robot).
> Decentralized RL (orange) leads at 4, 8, and 12 robots (+2.0% to +8.0%), while
> centralized RL (blue) gains a marginal advantage at 16 robots (+2.2%) due to implicit
> crowd avoidance in the joint policy. Both learned methods substantially exceed CPFA
> (grey) at all swarm sizes. Notches indicate 95% confidence intervals around the median.

---

## 8. Potential Reviewer Questions

**Q: The 16r difference is only 2.2% — is this statistically significant?**
A: The gap is small and may not reach significance at n=20 given the overlapping
distributions. A Welch's t-test should be reported; if p > 0.05, the correct
statement is "no statistically significant difference at 16 robots, with centralized
showing a non-significant trend of +2.2%." The honest finding is that they are
effectively tied at 16r, with decentralized leading at all smaller swarm sizes.

**Q: Can the decentralized system address the 16r crowding issue?**
A: Yes, by adding explicit peer-robot position information to the observation space
(e.g., GPS positions of nearby robots received over pheromone channel). This is left
as future work — the current 19D obs does not include teammate positions, matching
the hardware constraint of real e-puck robots without direct GPS-over-WiFi.

**Q: Why does CPFA's per-robot efficiency increase with swarm size?**
A: More robots means more random exploration coverage per unit time, increasing the
probability of a robot stumbling upon an undiscovered cluster. CPFA benefits from
this opportunistic crowdsourcing in a way that is not captured by the "independent
pheromone agent" model.

**Q: Does the 16 tags/robot scaling hold for real rescue missions?**
A: The scaling ensures that each robot's individual contribution remains constant
regardless of swarm size, which is a reasonable fairness constraint. Real deployments
would typically scale tags with area, not robots (as in Exp 1). Both experiments
together cover the two practical scenarios: fixed swarm in scaling environment (Exp 1)
and scaling swarm in fixed environment (Exp 2).

---

## 9. Raw Numbers for Reference

### 4 Robots, 7×7 arena, 64 tags, 15 min (n=20 each)
- Decen: mean=49.25, std=8.75, min=29, max=64
- Cntrl: mean=45.60, std=6.68, min=32, max=59
- CPFA:  mean=26.80, std=8.35, min=12, max=42

### 8 Robots, 7×7 arena, 128 tags, 15 min (n=20 each)
- Decen: mean=105.10, std=11.53, min=85, max=128
- Cntrl: mean=98.60, std=8.27, min=82, max=114
- CPFA:  mean=63.70, std=6.89, min=46, max=77

### 12 Robots, 7×7 arena, 192 tags, 15 min (n=20 each)
- Decen: mean=156.80, std=11.20, min=142, max=182
- Cntrl: mean=153.75, std=9.54, min=134, max=168
- CPFA:  mean=109.25, std=13.30, min=86, max=132

### 16 Robots, 7×7 arena, 256 tags, 15 min (n=20 each)
- Decen: mean=209.50, std=17.60, min=167, max=253
- Cntrl: mean=214.25, std=14.19, min=188, max=246
- CPFA:  mean=161.00, std=12.38, min=143, max=181
