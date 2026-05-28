# Testing Guide — CPFA-RL Centralized (CoRL 2026)

Evaluate the trained PPO model and compare against the hand-coded CPFA baseline.
Both systems run in the same world with the same fixed tag layout. Any performance
difference is attributable solely to learned vs. hand-coded navigation strategy.

---

## Prerequisites

```bash
export WEBOTS_HOME=/usr/local/webots
export PYTHONPATH=$PYTHONPATH:$WEBOTS_HOME/lib/controller/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WEBOTS_HOME/lib/controller
```

---

## Files Overview

| File | Role |
|------|------|
| `worlds/eval_best_5x5.wbt` | Shared evaluation world — 5×5m, 4 robots, 64 tags at fixed positions |
| `controllers/eval_best_model/eval_best_model_5x5.py` | Extern eval supervisor — loads PPO model, deterministic inference |
| `controllers/cpfa_baseline/cpfa_baseline.py` | Extern CPFA supervisor — hand-coded state machine, no PPO |
| `controllers/epuck_driver/epuck_driver.py` | Robot controller — shared by both, unchanged |

---

## Evaluating the Trained PPO Model

**Step 1: Launch Webots with the eval world (separate instance from training)**

```bash
webots worlds/eval_best_5x5.wbt &
sleep 10
```

**Step 2: Run the eval supervisor** (use port 1235 if training is running on 1234)

```bash
WEBOTS_PORT=1235 python3 controllers/eval_best_model/eval_best_model_5x5.py \
    logs/ppo_cpfa_v5/ppo_cpfa_v5_4000000_steps
```

Pass any checkpoint without `.zip` extension. Default path if none given:
`logs/ppo_cpfa_v5/ppo_cpfa_v5_200000_steps`

The model runs with `deterministic=True` — no action sampling noise during eval.

### Observation Space (must match training exactly)

```
18 values per robot × 4 robots = 72 total

[0:8]   proximity sensors (÷ 4096)
[8]     carrying
[9]     dist_to_base_norm    (÷ 3.5m)
[10]    angle_to_base_norm   (÷ π)
[11]    site_known
[12]    site_dist_norm       (÷ 3.5m)
[13]    site_angle_norm      (÷ π)
[14]    phero_known
[15]    phero_dist_norm      (÷ 3.5m)
[16]    phero_angle_norm     (÷ π)
[17]    search_duration_norm (steps_without_pickup ÷ 4000)

obs[11–17] zeroed when carrying=True
```

Tag sensing is **not** included — removed to ensure a fair comparison with CPFA baseline
which has no equivalent sensing advantage.

### Eval Log Format

Logged to console and `eval_cpfa_log.txt` every 500 steps:

```
======================================================================
Step 3500 (1.9 min) | Pickups: 3 | Deposits: 3 | Rate: 1.61 tags/min
phero_entries=3 phero_max=0.994
======================================================================
R1[BASE_ESC  ]: L=+1.00 R=+0.00 | carry=0 | base=0.07 ba=+0.17 |
               site=1 sd=0.04 sa=+0.83 | phero=0 pd=0.00 pa=+0.00 | srch=0.00 | wall=2.29
R2[SITE      ]: L=-0.95 R=-0.19 | carry=0 | base=0.08 ba=+0.78 |
               site=1 sd=0.09 sa=+0.05 | phero=0 pd=0.00 pa=+0.00 | srch=0.08 | wall=2.24
R3[GIVE_UP   ]: L=+1.00 R=+0.88 | carry=0 | base=0.08 ba=-0.05 |
               site=0 sd=0.00 sa=+0.00 | phero=0 pd=0.00 pa=+0.00 | srch=0.76 | wall=2.20
R4[BASE_ESC  ]: L=+1.00 R=+0.00 | carry=0 | base=0.07 ba=+0.09 |
               site=0 sd=0.00 sa=+0.00 | phero=1 pd=0.09 pa=+0.41 | srch=0.10 | wall=2.27
```

**Column meanings:**
- `L=, R=` — left/right motor command sent to robot (overridden action, not raw PPO output)
- `carry` — 1 if carrying food
- `base=` — normalised distance to nest (÷ 3.5m)
- `ba=` — signed angle to nest (÷ π)
- `site=` / `phero=` — 1 if target assigned
- `sd=, pd=` — normalised distance to site/phero target
- `sa=, pa=` — signed angle to site/phero target
- `srch=` — search_duration_norm (saturates at 1.0 near give-up)
- `wall=` — absolute distance to nearest wall face

**MODE values:**
- `WALL_ESC` — P1 override active (wall proximity or obstacle)
- `BASE_ESC` — BASE_ESC override (not carrying, not gave_up, dist < 0.25m)
- `RTB` — P2 override (carrying, returning to nest)
- `GIVE_UP` — gave_up=True, P2 steering robot home empty-handed
- `SITE` — PPO navigating toward site fidelity target
- `PHERO` — PPO navigating toward pheromone target
- `EXPLORE` — PPO in free exploration (no target assigned)

### Healthy Behaviour Signs

- Actions are varied small floats (e.g. `+0.12, +0.07`) — NOT `+1.00 / -1.00` oscillation
- Robots cycle: `SITE`/`PHERO` → (deposit) → `BASE_ESC` → `SITE`/`PHERO`
- `phero_entries` grows after first few deposits
- `srch=` approaching 1.0 triggers give-up return (robots don't stay lost forever)
- Rate should be stable or improving — not declining toward 0 after step 5000

### Warning Signs

- `L=+1.00 R=-1.00` or `L=-1.00 R=+1.00` repeating — entropy-diverged model (std too large)
- All robots in `EXPLORE` with `base=0.07–0.15` — robots stuck near nest, model undertrained
- Rate declining steeply after step 3000 — nearby clusters depleted, model not navigating farther

### Meaningful Eval Checkpoints

Early checkpoints (< 1M steps) show Phase 1 behaviour only — robots find nearby clusters
but don't navigate confidently to farther ones. Meaningful comparison against CPFA requires
training through Phase 3 or Phase 4:

| Checkpoint | Training phase | What to expect |
|------------|---------------|----------------|
| 600K steps | Phase 1 (~ep 37) | Basic pickup/deposit at 0.4–1.2m range |
| 1.5M steps | Phase 2 (~ep 92) | Learning navigation at 1.6m |
| 3M steps | Phase 3 (~ep 183) | Developing mid-range navigation |
| 6M+ steps | Phase 4 (~ep 366+) | Full arena, meaningful CPFA comparison |

---

## Running the CPFA Baseline

### Controller files

| Arena | Controller | Log file |
|-------|-----------|----------|
| 5×5 m | `controllers/cpfa_baseline/cpfa_baseline.py` | `cpfa_baseline_log.txt` |
| 7×7 m | `controllers/cpfa_baseline/cpfa_baseline_7x7.py` | `cpfa_baseline_log_7x7.txt` |
| 9×9 m | `controllers/cpfa_baseline/cpfa_baseline_9x9.py` | `cpfa_baseline_log_9x9.txt` |
| 12×12 m | `controllers/cpfa_baseline/cpfa_baseline_12x12.py` | `cpfa_baseline_log_12x12.txt` |

All four controllers share identical CPFA logic and parameters — only `num_tags`, `wall_dist`,
CRW clamp bounds, and wall-target bounds differ per arena.

### 5×5 arena (single fixed world)

```bash
webots worlds/eval_best_5x5.wbt &
sleep 10
WEBOTS_PORT=1235 python3 controllers/cpfa_baseline/cpfa_baseline.py
```

### 20-sample runs — 7×7 arena

```bash
webots --mode=fast --minimize --no-rendering worlds/eval_sample<N>_7x7.wbt &
sleep 10
WEBOTS_PORT=1235 python3 controllers/cpfa_baseline/cpfa_baseline_7x7.py
```

### 20-sample runs — 9×9 arena

```bash
webots --mode=fast --minimize --no-rendering worlds/eval_sample<N>_9x9.wbt &
sleep 10
WEBOTS_PORT=1235 python3 controllers/cpfa_baseline/cpfa_baseline_9x9.py
```

### 20-sample runs — 12×12 arena

```bash
webots --mode=fast --minimize --no-rendering worlds/eval_sample<N>_12x12.wbt &
sleep 10
WEBOTS_PORT=1235 python3 controllers/cpfa_baseline/cpfa_baseline_12x12.py
```

Replace `<N>` with 1–20.

### Per-arena controller differences

| Arena | `num_tags` | `wall_dist` formula | CRW clamp | Wall target bounds |
|-------|-----------|--------------------|-----------|--------------------|
| 5×5   | 64  | `2.5 − max\|pos\|` | ±2.2 m | ±2.415 m |
| 7×7   | 128 | `3.5 − max\|pos\|` | ±3.2 m | ±3.415 m |
| 9×9   | 208 | `4.5 − max\|pos\|` | ±4.2 m | ±4.415 m |
| 12×12 | 368 | `6.0 − max\|pos\|` | ±5.7 m | ±5.915 m |

### CPFA Parameters (identical across all arena sizes)

| Parameter | Value |
|-----------|-------|
| `RATE_OF_LAYING_PHEROMONE` | 3.0 |
| `RATE_OF_SITE_FIDELITY` | 1.376 (ARGoS-evolved) |
| `RATE_OF_PHEROMONE_DECAY` | 0.05 /sec (τ ≈ 20s) |
| `ProbabilityOfReturningToNest` | 0.0189 per 5-second check |
| `ProbabilityOfSwitchingToSearching` | 0.765 per 5-second check |
| `UninformedSearchVariation` | 3.67 rad (≈ 210°) |
| `RateOfInformedSearchDecay` | 0.346 /waypoint |

**CPFA MODE values in log:**
- `WALL_ESC` — collision avoidance
- `RTB` — carrying, returning to nest
- `GIVE_UP` — returning empty (give-up triggered)
- `SITE` — DEPARTING toward site fidelity target
- `PHERO` — DEPARTING toward pheromone roulette target
- `DEPART` — uninformed DEPARTING toward random wall position
- `SEARCH` — CRW random walk (uninformed or informed)
- `SURVEY` — post-pickup 360° rotation before returning

---

## Comparing the Two Systems

Primary metric: **tags deposited per simulated minute** (printed every 500 steps in both logs).

| Component | CPFA Baseline | PPO-CPFA (ppo_cpfa_c7) |
|-----------|--------------|------------------------|
| Pheromone model (list, Poisson CDF, roulette-wheel) | hand-coded | **identical** |
| Site fidelity (Poisson CDF priority) | hand-coded | **identical** |
| Pheromone decay (τ ≈ 20s) | hand-coded | **identical** |
| Nest-only information timing | yes | **identical** |
| Give-up mechanism (P=0.0189 per 5s) | hand-coded | **identical** |
| BASE_ESC (steer away, dist < 0.25m) | hand-coded | **identical** |
| P1 wall escape | hard override | **identical** |
| P2 RTB when carrying | hard override | **identical** |
| DEPARTING to site/phero target | deterministic heading | **PPO (learned)** |
| Local search at cluster | CRW random walk | **PPO (learned)** |
| Tag seek | none (removed) | **none (removed)** |
| Exploration strategy | CRW with informed decay | **PPO (learned)** |

The pheromone infrastructure is held constant across both systems and all arena sizes.
Performance difference = learned vs. hand-coded navigation strategy. This is the paper's core claim.

### Paired Comparison — 20-Sample Suite

Run both systems on the **same sample world** and compare rates directly.
Each world has a fixed seed (rotation-based layout) so results are reproducible.

| Arena | PPO controller | CPFA controller | World pattern |
|-------|---------------|-----------------|---------------|
| 5×5   | `eval_best_model_5x5.py` | `cpfa_baseline.py` | `eval_sample<N>_5x5.wbt` |
| 7×7   | `eval_best_model_7x7.py` | `cpfa_baseline_7x7.py` | `eval_sample<N>_7x7.wbt` |
| 9×9   | `eval_best_model_9x9.py` | `cpfa_baseline_9x9.py` | `eval_sample<N>_9x9.wbt` |
| 12×12 | `eval_best_model_12x12.py` | `cpfa_baseline_12x12.py` | `eval_sample<N>_12x12.wbt` |

### Procedure for Paper Results

1. For each arena size, run both systems on all 20 sample worlds (reload Webots between runs)
2. Each trial: let run for **30 simulated minutes** (≈ 28,125 steps at 64 ms/step)
3. Record total deposits per trial → compute tags/min
4. Report mean ± std over 20 samples for both systems, per arena
5. Plot PPO vs CPFA rate across arenas to show generalisation gap

---

## Generalization Evaluation — 20-Sample Suite (`ppo_cpfa_c7`)

Tests how well the model trained on a 5×5 arena transfers to larger, unseen arenas.
Obs normalizations are intentionally kept at 3.5 m (training value) in all eval controllers,
so any performance change reflects distribution-shift robustness, not recalibration.

### Arena Summary

| Arena | Eval controller | World files | Tags | Clusters | Log file |
|-------|----------------|-------------|------|----------|----------|
| 5×5 m | `eval_best_model_5x5.py` | `eval_sample1_5x5.wbt` … `eval_sample20_5x5.wbt` | 64 | 6 | `eval_cpfa_log.txt` |
| 7×7 m | `eval_best_model_7x7.py` | `eval_sample1_7x7.wbt` … `eval_sample20_7x7.wbt` | 128 | 11 | `eval_cpfa_log_7x7.txt` |
| 9×9 m | `eval_best_model_9x9.py` | `eval_sample1_9x9.wbt` … `eval_sample20_9x9.wbt` | 208 | 14 | `eval_cpfa_log_9x9.txt` |
| 12×12 m | `eval_best_model_12x12.py` | `eval_sample1_12x12.wbt` … `eval_sample20_12x12.wbt` | 368 | 19 | `eval_cpfa_log_12x12.txt` |

Sample rotations: samples 1–10 at 36° intervals (0°, 36°, …, 324°);
samples 11–20 interleaved at 18° offset (18°, 54°, …, 342°).

---

### Run Commands

Replace `<N>` with sample number 1–20 and `<ARENA>` with the arena suffix.

**5×5 arena (training distribution)**

```bash
webots --mode=fast --minimize --no-rendering worlds/eval_sample<N>_5x5.wbt &
sleep 10
WEBOTS_PORT=1235 python3 controllers/eval_best_model/eval_best_model_5x5.py \
    logs/ppo_cpfa_c7/ppo_cpfa_c7_5000000_steps
```

**7×7 arena (1.96× area)**

```bash
webots --mode=fast --minimize --no-rendering worlds/eval_sample<N>_7x7.wbt &
sleep 10
WEBOTS_PORT=1235 python3 controllers/eval_best_model/eval_best_model_7x7.py \
    logs/ppo_cpfa_c7/ppo_cpfa_c7_5000000_steps
```

**9×9 arena (3.24× area)**

```bash
webots --mode=fast --minimize --no-rendering worlds/eval_sample<N>_9x9.wbt &
sleep 10
WEBOTS_PORT=1235 python3 controllers/eval_best_model/eval_best_model_9x9.py \
    logs/ppo_cpfa_c7/ppo_cpfa_c7_5000000_steps
```

**12×12 arena (5.76× area)**

```bash
webots --mode=fast --minimize --no-rendering worlds/eval_sample<N>_12x12.wbt &
sleep 10
WEBOTS_PORT=1235 python3 controllers/eval_best_model/eval_best_model_12x12.py \
    logs/ppo_cpfa_c7/ppo_cpfa_c7_5000000_steps
```

> **Tip:** If port 1234 is in use by training, use `WEBOTS_PORT=1235`.
> If running multiple evals in parallel, use 1235, 1236, 1237, … for each instance.

---

### Sample Index

| Sample | Rotation | 5×5 world | 7×7 world | 9×9 world | 12×12 world |
|--------|----------|-----------|-----------|-----------|-------------|
| 1  | 0°   | eval_sample1_5x5.wbt  | eval_sample1_7x7.wbt  | eval_sample1_9x9.wbt  | eval_sample1_12x12.wbt  |
| 2  | 36°  | eval_sample2_5x5.wbt  | eval_sample2_7x7.wbt  | eval_sample2_9x9.wbt  | eval_sample2_12x12.wbt  |
| 3  | 72°  | eval_sample3_5x5.wbt  | eval_sample3_7x7.wbt  | eval_sample3_9x9.wbt  | eval_sample3_12x12.wbt  |
| 4  | 108° | eval_sample4_5x5.wbt  | eval_sample4_7x7.wbt  | eval_sample4_9x9.wbt  | eval_sample4_12x12.wbt  |
| 5  | 144° | eval_sample5_5x5.wbt  | eval_sample5_7x7.wbt  | eval_sample5_9x9.wbt  | eval_sample5_12x12.wbt  |
| 6  | 180° | eval_sample6_5x5.wbt  | eval_sample6_7x7.wbt  | eval_sample6_9x9.wbt  | eval_sample6_12x12.wbt  |
| 7  | 216° | eval_sample7_5x5.wbt  | eval_sample7_7x7.wbt  | eval_sample7_9x9.wbt  | eval_sample7_12x12.wbt  |
| 8  | 252° | eval_sample8_5x5.wbt  | eval_sample8_7x7.wbt  | eval_sample8_9x9.wbt  | eval_sample8_12x12.wbt  |
| 9  | 288° | eval_sample9_5x5.wbt  | eval_sample9_7x7.wbt  | eval_sample9_9x9.wbt  | eval_sample9_12x12.wbt  |
| 10 | 324° | eval_sample10_5x5.wbt | eval_sample10_7x7.wbt | eval_sample10_9x9.wbt | eval_sample10_12x12.wbt |
| 11 | 18°  | eval_sample11_5x5.wbt | eval_sample11_7x7.wbt | eval_sample11_9x9.wbt | eval_sample11_12x12.wbt |
| 12 | 54°  | eval_sample12_5x5.wbt | eval_sample12_7x7.wbt | eval_sample12_9x9.wbt | eval_sample12_12x12.wbt |
| 13 | 90°  | eval_sample13_5x5.wbt | eval_sample13_7x7.wbt | eval_sample13_9x9.wbt | eval_sample13_12x12.wbt |
| 14 | 126° | eval_sample14_5x5.wbt | eval_sample14_7x7.wbt | eval_sample14_9x9.wbt | eval_sample14_12x12.wbt |
| 15 | 162° | eval_sample15_5x5.wbt | eval_sample15_7x7.wbt | eval_sample15_9x9.wbt | eval_sample15_12x12.wbt |
| 16 | 198° | eval_sample16_5x5.wbt | eval_sample16_7x7.wbt | eval_sample16_9x9.wbt | eval_sample16_12x12.wbt |
| 17 | 234° | eval_sample17_5x5.wbt | eval_sample17_7x7.wbt | eval_sample17_9x9.wbt | eval_sample17_12x12.wbt |
| 18 | 270° | eval_sample18_5x5.wbt | eval_sample18_7x7.wbt | eval_sample18_9x9.wbt | eval_sample18_12x12.wbt |
| 19 | 306° | eval_sample19_5x5.wbt | eval_sample19_7x7.wbt | eval_sample19_9x9.wbt | eval_sample19_12x12.wbt |
| 20 | 342° | eval_sample20_5x5.wbt | eval_sample20_7x7.wbt | eval_sample20_9x9.wbt | eval_sample20_12x12.wbt |

---

### Measurement Protocol

Each trial: let run for **30 simulated minutes** (≈ 28,125 steps at 64 ms/step).
Primary metric: **tags deposited per simulated minute** (printed every 500 steps in the log).

**Record per sample:**

```
Arena | Sample | Deposits | Duration (min) | Rate (tags/min)
```

**Aggregate per arena:**

```
mean ± std over 20 samples
```

**Results table template:**

| Arena | Mean rate (tags/min) | Std | Min | Max |
|-------|---------------------|-----|-----|-----|
| 5×5   |                     |     |     |     |
| 7×7   |                     |     |     |     |
| 9×9   |                     |     |     |     |
| 12×12 |                     |     |     |     |

---

### What Each Arena Tests

| Arena | What it probes |
|-------|---------------|
| 5×5   | In-distribution performance — clusters at trained distances (up to 2.1 m) |
| 7×7   | Mild OOD shift — VFAR clusters at 3.0 m, obs normalised at 3.5 m (same as training) |
| 9×9   | Moderate OOD — VFAR at 3.8 m, clusters at radii the policy never trained on |
| 12×12 | Severe OOD — VFAR at 5.2 m, 5.76× training area; tests pheromone-guided extrapolation |

A drop in rate from 5×5 → 12×12 is expected; the paper claim is that the PPO policy
degrades more gracefully than the CPFA baseline under distribution shift, thanks to
learned rather than hard-coded navigation.

---

## Distribution-Type Evaluation — 5×5 Arena (ppo_cpfa_c7)

These 10 worlds hold the arena size **fixed at 5×5 m** and the tag count **fixed at 64**,
but vary *how* tags are spatially distributed. They probe robustness to tag layout rather
than arena scale.

### World Files

| Type | Files | Tags | Distribution |
|------|-------|------|--------------|
| Power-law clusters | `eval_powerlaw1_5x5.wbt` … `eval_powerlaw10_5x5.wbt` | 64 | Few large, many small clusters (sizes: 20, 12, 9, 9, 8, 6) |
| Uniform random | `eval_random1_5x5.wbt` … `eval_random10_5x5.wbt` | 64 | No clustering — tags placed independently at random |

**Power-law layout:** 6 clusters at 72° rotation intervals. Samples 1–5 at 0°, 72°, 144°, 216°, 288°;
samples 6–10 interleaved at 36°, 108°, 180°, 252°, 324°.
`min_spacing=0.0775 m`, placement zone ±2.1 m (same safe zone as training).

**Random layout:** Uniform random placement, no clusters. Seeds 1–5: 42, 137, 271, 512, 999; seeds 6–10: 1337, 2718, 3141, 4321, 8675.
`min_spacing=0.08 m`, `nest_excl=0.35 m`, placement zone ±2.1 m.

---

### Run Commands

**Power-law samples (same controller as 5×5 eval — use `eval_best_model_5x5.py`)**

```bash
webots --mode=fast --minimize --no-rendering worlds/eval_powerlaw<N>_5x5.wbt &
sleep 10
WEBOTS_PORT=1235 python3 controllers/eval_best_model/eval_best_model_5x5.py \
    logs/ppo_cpfa_c7/ppo_cpfa_c7_5000000_steps
```

Replace `<N>` with 1–10.

**Random samples**

```bash
webots --mode=fast --minimize --no-rendering worlds/eval_random<N>_5x5.wbt &
sleep 10
WEBOTS_PORT=1235 python3 controllers/eval_best_model/eval_best_model_5x5.py \
    logs/ppo_cpfa_c7/ppo_cpfa_c7_5000000_steps
```

Replace `<N>` with 1–10.

**CPFA baseline — same worlds, different controller**

```bash
webots --mode=fast --minimize --no-rendering worlds/eval_powerlaw<N>_5x5.wbt &
sleep 10
python3 controllers/cpfa_baseline/cpfa_baseline.py

# or for random:
webots --mode=fast --minimize --no-rendering worlds/eval_random<N>_5x5.wbt &
sleep 10
python3 controllers/cpfa_baseline/cpfa_baseline.py
```

---

### What Each Distribution Tests

| World set | What it probes |
|-----------|---------------|
| Power-law clusters | Biased density — one large dominant cluster; pheromone exploitation heavily rewarded |
| Uniform random | No pheromone signal — site fidelity provides no advantage; CPFA degenerates to random walk |

**Hypothesis:** PPO policy maintains higher collection rates on uniform random layouts
because it learns a general spatial coverage strategy, while CPFA wastes time revisiting
depleted cluster centres and site-fidelity shortcuts that don't exist.

### Measurement Protocol

Same 30-minute trials as the generalization suite. Record tags/min per sample (10 samples each).

**Results table template:**

| Distribution | System | Mean rate (tags/min) | Std | Min | Max |
|-------------|--------|---------------------|-----|-----|-----|
| Power-law   | PPO    |                     |     |     |     |
| Power-law   | CPFA   |                     |     |     |     |
| Random      | PPO    |                     |     |     |     |
| Random      | CPFA   |                     |     |     |     |

---

## Headless Evaluation

```bash
webots --mode=fast --minimize --no-rendering worlds/eval_best_5x5.wbt &
sleep 10
WEBOTS_PORT=1235 python3 controllers/eval_best_model/eval_best_model_5x5.py \
    logs/ppo_cpfa_v5/ppo_cpfa_v5_6000000_steps
```

---

## Troubleshooting

**`[ERROR]` on model load mentioning obs size mismatch** — Model must match `obs_per_robot=18`
(72D total). Do not load v3 or earlier checkpoints trained with 21D (84D total).

**Port conflict** — If training is on port 1234, run eval on port 1235 with `WEBOTS_PORT=1235`.

**Robots not moving** — Press Play in Webots before running the supervisor.

**Rate drops to 0 after several minutes** — All 64 tags collected. Reload Webots to reset.

**CPFA robots repeatedly exploring depleted areas** — Normal. Give-up will redirect them.
Expected behaviour after dense cluster depletion.

**`numpy.dtype size changed`** — Run `pip install "numpy<2"`.