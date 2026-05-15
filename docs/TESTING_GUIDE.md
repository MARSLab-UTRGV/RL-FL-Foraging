# Testing Guide — CPFA-RL Centralized (CoRL 2026)

This guide covers evaluating the trained PPO model and the hand-coded CPFA baseline for comparison.

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
| `controllers/eval_best_model/eval_best_model_5x5.py` | Extern eval supervisor — loads PPO model, runs deterministic inference |
| `controllers/cpfa_baseline/cpfa_baseline.py` | Extern CPFA supervisor — hand-coded CPFA state machine, no PPO |
| `controllers/epuck_driver/epuck_driver.py` | Robot controller — unchanged, runs inside Webots for both evaluations |

Both the RL eval and CPFA baseline run in the **same world** (`eval_best_5x5.wbt`) with the same fixed tag layout and same 4 robots. This ensures a direct, fair comparison.

---

## Evaluating the Trained PPO Model

**Step 1: Launch Webots with the eval world**

```bash
webots worlds/eval_best_5x5.wbt &
sleep 10
```

**Step 2: Run the eval supervisor**

```bash
python3 controllers/eval_best_model/eval_best_model_5x5.py ppo_cpfa_5x5.zip
```

Or pass any checkpoint:

```bash
python3 controllers/eval_best_model/eval_best_model_5x5.py \
    logs/ppo_cpfa_5x5/ppo_cpfa_5x5_2000000_steps.zip
```

The model is loaded with `deterministic=True` — no action sampling noise.

### What to Expect

```
[PICKUP] Robot 2 picked up tag (density=6) | Total: 1
  [TARGET] Robot 2 → SITE (1.54, -1.61)
[DEPOSIT] Robot 2 deposited! Total: 1
  [PHERO] Laid at (1.54, -1.61) density=6 prob=0.97 total_entries=1
  [TARGET] Robot 2 → SITE (1.54, -1.61)
```

**Logged every 500 steps to console and `eval_cpfa_log.txt`:**

```
======================================================================
Step 500 (0.3 min) | Pickups: 4 | Deposits: 3 | Rate: 10.00 tags/min
phero_entries=2 phero_max=0.984
R1[SITE      ]: L=+0.98 R=+0.82 | carry=0 | tag_vis=0 td=0.00 | base=1.12 ba=+0.31
               | site=1 sd=0.89 sa=-0.12 | phero=0 pd=0.00 pa=+0.00 | srch=0.04 | wall=1.23
```

**MODE values in log:**
- `WALL_ESC` — P1 override active
- `BASE_ESC` — escaping nest (not carrying, dist_to_base < 0.25m) — same in both systems
- `RTB` — P2 override active (carrying, returning to nest)
- `SITE` — PPO navigating toward site fidelity target (full trip, nest to cluster)
- `PHERO` — PPO navigating toward pheromone target (full trip, nest to cluster)
- `EXPLORE` — PPO in free exploration (no target assigned)
- `GIVE_UP` — search_duration_norm near 1.0, PPO heading back empty-handed

### Healthy Behaviour

- Robots cycle: SITE/PHERO → (find tags) → RTB → SITE/PHERO
- `phero_entries` grows after first few deposits
- Rate should exceed CPFA baseline (see below)
- `srch` values near 0.7–1.0 trigger give-up returns (robots don't stay lost forever)

---

## Running the CPFA Baseline

**Step 1: Same eval world (reload Webots to reset tags)**

```bash
webots worlds/eval_best_5x5.wbt &
sleep 10
```

**Step 2: Run the CPFA supervisor**

```bash
python3 controllers/cpfa_baseline/cpfa_baseline.py
```

No model path needed — CPFA is fully hand-coded.

### What to Expect

```
[PICKUP] R1 picked up tag (density=5) | Total: 1
  [PHERO] Laid at (1.55, -1.58) density=5 prob=0.93 entries=1
  [TARGET] R1 → SITE (1.55, -1.58)
[GIVE-UP] R3 returned empty after 312 searching steps
  [TARGET] R3 → PHERO (1.55, -1.58)
```

**Logged every 500 steps to console and `cpfa_baseline_log.txt`:**

```
======================================================================
Step 500 (0.3 min) | Pickups: 3 | Deposits: 2 | Rate: 6.67 tags/min
phero_entries=1 max_w=0.984
R1[SITE      ]: carry=0 | base=1.18 | wall=1.32 | search= 47 | target=site(1.55,-1.58)
R2[RTB       ]: carry=1 | base=0.61 | wall=2.11 | search=  0 | target=none
```

**MODE values in log:**
- `WALL_ESC` — P1 override active
- `RTB` — carrying, returning to nest
- `GIVE_UP` — returning empty (give-up triggered)
- `SITE` — DEPARTING toward site fidelity target
- `PHERO` — DEPARTING toward pheromone roulette target
- `TAG_SEEK` — tag detected in FOV during local search
- `SEARCH` — CRW random walk (uninformed or informed)

### CPFA Parameters

| Parameter | Value |
|-----------|-------|
| `RATE_OF_LAYING_PHEROMONE` | 3.0 |
| `RATE_OF_SITE_FIDELITY` | 3.0 |
| `RATE_OF_PHEROMONE_DECAY` | 0.01 /sec |
| `ProbabilityOfReturningToNest` | 0.1 (checked every 5 sim-sec) |
| `UninformedSearchVariation` | 30° Gaussian CRW |
| `RateOfInformedSearchDecay` | 0.0002 /step |

---

## Comparing the Two

Both supervisors run in `eval_best_5x5.wbt` with the same fixed tag positions. The key metric is **tags deposited per simulated minute**.

| Component | CPFA Baseline | PPO-CPFA (trained) |
|-----------|--------------|---------------------|
| Pheromone model | list, Poisson CDF, roulette-wheel | **identical** |
| Site fidelity | Poisson CDF priority | **identical** |
| Nest-only info | yes | **identical** |
| Pheromone decay | exp(-0.01 × dt) | **identical** |
| Nest escape after deposit | BASE_ESC (dist<0.25m → steer away) | identical |
| Departure + navigation to target | DEPARTING state (deterministic heading) | PPO (learned from obs[14-19]) |
| Local search at target | SEARCHING state (CRW) | PPO (learned) |
| Tag seek | hard-coded FOV seek | PPO (learned from obs[8-10]) |
| Give-up | probabilistic (P=0.1 / 5 sec) | learned (obs[20] signal) |
| Exploration (no target) | CRW with informed-search decay | PPO (learned) |

The pheromone infrastructure is held constant. Any performance difference is **purely attributable to the learned policy vs. the hand-coded state machine** — which is the paper's core claim.

### Reading the Logs Side by Side

Both log `Rate: X tags/min` every 500 steps (≈16 sim-seconds apart). To compare at the same evaluation time point, check the step number column. Both use `basicTimeStep 32ms` in the same world.

**Target:** PPO-CPFA rate should exceed CPFA baseline rate across a sustained evaluation window.

---

## Headless Evaluation (No GUI)

```bash
webots --mode=fast --minimize --no-rendering worlds/eval_best_5x5.wbt &
sleep 10
python3 controllers/eval_best_model/eval_best_model_5x5.py ppo_cpfa_5x5.zip
```

---

## Troubleshooting

**`[ERROR] Model was trained on obs size X, but env has Y`** — The model must match `obs_per_robot=21` (84D total). Do not load old models trained with 20D obs.

**Robots not moving** — Press Play in Webots before running the supervisor. The extern controller connects after Webots loads.

**`numpy.dtype size changed`** — Run `pip install "numpy<2"`.

**Rate drops to 0 after a while** — All 64 tags in the world have been collected. Reload Webots to reset the world for a new run.

**CPFA robots stuck exploring same area repeatedly** — Normal after cluster depletion. The give-up mechanic (`PROB_RETURN_TO_NEST`) will eventually redirect them.