# Project Progress — CPFA-RL Foraging (CoRL 2026)

**Last updated:** 2026-05-13

---

## Research Goal

Show that a learned PPO policy, using **CPFA's exact pheromone infrastructure**, outperforms CPFA itself on a multi-robot foraging task in a 5×5m arena. The pheromone model is held identical between both systems — only the navigation/strategy is learned vs. hand-coded.

**Core claim:** Replacing CPFA's 4-state machine with PPO, while preserving the full CPFA pheromone model (list, Poisson CDF, roulette-wheel, site fidelity, nest-only information), produces a faster forager.

---

## System Overview

| Component | Description |
|-----------|-------------|
| Arena | 5×5m Webots arena, walls at ±2.5m |
| Robots | 4 e-puck robots with proximity sensors |
| Tags | 64 AprilTags distributed in clusters |
| Controller | `epuck_driver.py` — sends 8 proximity values, receives [left, right] motor commands |
| Pheromone | List-based (not grid): `{x, y, weight, resource_density}` per entry |

---

## Current Status

### CPFA-RL Centralized Supervisor — Retrain Required

**File:** `controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_cpfa.py`
**World:** `worlds/epuck_foraging_shaping_5x5.wbt`
**Status:** P3 (DEPARTING) restored. First 3M-step run failed (policy didn't learn steering). Retrain with updated supervisor.

Key design:
- Obs space: 21D per robot × 4 robots = **84D total**
- CPFA pheromone model identical to baseline (list, Poisson CDF, roulette-wheel, site fidelity)
- Pheromone information assigned **only at nest return** (matches CPFA exactly)
- PPO replaces DEPARTING + SEARCHING states (full navigation + local search)
- BASE_ESC override: steer away from nest when not carrying and dist < 0.25m (identical in both systems)
- P2 override (RTB when carrying) preserved = CPFA RETURNING state
- 3-phase distance curriculum: 1.0m → 1.8m → 2.3m cluster distance
- Give-up behaviour: obs[20] `search_duration_norm` + two-phase reward (explore outward, then reward nest approach after 500 steps without food)

### CPFA Baseline — Complete

**File:** `controllers/cpfa_baseline/cpfa_baseline.py`
**World:** `worlds/eval_best_5x5.wbt`
**Status:** Complete. Matches ARGoS CPFA algorithm with Webots-specific adaptations.

Implements:
- State machine: DEPARTING → SEARCHING → RETURNING (SURVEYING merged into pickup — Webots constraint)
- CRW (Correlated Random Walk) with 30° Gaussian heading variation for uninformed search
- Informed search decay: correlation width widens exponentially (`RATE_OF_INFORMED_SEARCH_DECAY=0.0002/step`)
- ProbabilityOfReturningToNest = 0.1, checked every 5 sim-seconds
- Identical pheromone model to training supervisor (same parameters, same list structure)

### RL Evaluation Script — Complete

**File:** `controllers/eval_best_model/eval_best_model_5x5.py`
**World:** `worlds/eval_best_5x5.wbt`
**Status:** Complete. Obs space matches training supervisor exactly (21D per robot, 84D total, CPFA pheromone signals).

---

## What Remains

### Training
- [ ] Run training: `python3 controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_cpfa.py --run_name ppo_cpfa_5x5 --total_timesteps 3000000`
- [ ] Monitor `training_log.txt` — check phase transitions, pheromone list growth, deposit rate per episode

### Evaluation
- [ ] Run RL eval on `eval_best_5x5.wbt` with trained `ppo_cpfa_5x5.zip`
- [ ] Run CPFA baseline on same world
- [ ] Run 5+ trials of each (reload world between trials) and record tags/min
- [ ] Report mean ± std for both

### Analysis (Paper)
- [ ] Results table: PPO-CPFA vs CPFA baseline (tags/min, mean ± std)
- [ ] Ablation: PPO with vs without pheromone signals (zero obs[14-19]) — isolates pheromone contribution
- [ ] Learning curve: deposits/episode across 3M timesteps — show curriculum progression
- [ ] Pheromone activity log: `entries` and `max_weight` over training — show pheromone list being used

---

## Performance Targets

| System | Type | Tags/min | Status |
|--------|------|----------|--------|
| CPFA baseline | Hand-coded | TBD (run to get number) | Ready to run |
| PPO-CPFA (`ppo_cpfa_5x5.zip`) | Learned | TBD | Training not started |

The paper needs PPO-CPFA to exceed the CPFA baseline. Both systems use identical pheromone infrastructure, so any gap is attributable to the learned policy.

---

## Key Files

| Purpose | File |
|---------|------|
| **RL training supervisor** | `controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_cpfa.py` |
| **RL evaluation supervisor** | `controllers/eval_best_model/eval_best_model_5x5.py` |
| **CPFA baseline supervisor** | `controllers/cpfa_baseline/cpfa_baseline.py` |
| **Robot controller** (shared by all) | `controllers/epuck_driver/epuck_driver.py` |
| **Training world** | `worlds/epuck_foraging_shaping_5x5.wbt` |
| **Evaluation world** (shared) | `worlds/eval_best_5x5.wbt` |

---

## Pheromone Model (both systems, identical)

| Parameter | Value |
|-----------|-------|
| Structure | List of `{x, y, weight, resource_density}` |
| Created at | Nest deposit (not at pickup) |
| Gate | Poisson CDF: `P(lay) = CDF(density, λ=3.0)` |
| Selection | Roulette-wheel weighted by weight |
| Site fidelity | Priority 1: Poisson CDF test. Priority 2: pheromone. Priority 3: explore |
| Decay | `weight *= exp(-0.01 * dt_sec)` per step |
| Pruning | Removed when `weight < 0.001` |
| Info timing | Assigned only at nest return |

---

## Observation Space Detail (21D per robot)

```
[0:8]   proximity sensors (÷ 4096)
[8]     tag_visible
[9]     tag_dist_norm     (÷ 1.0m)
[10]    tag_angle_norm    (÷ π)
[11]    carrying
[12]    dist_to_base_norm (÷ 3.5m)
[13]    angle_to_base_norm (÷ π)
[14]    site_known        ← CPFA site fidelity (assigned at nest)
[15]    site_dist_norm    (÷ 3.5m)
[16]    site_angle_norm   (÷ π)
[17]    phero_known       ← CPFA pheromone target (roulette-wheel, at nest)
[18]    phero_dist_norm   (÷ 3.5m)
[19]    phero_angle_norm  (÷ π)
[20]    search_duration_norm  (steps_without_pickup ÷ 700)

obs[14-20] zeroed when carrying=True
```

For training and testing commands, see `TRAINING_GUIDE.md` and `TESTING_GUIDE.md`.