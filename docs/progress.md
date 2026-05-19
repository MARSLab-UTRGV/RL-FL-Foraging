# Project Progress — CPFA-RL Foraging (CoRL 2026)

**Last updated:** 2026-05-19

---

## Research Goal

Show that a learned PPO policy using **CPFA's exact pheromone infrastructure** outperforms
CPFA itself on a 5×5m multi-robot foraging task. The pheromone model is held identical
between both systems — any performance gap is purely attributable to learned vs. hand-coded
navigation strategy.

**Core claim:** Replacing CPFA's 4-state machine with PPO, while preserving the full
pheromone model (list, Poisson CDF, roulette-wheel, site fidelity, nest-only information),
produces a faster forager.

---

## System Overview

| Component | Value |
|-----------|-------|
| Arena | 5×5m Webots arena, walls at ±2.5m |
| Robots | 4 e-puck robots with 8 proximity sensors |
| Tags | 64 AprilTags in 11 clusters per episode |
| Timestep | 64ms (training), matching baseline |
| Pheromone | List: `{x, y, weight, resource_density}` — not a grid |
| Obs space | 18D per robot × 4 = 72D total (tag sensing removed) |
| Actions | `[left_motor, right_motor]` ∈ [−1, 1] per robot |

---

## Training Versions History

| Version | ent_coef | Status | Notes |
|---------|----------|--------|-------|
| v1–v2 | various | Abandoned | Early obs space experiments |
| v3 | 0.01 | Abandoned | 21D obs (with tag sensing); robot stuck still after full arena |
| v4 | 0.05 | Failed — entropy diverged | std grew 1.91 → 540 by ep283; policy mean irrelevant |
| **v5** | **0.03** | **In progress** | 18D obs (tag sensing removed); clean training, ep35+ healthy |

---

## Current Status

### v5 Training — In Progress

**File:** `controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_cpfa.py`
**World:** `worlds/epuck_foraging_shaping_5x5.wbt`
**Run name:** `ppo_cpfa_v5`
**Total timesteps:** 10,000,000
**Current:** ~573,440 steps (ep 35), Phase 1 (max_dist=1.2m)

**Key design decisions for v5:**
- Obs space reduced to 18D (removed tag sensing obs[8–10] — fairness vs CPFA baseline)
- Outward positional bonus removed — was the standing-still local optimum in v3/v4
- Forward motion bias scaled 0.01 → 0.15 — only reward for actual movement
- ent_coef = 0.03: entropy (0.168/step) maintains free-explore diversity; DEPARTING
  reward (0.65/step) dominates during navigation (4× entropy)
- Curriculum: 4 phases (1.2m → 1.6m → 2.0m → 2.3m), 11 fixed clusters

**v5 training metrics at ep 35 (573K steps):**

| Metric | Value | Status |
|--------|-------|--------|
| `std` | 1.65 | Healthy — slow growth, not diverging |
| `ep_rew_mean` | −1.00e+04 | Improving — halved from −2.08e+04 at ep1 |
| `explained_variance` | 0.807 | Excellent — value network well-calibrated |
| `approx_kl` | 0.008 | Normal — stable policy updates |
| `entropy_loss` | −15.3 | Healthy |
| Deposit rate (ep 28) | 2.17 tags/min | Best Phase 1 episode |

**Phase 2 transition at ep 60 (≈983K steps):** cluster max_dist jumps to 1.6m. Expect
a temporary deposit rate drop as policy adapts to longer navigation distances.

---

### CPFA Baseline — Complete

**File:** `controllers/cpfa_baseline/cpfa_baseline.py`
**World:** `worlds/eval_best_5x5.wbt`
**Status:** Complete and verified.

Implements:
- 3-state machine: DEPARTING → SEARCHING → RETURNING (SURVEYING merged into pickup)
- CRW (Correlated Random Walk) local search with informed-search decay
- CPFA pheromone: identical parameters to training supervisor
- Give-up: P=0.0189 per 5-second check
- **No tag sensing** — removed for fair comparison with RL system

### RL Evaluation Supervisor — Complete

**File:** `controllers/eval_best_model/eval_best_model_5x5.py`
**World:** `worlds/eval_best_5x5.wbt`
**Status:** Complete. Obs space matches training exactly (18D per robot, 72D total).

v5 eval at 600K steps showed:
- No `±1.00` oscillation (contrast with broken v4 at 5.2M steps)
- Rate ≈ 1.6–2.0 tags/min
- Robots cluster near base (expected — model is Phase 1 only, hasn't learned farther navigation)
- Meaningful nav comparison requires ≥ 4M steps checkpoint

---

## Observation Space (current — v5)

```
[0:8]   proximity sensors                  ÷ 4096, range [0, 1]
[8]     carrying                           1.0 if holding food
[9]     dist_to_base_norm                  dist / 3.5m
[10]    angle_to_base_norm                 signed angle / π
[11]    site_known                         1.0 if site fidelity target at nest
[12]    site_dist_norm                     dist to site / 3.5m
[13]    site_angle_norm                    signed angle to site / π
[14]    phero_known                        1.0 if pheromone target at nest
[15]    phero_dist_norm                    dist to phero / 3.5m
[16]    phero_angle_norm                   signed angle to phero / π
[17]    search_duration_norm               steps_without_pickup / 4000

obs[11–17] zeroed when carrying=True
```

---

## Reward Shaping (current — v5)

### Per robot, always active
- **Proximity penalty**: `−max(prox) × 0.5` when `max(prox) > 0.1`
- **Wall penalty**: `−(0.35 − wall_dist) × 0.5` when `wall_dist < 0.35m`
- **Time penalty**: `−0.005` every step

### Per robot, exploration branch (not carrying)
- **Pickup**: `+5.0` when tag within 0.15m
- **Forward motion bias**: `avg_speed × 0.15` when `avg_speed > 0` and `wall_dist ≥ 0.35m`
  — the **only** per-step reward available in free exploration; standing still earns nothing
- **SITE approach shaping**: `(prev_site_dist − curr_site_dist) × 15.0`
  — pre-seeded at deposit, cleared at cluster arrival (0.05m) or pickup
- **SITE orientation**: `dot(forward, site_dir) × 0.5`
- **PHERO approach shaping**: `(prev_phero_dist − curr_phero_dist) × 15.0`
- **PHERO orientation**: `dot(forward, phero_dir) × 0.5`

### Per robot, carrying branch
- **Deposit**: `+20.0` when `dist_to_base < 0.25m`
- **RTB approach shaping**: `(prev_base_dist − curr_base_dist) × 8.0`
  — pre-seeded at pickup to prevent spike on first carry step
- **RTB orientation**: `dot(forward, base_dir) × 0.5`

### Multi-robot
- **Separation penalty**: `−(1.0 − sep) × 0.5` per pair when both within 1.5m of base
  and inter-robot separation `sep < 1.0m`

---

## CPFA Pheromone Parameters (both systems — identical)

| Parameter | Value |
|-----------|-------|
| `RATE_OF_LAYING_PHEROMONE` | 3.0 |
| `RATE_OF_SITE_FIDELITY` | 1.376 (ARGoS-evolved) |
| `RATE_OF_PHEROMONE_DECAY` | 0.05 /sec (τ ≈ 20s) |
| `PHEROMONE_MIN` | 0.001 |
| `PROB_RETURN_TO_NEST` | 0.0189 |
| `GIVE_UP_CHECK_STEPS` | 78 (5s at 64ms) |

---

## Training Curriculum (v5)

| Phase | Episodes | max_dist | Share of training |
|-------|----------|----------|-------------------|
| 1 — Near | 1–59 | 1.2m | ~10% |
| 2 — Medium | 60–149 | 1.6m | ~15% |
| 3 — Far | 150–299 | 2.0m | ~25% |
| 4 — Full | 300+ | 2.3m | ~51% |

11 clusters fixed per episode. Only max cluster distance changes across phases.

---

## Key Files

| Purpose | File |
|---------|------|
| RL training supervisor | `controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_cpfa.py` |
| RL eval supervisor | `controllers/eval_best_model/eval_best_model_5x5.py` |
| CPFA baseline | `controllers/cpfa_baseline/cpfa_baseline.py` |
| Robot controller (shared) | `controllers/epuck_driver/epuck_driver.py` |
| Training world | `worlds/epuck_foraging_shaping_5x5.wbt` |
| Eval world (shared) | `worlds/eval_best_5x5.wbt` |
| Training log | `training_log.txt` |
| Eval log | `eval_cpfa_log.txt` |
| v5 checkpoints | `logs/ppo_cpfa_v5/ppo_cpfa_v5_*_steps.zip` |

---

## What Remains

### Training
- [x] Remove tag sensing from both RL and baseline (fairness)
- [x] Fix standing-still reward (remove positional bonus, scale forward bias)
- [x] Fix entropy divergence (ent_coef 0.05 → 0.03)
- [x] v5 training started — healthy at ep 35 (573K steps)
- [ ] Let v5 reach Phase 2 (ep 60, ~1M steps) — first real navigation test
- [ ] Let v5 complete Phase 3 (ep 150, ~2.5M steps) — meaningful checkpoint
- [ ] Let v5 complete Phase 4 (ep 300+, ~5M steps) — full arena behaviour

### Evaluation
- [ ] Eval v5 at 4M, 6M, 8M step checkpoints against CPFA baseline
- [ ] 5+ trials of each system, record tags/min per trial
- [ ] Report mean ± std for both

### Analysis (Paper)
- [ ] Results table: PPO-CPFA v5 vs CPFA baseline (tags/min, mean ± std)
- [ ] Learning curve: deposits/episode across 10M timesteps — show curriculum progression
- [ ] Pheromone activity: entries/max_weight over training — confirm pheromone list is used
- [ ] Ablation: PPO with pheromone signals zeroed — isolates pheromone contribution

---

## Performance Targets

| System | Type | Tags/min | Status |
|--------|------|----------|--------|
| CPFA baseline | Hand-coded | TBD — run to get number | Ready |
| PPO-CPFA v5 (early, 600K) | Learned | ~1.6–2.0 | Phase 1 only |
| PPO-CPFA v5 (final, 6M+) | Learned | TBD | Training in progress |