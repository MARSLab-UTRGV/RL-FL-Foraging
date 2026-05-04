# Project Progress — Multi-Agent E-puck Foraging (CoRL 2026)

**Last updated:** 2026-05-04

---

## Project Goal

Train 4 e-puck robots to autonomously forage 70 AprilTags scattered in 6 clusters inside a 5×5 m Webots arena, simulating a **rescue mission** where robots must find and collect items from unknown cluster locations.

The core research question is: **can a decentralized RL policy — where each robot acts using only its own onboard sensors — match or outperform a fully centralized RL policy that has access to global information?**

The centralized version serves as the performance upper bound. The decentralized version must demonstrate comparable or better foraging performance while requiring no central supervisor at execution time. The **CPFA baseline (5.94 tags/min)** is a secondary reference showing where classical swarm methods sit.

The research contribution for **CoRL 2026** is a **Centralized Training, Decentralized Execution (CTDE)** architecture where robots learn to coordinate using **peer-to-peer pheromone signals** without any central brain at execution time — and where 14/18 observation dimensions are computed entirely onboard, making the policy directly deployable on physical robots.

---

## Two Versions

### 1. Centralized (Baseline)
Robots are controlled by a single shared PPO policy trained and executed with full supervisor involvement. The supervisor maintains a global pheromone grid and provides all observations. This serves as the **research baseline**.

### 2. Decentralized — Hybrid Option A (CoRL Contribution)
Robots are trained with a shared PPO policy but each robot computes most of its own observations autonomously using onboard sensors (GPS, InertialUnit, peer-to-peer pheromone radio). At execution time, each robot runs the policy locally with no central supervisor — robots coordinate only by broadcasting pheromone signals to neighbors within 2 m.

This is the novel contribution: **14 out of 18 observation dimensions are computed onboard**, making the policy deployable on physical robots with no architectural change.

---

## How It Works

**Exploration:** Robots explore the arena. When no pheromone signal is known, they are rewarded for moving outward from the base.

**Pheromone:** When a robot picks up a tag from a cluster, it broadcasts a pheromone signal encoding the cluster location and density. Nearby robots receive this signal and navigate toward the cluster. The stronger the signal (denser the cluster), the more robots are attracted. Pheromone persists for `INITIAL_TTL=2000` steps (~64 sec sim time, 5-10 round trips).

**Foraging loop:** Robot finds cluster → picks up tag → returns to base (centre of arena) → deposits → goes back to cluster (guided by pheromone memory).

**P1-P4 Hard-coded overrides:** Applied after PPO inference, fully onboard. Rewards fire normally regardless.
- P1: Wall escape (< 0.6 m from wall) — steer to centre
- P2: Return to base when carrying
- P3: Push away from base (< 0.3 m) to target at 1.2 m
- P4: Steer toward tag when visible

---

## Current Status

### Centralized Version — Complete (baseline)

- Training complete (7 million steps)
- **Best model: `ppo_v16_phero.zip`**
- Performance: ~11 tags/min — nearly **2× the CPFA baseline**
- v17 retrain (`ppo_v17_phero2.zip`) in progress with improved reward shaping

### Decentralized Version — v4 In Training

- v1-v3 complete (see failure analysis below)
- **v4 training in progress** with fixes for all identified failure modes
- Model will be at `decentralized_optA_v4.zip`

---

## Decentralized Training History

### v1 — First Complete Run (7M steps)

**Result:** 3.1 tags/min (below CPFA 5.94 tags/min)

**Failure analysis from eval log:**
- **Phase 1 (0–7 min sim):** All robots orbited at base=0.23-0.24m (`BASE_AVOID`). Tag visibility=1 at 0.77-0.96m. Rate peaked at 6.12 tags/min by fishing nearby tags via P4. This is a base-orbit local optimum.
- **Phase 2 (8–15 min sim):** All 4 robots transitioned to `WALL_ESC` at wall=0.06m. Deposits frozen at 45. Rate dropped to 3.1 tags/min overall.
- **Root cause:** P3 multiplier=3.0 pushed robots to 0.72m (below 0.8m exploration zone). PPO never received zone-entry rewards simultaneously with pheromone gradient → pheromone following not learned. When nearby tags depleted, robots had no learned strategy for navigating to distant clusters.

### v2 — Old P1/P3 Thresholds

**Result:** Wall-hugging confirmed (trained with P1=0.35m, P3=0.6m)

Thresholds too permissive: robots learned to navigate near walls where P1 rarely triggered, resulting in wall-dependent movement patterns rather than open-field exploration.

### v3 — Fixed P1/P3 Thresholds

**Result:** Pheromone following still not learned

P1=0.6m, P3=0.3m applied, but P3 target still 0.72m (multiplier=3.0). PPO trained almost entirely below the active exploration zone → zone rewards never fire during pheromone approach → policy never associates pheromone with reward.

### v4 — All Fixes Applied (In Training)

Changes from v3:
- **P3 multiplier 5.0** → target 1.2m (inside active zone 0.8–2.4m)
- **Near-base penalty ×4.0** (was ×2.0) — stronger discouragement from base-orbit
- **Pheromone approach reward ×5.0** (was ×3.0) — direct incentive for following signal
- **INITIAL_TTL=2000** (was 400) — pheromone persists ~64 sec sim = 5-10 round trips per cluster

Also added per-episode monitoring to training supervisor for live debugging.

---

## Pheromone TTL Bug (Fixed in v4)

The original `INITIAL_TTL=400` equates to ~12.8 seconds of sim time. A round trip (base to cluster at 1.5m and back) takes roughly 5-8 seconds. This means pheromone expired after only 1-2 trips to each cluster, causing robots to forget cluster locations entirely between return visits.

Fix: `INITIAL_TTL=2000` in `controllers/epuck_decentralized/epuck_decentralized.py` — automatically propagates to the eval robot via import.

---

## What Remains

### Immediate
- [ ] Wait for v4 training to complete
- [ ] Evaluate `decentralized_optA_v4.zip` — measure tags/min
- [ ] Run 5+ evaluation seeds (30 min each) for statistical significance
- [ ] Evaluate `ppo_v17_phero2.zip` — compare with v16 centralized baseline
- [ ] Report mean ± std tags/min for all models

### Analysis
- [ ] Ablation study: pheromone ON vs pheromone OFF — shows pheromone's contribution
- [ ] Compare v1-v4 learning curves to document reward shaping progression

### Paper (CoRL 2026)
- [ ] Results section: centralized vs decentralized vs CPFA baseline table
- [ ] Ablation table: with and without pheromone
- [ ] Architecture diagram showing the CTDE obs flow
- [ ] Demo video: 4 robots coordinating via P2P pheromone with no central supervisor

### Future / Optional
- [ ] Physical robot deployment test (replace simulated GPS/camera with real sensors)
- [ ] If v4 underperforms: tune reward scales or increase training timesteps

---

## Performance Summary

| Model | Type | Tags/min | vs CPFA | Status |
|-------|------|----------|---------|--------|
| CPFA baseline | Classical | 5.94 | — | Reference |
| `ppo_v16_phero.zip` | Centralized | ~11 | +85% | Complete |
| `ppo_v17_phero2.zip` | Centralized | TBD | TBD | In training |
| `decentralized_optA_v1.zip` | Decentralized | 3.1 | −48% | Failed |
| `decentralized_optA_v2.zip` | Decentralized | — | — | Failed (wall-hugging) |
| `decentralized_optA_v3.zip` | Decentralized | — | — | Failed (phero not learned) |
| `decentralized_optA_v4.zip` | Decentralized | TBD | TBD | In training |

---

## Key Files

| Purpose | File |
|---------|------|
| Centralized training | `controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_shaping.py` |
| Centralized evaluation | `controllers/eval_best_model/eval_best_model.py` |
| Centralized robot | `controllers/epuck_driver/epuck_driver.py` |
| Decentralized training supervisor | `controllers/decentralized_supervisor/decentralized_supervisor.py` |
| Decentralized training robot | `controllers/epuck_decentralized/epuck_decentralized.py` |
| Decentralized eval supervisor | `controllers/eval_decentralized/eval_decentralized.py` |
| Decentralized eval robot | `controllers/epuck_decentralized_eval/epuck_decentralized_eval.py` |
| Centralized training world | `worlds/epuck_foraging_shaping.wbt` |
| Centralized eval world | `worlds/eval_best.wbt` |
| Decentralized training world | `worlds/epuck_foraging_decentralized.wbt` |
| Decentralized eval world | `worlds/eval_decentralized.wbt` |

For how to run training and evaluation, see `TRAINING_GUIDE.md` and `TESTING_GUIDE.md`.
