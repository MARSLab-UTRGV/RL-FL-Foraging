# Project Progress — Multi-Agent E-puck Foraging (CoRL 2026)

**Last updated:** 2026-05-01

---

## Project Goal

Train 4 e-puck robots to autonomously forage 70 AprilTags scattered in 6 clusters inside a 5×5 m Webots arena, simulating a **rescue mission** where robots must find and collect items from unknown cluster locations.

The core research question is: **can a decentralized RL policy — where each robot acts using only its own onboard sensors — match or outperform a fully centralized RL policy that has access to global information?**

The centralized version serves as the performance upper bound. The decentralized version must demonstrate comparable or better foraging performance while requiring no central supervisor at execution time. The **CPFA baseline (5.94 tags/min)** is a secondary reference point showing both RL models improve over classical swarm methods.

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

**Pheromone:** When a robot picks up a tag from a cluster, it broadcasts a pheromone signal encoding the cluster location and density. Nearby robots receive this signal and navigate toward the cluster. The stronger the signal (denser the cluster), the more robots are attracted.

**Foraging loop:** Robot finds cluster → picks up tag → returns to base (centre of arena) → deposits → goes back to cluster (guided by pheromone memory).

**Curriculum:** Training starts with clusters close to the base and gradually expands to the full arena as the robots improve.

---

## Current Status

### Centralized Version ✅ Complete

- Training complete (7 million steps)
- **Best model: `ppo_v16_phero.zip`**
- Performance: ~11 tags/min — nearly **2× the CPFA baseline**
- Serves as the centralized baseline for comparison with the decentralized version

A refined retrain (`ppo_v16_retrain.zip`) is currently running with an improved reward shaping that fixes a gradient blind spot after each deposit. This is expected to perform at least as well as v16.

### Decentralized Version 🔄 In Training

- All code is complete and verified running
- Training in progress simultaneously on two machines (LD39052 and DL39053)
- Model: `decentralized_optA_v1.zip`
- Expected to complete in ~7–8 hours per machine
- Early training shows all 4 robots picking up and depositing tags from the first episode

---

## What Remains

### Immediate (once training completes)
- [ ] Evaluate `ppo_v16_retrain.zip` — compare with v16 baseline
- [ ] Evaluate `decentralized_optA_v1.zip` — measure tags/min
- [ ] Run 5+ evaluation seeds (30 min each) per model for statistical significance
- [ ] Report mean ± std tags/min for all models

### Analysis
- [ ] Ablation study: pheromone ON vs pheromone OFF — to show pheromone's contribution
- [ ] Create a dedicated evaluation script for the decentralized model

### Paper (CoRL 2026)
- [ ] Results section: centralized vs decentralized vs CPFA baseline table
- [ ] Ablation table: with and without pheromone
- [ ] Architecture diagram showing the CTDE obs flow
- [ ] Demo video: 4 robots coordinating via P2P pheromone with no central supervisor

### Future / Optional
- [ ] Further decentralized training if v1 underperforms (tune reward scales)
- [ ] Physical robot deployment test (replace simulated GPS/camera with real sensors)

---

## Key Files

| Purpose | File |
|---------|------|
| Centralized training | `controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_shaping.py` |
| Centralized evaluation | `controllers/eval_best_model/eval_best_model.py` |
| Centralized robot | `controllers/epuck_driver/epuck_driver.py` |
| Decentralized training | `controllers/decentralized_supervisor/decentralized_supervisor.py` |
| Decentralized robot | `controllers/epuck_decentralized/epuck_decentralized.py` |
| Centralized training world | `worlds/epuck_foraging_shaping.wbt` |
| Centralized eval world | `worlds/eval_best.wbt` |
| Decentralized world | `worlds/epuck_foraging_decentralized.wbt` |
| Best centralized model | `ppo_v16_phero.zip` |
| Decentralized model (in training) | `decentralized_optA_v1.zip` |

For how to run training and evaluation, see `TRAINING_GUIDE.md` and `TESTING_GUIDE.md`.
