# Project Progress — Multi-Agent E-puck Foraging (CoRL 2026)

**Last updated:** 2026-05-15

---

## Project Goal

Train 4 e-puck robots to autonomously forage 64 AprilTags scattered in clusters inside a 5×5 m Webots arena, simulating a **rescue mission** where robots must find and collect items from unknown cluster locations.

The core research question: **can fully decentralized RL — where each robot trains its own independent policy from local experience only — match or outperform centralized RL and the CPFA baseline?**

The **CPFA baseline (5.94 tags/min)** represents classical swarm performance. The centralized version (~11 tags/min) is the upper-bound reference. The decentralized version must coordinate exclusively via peer-to-peer pheromone with no shared model, no global information, and no central supervisor.

The **CoRL 2026 contribution** is a **fully decentralized independent PPO** architecture where 4 robots each train their own policy from their own experience, coordinate via P2P pheromone (2 m range), and optionally share model weights via gossip federated learning. 14/20 observation dimensions are computed entirely onboard — the policy is deployable on physical robots without architectural change.

---

## Three Versions

### 1. Centralized (Performance Upper Bound)
Single shared PPO policy trained and run by the supervisor. The supervisor maintains a global pheromone list and provides all 21D observations per robot (84D total). Robots are simple sensor/actuator bridges.

**Status: Complete. Best model `ppo_cpfa_5x5.zip` (~11 tags/min).**

### 2. CTDE — Centralized Training, Decentralized Execution (Archived)
Single shared PPO trained centrally, deployed locally on each robot. Robots compute 18D observations onboard using GPS/IMU/P2P pheromone. Archived as `decentralized_optA_v1.zip` through `decentralized_optA_v4.zip`.

**Status: Archived. v4 = best CTDE result.**

### 3. Fully Decentralized Independent PPO (CoRL 2026 Contribution)
Each robot trains its own independent PPO from its own local experience only. No shared rollout buffer, no parameter sharing during training except optional gossip FL after each update. Robots coordinate exclusively via P2P pheromone. The supervisor is a thin simulation shim — no PPO, no rewards.

**Status: `decentralized_indep_v4` — IN TRAINING (7M steps, ~427 episodes).**

---

## How It Works

**Exploration:** Each robot independently learns when and where to explore. When no target is known, PPO is rewarded for moving outward from base.

**Pheromone (CPFA-style, fully P2P):**
- At tag pickup: robot adds cluster location to its pheromone list with a density-based weight (0.2–1.0)
- Each step: broadcasts strongest known cluster to neighbors within 2 m (channel 10)
- Each step: receives broadcasts from neighbors, merges by location (accept-if-stronger)
- Decays exponentially (rate=0.01/sec), pruned when weight < 0.001

**Target assignment (CPFA Poisson gate — at deposit):**
1. **Site fidelity** — probabilistic return to own last pickup (probability = pickup_signal 0.2–1.0)
2. **Pheromone roulette** — weighted random selection from received cluster list
3. **Free exploration** — PPO learns efficient search (advantage over CPFA random walk)

**Gossip FL (optional, channel 11, 2 m):** After each PPO update, robots broadcast their model weights and merge with any received neighbor weights (FedAvg, α=0.2).

**Hard-coded overrides (after PPO inference, reward fires regardless):**
- **P1:** wall_dist < 0.35 m or max(prox) > 0.55 → steer to centre (gain=4)
- **BASE_ESC:** not carrying and dist_to_base < 0.25 m → nudge outward 0.5 m (gain=4)
- **P2:** carrying → steer to centre (gain=2.5)
- No P3, No P4

---

## Current Status

### Centralized — Complete
- 7M steps, `ppo_cpfa_5x5.zip`
- Performance: ~11 tags/min (~+85% over CPFA baseline)

### CTDE (archived) — v1–v4 Complete
- `decentralized_optA_v4.zip` — best CTDE model (7M steps), eval pending

### Independent Training — IN PROGRESS
- `decentralized_indep_v1`, `v2` — completed earlier runs
- **`decentralized_indep_v4` — currently training** (7M steps, ~427 episodes, STEPS_PER_EPISODE=16384)
- Architecture: 4 independent PPOs (2×256 Tanh, lr=3e-4, ent=0.15, batch=256), 1 update per episode
- Curriculum: ep<60 close, ep<201 medium, ep≥201 full arena

---

## Independent Training History

### decentralized_indep_v1, v2
Earlier independent training runs. Architecture not yet aligned with centralized version (old overrides, Gaussian tag scatter, 3M steps).

### decentralized_indep_v4 (current)
All design decisions aligned with centralized CPFA supervisor:
- **BASE_ESC** (0.25 m, 0.5 m nudge) replaces old P3 wide push — matches centralized exactly
- **No P3** — PPO controls all post-deposit navigation
- **Rotated grid tag placement** (0.10 m uniform spacing, random orientation per episode) — matches eval world geometry, eliminates sim-to-eval tag distribution gap
- **CPFA list-based pheromone** (one entry per cluster, roulette selection, site fidelity gate)
- **Target depletion feedback** — failed arrival at cluster halves weight, accelerates natural decay
- **Curriculum** calibrated to 427 episodes (14%/33%/53% = ep<60/ep<201/ep≥201)
- **7M steps** — matches centralized total

---

## CTDE Failure Analysis (Archived)

### CTDE v1 — 3.1 tags/min (below CPFA 5.94)
- Robots orbited base at 0.23 m during early episodes (base-orbit optimum via P4)
- All 4 robots transitioned to WALL_ESC after nearby tags depleted
- Root cause: P3 multiplier=3.0 pushed to 0.72 m (below 0.8 m exploration zone) → pheromone following never learned

### CTDE v2 — Wall-hugging
- P1=0.35 m, P3=0.6 m — too permissive, robots navigated near walls

### CTDE v3 — Pheromone not learned
- P3 target 0.72 m (multiplier=3.0) still below active zone → same root cause as v1

### CTDE v4 — Best CTDE (7M steps, eval pending)
- P3 target 1.2 m (multiplier=5.0), INITIAL_TTL=2000, phero approach ×5.0, near-base ×4.0

---

## What Remains

### Immediate
- [ ] Wait for `decentralized_indep_v4` training to complete (~427 episodes)
- [ ] Evaluate `decentralized_indep_v4` — measure tags/min per robot and combined
- [ ] Evaluate `decentralized_optA_v4.zip` — compare CTDE vs independent vs centralized
- [ ] Run 5+ evaluation seeds (30 min each) for statistical significance

### Analysis
- [ ] Ablation: gossip FL ON vs OFF — quantify model sharing contribution
- [ ] Ablation: pheromone ON vs OFF — quantify coordination contribution
- [ ] Compare independent vs CTDE learning curves

### Paper (CoRL 2026)
- [ ] Results table: centralized vs CTDE vs independent vs CPFA baseline
- [ ] Architecture diagram: independent PPO + P2P pheromone + gossip FL
- [ ] Deployment analysis: which obs dims require supervisor vs onboard only
- [ ] Demo video: 4 robots coordinating via P2P pheromone, no central supervisor

---

## Performance Summary

| Model | Type | Tags/min | vs CPFA | Status |
|-------|------|----------|---------|--------|
| CPFA baseline | Classical | 5.94 | — | Reference |
| `ppo_cpfa_5x5.zip` | Centralized | ~11 | +85% | Complete |
| `decentralized_optA_v4.zip` | CTDE | TBD | TBD | Eval pending |
| `decentralized_indep_v1.zip` | Independent | TBD | TBD | Completed |
| `decentralized_indep_v2.zip` | Independent | TBD | TBD | Completed |
| `robot{1-4}_decentralized_indep_v4.zip` | Independent | TBD | TBD | **In training** |

---

## Key Files

| Purpose | File |
|---------|------|
| Centralized training supervisor | `controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_cpfa.py` |
| Centralized robot | `controllers/epuck_driver/epuck_driver.py` |
| Centralized eval | `controllers/eval_best_model/eval_best_model_5x5.py` |
| Independent training supervisor (thin shim) | `controllers/decentralized_supervisor/decentralized_supervisor.py` |
| Independent training robot | `controllers/epuck_decentralized_train_v4/epuck_decentralized_train_v4.py` |
| Robot base class (sensors + pheromone) | `controllers/epuck_decentralized/epuck_decentralized_v4.py` |
| Decentralized eval supervisor | `controllers/eval_decentralized/eval_decentralized.py` |
| Decentralized eval robot | `controllers/epuck_decentralized_eval/epuck_decentralized_eval.py` |
| Centralized training world | `worlds/epuck_foraging_shaping_5x5.wbt` |
| Independent training world | `worlds/epuck_foraging_decentralized.wbt` |
| Decentralized eval world | `worlds/eval_decentralized.wbt` |

For how to run training and evaluation, see `TRAINING_GUIDE.md` and `TESTING_GUIDE.md`.
