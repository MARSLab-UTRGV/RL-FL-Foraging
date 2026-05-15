# Training Guide — Multi-Agent E-puck Foraging

This project supports two training modes:
- **Centralized** — single shared PPO in supervisor, 21D per-robot obs (84D total), supervisor-side CPFA pheromone list
- **Fully Decentralized Independent** — 4 independent PPOs (one per robot), 20D local obs, P2P pheromone + optional gossip FL (CoRL 2026)

---

## Prerequisites

Add to `~/.bashrc` (or run before each session):

```bash
export WEBOTS_HOME=/usr/local/webots
export PYTHONPATH=$PYTHONPATH:$WEBOTS_HOME/lib/controller/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WEBOTS_HOME/lib/controller
```

Install dependencies:

```bash
pip install "stable-baselines3" "numpy<2" deepbots torch gym
```

---

## Files Overview

### Centralized

| File | Role |
|------|------|
| `worlds/epuck_foraging_shaping_5x5.wbt` | Training world (DO NOT modify) |
| `controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_cpfa.py` | Extern supervisor — PPO + rewards + pheromone |
| `controllers/epuck_driver/epuck_driver.py` | Robot controller — prox sensors only, receives motor commands |

### Fully Decentralized Independent

| File | Role |
|------|------|
| `worlds/epuck_foraging_decentralized.wbt` | Training world — each robot has GPS, IMU, phero emitter/receiver |
| `controllers/decentralized_supervisor/decentralized_supervisor.py` | Thin shim — tag mechanics, camera sim, episode resets only. No PPO, no rewards |
| `controllers/epuck_decentralized_train_v4/epuck_decentralized_train_v4.py` | Training robot — runs own SB3 PPO, computes own reward, gossip FL |
| `controllers/epuck_decentralized/epuck_decentralized_v4.py` | Base class — GPS, IMU, prox, CPFA pheromone list, P2P broadcast |

---

## Centralized Training

**Step 1: Launch Webots in fast mode**

```bash
webots --mode=fast worlds/epuck_foraging_shaping_5x5.wbt &
sleep 10
```

**Step 2: Run the extern supervisor** (from project root)

```bash
cd controllers/epuck_foraging_supervisor_shaping
python3 epuck_foraging_supervisor_cpfa.py \
    --run_name ppo_cpfa_5x5 \
    --total_timesteps 7000000 \
    --lr 3e-4 \
    --ent_coef 0.15
```

**Resume from checkpoint:**

```bash
python3 epuck_foraging_supervisor_cpfa.py \
    --run_name ppo_cpfa_5x5 \
    --total_timesteps 7000000 \
    --resume ../../ppo_cpfa_5x5.zip
```

### Centralized Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--run_name` | `ppo_cpfa_5x5` | Output model name |
| `--lr` | `3e-4` | Learning rate |
| `--ent_coef` | `0.15` | Entropy coefficient |
| `--batch_size` | `1024` | PPO minibatch size |
| `--total_timesteps` | `7000000` | Total training steps |
| `--resume` | None | Path to `.zip` to resume from |

### Centralized Obs Space (21D per robot, 84D total)

```
[0:8]  proximity sensors (÷4096)
[8]    tag_visible         (supervisor camera sim)
[9]    tag_dist_norm       (÷ 1.0m)
[10]   tag_angle_norm      (÷ π)
[11]   carrying            (0/1)
[12]   base_dist_norm      (÷ 3.5m)
[13]   base_angle_norm     (÷ π)
[14]   site_known          (CPFA site fidelity target assigned at nest)
[15]   site_dist_norm      (÷ 3.5m)
[16]   site_angle_norm     (÷ π)
[17]   phero_known         (CPFA pheromone roulette target assigned at nest)
[18]   phero_dist_norm     (÷ 3.5m)
[19]   phero_angle_norm    (÷ π)
[20]   search_duration_norm (steps since last pickup ÷ 700 — give-up signal)
```

---

## Fully Decentralized Independent Training

Each robot runs its own SB3 PPO onboard. The supervisor is a thin simulation shim — it handles only tag mechanics, camera simulation, and episode resets. No PPO, no rewards, no pheromone grid.

**Step 1: Write run name and launch Webots**

```bash
cd "/home/sara/Documents/Centralized Learning/RL-FL-Foraging"
echo "decentralized_indep_v4" > current_run_name.txt
webots --mode=fast worlds/epuck_foraging_decentralized.wbt &
sleep 10
```

**Step 2: Run the thin supervisor** (from project root)

```bash
cd controllers/decentralized_supervisor
python3 decentralized_supervisor.py --run_name decentralized_indep_v4
```

The supervisor writes the run name to `current_run_name.txt`. Each robot reads it at init and saves models as `robot{N}_{run_name}.zip`.

### Training Parameters (hardcoded in robot controller)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TOTAL_STEPS` | 7,000,000 | Per-robot training steps |
| `N_STEPS` | 16,384 | Rollout buffer size = steps per episode |
| `BATCH_SIZE` | 256 | PPO minibatch size |
| `N_EPOCHS` | 10 | PPO epochs per update |
| `GAMMA` | 0.99 | Discount factor |
| `ENT_COEF` | 0.15 | Entropy coefficient |
| `LR` | 3e-4 | Learning rate |
| `GOSSIP_ALPHA` | 0.2 | FedAvg weight for gossip FL |

### Decentralized Obs Space (20D per robot — all local)

```
[0:8]  proximity sensors (÷4096)          ← robot onboard
[8]    tag_visible                         ← supervisor (camera sim → real camera at deploy)
[9]    tag_dist_norm     (÷ 1.0m)          ← supervisor
[10]   tag_angle_norm    (÷ π)             ← supervisor
[11]   carrying          (0/1)             ← robot onboard
[12]   base_dist_norm    (GPS ÷ 3.5m)      ← robot GPS
[13]   base_angle_norm   (GPS+IMU ÷ π)     ← robot GPS + InertialUnit
[14]   site_known        (own last pickup) ← robot onboard
[15]   site_dist_norm    (÷ 3.5m)          ← robot onboard
[16]   site_angle_norm   (÷ π)             ← robot onboard
[17]   phero_known       (roulette target) ← robot P2P receiver
[18]   phero_dist_norm   (÷ 3.5m)          ← robot P2P receiver
[19]   phero_angle_norm  (÷ π)             ← robot P2P receiver
```

14/20 dims computed fully onboard. Supervisor provides only [8–10] (camera sim). At physical deployment, replace [8–10] with onboard camera + AprilTag detector.

### Pheromone Design (CPFA-style, fully P2P)

- **Created at pickup** with density-based weight: `0.2 + 0.8 × min(nearby_tags / 5, 1.0)`
- **List-based** (one entry per distinct cluster, `MERGE_RADIUS=0.3 m`)
- **Broadcast:** strongest entry every step via `phero_emitter` (channel 10, 2 m range)
- **Receive:** accept-if-stronger per cluster location — prevents stale overwrites
- **Decay:** `weight × exp(−0.01 × dt_sec)`, pruned when weight < 0.001
- **Site fidelity:** probabilistic return to own last pickup (probability = pickup_signal)
- **Roulette selection:** weighted random pick from received cluster list at deposit

### Target Assignment at Deposit (CPFA Priority Order)

1. **Site fidelity** — `random() < last_pickup_weight` → return to own last pickup cluster
2. **Pheromone roulette** — weighted random from pheromone list (higher weight = more likely)
3. **Free exploration** — PPO learns efficient search (beats CPFA random walk)

### Target Depletion Feedback

If robot arrives at target cluster (< 0.18 m) with no pickup:
- Site fidelity: halve `last_pickup_weight` (floor 0.05) → reduces future return probability
- Pheromone: halve matching list entry weight → roulette shifts to fresher clusters

### Hard-coded Overrides (P1 / BASE_ESC / P2)

Applied after PPO inference. Rewards fire normally regardless of which override is active.

| Override | Condition | Action |
|----------|-----------|--------|
| **P1 Wall escape** | wall_dist < 0.35 m or max(prox) > 0.55 | Steer to centre (gain=4.0) |
| **BASE_ESC** | not carrying and dist_to_base < 0.25 m | Nudge outward 0.5 m (gain=4.0) |
| **P2 Return to base** | carrying=True | Steer to centre (gain=2.5) |

No P3, no P4 — all behavioral decisions are PPO-learned.

### Gossip Federated Learning (channel 11, 2 m)

After each PPO update, each robot:
1. Broadcasts its full policy weights (base64-encoded pickle, ~750 KB)
2. Merges any received neighbor weights: `merged = (1 − α) × local + α × neighbor` (α=0.2)

Gossip is hardware-range-limited (2 m) — only nearby robots share models, not all 4 globally.

### Reward Structure (computed fully onboard)

| Component | Value |
|-----------|-------|
| Tag pickup | +5.0 |
| Tag deposit | +20.0 |
| Tag approach shaping | ×8.0 (÷ distance reduction) |
| Site/phero target approach | ×15.0 |
| Orientation toward target | ×0.5 (cosine) |
| Zone reward (0.8–2.4 m from base) | +0.10/step |
| Near-base penalty (< 0.8 m) | −(0.8 − dist) × 4.0 |
| Exploration (no target, dist > 0.5 m) | +min(dist/2.3, 1.0) × 0.15 |
| Wall proximity penalty (< 0.35 m) | −(0.35 − wall_dist) × 0.5 |
| Obstacle proximity penalty | −max_prox × 0.5 |
| Time penalty | −0.005/step |

### Curriculum (supervisor-controlled tag placement)

| Phase | Episodes | Clusters | Max radius |
|-------|----------|----------|------------|
| Close | ep < 60 (14%) | 2 | 1.0 m |
| Medium | ep < 201 (33%) | 3–5 | 1.8 m |
| Full | ep ≥ 201 (53%) | 6–8 | 2.3 m |

Tag placement: rotated grid (0.10 m uniform spacing, random orientation 0–π/2 per episode). Matches eval world geometry.

### Episode Sync

- `STEPS_PER_EPISODE = N_STEPS = 16384` — buffer fills exactly once per episode
- Supervisor and robots independently count steps — no special reset signal
- 1 PPO update per episode, gossip after each update
- ~427 total episodes at 7M steps

---

## Per-Episode Monitoring

### Supervisor terminal

```
=================================================================
[EP 87] Picks: 12 | Deps: 9 | Rate: 1.04 tags/min (sim) | TotalDeps: 412 | Wall: 5.2 min
  Curriculum: MEDIUM  (3-5 clusters, max_dist=1.8m)
  Pheromone:  P2P per-robot (not tracked by supervisor)
  R1[PPO     ]: carry=0 | base=1.43 | wall=1.87
  R2[RTB     ]: carry=1 | base=0.82 | wall=2.11
  R3[WALL_ESC]: carry=0 | base=2.21 | wall=0.28
  R4[BASE_ESC]: carry=0 | base=0.18 | wall=2.31
=================================================================
```

**Mode labels:** `WALL_ESC` (P1 active) | `BASE_ESC` (post-deposit nudge) | `RTB` (P2, carrying) | `PPO` (free policy)

### Per-robot log (`logs/robot{N}_decentralized_indep_v4/training_log.txt`)

```
[PICKUP] robot1 at (1.23,-0.45) | strength=0.68 approx_density~3 | Ep picks: 5
  [PHERO] Added (1.23,-0.45) weight=0.68 | list_size=2
[DEPOSIT] robot1 | Ep deps: 3
  [SITE_FID] last_pickup=(1.23,-0.45) weight=0.68
  [TARGET]   → SITE (1.23,-0.45)
───────────────────────────────────────────
| train/approx_kl        0.01136665       |
| train/clip_fraction         0.0412      |
| train/entropy_loss          -94.4       |
| train/explained_var          0.888      |
| train/loss                  -2.8800     |
| train/value_loss              33.6      |
───────────────────────────────────────────
[robot1] EP END | Picks: 5 | Deps: 3 | Steps: 212992 | Updates: 13 | GossipMerges: 2 | phero_entries=2 max_w=0.412 | target=SITE
```

**What to look for:**
- `[TARGET] → SITE/PHERO` appearing after deposits → site fidelity and pheromone working
- `list_size` growing over episode → pheromone accumulating from neighbors
- `explained_var` climbing toward 0.8+ → value function learning well
- `Deps` per episode increasing over training → foraging improving
- `GossipMerges` > 0 → model sharing active

---

## Output Files

```
./
├── robot1_{run_name}.zip                         # Latest model — robot 1
├── robot2_{run_name}.zip                         # Latest model — robot 2
├── robot3_{run_name}.zip                         # Latest model — robot 3
├── robot4_{run_name}.zip                         # Latest model — robot 4
└── logs/
    ├── robot1_{run_name}/
    │   ├── training_log.txt                      # Per-step events + PPO stats
    │   ├── robot1_{run_name}_50000_steps.zip     # Checkpoint every 50k steps
    │   └── ...
    ├── robot2_{run_name}/
    │   └── ...
    └── {supervisor_log}/
        └── supervisor_log.txt                    # Episode summaries
```

---

## Trained Models (Current)

| Model | Type | Steps | Status |
|-------|------|-------|--------|
| `ppo_cpfa_5x5.zip` | Centralized | 7M | Complete — ~11 tags/min |
| `decentralized_optA_v4.zip` | CTDE (archived) | 7M | Complete — eval pending |
| `robot{1-4}_decentralized_indep_v4.zip` | Independent | 7M | **In training** |

---

## Troubleshooting

**`Device "gps" was not found`** — Each E-puck in the decentralized world must use `turretSlot` (not `extensionSlot`). Check `worlds/epuck_foraging_decentralized.wbt`.

**`numpy.dtype size changed`** — Run `pip install "numpy<2"` to fix NumPy 2.x binary incompatibility.

**Robots not moving** — Webots simulation must be running (press Play). The extern supervisor connects after Webots loads.

**`[PICKUP]` events not appearing in first episodes** — Normal in Phase 1 (close clusters). If absent after 20+ episodes, check that `current_run_name.txt` was written before Webots started.

**No `[TARGET] → SITE/PHERO` after deposits** — Check that `_assign_target()` is being called. Should appear if `[DEPOSIT]` lines are present.

**`GossipMerges: 0` every episode** — Gossip emitter/receiver devices may not be in the world file. Check that `model_emitter` and `model_receiver` (channel 11, range 2 m) are in each robot's `turretSlot`.

**Low `explained_variance` (< 0.3) after 50 episodes** — Value function not learning. Check that reward signals are firing: `[PICKUP]` and `[DEPOSIT]` lines must appear in logs.

**All robots show `BASE_ESC` constantly** — BASE_ESC threshold is 0.25 m (very small). If this persists, check that robots are being respawned correctly at episode reset (positions [±0.5, 0], [0, ±0.5]).
