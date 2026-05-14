# Training Guide — CPFA-RL Centralized (CoRL 2026)

Train 4 e-puck robots using PPO with CPFA's exact pheromone model as the centralized baseline for the CoRL 2026 paper.

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
pip install "stable-baselines3" "numpy<2" torch gymnasium
```

---

## Files Overview

| File | Role |
|------|------|
| `worlds/epuck_foraging_shaping_5x5.wbt` | Training world — 5×5m, 4 robots, 64 tags, `basicTimeStep 32` |
| `controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_cpfa.py` | Extern supervisor — PPO training loop with CPFA pheromone model |
| `controllers/epuck_driver/epuck_driver.py` | Robot controller — runs inside Webots, sends proximity sensors, receives motor commands |

---

## Training

**Step 1: Launch Webots**

```bash
webots --mode=fast worlds/epuck_foraging_shaping_5x5.wbt &
sleep 10
```

**Step 2: Run the extern supervisor** (from project root)

```bash
python3 controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_cpfa.py \
    --run_name ppo_cpfa_5x5 \
    --total_timesteps 3000000
```

**Resume from checkpoint:**

```bash
python3 controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_cpfa.py \
    --run_name ppo_cpfa_5x5 \
    --total_timesteps 3000000 \
    --resume logs/ppo_cpfa_5x5/ppo_cpfa_5x5_1000000_steps.zip
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--run_name` | `ppo_cpfa_5x5` | Output model name |
| `--lr` | `3e-4` | Learning rate |
| `--ent_coef` | `0.15` | Entropy coefficient |
| `--batch_size` | `1024` | PPO minibatch size |
| `--total_timesteps` | `3000000` | Total training steps (~366 episodes at 8192 steps/ep) |
| `--resume` | None | Path to `.zip` checkpoint to resume from |

### PPO Architecture

```
net_arch   = [256, 256]
activation = Tanh
n_steps    = 8192          # one full episode per rollout
gamma      = 0.99
gae_lambda = 0.95
clip_range = 0.2
```

---

## Observation Space (21D per robot, 84D total)

```
[0:8]  proximity sensors (8 values, normalised by 4096)
[8]    tag_visible       — nearest tag in FOV (1.0m range, ±1.2 rad)
[9]    tag_dist_norm     — distance to nearest tag / 1.0m
[10]   tag_angle_norm    — signed angle to nearest tag / π
[11]   carrying          — 1 if holding food, else 0
[12]   dist_to_base_norm — distance to nest / 3.5m
[13]   angle_to_base_norm — signed angle to nest / π
[14]   site_known        — 1 if site fidelity target assigned at nest
[15]   site_dist_norm    — distance to site fidelity target / 3.5m
[16]   site_angle_norm   — signed angle to site fidelity target / π
[17]   phero_known       — 1 if pheromone target assigned at nest (roulette-wheel)
[18]   phero_dist_norm   — distance to pheromone target / 3.5m
[19]   phero_angle_norm  — signed angle to pheromone target / π
[20]   search_duration_norm — steps without pickup / 700 (give-up signal for PPO)

obs[14-20] zeroed when carrying=True
```

---

## CPFA Pheromone Model

The pheromone model is **identical to CPFA** — PPO replaces only the 4-state machine navigation.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `RATE_OF_LAYING_PHEROMONE` | 3.0 | Poisson λ: higher density → more likely to lay |
| `RATE_OF_SITE_FIDELITY` | 3.0 | Poisson λ: higher density → more likely to return |
| `RATE_OF_PHEROMONE_DECAY` | 0.01 /sec | Exponential decay: `weight *= exp(-0.01 * dt_sec)` |
| `PHEROMONE_MIN` | 0.001 | Entries pruned below this weight |
| `SEARCH_GIVE_UP_STEPS` | 500 | Steps without pickup before give-up reward kicks in |
| `SEARCH_DURATION_NORM` | 700 | obs[20] saturates at 1.0 after this many steps |

Pheromone is a **list** of `{x, y, weight, resource_density}` entries. Created at nest deposit via Poisson CDF gate. Robot receives its nest target (site fidelity or pheromone roulette-wheel) only when it returns to the nest — identical to CPFA's nest-only information model.

---

## Hard-Coded Overrides

| Override | Condition | Action | CPFA Equivalent |
|----------|-----------|--------|-----------------|
| **P1 Wall escape** | `wall < 0.35m` or `max(prox) > 0.55` | Steer to centre, gain 4.0 | Collision avoidance |
| **P2 Return to base** | `carrying = True` | Steer to nest, gain 2.5 | CPFA RETURNING state |
| P3 Base avoidance | **Removed** — reward penalty used instead | — | — |

PPO learns everything else: tag approach, site fidelity navigation, pheromone following, arena exploration, and give-up timing.

---

## Curriculum

Training uses a 3-phase distance curriculum (~366 total episodes):

| Phase | Episodes | Cluster distance | Clusters | Share |
|-------|----------|-----------------|----------|-------|
| 1 — Close | 1–59 | up to 1.0m | 2 | 16% |
| 2 — Medium | 60–199 | up to 1.8m | 3–5 | 38% |
| 3 — Full | 200+ | up to 2.3m | 6–8 | 46% |

Tags are distributed as 64 items across clusters using Gaussian scatter (σ=0.20m, clamped to ±2.3m).

---

## Per-Episode Monitoring

The supervisor prints a summary at each episode end and every 500 steps to `training_log.txt`:

```
=================================================================
[EP 42] Picks: 18 | Deps: 14 | Rate: 3.21 tags/min (sim) | TotalDeps: 312
  Curriculum: MEDIUM  (3-5 clusters, max_dist=1.8m)
  Pheromone:  entries=7 | max_weight=0.843
  R1[PHERO  ]: carry=0 | base=1.42 | tag_vis=0 | site=0 sd=0.00 | phero=1 pd=1.12 | search=0.23
  R2[RTB    ]: carry=1 | base=0.67 | ...
=================================================================
```

**MODE values:**
- `WALL_ESC` — P1 active
- `RTB` — P2 active (carrying, returning to nest)
- `SITE` — PPO navigating toward site fidelity target
- `PHERO` — PPO navigating toward pheromone roulette target
- `EXPLORE` — PPO in free exploration (no nest target assigned)

**What to watch:**
- `Rate` should increase across episodes (target: beat CPFA baseline)
- `entries` in pheromone list should grow after early episodes
- Robots should cycle through `SITE`/`PHERO` → `RTB` → `SITE`/`PHERO` patterns
- `search` values near 1.0 indicate give-up behaviour is engaging

---

## Output Files

```
./
├── ppo_cpfa_5x5.zip                         # Final trained model
├── training_log.txt                         # Per-step and per-episode log
└── logs/ppo_cpfa_5x5/
    ├── ppo_cpfa_5x5_200000_steps.zip        # Checkpoint every 200k steps
    ├── ppo_cpfa_5x5_400000_steps.zip
    └── ...
```

---

## SB3 Metrics to Watch

| Metric | Early training | Healthy |
|--------|---------------|---------|
| `ep_rew_mean` | negative | increasing toward positive |
| `explained_variance` | near 0 | climbing toward 0.8+ |
| `entropy_loss` | high (−2.8) | gradually decreasing |
| `value_loss` | high | stabilising then decreasing |

---

## Troubleshooting

**No `[PICKUP]` events in first 5 minutes** — Phase 1 clusters too sparse for the random policy. Normal. First pickups usually appear around episode 3–5.

**`ep_rew_mean` stays strongly negative** — Increase `--ent_coef 0.20` for more exploration early.

**Robots stuck in WALL_ESC for entire episodes** — Check P1 threshold (`wall_dist < 0.35`). Should be based on `2.5 - max(abs(x), abs(y))` for the 5×5m arena.

**`numpy.dtype size changed`** — Run `pip install "numpy<2"` to fix NumPy 2.x binary incompatibility.

**Webots disconnects during training** — Ensure Webots is fully loaded before running the supervisor (`sleep 10` after launch).