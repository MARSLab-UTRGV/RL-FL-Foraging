# Training Guide — CPFA-RL Centralized (CoRL 2026)

Train 4 e-puck robots using PPO with CPFA's exact pheromone model. PPO replaces CPFA's
hand-coded 4-state machine while keeping the full pheromone infrastructure identical to
the baseline. Any performance gap is purely attributable to the learned policy.

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
pip install stable-baselines3 torch gymnasium
```

---

## Files Overview

| File | Role |
|------|------|
| `worlds/epuck_foraging_shaping_5x5.wbt` | Training world — 5×5m, 4 robots, 64 tags, `basicTimeStep 64ms` |
| `controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_cpfa.py` | Extern supervisor — PPO training loop + CPFA pheromone system |
| `controllers/epuck_driver/epuck_driver.py` | Robot controller — sends 8 proximity sensors, receives [left, right] motor commands |

---

## Training

**Step 1: Launch Webots**

```bash
webots --mode=fast worlds/epuck_foraging_shaping_5x5.wbt &
sleep 10
```

**Step 2: Run the extern supervisor** (from project root)

```bash
WEBOTS_PORT=1234 python3 controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_cpfa.py \
    --run_name ppo_cpfa_v5 \
    --total_timesteps 10000000
```

**Resume from checkpoint:**

```bash
WEBOTS_PORT=1234 python3 controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_cpfa.py \
    --run_name ppo_cpfa_v5 \
    --total_timesteps 10000000 \
    --resume logs/ppo_cpfa_v5/ppo_cpfa_v5_2000000_steps.zip
```

### Training Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--run_name` | `ppo_cpfa_v5` | Output model name / checkpoint folder |
| `--lr` | `3e-4` | Learning rate |
| `--ent_coef` | `0.03` | Entropy coefficient — chosen so entropy (0.168/step) maintains exploration in free search while DEPARTING signal (0.65/step) dominates navigation |
| `--batch_size` | `1024` | PPO minibatch size |
| `--total_timesteps` | `10000000` | 10M steps ≈ 610 episodes at 16384 steps/ep |
| `--resume` | None | Path to `.zip` checkpoint to resume from |

### PPO Architecture

```
net_arch     = [256, 256]
activation   = Tanh
n_steps      = 16384       # one full episode per rollout (~17.5 min sim time)
batch_size   = 1024
gamma        = 0.99
gae_lambda   = 0.95
clip_range   = 0.2
vf_coef      = 0.5
max_grad_norm = 0.5
device       = cpu
```

Checkpoints saved every 200,000 steps to `logs/ppo_cpfa_v5/`.

---

## Observation Space (18D per robot, 72D total)

Tag sensing removed entirely — RL advantage must come from learned pheromone-based
navigation, not a sensing advantage over CPFA baseline.

```
[0:8]   proximity sensors             8 values, normalised to [0, 1] by ÷ 4096
[8]     carrying                      1.0 if holding food, else 0.0
[9]     dist_to_base_norm             distance to nest / 3.5m
[10]    angle_to_base_norm            signed angle to nest / π   (range [-1, 1])
[11]    site_known                    1.0 if site fidelity target assigned at nest
[12]    site_dist_norm                distance to site target / 3.5m
[13]    site_angle_norm               signed angle to site target / π
[14]    phero_known                   1.0 if pheromone target assigned at nest
[15]    phero_dist_norm               distance to pheromone target / 3.5m
[16]    phero_angle_norm              signed angle to pheromone target / π
[17]    search_duration_norm          steps_without_pickup / 4000  (saturates at 1.0)
                                      give-up signal: high value = long failed search

obs[11–17] zeroed when carrying=True
```

Action space: `[left_motor, right_motor]` per robot, range `[-1, 1]`, clipped in driver to
`±6.28 rad/s` with a 6.0× scale factor.

---

## Reward Shaping — Complete Definition

All rewards summed into a single scalar per environment step across all 4 robots.

### Always Active (all modes, per robot)

| Component | Condition | Value | Purpose |
|-----------|-----------|-------|---------|
| **Proximity penalty** | `max(prox) > 0.1` | `−max(prox) × 0.5` (0 to −0.5) | Discourages obstacle collisions |
| **Wall penalty** | `wall_dist < 0.35m` | `−(0.35 − wall_dist) × 0.5` (0 to −0.175) | Fires with P1 override; teaches wall avoidance |
| **Time penalty** | every step | `−0.005` | Penalises idle time; encourages efficiency |

### Exploration Branch (not carrying, per robot)

| Component | Condition | Value | Purpose |
|-----------|-----------|-------|---------|
| **Pickup reward** | tag within 0.15m | `+5.0` | Primary food-finding signal |
| **Forward motion bias** | `wall_dist ≥ 0.35m` AND `avg_speed > 0` | `avg_speed × 0.15` (0 to +0.15) | Requires actual movement — standing still earns nothing |
| **SITE approach shaping** | target is SITE | `(prev_site_dist − curr_site_dist) × 15.0` | Dense step-by-step gradient toward site fidelity target |
| **SITE orientation reward** | target is SITE AND `dist > 0.001m` | `dot(forward, target_dir) × 0.5` (−0.5 to +0.5) | Rewards facing toward target; combined with approach ≈ 0.65/step |
| **PHERO approach shaping** | target is PHERO | `(prev_phero_dist − curr_phero_dist) × 15.0` | Same structure as SITE shaping |
| **PHERO orientation reward** | target is PHERO AND `dist > 0.001m` | `dot(forward, target_dir) × 0.5` | Rewards facing toward pheromone target |

Notes on approach shaping:
- Seeded at nest deposit so reward fires from the first DEPARTING step (no cold-start spike).
- Cleared at cluster arrival (`dist_to_target < 0.05m`) — robot enters free local search.
- Cleared at pickup — robot transitions to carrying.
- `prev_dist = None` guard: reward skipped on the very first step to avoid undefined delta.

### Carrying Branch (per robot)

| Component | Condition | Value | Purpose |
|-----------|-----------|-------|---------|
| **Deposit reward** | `dist_to_base < 0.25m` while carrying | `+20.0` | Primary task-completion signal |
| **RTB approach shaping** | every carry step | `(prev_base_dist − curr_base_dist) × 8.0` | Dense gradient toward nest while returning |
| **RTB orientation reward** | `dist_to_base > 0.001m` | `dot(forward, base_dir) × 0.5` (−0.5 to +0.5) | Rewards facing toward nest |

RTB shaping is seeded at pickup (`prev_base_dists[i] = dist_from_base_at_pickup`) to
prevent a large negative spike on the first carry step.

### Multi-Robot (summed across pairs)

| Component | Condition | Value | Purpose |
|-----------|-----------|-------|---------|
| **Separation penalty** | both robots within 1.5m of base AND inter-robot `sep < 1.0m` | `−(1.0 − sep) × 0.5` per pair | Discourages crowding near nest; encourages spread |

### Reward Budget Summary

| Situation | Net per robot per step | Standing still possible? |
|-----------|------------------------|--------------------------|
| Standing still anywhere | −0.005 | **No — guaranteed loss** |
| Moving forward (free explore) | up to +0.145 | must move |
| DEPARTING to target (direct approach) | ~+0.645 | must move toward target |
| RTB (carrying, direct path) | ~+0.745 | P2 handles steering |
| Pickup event | +5.0 | — |
| Deposit event | +20.0 | — |

There is no positional bonus for being at any distance from the base. The only way to
earn positive net reward per step is actual forward movement.

---

## CPFA Pheromone Model

Identical to the CPFA baseline — PPO replaces only the navigation strategy.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `RATE_OF_LAYING_PHEROMONE` | 3.0 | Poisson λ for deposit gate: higher resource density → higher P(lay) |
| `RATE_OF_SITE_FIDELITY` | 1.376 | ARGoS-evolved value — matches cpfa_baseline exactly |
| `RATE_OF_PHEROMONE_DECAY` | 0.05 /sec | Exponential: `weight *= exp(−0.05 × dt_sec)` → τ ≈ 20s |
| `PHEROMONE_MIN` | 0.001 | Entries pruned below this weight |
| `PROB_RETURN_TO_NEST` | 0.0189 | Give-up probability per 5-second check (identical to ARGoS) |
| `GIVE_UP_CHECK_STEPS` | 78 | 5s at 64ms/step — matches baseline's 156 steps at 32ms |
| `SEARCH_DURATION_NORM` | 4000 | obs[17] saturates at 1.0 near E[give-up] = 4127 steps |

Pheromone is a **list** of `{x, y, weight, resource_density}` entries. Created at nest
deposit (not at pickup) via Poisson CDF gate. Target assigned only at nest return
(site fidelity priority 1, pheromone roulette-wheel priority 2, free explore priority 3).

---

## Hard-Coded Overrides

PPO controls all navigation. Three overrides mirror CPFA's hardwired behaviours exactly:

| Override | Condition | Action | CPFA Equivalent |
|----------|-----------|--------|-----------------|
| **P1 Wall escape** | `wall_dist < 0.35m` OR `max(prox) > 0.55` | Steer to arena centre, gain 4.0 | Collision avoidance |
| **BASE_ESC** | not carrying AND not `gave_up` AND `dist_to_base < 0.25m` | Steer directly away from nest, gain 4.0 | Implicit departure after deposit. Suppressed during give-up (robot must return to nest) |
| **P2 Return to base** | `carrying = True` OR `gave_up = True` | Steer to nest, gain 2.5 | CPFA RETURNING state |

PPO controls everything else: DEPARTING (navigate to SITE/PHERO target), local search at
cluster, give-up timing, and free exploration.

---

## Training Curriculum

4 phases across ~610 total episodes (10M steps ÷ 16384 steps/ep):

| Phase | Episodes | Cluster max distance | Min separation | Duration |
|-------|----------|---------------------|----------------|----------|
| 1 — Near | 1–59 | 1.2m | 0.40m | ~10% |
| 2 — Medium | 60–149 | 1.6m | 0.45m | ~15% |
| 3 — Far | 150–299 | 2.0m | 0.50m | ~25% |
| 4 — Full | 300+ | 2.3m | 0.50m | ~51% |

Fixed layout: 11 clusters per episode, 5 clusters × 8 tags + 6 clusters × 4 tags = 64 tags.
Only cluster distance changes — count and tag count are fixed across all phases.
Wall is at 2.5m; P1 fires at 2.15m on-axis → 2.3m is the practical maximum.

---

## Per-Episode Monitoring

Training log is written to `training_log.txt`. Every 500 steps and at episode end:

```
=================================================================
[EP 28] Picks: 19 | Deps: 19 | Rate: 2.17 tags/min (sim) | TotalDeps: 296
  Curriculum: NEAR    (11 clusters, max_dist=1.2m)
  Pheromone:  entries=2 | max_weight=0.007
  R1[EXPLORE ]: carry=0 | base=0.47 | target=explore | density=5
  R2[PHERO   ]: carry=0 | base=0.47 | target=phero(-0.56,-0.18) | density=4
  R3[EXPLORE ]: carry=0 | base=0.52 | target=explore | density=3
  R4[EXPLORE ]: carry=0 | base=0.31 | target=explore | density=11
=================================================================
```

**MODE values:**
- `WALL_ESC` — P1 active
- `BASE_ESC` — escaping nest (not carrying, dist < 0.25m)
- `RTB` — P2 active (carrying or gave_up, returning to nest)
- `GIVE_UP` — gave_up=True, P2 steering robot home empty-handed
- `SITE` — PPO navigating toward site fidelity target
- `PHERO` — PPO navigating toward pheromone target
- `EXPLORE` — PPO in free exploration (no target assigned)

**SB3 metrics to watch:**

| Metric | Healthy range | Warning |
|--------|--------------|---------|
| `std` | 1.0–2.5 | > 3.0 growing = entropy divergence risk |
| `ep_rew_mean` | improving (less negative) | plateau or regression |
| `explained_variance` | > 0.5 by ep 30, > 0.8 by ep 60 | < 0.3 = value net not learning |
| `approx_kl` | 0.005–0.015 | > 0.05 = policy update too large |
| `entropy_loss` | −14 to −16 | growing toward 0 = policy collapsing |

**Phase 2 transition warning (ep 60):** Deposit rate will drop when clusters jump from
1.2m to 1.6m. This is normal curriculum pressure, not regression.

---

## Output Files

```
./
├── training_log.txt                          # Per-step and per-episode log
└── logs/ppo_cpfa_v5/
    ├── ppo_cpfa_v5_200000_steps.zip          # Checkpoint every 200k steps
    ├── ppo_cpfa_v5_400000_steps.zip
    └── ...
```

---

## Troubleshooting

**`std` growing past 3.0** — Entropy divergence. Reduce `--ent_coef` to 0.01 and restart.
v4 failure mode: ent_coef=0.05 caused std → 540, making policy mean irrelevant.

**No deposits in first 3 episodes** — Normal. Phase 1 clusters are at 0.4–1.2m; random
policy takes a few episodes to stumble onto them.

**`ep_rew_mean` plateau at −10,000** — Value network has learned to predict rewards
but policy hasn't improved. Check `explained_variance`; if > 0.8, the problem is
policy gradient, not critic. Try lowering `--lr` to `1e-4`.

**Port conflict** — If port 1234 is in use by training, run eval on port 1235:
`WEBOTS_PORT=1235 python3 controllers/eval_best_model/eval_best_model_5x5.py`

**`numpy.dtype size changed`** — Run `pip install "numpy<2"` to fix NumPy 2.x binary
incompatibility with older compiled extensions.