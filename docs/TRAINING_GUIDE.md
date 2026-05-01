# Training Guide — Multi-Agent E-puck Foraging

This project supports two training modes:
- **Centralized** — single shared PPO policy, 80D stacked obs, supervisor-side pheromone grid
- **Decentralized (Option A / CTDE)** — single shared PPO policy with parameter sharing, 18D per-robot obs, robot-side GPS/IMU/pheromone (CoRL 2026)

---

## Prerequisites

Add to `~/.bashrc` (or run before each session):

```bash
export WEBOTS_HOME=/usr/local/webots
export PYTHONPATH=$PYTHONPATH:$WEBOTS_HOME/lib/controller/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WEBOTS_HOME/lib/controller
```

Install dependencies (system Python, no venv needed):

```bash
pip install "stable-baselines3" "numpy<2" deepbots torch gym
```

---

## Files Overview

### Centralized

| File | Role |
|------|------|
| `worlds/epuck_foraging_shaping.wbt` | Training world (DO NOT modify) |
| `controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_shaping.py` | Extern supervisor — PPO training loop |
| `controllers/epuck_driver/epuck_driver.py` | Robot controller (runs inside Webots) |

### Decentralized (Option A)

| File | Role |
|------|------|
| `worlds/epuck_foraging_decentralized.wbt` | Training world with GPS/IMU/phero on each robot |
| `controllers/decentralized_supervisor/decentralized_supervisor.py` | Extern supervisor — PPO training loop (SB3 VecEnv, n_envs=4) |
| `controllers/epuck_decentralized/epuck_decentralized.py` | Robot controller — computes GPS/IMU/pheromone obs onboard |

---

## Centralized Training

**Step 1: Launch Webots in fast mode**

```bash
webots --mode=fast worlds/epuck_foraging_shaping.wbt &
sleep 10
```

**Step 2: Run the extern supervisor**

```bash
cd controllers/epuck_foraging_supervisor_shaping
python3 epuck_foraging_supervisor_shaping.py \
    --run_name ppo_v16_retrain \
    --total_timesteps 7000000 \
    --lr 3e-4 \
    --ent_coef 0.10
```

**Resume from checkpoint:**

```bash
python3 epuck_foraging_supervisor_shaping.py \
    --run_name ppo_v16_retrain \
    --total_timesteps 7000000 \
    --resume ../../ppo_v16_retrain.zip
```

### Centralized Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--run_name` | `ppo_v16_retrain` | Output model name |
| `--lr` | `3e-4` | Learning rate |
| `--ent_coef` | `0.10` | Entropy coefficient |
| `--batch_size` | `4096` | PPO minibatch size |
| `--total_timesteps` | `7000000` | Total training steps |
| `--resume` | None | Path to `.zip` to resume from |

### Centralized Obs Space (20D per robot, 80D total)

```
[0:8]  proximity sensors (÷4096)
[8]    tag_visible       (supervisor camera sim)
[9]    tag_dist_norm     (÷ 1.0)
[10]   tag_angle_norm    (÷ π)
[11]   carrying          (0/1)
[12]   base_dist_norm    (÷ 3.5)
[13]   base_angle_norm   (÷ π)
[14]   cluster_known     (pheromone grid hotspot exists)
[15]   cluster_dist_norm (÷ 3.5)
[16]   cluster_angle_norm(÷ π)
[17]   phero_front_norm  (local grid ahead ÷ 10)
[18]   phero_left_norm
[19]   phero_right_norm
```

---

## Decentralized Training (Option A / CTDE)

**Step 1: Launch Webots in fast mode**

```bash
webots --mode=fast worlds/epuck_foraging_decentralized.wbt &
sleep 10
```

**Step 2: Run the extern supervisor**

```bash
cd controllers/decentralized_supervisor
python3 decentralized_supervisor.py \
    --run_name decentralized_optA_v1 \
    --total_timesteps 7000000 \
    --lr 3e-4 \
    --ent_coef 0.10
```

**Resume from checkpoint:**

```bash
python3 decentralized_supervisor.py \
    --run_name decentralized_optA_v1 \
    --total_timesteps 7000000 \
    --resume ./logs/decentralized_optA_v1/decentralized_optA_v1_200000_steps.zip
```

### Decentralized Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--run_name` | `decentralized_optA_v1` | Output model name |
| `--lr` | `3e-4` | Learning rate |
| `--ent_coef` | `0.10` | Entropy coefficient |
| `--total_timesteps` | `7000000` | Total training steps |
| `--resume` | None | Path to `.zip` to resume from |

### Decentralized Obs Space (18D per robot — CTDE)

```
[0:8]  proximity sensors (÷4096)          ← robot onboard
[8]    tag_visible                         ← supervisor (camera sim)
[9]    tag_dist_norm     (÷ 1.0)           ← supervisor
[10]   tag_angle_norm    (÷ π)             ← supervisor
[11]   carrying          (0/1)             ← robot onboard
[12]   base_dist_norm    (GPS ÷ 3.5)       ← robot GPS
[13]   base_angle_norm   (GPS+IMU ÷ π)    ← robot GPS + InertialUnit
[14]   phero_known       (1 if hotspot)    ← robot P2P receiver
[15]   phero_dist_norm   (÷ 3.5)          ← robot P2P receiver
[16]   phero_angle_norm  (÷ π)            ← robot P2P receiver
[17]   phero_strength    (0–1)             ← robot P2P receiver
```

14/18 dims computed onboard → true CTDE. Supervisor only provides tag obs [8-10] + pickup mechanics.

### How Parameter Sharing Works

One policy is shared across all 4 robots (`n_envs=4`). Each Webots step:
1. All 4 robots send their individual 18D obs to the supervisor
2. Supervisor runs the shared policy on all 4 obs as a batch
3. Each robot receives its own 2D motor action independently
4. All 4 `(obs, action, reward)` tuples go into the shared rollout buffer
5. PPO updates the single shared policy from all 4 robots' experience

### Pheromone in Decentralized Mode

- **No supervisor pheromone grid** — fully peer-to-peer
- Each robot broadcasts its hotspot via Webots Emitter (channel 10, range 2 m)
- On tag pickup: supervisor sends pheromone strength (0.2–1.0) based on local cluster density
- Robot creates hotspot at its GPS position and broadcasts to neighbors
- Robots within 2 m receive and follow the hotspot signal autonomously

---

## Output Files

Both modes produce the same output structure:

```
./
├── {run_name}.zip                         # Final trained model (SB3 format)
└── logs/{run_name}/
    ├── {run_name}_200000_steps.zip        # Checkpoint every 200k env steps
    ├── {run_name}_400000_steps.zip
    └── ...
```

---

## Monitoring Training

SB3 prints a table every rollout. Key metrics to watch:

| Metric | Early training | Healthy training |
|--------|---------------|------------------|
| `ep_rew_mean` | negative | increasing toward positive |
| `explained_variance` | near 0 | climbing toward 0.8+ |
| `entropy_loss` | high (−2.8) | gradually decreasing |
| `value_loss` | high | stabilizing then decreasing |
| `std` | ~1.0 | decreasing as policy focuses |

Also watch for `[PICKUP]` and `[DEPOSIT]` print lines — deposits per episode should increase over training.

---

## Recommended Training Steps

| Goal | Timesteps |
|------|-----------|
| Quick sanity check | 200,000 |
| Short experiment | 1,000,000 |
| Standard training | 7,000,000 |
| Full convergence | 10,000,000+ |

---

## Trained Models (Current)

| Model | Type | Notes |
|-------|------|-------|
| `ppo_v16_phero.zip` | Centralized | Best centralized baseline (~11 tags/min) |
| `ppo_v16_retrain.zip` | Centralized | Re-run with Fix 1 (post-deposit pheromone gradient) |
| `decentralized_optA_v1.zip` | Decentralized | Option A — in training |

---

## Troubleshooting

**`Device "gps" was not found`** — The E-puck PROTO uses `turretSlot`, not `extensionSlot`. Verify `worlds/epuck_foraging_decentralized.wbt` uses `turretSlot [...]`.

**`numpy.dtype size changed`** — NumPy 2.x conflicts with system packages. Fix: `pip install "numpy<2"`.

**Robots not moving** — Webots simulation must be running (press Play). The extern supervisor connects after Webots loads.

**Training stuck at low reward** — Increase `--ent_coef 0.15` for more exploration. Check that `[PICKUP]` events appear in the first few minutes.

**Out of GPU memory** — Reduce batch size with `--batch_size 2048`.
