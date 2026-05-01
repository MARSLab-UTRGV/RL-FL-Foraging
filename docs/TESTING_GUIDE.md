# Testing Guide — Multi-Agent E-puck Foraging

This guide covers evaluating trained models for both the **centralized** and **decentralized** versions.

---

## Prerequisites

```bash
export WEBOTS_HOME=/usr/local/webots
export PYTHONPATH=$PYTHONPATH:$WEBOTS_HOME/lib/controller/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WEBOTS_HOME/lib/controller
```

---

## Files Overview

### Centralized Evaluation

| File | Role |
|------|------|
| `worlds/eval_best.wbt` | Evaluation world (4 robots, 70 tags, no training hooks) |
| `controllers/eval_best_model/eval_best_model.py` | Extern eval script — loads model and runs episodes |
| `controllers/epuck_driver/epuck_driver.py` | Robot controller (runs inside Webots, unchanged) |

### Decentralized Evaluation

| File | Role |
|------|------|
| `worlds/epuck_foraging_decentralized.wbt` | World with GPS/IMU/phero on each robot |
| `controllers/decentralized_supervisor/decentralized_supervisor.py` | Supervisor (used for training; evaluation uses the same world) |
| `controllers/epuck_decentralized/epuck_decentralized.py` | Robot controller — runs autonomously with onboard sensors |

---

## Centralized Evaluation

**Step 1: Launch Webots with the eval world**

```bash
webots worlds/eval_best.wbt &
sleep 10
```

**Step 2: Run the evaluation script**

```bash
cd controllers/eval_best_model
python3 eval_best_model.py ../../ppo_v16_phero.zip
```

Pass any `.zip` model path as the argument:

```bash
# Best centralized model
python3 eval_best_model.py ../../ppo_v16_phero.zip

# Retrained model
python3 eval_best_model.py ../../ppo_v16_retrain.zip
```

### What to Expect

```
[EVAL] Model: ppo_v16_phero.zip
[EVAL] Step 100 | Deposits: 3 | Reward: 142.5
[EVAL] Step 200 | Deposits: 7 | Reward: 289.0
...
[PICKUP] Robot 2 picked up tag. Total: 8
[DEPOSIT] Robot 2 deposited! Total: 8
```

- Robots navigate toward tag clusters
- On tag pickup: robot carries it back to base (centre of arena)
- Deposit increases the total count


### Centralized Obs Space (20D per robot, 80D stacked total)

The eval script expects the same 20D obs as training:

```
[0:8]  proximity sensors
[8]    tag_visible
[9]    tag_dist_norm
[10]   tag_angle_norm
[11]   carrying
[12]   base_dist_norm
[13]   base_angle_norm
[14]   cluster_known     (supervisor pheromone grid)
[15]   cluster_dist_norm
[16]   cluster_angle_norm
[17]   phero_front_norm
[18]   phero_left_norm
[19]   phero_right_norm
```

**Note:** The `eval_best_model.py` script is compatible with both v16 and v16_retrain models — both use the same 20D obs layout.

---

## Decentralized Evaluation

The decentralized model (`decentralized_optA_v1.zip`) is an SB3 PPO model trained with parameter sharing. To evaluate it, launch the decentralized world and load the model.

**Step 1: Launch Webots with the decentralized world**

```bash
webots worlds/epuck_foraging_decentralized.wbt &
sleep 10
```

**Step 2: Run a quick eval using the supervisor in eval mode**

Create a short eval script or modify `decentralized_supervisor.py`'s `__main__` block. Alternatively, load the model directly:

```python
from stable_baselines3 import PPO
import numpy as np

model = PPO.load("decentralized_optA_v1.zip")

# In your step loop:
obs = env.reset()                           # (4, 18) array
actions, _ = model.predict(obs, deterministic=True)   # (4, 2) array
obs, rewards, dones, infos = env.step(actions)
```

The decentralized world uses:
- `controllers/epuck_decentralized/epuck_decentralized.py` — robot controller (runs inside Webots)
- `controllers/decentralized_supervisor/decentralized_supervisor.py` — handles tag mechanics and obs assembly

### Decentralized Obs Space (18D per robot)

```
[0:8]  proximity sensors        ← robot onboard
[8]    tag_visible              ← supervisor (camera sim / real camera at deployment)
[9]    tag_dist_norm
[10]   tag_angle_norm
[11]   carrying                 ← robot onboard
[12]   base_dist_norm           ← robot GPS
[13]   base_angle_norm          ← robot GPS + InertialUnit
[14]   phero_known              ← robot P2P pheromone receiver
[15]   phero_dist_norm
[16]   phero_angle_norm
[17]   phero_strength
```

### CTDE Deployment Note

At deployment on real robots:
- Dims [0:8], [11:18] — already computed onboard (no supervisor needed)
- Dims [8:11] — replace supervisor camera sim with onboard camera + AprilTag detector
- Each robot loads the same `decentralized_optA_v1.zip` policy and runs inference locally
- Robots coordinate only via P2P pheromone broadcasts (channel 10, 2 m range)

---

## Comparing Centralized vs Decentralized

Run both in identical arena configurations and measure:

| Metric | Centralized | Decentralized |
|--------|-------------|---------------|
| Tags/min | ~11 (v16) | TBD after training |
| Obs computed onboard | 0/20 | 14/18 |
| Supervisor at execution | Required | Not required |
| Pheromone type | Global grid (supervisor) | P2P broadcast (robot) |
| CPFA baseline | 5.94 tags/min | 5.94 tags/min |

For statistical significance, run at least **5 evaluation seeds** of 30 minutes each and report mean ± std.

---

## Headless Evaluation (No GUI)

```bash
webots --mode=fast --minimize --no-rendering worlds/eval_best.wbt &
sleep 10
python3 controllers/eval_best_model/eval_best_model.py ppo_v16_phero.zip
```

---

## Changing Simulation Speed

In Webots GUI: drag the speed slider in the toolbar, or launch with `--mode=fast` for maximum speed (no rendering).

---

## Troubleshooting

**`Device "gps" was not found`** — The decentralized world must use `turretSlot` (not `extensionSlot`) in each E-puck node. Check `worlds/epuck_foraging_decentralized.wbt`.

**Robots not moving** — Press Play in Webots before running the eval script. The extern controller connects after Webots loads.

**Wrong obs size error** — Centralized models expect 80D input; decentralized models expect 18D per robot. Do not mix worlds and models.

**`numpy.dtype size changed`** — Run `pip install "numpy<2"` to fix NumPy 2.x binary incompatibility.

**Model file not found** — All trained models are `.zip` files in the project root. Pass the full or relative path as argument.
