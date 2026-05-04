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
| `worlds/eval_decentralized.wbt` | Eval world — fixed 6-cluster layout, base radius 0.1, decentralized robots |
| `controllers/eval_decentralized/eval_decentralized.py` | Lightweight extern supervisor — tag mechanics + camera sim only (no PPO) |
| `controllers/epuck_decentralized_eval/epuck_decentralized_eval.py` | Robot controller — loads PPO, runs inference locally, drives own motors |
| `controllers/epuck_decentralized/epuck_decentralized.py` | Base robot controller (sensor/pheromone infrastructure, imported by eval robot) |

---

## Centralized Evaluation

**Step 1: Launch Webots with the eval world**

```bash
webots worlds/eval_best.wbt &
sleep 10
```

**Step 2: Run the evaluation script** (from project root)

```bash
python3 controllers/eval_best_model/eval_best_model.py ppo_v16_phero.zip
```

Or pass any `.zip` model path:

```bash
# Best centralized model
python3 controllers/eval_best_model/eval_best_model.py ppo_v16_phero.zip

# v17 model
python3 controllers/eval_best_model/eval_best_model.py ppo_v17_phero2.zip
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
- Target: > 5.94 tags/min (CPFA baseline)

### Centralized Obs Space (20D per robot, 80D stacked total)

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

---

## Decentralized Evaluation

The decentralized eval uses **CTDE (Centralized Training, Decentralized Execution)**:
- Each robot loads the PPO model and runs inference locally
- The supervisor only provides tag observations (simulating a camera) and handles pickup/deposit mechanics
- No motor commands are sent from the supervisor — all P1-P4 overrides run onboard

**Step 1: Launch Webots with the decentralized eval world**

```bash
webots worlds/eval_decentralized.wbt &
sleep 10
```

**Step 2: Run the lightweight supervisor** (from project root)

```bash
python3 controllers/eval_decentralized/eval_decentralized.py
```

The robot controller (`epuck_decentralized_eval`) automatically loads the model specified in `current_eval_model.txt`, or falls back to `decentralized_optA_v1` if that file doesn't exist.

**To run a specific model:**

```bash
# Write model path to config file, then launch supervisor
echo "/home/sara/Documents/Centralized Learning/RL-FL-Foraging/decentralized_optA_v4.zip" \
    > current_eval_model.txt
python3 controllers/eval_decentralized/eval_decentralized.py
```

### What to Expect

```
[ROBOT] Loading model: decentralized_optA_v4.zip
[ROBOT] Model loaded. Running inference locally.
[PICKUP] Robot 1 picked up tag at (1.23, 0.45). Strength: 0.60
[DEPOSIT] Robot 1 deposited. Total: 1
```

Each robot prints its own model load message at startup. Pickups and deposits are logged by the supervisor.

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

### P1–P4 Override Modes (fully onboard at eval)

The eval robot applies these overrides in priority order after PPO inference. All run locally using GPS + IMU — no supervisor dependency.

| Mode label | Condition | Action |
|-----------|-----------|--------|
| `WALL_ESC` | wall_dist < 0.6m | Steer to centre (gain 4.0) |
| `RTB` | carrying=True | Steer to centre (gain 2.5) |
| `BASE_AVOID` | dist_to_base < 0.3m | Steer to 1.2m target (gain 3.0) |
| `PPO` | otherwise | Raw PPO output |

### Interpreting Eval Logs

**Healthy behaviour:**
- Robots spend most time in `PPO` mode, with brief `RTB` periods when carrying
- `phero=1` appears frequently after first deposits (robots following pheromone)
- Rate increases as robots learn cluster locations
- No robots stuck in `WALL_ESC` for extended periods

**Failure patterns seen in v1–v3:**
- All robots in `BASE_AVOID` for entire episodes → P3 not pushing far enough (multiplier too low)
- All robots in `WALL_ESC` after phase 1 → wall-stuck collapse (P1 working but too late)
- `phero=0 str=0.00` always → pheromone not being followed (TTL too short or reward too weak)

### CTDE Deployment Note

At deployment on real robots:
- Dims [0:8], [11:18] — already computed onboard (no supervisor needed)
- Dims [8:11] — replace supervisor camera sim with onboard camera + AprilTag detector
- Each robot loads `decentralized_optA_v4.zip` and runs inference locally (no central supervisor)
- Robots coordinate only via P2P pheromone broadcasts (channel 10, 2 m range)

---

## Comparing Centralized vs Decentralized

Run both in identical arena configurations and measure:

| Metric | Centralized | Decentralized |
|--------|-------------|---------------|
| Tags/min | ~11 (v16) | TBD (v4 in training) |
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

**Model file not found** — All trained models are `.zip` files in the project root. Pass the full or relative path, or write the path to `current_eval_model.txt`.

**Robots stuck at wall immediately** — Check that `INITIAL_TTL=2000` in `controllers/epuck_decentralized/epuck_decentralized.py`. Old value of 400 causes pheromone to expire after 1-2 trips.

**Low deposit rate despite robots moving** — If all robots orbit near base, P3 is not pushing far enough. Verify P3 multiplier=5.0 in `_apply_overrides` in the eval robot.
