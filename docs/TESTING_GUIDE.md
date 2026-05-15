# Testing Guide — Multi-Agent E-puck Foraging

This guide covers evaluating trained models for both the **centralized** and **fully decentralized independent** versions.

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
| `worlds/eval_best_5x5.wbt` | Eval world — fixed 6-cluster layout, 64 tags |
| `controllers/eval_best_model/eval_best_model_5x5.py` | Extern eval script — loads shared model, runs episodes |
| `controllers/epuck_driver/epuck_driver.py` | Robot controller (runs inside Webots, unchanged) |

### Decentralized Evaluation

| File | Role |
|------|------|
| `worlds/eval_decentralized.wbt` | Eval world — fixed 6-cluster layout, 64 tags |
| `controllers/eval_decentralized/eval_decentralized.py` | Lightweight extern supervisor — tag mechanics + camera sim only |
| `controllers/epuck_decentralized_eval/epuck_decentralized_eval.py` | Robot controller — loads own PPO, runs inference locally |
| `controllers/epuck_decentralized/epuck_decentralized_v4.py` | Base class — sensors, CPFA pheromone, P2P broadcast (imported by eval robot) |

---

## Centralized Evaluation

**Step 1: Launch Webots with the eval world**

```bash
webots worlds/eval_best_5x5.wbt &
sleep 10
```

**Step 2: Run the evaluation script** (from project root)

```bash
python3 controllers/eval_best_model/eval_best_model_5x5.py ppo_cpfa_5x5.zip
```

### What to Expect

```
[EVAL] Model: ppo_cpfa_5x5.zip
[PICKUP] Robot 2 picked up tag (density=3) | Total: 8
[DEPOSIT] Robot 2 deposited! Total: 8
  [PHERO] Laid at (1.23, -0.45) density=3 prob=0.65 total_entries=2
  [TARGET] Robot 2 → SITE (1.23,-0.45)
[EP 1] Picks: 47 | Deps: 38 | Rate: 2.19 tags/min (sim) | TotalDeps: 38
```

Target: > 5.94 tags/min (CPFA baseline). Best centralized model achieves ~11 tags/min.

### Centralized Obs Space (21D per robot, 84D total)

```
[0:8]  proximity sensors
[8]    tag_visible
[9]    tag_dist_norm      (÷ 1.0m)
[10]   tag_angle_norm     (÷ π)
[11]   carrying
[12]   base_dist_norm     (÷ 3.5m)
[13]   base_angle_norm    (÷ π)
[14]   site_known         (CPFA site fidelity target)
[15]   site_dist_norm
[16]   site_angle_norm
[17]   phero_known        (CPFA roulette pheromone target)
[18]   phero_dist_norm
[19]   phero_angle_norm
[20]   search_duration_norm (give-up timer ÷ 700)
```

---

## Decentralized Evaluation

Each robot loads its own per-robot PPO model and runs inference locally. The supervisor only provides camera simulation (tag obs [8–10]) and handles pickup/deposit mechanics.

### Model Loading Priority (eval robot)

1. `current_eval_run.txt` exists → loads `robot{N}_{run_name}.zip` (per-robot independent models)
2. `current_eval_model.txt` exists → loads shared model path (CTDE / fallback)
3. Hard-coded fallback: `decentralized_optA_v1`

**Step 1: Write the run name**

```bash
# For independent models (one per robot)
echo "decentralized_indep_v4" > current_eval_run.txt

# OR for a shared CTDE model
echo "/full/path/to/decentralized_optA_v4.zip" > current_eval_model.txt
```

**Step 2: Launch Webots with the decentralized eval world**

```bash
webots worlds/eval_decentralized.wbt &
sleep 10
```

**Step 3: Run the lightweight supervisor** (from project root)

```bash
python3 controllers/eval_decentralized/eval_decentralized.py
```

### What to Expect

```
[robot1] Loading model: robot1_decentralized_indep_v4.zip
[robot2] Loading model: robot2_decentralized_indep_v4.zip
[robot3] Loading model: robot3_decentralized_indep_v4.zip
[robot4] Loading model: robot4_decentralized_indep_v4.zip

[PICKUP] robot1 at (1.23,-0.45) | strength=0.68 approx_density~3
  [PHERO] Added (1.23,-0.45) weight=0.68 | list_size=1
[DEPOSIT] robot1 | Total deps: 1
  [SITE_FID] last_pickup=(1.23,-0.45) weight=0.68
  [TARGET]   → SITE (1.23,-0.45)

=================================================================
[EP 1] Picks: 12 | Deps: 9 | Rate: 1.04 tags/min (sim) | TotalDeps: 9
  Curriculum: N/A (eval — fixed layout)
  Pheromone:  P2P per-robot (not tracked by supervisor)
  R1[PPO     ]: carry=0 | base=1.43 | wall=1.87
  R2[RTB     ]: carry=1 | base=0.82 | wall=2.11
  R3[PPO     ]: carry=0 | base=1.67 | wall=1.23
  R4[BASE_ESC]: carry=0 | base=0.18 | wall=2.31
=================================================================
```

### Decentralized Obs Space (20D per robot)

```
[0:8]  proximity sensors (÷4096)          ← robot onboard
[8]    tag_visible                         ← supervisor (camera sim / real camera at deploy)
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

14/20 dims computed fully onboard — no supervisor needed at execution.

### Override Modes (fully onboard at eval)

| Mode label | Condition | Action |
|-----------|-----------|--------|
| `WALL_ESC` | wall_dist < 0.35 m or max(prox) > 0.55 | Steer to centre (gain=4.0) |
| `BASE_ESC` | not carrying and dist_to_base < 0.25 m | Nudge outward 0.5 m (gain=4.0) |
| `RTB` | carrying=True | Steer to centre (gain=2.5) |
| `PPO` | otherwise | Raw PPO output |

### Interpreting Eval Logs

**Healthy behaviour:**
- `[TARGET] → SITE/PHERO` appearing after every deposit → coordination working
- `RTB` mode when carrying, brief `BASE_ESC` after deposit, mostly `PPO` mode
- `list_size` growing within an episode → pheromone broadcast working
- Deposit rate increasing over the first 5–10 minutes of eval

**Failure patterns:**
- `[TARGET] → EXPLORE` always → pheromone list empty, P2P broadcast not working
- All robots in `WALL_ESC` for extended periods → obstacle avoidance collapse
- `[TARGET] → SITE` always, same location → site fidelity stuck, cluster fully depleted

### Verifying Pheromone and Site Fidelity

Check per-robot logs or terminal for:
```
[PHERO] Added (X,Y) weight=W | list_size=N       # Pheromone created at pickup ✓
[SITE_FID] last_pickup=(X,Y) weight=W             # Site fidelity stored ✓
[TARGET] → SITE (X,Y)                             # Site fidelity used ✓
[TARGET] → PHERO (X,Y)                            # Pheromone roulette used ✓
[TARGET] → EXPLORE                                # Free exploration ✓
```

---

## Statistical Evaluation Protocol

For CoRL results, run at least **5 seeds × 30 minutes** per model and report mean ± std tags/min.

```bash
for seed in 1 2 3 4 5; do
    echo "decentralized_indep_v4" > current_eval_run.txt
    webots --mode=fast worlds/eval_decentralized.wbt &
    sleep 10
    timeout 1800 python3 controllers/eval_decentralized/eval_decentralized.py \
        2>&1 | tee logs/eval_seed${seed}.txt
    pkill -f webots
    sleep 5
done
```

---

## Comparing Centralized vs Decentralized

Run both in the same fixed eval world and measure:

| Metric | Centralized | Decentralized Independent |
|--------|-------------|--------------------------|
| Tags/min | ~11 (v16) | TBD (`decentralized_indep_v4`) |
| Obs dims per robot | 21 | 20 |
| Obs computed onboard | 0/21 (all supervisor) | 14/20 |
| Supervisor at execution | Required | Not required |
| Pheromone type | Global list (supervisor) | P2P broadcast (robot) |
| Models | 1 shared | 4 independent (+ optional gossip) |
| CPFA baseline | 5.94 tags/min | 5.94 tags/min |

---

## Headless Evaluation (No GUI)

```bash
webots --mode=fast --minimize --no-rendering worlds/eval_decentralized.wbt &
sleep 10
python3 controllers/eval_decentralized/eval_decentralized.py
```

---

## Physical Robot Deployment

At deployment on real robots, only 3 obs dims require supervisor:
- `[8] tag_visible`, `[9] tag_dist_norm`, `[10] tag_angle_norm` → replace with onboard camera + AprilTag detector

All other 17 dims are already computed onboard. Load `robot{N}_{run_name}.zip` on each robot, point the camera at the arena, run inference locally.

---

## Troubleshooting

**`Device "gps" was not found`** — The decentralized world must use `turretSlot` (not `extensionSlot`) in each E-puck node. Check `worlds/eval_decentralized.wbt`.

**Robots not moving** — Press Play in Webots before running the eval script.

**Wrong obs size error** — Centralized models expect 84D input; decentralized models expect 20D per robot. Do not mix worlds and models.

**`numpy.dtype size changed`** — Run `pip install "numpy<2"`.

**Model file not found** — Trained models are in the project root as `robot{N}_{run_name}.zip`. Verify `current_eval_run.txt` contains the correct run name (not a full path).

**`[TARGET] → EXPLORE` always** — Pheromone list is empty. Either no pickups have occurred yet (normal in first few minutes) or P2P broadcast is not working (check phero_emitter/receiver in world file).

**All robots `RTB` but deposit rate = 0** — Deposit detection threshold is 0.25 m from base centre. Check that base is at (0, 0, 0) in the eval world.
