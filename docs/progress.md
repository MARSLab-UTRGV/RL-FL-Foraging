# Training Progress & Model Registry
This document tracks all PPO models trained for the Multi-Agent e-puck Foraging task.

## System Architecture (CTDE)
We use **Centralized Training, Decentralized Execution**.

- **The "Brain" (PPO Agent):** A single neural network controls ALL 4 robots. It receives all their observations combined and outputs all their actions combined.
- **The "Server" (Supervisor):** The Webots Supervisor acts as the central hub. It collects data from all robots, feeds it to the PPO brain, and sends actions back to the robots.
- **Is it Cheating?:** No. Even though the training is centralized, each robot's portion of the input only contains its own local sensor data. It does NOT see what other robots see.

## Observation Space (Input)
Each robot sees **15 numbers**. The total input to the brain is **15 * 4 = 60 numbers**.

- **Proximity (8):** Distance to obstacles around the robot.
- **Tag Visible (1):** 1.0 if a tag is seen, 0.0 otherwise.
- **Tag Distance (1):** Distance to the closest visible tag (max 2m).
- **Tag Angle (1):** Angle to the closest visible tag.
- **Carrying (1):** 1.0 if holding a box, 0.0 otherwise.
- **Pheromones (3):** Smell of "food trail" in front/left/right.

## Action Space (Output)
The brain outputs **2 numbers per robot**. Total output is **2 * 4 = 8 numbers**.

- **Left Wheel Speed:** -1.0 to +1.0
- **Right Wheel Speed:** -1.0 to +1.0

## Reward Function (The Goal)
How we teach them what to do:

- **Pickup:** +1.0 (Big reward!)
- **Deposit:** +10.0 (Huge reward!)
- **Shaping (Approaching Tag):** +10 * (OldDist - NewDist) (Points for getting closer)
- **Shaping (Approaching Base):** +10 * (OldDist - NewDist) (Points for getting closer)
- **Collision:** -0.01 (Don't crash)
- **Time Penalty:** -0.001 per step (Hurry up!)

## Current Best Model
**Model ID:** `epuck_top5_baseline2M`
**File Path:** `.../epuck_top5_baseline2M/ppo_epuck_top5_baseline2M.zip`
**Status:** Verified Working (Robots approach tags and forage)

### Run Command
Run this from the project root (You need to have webots open already with the world loaded, this world:/home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/eval_best.wbt):

```bash
cd /home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project && \
export WEBOTS_HOME=/usr/local/webots && \
export PYTHONPATH=$PYTHONPATH:$WEBOTS_HOME/lib/controller/python && \
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WEBOTS_HOME/lib/controller && \
python3 controllers/eval_best_model/eval_best_model.py controllers/epuck_top5_baseline2M/ppo_epuck_top5_baseline2M.zip
```

## Latest Models (With Reward Shaping)
Trained on Nov 21, 2025. These models use dense rewards for approaching tags and collision penalties.

| Model ID | Timesteps | Batch Size | Learning Rate | Ent. Coef | Description | Status | Observations |
|----------|-----------|------------|---------------|-----------|-------------|--------|--------------|
| `epuck_top5_quick100k` | 100,000 | 2048 | 5e-4 | 0.02 | Quick Test: Fast learning rate, small batch for rapid feedback. | Ready | To be tested |
| `epuck_top5_short250k` | 250,000 | 4096 | 3e-4 | 0.01 | Short Baseline: Standard params, short duration. | Ready | To be tested |
| `epuck_top5_baseline2M` | 2,000,000 | 4096 | 3e-4 | 0.01 | Baseline 2M: The standard "Gold" configuration. Long training. | Ready | Working! Robots approach tags directly. |
| `epuck_top5_smallbatch2M` | 2,000,000 | 2048 | 3e-4 | 0.01 | Small Batch: More frequent updates (2x more updates per epoch). | Ready | To be tested |
| `epuck_top5_highent2M` | 2,000,000 | 4096 | 3e-4 | 0.05 | High Entropy: Encourages 5x more exploration. Good for escaping local optima. | Ready | To be tested |

## Legacy Models (No Reward Shaping)
Trained previously. These models struggled with the sparse reward signal (spinning behavior).

| Model ID | Timesteps | Batch Size | Description | Status |
|----------|-----------|------------|-------------|--------|
| `ppo_epuck_heavy` | 100,000 | 4096 | Initial GUI test run. | Suboptimal |
| `ppo_epuck_FAST` | 200,000 | 4096 | Fast mode test. | Suboptimal |
| `ppo_epuck_HEADLESS` | 300,000 | 4096 | Headless mode test. | Suboptimal |
| `ppo_epuck_2M` | 2,000,000 | 4096 | Long run (failed to converge due to sparse rewards). | Suboptimal |
| `ppo_2048` | 2,000,000 | 2048 | Small batch variant. | Suboptimal |
| `ppo_1024` | 2,000,000 | 1024 | Tiny batch variant. | Suboptimal |
| `ppo_epuck_foraging_centralized` | 100,000 | 4096 | Very first test run. | Suboptimal |

## Cancelled Models
These configurations were generated but crashed or were cancelled before completion.

- `epuck_shaping_1_baseline` to `epuck_shaping_7_aggressive`: Cancelled due to crash. Replaced by "Top 5" run.
- `epuck_shaping_8_quick100k` to `epuck_shaping_11_long1M`: Cancelled/Merged into "Top 5" run.

## File Locations
All models are stored in: `/home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/controllers/`

Example path: `.../controllers/epuck_top5_baseline2M/ppo_epuck_top5_baseline2M.zip`

## 7 Parallel Models (Dec 3-5, 2025)

We are addressing the limitations found in previous runs (slow speed, collisions, easy task difficulty).

### Environment Updates
1. **Robot Speed:** Increased motor command scaling by 6x (using full speed range).
2. **Collision Penalty:** Increased from -0.01 to -0.1 (10x penalty for bumping).
3. **Vision Range:** Reduced from 2.0m to 0.8m (forcing exploration).
4. **Deposit Range:** Reduced from 0.3m to 0.15m (precision required).
5. **World Changes:** Base station radius reduced (0.2 -> 0.1), AprilTags reduced (0.0275 -> 0.02).

### Trained Models (Awaiting Testing)
Training completed Dec 3-5, 2025. All models saved to project root as `.zip` files. These models have not been tested yet.

| Model ID | Timesteps | Batch Size | LR | Ent. Coef | Description | Status |
|----------|-----------|------------|----|-----------|-------------|--------|
| `ppo_5models_baseline` | 2M | 4096 | 3e-4 | 0.01 | Baseline with new hard environment. | Trained, not tested |
| `ppo_5models_high_ent` | 2M | 4096 | 3e-4 | 0.05 | High Entropy (0.05) to force exploration. | Trained, not tested |
| `ppo_5models_high_lr` | 2M | 4096 | 5e-4 | 0.01 | Higher Learning Rate. | Trained, not tested |
| `ppo_5models_small_batch` | 2M | 2048 | 3e-4 | 0.01 | Small Batch (2048). | Trained, not tested |
| `ppo_5models_verysmall_batch` | 2M | 1024 | 3e-4 | 0.01 | Very Small Batch (1024). | Trained, not tested |
| `ppo_5models_long_5M` | 5M | 4096 | 3e-4 | 0.01 | Long Run: 5 million steps for convergence. | Trained, not tested |
| `ppo_5models_extralong_10M` | 10M | 4096 | 3e-4 | 0.01 | Extra Long Run: 10 million steps. | Trained, not tested |

### How to Test These Models

For full instructions, see TESTING_GUIDE.md.

### File Locations
Models saved to: `/home/andres2020/Dev/deepbots_test/RL-FL-Foraging/`
- `ppo_5models_baseline.zip`
- `ppo_5models_high_ent.zip`
- `ppo_5models_high_lr.zip`
- `ppo_5models_small_batch.zip`
- `ppo_5models_verysmall_batch.zip`
- `ppo_5models_long_5M.zip`
- `ppo_5models_extralong_10M.zip`

Checkpoints in `logs/ppo_5models_*/` directories.
