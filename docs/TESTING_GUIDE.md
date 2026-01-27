# Testing a Trained Model

This guide explains how to evaluate and visualize a trained PPO model.

## Prerequisites

```bash
# Set environment variables (add to ~/.bashrc for permanence)
export WEBOTS_HOME=/usr/local/webots
export PYTHONPATH=$PYTHONPATH:$WEBOTS_HOME/lib/controller/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WEBOTS_HOME/lib/controller
```

## Quick Start

### Option 1: Use the Visualization Script (Recommended)

```bash
./scripts/run_viz.sh
```

This launches Webots and runs the best model automatically.

To change which model is visualized, edit `scripts/run_viz.sh` and change the `MODEL` variable.

### Option 2: Manual Steps

**Step 1: Start Webots**

```bash
webots worlds/eval_best.wbt &
```

Wait for Webots to fully load (~10 seconds).

**Step 2: Run the evaluation controller**

```bash
python3 controllers/eval_best_model/eval_best_model.py models/epuck_top5_baseline2M/ppo_epuck_top5_baseline2M.zip
```

Replace the path with any trained model `.zip` file.

## Available Trained Models

All models are in the project root directory:

| Model | Description |
|-------|-------------|
| `ppo_5models_baseline.zip` | Standard config (2M steps) |
| `ppo_5models_high_ent.zip` | High exploration (entropy=0.05) |
| `ppo_5models_high_lr.zip` | High learning rate |
| `ppo_5models_small_batch.zip` | Batch size 2048 |
| `ppo_5models_verysmall_batch.zip` | Batch size 1024 |
| `ppo_5models_long_5M.zip` | Extended training (5M steps) |
| `ppo_5models_extralong_10M.zip` | Extended training (10M steps) |

Older models are in `models/epuck_top5_*/` directories.

## What to Expect

When the model runs, you should see:
- 4 e-puck robots moving in the arena
- Robots navigating toward yellow AprilTag boxes
- Robots picking up boxes (tags disappear)
- Robots returning to the red base station at center
- Robots depositing boxes (score increases)

The terminal shows step count and cumulative reward every 100 steps.

## Changing Simulation Speed

In Webots GUI:
1. Click the speed slider in the toolbar
2. Drag to increase speed (up to 10x or more)
3. Or use `--mode=fast` when launching Webots for maximum speed

## Headless Evaluation (No GUI)

```bash
webots --mode=fast --minimize --no-rendering worlds/eval_best.wbt &
sleep 10
python3 controllers/eval_best_model/eval_best_model.py ppo_5models_baseline.zip
```

## Troubleshooting

**Webots not found:**
- Verify `WEBOTS_HOME` points to your Webots installation

**Model file not found:**
- Check the path is correct and the `.zip` file exists

**Robots not moving:**
- Ensure Webots simulation is running (click play button)
- Check that the controller is connected (look for print output)

**Import errors:**
- Run `pip install stable-baselines3 deepbots torch numpy`
