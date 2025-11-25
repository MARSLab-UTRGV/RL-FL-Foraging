# Multi-Agent E-puck Foraging with Reinforcement Learning

A multi-agent reinforcement learning system for simulated e-puck robots performing collaborative foraging tasks in Webots. This project uses Proximal Policy Optimization (PPO) with Centralized Training, Decentralized Execution (CTDE).

## Overview

This project trains 4 e-puck robots to:
- Search for boxes marked with AprilTags
- Pick up boxes and carry them back to a base station
- Navigate around obstacles
- Coordinate implicitly through a shared policy

## Project Structure

```
RL-FL-Foraging/
├── models/                 # Trained PPO models
│   ├── epuck_top5_*/      # Latest models with reward shaping
│   └── legacy/            # Earlier models (sparse rewards)
├── controllers/           # Webots controller code
│   ├── epuck_driver/      # Individual robot controller
│   ├── epuck_foraging_supervisor_shaping/  # Training supervisor
│   ├── eval_best_model/   # Evaluation controller
│   └── vectorized_supervisor/  # Vectorized training (experimental)
├── worlds/                # Webots world files (.wbt)
├── scripts/               # Training and evaluation scripts
├── configs/               # Configuration generators
├── logs/                  # Training logs
└── docs/                  # Documentation
    ├── progress.md        # Model registry & training history
    └── SPEEDUP_GUIDE.md   # Webots speedup techniques
```

## Quick Start

### Prerequisites

- Python 3.8+
- Webots R2023b or later
- Dependencies: `pip install -r requirements.txt`

### Environment Setup

```bash
export WEBOTS_HOME=/usr/local/webots
export PYTHONPATH=$PYTHONPATH:$WEBOTS_HOME/lib/controller/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WEBOTS_HOME/lib/controller
```

### Running the Best Model

1. Open Webots and load: `worlds/eval_best.wbt`
2. Run evaluation:

```bash
python3 controllers/eval_best_model/eval_best_model.py \
  models/epuck_top5_baseline2M/ppo_epuck_top5_baseline2M.zip
```

### Training New Models

See `scripts/` directory for training scripts and `configs/` for configuration generators.

## System Architecture

**Centralized Training, Decentralized Execution (CTDE)**

- **The "Brain" (PPO Agent):** A single neural network controls all 4 robots
- **The "Server" (Supervisor):** Webots Supervisor collects observations and distributes actions
- **No Cheating:** Each robot only receives its own local sensor data (no global information)

## Observation Space (60 values total)

Each robot observes 15 values:
- Proximity sensors (8): Distance to obstacles
- Tag visible (1): Whether an AprilTag is detected
- Tag distance (1): Distance to nearest tag
- Tag angle (1): Angle to nearest tag
- Carrying (1): Whether holding a box
- Pheromones (3): Artificial "scent trail" (front/left/right)

## Action Space (8 values total)

Each robot outputs 2 values:
- Left wheel speed: [-1.0, +1.0]
- Right wheel speed: [-1.0, +1.0]

## Reward Function

- **Pickup:** +1.0
- **Deposit:** +10.0
- **Approach tag (shaping):** +10 × (old_dist - new_dist)
- **Approach base (shaping):** +10 × (old_dist - new_dist)
- **Collision:** -0.01
- **Time penalty:** -0.001 per step

## Current Best Model

**Model:** `epuck_top5_baseline2M`
**Timesteps:** 2,000,000
**Status:** Verified Working
**Behavior:** Robots successfully approach tags, pick up boxes, and navigate to base

See `docs/progress.md` for complete model registry and training history.

## Key Features

- Multi-agent coordination through shared policy
- Dense reward shaping for faster learning
- Parallel training configuration generators
- Comprehensive logging and evaluation tools
- Headless and GUI training modes

## Documentation

- `docs/progress.md` - Complete model registry and training history
- `docs/SPEEDUP_GUIDE.md` - Techniques to accelerate Webots simulation

## Results

The best model (`epuck_top5_baseline2M`) demonstrates:
- Direct navigation to AprilTag-marked boxes
- Successful pickup behavior
- Goal-directed movement toward base station
- Obstacle avoidance

Previous models with sparse rewards showed spinning behavior and failed to converge.

## Citation

If you use this code in your research, please cite:
- [Deepbots Framework](https://github.com/aidudezzz/deepbots)
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)

## License

This project builds on the deepbots-tutorials examples.

## Author

MARSLab - UTRGV

## Acknowledgments

- Webots robotics simulator
- Stable-Baselines3 RL library
- Deepbots framework
