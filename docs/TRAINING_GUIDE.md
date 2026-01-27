# Training a Model

This guide explains how to train a new PPO model and then test it.

## Prerequisites

```bash
# Set environment variables (add to ~/.bashrc for permanence)
export WEBOTS_HOME=/usr/local/webots
export PYTHONPATH=$PYTHONPATH:$WEBOTS_HOME/lib/controller/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WEBOTS_HOME/lib/controller

# Install dependencies
pip install stable-baselines3 deepbots torch numpy gym
```

## Quick Start: Train a Single Model

**Step 1: Start Webots in fast mode**

```bash
webots --mode=fast --minimize --no-rendering worlds/epuck_5models.wbt &
sleep 10
```

**Step 2: Run training**

```bash
python3 controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_shaping.py \
    --run_name my_model \
    --total_timesteps 2000000
```

**Step 3: Wait for training to complete**

Training 2M steps takes approximately 12-24 hours depending on hardware.

**Step 4: Test the trained model**

```bash
# Kill the training Webots instance
pkill -f webots

# Start Webots in GUI mode
webots worlds/eval_best.wbt &
sleep 10

# Run evaluation with your new model
python3 controllers/eval_best_model/eval_best_model.py my_model.zip
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--run_name` | `ppo_epuck_shaping` | Name for this training run |
| `--lr` | `3e-4` | Learning rate |
| `--ent_coef` | `0.01` | Entropy coefficient (exploration) |
| `--batch_size` | `4096` | PPO batch size |
| `--total_timesteps` | `2000000` | Total training steps |

### Example Configurations

**Baseline (recommended for first run):**
```bash
python3 controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_shaping.py \
    --run_name baseline \
    --lr 3e-4 \
    --ent_coef 0.01 \
    --batch_size 4096 \
    --total_timesteps 2000000
```

**High exploration (if robots aren't exploring enough):**
```bash
python3 controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_shaping.py \
    --run_name high_exploration \
    --lr 3e-4 \
    --ent_coef 0.05 \
    --batch_size 4096 \
    --total_timesteps 2000000
```

**Quick test (100k steps, ~1-2 hours):**
```bash
python3 controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_shaping.py \
    --run_name quick_test \
    --lr 5e-4 \
    --ent_coef 0.02 \
    --batch_size 2048 \
    --total_timesteps 100000
```

## Output Files

Training creates:

```
./
├── {run_name}.zip                    # Final trained model
├── logs/{run_name}/                  # Checkpoints (every 100k steps)
│   ├── {run_name}_100000_steps.zip
│   ├── {run_name}_200000_steps.zip
│   └── ...
└── tensorboard/{run_name}/           # Training metrics
```

## Monitoring Training

### View TensorBoard logs

```bash
tensorboard --logdir=tensorboard/
```

Open http://localhost:6006 in your browser.

### Watch training output

Training prints progress every episode:
```
[TRAINING] Starting PPO Training: my_model
| rollout/                |           |
|    ep_len_mean          | 4096      |
|    ep_rew_mean          | -2.34     |
| time/                   |           |
|    fps                  | 312       |
|    iterations           | 1         |
|    time_elapsed         | 13        |
|    total_timesteps      | 4096      |
```

The `ep_rew_mean` should increase over time. Positive rewards indicate successful foraging.

## GPU Training

Training automatically uses GPU if available. To specify a GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python3 controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_shaping.py \
    --run_name my_model \
    --total_timesteps 2000000
```

## Training Multiple Models

### Sequential (one at a time)

```bash
./scripts/run_5_models.sh
```

### Parallel (multiple GPUs)

```bash
./scripts/run_5_models_parallel.sh
```

This launches 7 training runs with different hyperparameters on ports 4001-4007.

## Full Workflow Example

```bash
# 1. Setup
export WEBOTS_HOME=/usr/local/webots
export PYTHONPATH=$PYTHONPATH:$WEBOTS_HOME/lib/controller/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WEBOTS_HOME/lib/controller

# 2. Start Webots for training
webots --mode=fast --minimize --no-rendering worlds/epuck_5models.wbt &
WEBOTS_PID=$!
sleep 10

# 3. Train (this will take hours)
python3 controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_shaping.py \
    --run_name my_experiment \
    --total_timesteps 500000

# 4. Stop training Webots
kill $WEBOTS_PID

# 5. Start Webots for visualization
webots worlds/eval_best.wbt &
sleep 10

# 6. Test the model
python3 controllers/eval_best_model/eval_best_model.py my_experiment.zip
```

## Troubleshooting

**Training stuck at 0 reward:**
- Increase entropy coefficient (`--ent_coef 0.05`)
- Try smaller batch size (`--batch_size 2048`)
- Ensure robots are moving (check Webots visualization)

**Out of GPU memory:**
- Reduce batch size (`--batch_size 2048` or `--batch_size 1024`)

**Webots crashes:**
- Ensure only one Webots instance is running on that port
- Check system memory usage

**Model not improving:**
- Train longer (2M+ steps)
- Check TensorBoard for loss trends
- Try different learning rates

## Recommended Training Steps

| Goal | Timesteps | Time Estimate |
|------|-----------|---------------|
| Quick validation | 100,000 | 1-2 hours |
| Short experiment | 250,000 | 3-5 hours |
| Standard training | 2,000,000 | 12-24 hours |
| Extended training | 5,000,000 | 2-3 days |
| Full convergence | 10,000,000 | 5-7 days |
