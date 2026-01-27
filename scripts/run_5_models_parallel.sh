#!/bin/bash

# Parallel Automation Script for Training 5 Models + 2 Long Runs
# Usage: ./run_5_models_parallel.sh

# Setup paths
export WEBOTS_HOME=/usr/local/webots
export PYTHONPATH=$PYTHONPATH:$WEBOTS_HOME/lib/controller/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WEBOTS_HOME/lib/controller
BASE_DIR=$(pwd)

echo "================================================================"
echo "LAUNCHING 7 PARALLEL TRAINING SESSIONS (Staggered Start)"
echo "================================================================"

# Function to launch a single training instance
launch_training() {
    NAME=$1
    LR=$2
    ENT=$3
    BATCH=$4
    STEPS=$5
    PORT=$6
    GPU_ID=$7

    echo "Launching $NAME ($STEPS steps) on Port $PORT (GPU $GPU_ID)..."
    
    # Create a unique log file for this run
    LOG_FILE="logs/${NAME}_training.log"
    mkdir -p logs
    
    # Clean log file
    echo "Starting log for $NAME" > "$LOG_FILE"

    (
        # Export unique environment variables for this sub-shell
        export WEBOTS_PORT=$PORT
        export WEBOTS_CONTROLLER_URL="tcp://127.0.0.1:$PORT/supervisor"
        export CUDA_VISIBLE_DEVICES=$GPU_ID
        
        # Debug info
        echo "CWD: $(pwd)" >> "$LOG_FILE"
        
        # Start Webots for this specific instance
        echo "Starting Webots on port $PORT..." >> "$LOG_FILE"
        webots --mode=fast --minimize --no-rendering --port=$PORT worlds/epuck_5models.wbt >> "$LOG_FILE" 2>&1 &
        WEBOTS_PID=$!
        
        echo "Webots PID: $WEBOTS_PID" >> "$LOG_FILE"
        sleep 20 # Wait 20s for Webots to fully init

        # Run the training script
        export WEBOTS_PID=$WEBOTS_PID
        echo "Starting Python training script..." >> "$LOG_FILE"
        
        python3 controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_shaping.py \
            --run_name "$NAME" \
            --lr "$LR" \
            --ent_coef "$ENT" \
            --batch_size "$BATCH" \
            --total_timesteps "$STEPS" >> "$LOG_FILE" 2>&1
            
        echo "Training Finished for $NAME" >> "$LOG_FILE"
        
        # Cleanup
        kill $WEBOTS_PID
    ) &
}

# GPU Distribution (2x A6000)
# Stagger launches by 10s to prevent startup race conditions

# 1. Baseline (2M) - GPU 0
launch_training "ppo_5models_baseline" 3e-4 0.01 4096 2000000 4001 0
sleep 10

# 2. High Entropy (2M) - GPU 0
launch_training "ppo_5models_high_ent" 3e-4 0.05 4096 2000000 4002 0
sleep 10

# 3. High LR (2M) - GPU 0
launch_training "ppo_5models_high_lr" 5e-4 0.01 4096 2000000 4003 0
sleep 10

# 4. Small Batch (2M) - GPU 0
launch_training "ppo_5models_small_batch" 3e-4 0.01 2048 2000000 4004 0
sleep 10

# 5. Very Small Batch (2M) - GPU 1
launch_training "ppo_5models_verysmall_batch" 3e-4 0.01 1024 2000000 4005 1
sleep 10

# 6. Long Run 5M - GPU 1
launch_training "ppo_5models_long_5M" 3e-4 0.01 4096 5000000 4006 1
sleep 10

# 7. Extra Long Run 10M - GPU 1
launch_training "ppo_5models_extralong_10M" 3e-4 0.01 4096 10000000 4007 1

echo "All 7 instances launched in background!"
echo "Check 'logs/*_training.log' for progress."
