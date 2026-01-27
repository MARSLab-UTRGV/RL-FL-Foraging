#!/bin/bash

# Automation Script for Training 5 Models
# Usage: ./run_5_models.sh

# Setup paths
export WEBOTS_HOME=/usr/local/webots
export PYTHONPATH=$PYTHONPATH:$WEBOTS_HOME/lib/controller/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WEBOTS_HOME/lib/controller
BASE_DIR=$(pwd)

# Launch Webots in background with extern controller
# We use --mode=fast for training speed and --minimize/--no-rendering to save resources
echo "Starting Webots..."
webots --mode=fast --minimize --no-rendering --port=1234 worlds/epuck_5models.wbt &
WEBOTS_PID=$!

echo "Webots started with PID: $WEBOTS_PID"
echo "Waiting 10s for Webots to initialize..."
sleep 10

# Function to run training
run_training() {
    NAME=$1
    LR=$2
    ENT=$3
    BATCH=$4
    
    echo "----------------------------------------------------------------"
    echo "Starting Training: $NAME"
    echo "Params: LR=$LR, ENT=$ENT, BATCH=$BATCH"
    echo "----------------------------------------------------------------"
    
    export WEBOTS_PID=$WEBOTS_PID
    export WEBOTS_PORT=1234
    
    python3 controllers/epuck_foraging_supervisor_shaping/epuck_foraging_supervisor_shaping.py \
        --run_name "$NAME" \
        --lr "$LR" \
        --ent_coef "$ENT" \
        --batch_size "$BATCH"
        
    echo "Finished: $NAME"
    echo "----------------------------------------------------------------"
    sleep 5
}

# 1. Baseline
run_training "ppo_5models_baseline" 3e-4 0.01 4096

# 2. High Entropy (More Exploration)
run_training "ppo_5models_high_ent" 3e-4 0.05 4096

# 3. High Learning Rate
run_training "ppo_5models_high_lr" 5e-4 0.01 4096

# 4. Small Batch (More Updates)
run_training "ppo_5models_small_batch" 3e-4 0.01 2048

# 5. Very Small Batch (Rapid Updates)
run_training "ppo_5models_verysmall_batch" 3e-4 0.01 1024

echo "All 5 models trained!"

# Kill Webots
kill $WEBOTS_PID
echo "Webots terminated."


