#!/bin/bash

# Visualization Script
# Usage: ./scripts/run_viz.sh

export WEBOTS_HOME=/usr/local/webots
export PYTHONPATH=$PYTHONPATH:$WEBOTS_HOME/lib/controller/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$WEBOTS_HOME/lib/controller

PORT=5000
MODEL="ppo_5models_verysmall_batch" 

echo "----------------------------------------------------------------"
echo "VISUALIZING BEST MODEL: $MODEL"
echo "----------------------------------------------------------------"

# Start Webots with GUI (Realtime)
# We use a new port (5000) to avoid conflict with training
webots --mode=realtime --port=$PORT worlds/epuck_5models.wbt &
WEBOTS_PID=$!

echo "Webots launched (PID: $WEBOTS_PID). Waiting 10s..."
sleep 10

# Run Visualization Controller
export WEBOTS_CONTROLLER_URL="tcp://127.0.0.1:$PORT/supervisor"

python3 scripts/visualize_best.py --model_path "${MODEL}.zip"

# Cleanup
echo "Closing..."
kill $WEBOTS_PID


