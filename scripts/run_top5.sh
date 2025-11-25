#!/bin/bash
echo '🔥 Starting 5 Key Training Runs...'

echo '▶️  [1/5] Quick 100k Test (GUI) - Port 1234'
webots --port=1234 /home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/top5_quick100k.wbt > epuck_top5_quick100k.log 2>&1 &
sleep 5  # Stagger starts to avoid overload

echo '▶️  [2/5] Short 250k Baseline (Background) - Port 1235'
webots --mode=fast --no-rendering --minimize --port=1235 /home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/top5_short250k.wbt > epuck_top5_short250k.log 2>&1 &
sleep 5  # Stagger starts to avoid overload

echo '▶️  [3/5] Baseline 2M (Background) - Port 1236'
webots --mode=fast --no-rendering --minimize --port=1236 /home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/top5_baseline2M.wbt > epuck_top5_baseline2M.log 2>&1 &
sleep 5  # Stagger starts to avoid overload

echo '▶️  [4/5] Small Batch 2M (Background) - Port 1237'
webots --mode=fast --no-rendering --minimize --port=1237 /home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/top5_smallbatch2M.wbt > epuck_top5_smallbatch2M.log 2>&1 &
sleep 5  # Stagger starts to avoid overload

echo '▶️  [5/5] High Entropy 2M (Background) - Port 1238'
webots --mode=fast --no-rendering --minimize --port=1238 /home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/top5_highent2M.wbt > epuck_top5_highent2M.log 2>&1 &
sleep 5  # Stagger starts to avoid overload

echo '✅ All 5 instances launched!'
echo '📊 Monitor: tail -f epuck_top5_*.log'