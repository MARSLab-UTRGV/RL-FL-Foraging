#!/bin/bash
echo 'Starting 4 Additional Training Instances...'
echo 'Launching Quick 100k (Fast LR, Small Batch) on port 1241...'
webots --mode=fast --no-rendering --minimize --port=1241 /home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/shaping_8_quick100k.wbt > epuck_shaping_8_quick100k.log 2>&1 &
sleep 2
echo 'Launching Short 250k (Baseline Config) on port 1242...'
webots --mode=fast --no-rendering --minimize --port=1242 /home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/shaping_9_short250k.wbt > epuck_shaping_9_short250k.log 2>&1 &
sleep 2
echo 'Launching Medium 500k (Baseline Config) on port 1243...'
webots --mode=fast --no-rendering --minimize --port=1243 /home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/shaping_10_med500k.wbt > epuck_shaping_10_med500k.log 2>&1 &
sleep 2
echo 'Launching Long 1M (Baseline Config) on port 1244...'
webots --mode=fast --no-rendering --minimize --port=1244 /home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/shaping_11_long1M.wbt > epuck_shaping_11_long1M.log 2>&1 &
sleep 2
echo '4 Additional instances launched!'