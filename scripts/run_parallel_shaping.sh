#!/bin/bash
pkill -f webots
echo 'Starting 7 Parallel Training Instances...'
echo 'Launching Baseline (Batch 4096) on port 1234...'
webots --mode=fast --no-rendering --minimize --port=1234 /home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/shaping_1_baseline.wbt > epuck_shaping_1_baseline.log 2>&1 &
sleep 2
echo 'Launching Small Batch (2048) on port 1235...'
webots --mode=fast --no-rendering --minimize --port=1235 /home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/shaping_2_smallbatch.wbt > epuck_shaping_2_smallbatch.log 2>&1 &
sleep 2
echo 'Launching Large Batch (8192) on port 1236...'
webots --mode=fast --no-rendering --minimize --port=1236 /home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/shaping_3_largebatch.wbt > epuck_shaping_3_largebatch.log 2>&1 &
sleep 2
echo 'Launching High Entropy (0.05) on port 1237...'
webots --mode=fast --no-rendering --minimize --port=1237 /home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/shaping_4_highent.wbt > epuck_shaping_4_highent.log 2>&1 &
sleep 2
echo 'Launching Low LR (1e-4) on port 1238...'
webots --mode=fast --no-rendering --minimize --port=1238 /home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/shaping_5_lowlr.wbt > epuck_shaping_5_lowlr.log 2>&1 &
sleep 2
echo 'Launching High LR (5e-4) on port 1239...'
webots --mode=fast --no-rendering --minimize --port=1239 /home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/shaping_6_highlr.wbt > epuck_shaping_6_highlr.log 2>&1 &
sleep 2
echo 'Launching Aggressive (Small Batch + High LR + Med Ent) on port 1240...'
webots --mode=fast --no-rendering --minimize --port=1240 /home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/shaping_7_aggressive.wbt > epuck_shaping_7_aggressive.log 2>&1 &
sleep 2
echo 'All instances launched! Check logs for progress.'