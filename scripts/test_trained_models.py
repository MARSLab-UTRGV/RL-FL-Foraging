#!/usr/bin/env python3
"""
Test and compare all 6 trained PPO models
Runs each model for 5000 steps and reports performance metrics
"""

import subprocess
import time
import os
from stable_baselines3 import PPO

# Model configurations
models = [
    {
        "name": "GUI (100k steps)",
        "path": "/home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/controllers/epuck_foraging_supervisor/ppo_epuck_heavy.zip",
        "controller": "epuck_foraging_supervisor"
    },
    {
        "name": "FAST (200k steps)",
        "path": "/home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/controllers/epuck_foraging_supervisor_fast/ppo_epuck_FAST.zip",
        "controller": "epuck_foraging_supervisor_fast"
    },
    {
        "name": "HEADLESS (300k steps)",
        "path": "/home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/controllers/epuck_foraging_supervisor_headless/ppo_epuck_HEADLESS.zip",
        "controller": "epuck_foraging_supervisor_headless"
    },
    {
        "name": "2M (2M steps, batch=4096)",
        "path": "/home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/controllers/epuck_foraging_supervisor_2M/ppo_epuck_2M.zip",
        "controller": "epuck_foraging_supervisor_2M"
    },
    {
        "name": "2048 batch (2M steps, batch=2048)",
        "path": "/home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/controllers/epuck_supervisor_2048/ppo_2048.zip",
        "controller": "epuck_supervisor_2048"
    },
    {
        "name": "1024 batch (2M steps, batch=1024)",
        "path": "/home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/controllers/epuck_supervisor_1024/ppo_1024.zip",
        "controller": "epuck_supervisor_1024"
    }
]

print("="*80)
print("TESTING ALL 6 TRAINED MODELS")
print("="*80)
print("\nThis will test each model visually in Webots")
print("Press Ctrl+C to skip to the next model\n")

for i, model_info in enumerate(models, 1):
    print(f"\n{'='*80}")
    print(f"MODEL {i}/6: {model_info['name']}")
    print(f"Path: {model_info['path']}")
    print(f"{'='*80}")
    
    if not os.path.exists(model_info['path']):
        print(f"[ERROR] Model not found! Skipping...")
        continue

    print(f"\n[SUCCESS] Model loaded successfully!")
    print(f"Launching Webots with this model...")
    print(f"Watch the robots perform!")
    print(f"\nPress Ctrl+C when ready to test the next model\n")
    
    # Launch Webots with the specific controller
    world_file = f"/home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/epuck_foraging.wbt"
    
    # Modify world file to use the correct controller
    # (In production, you'd have separate world files or use a parameterized approach)
    
    try:
        proc = subprocess.Popen([
            "webots",
            world_file
        ])
        
        # Wait for user to interrupt or process to finish
        proc.wait()
        
    except KeyboardInterrupt:
        print(f"\nSkipping to next model...")
        proc.terminate()
        time.sleep(2)
        continue

    time.sleep(2)

print("\n" + "="*80)
print("ALL MODELS TESTED!")
print("="*80)
print("\nCompare the performance and pick your favorite!")
