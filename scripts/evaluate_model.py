#!/usr/bin/env python3
"""
Single model evaluation script
Usage: python evaluate_model.py <model_name>
Example: python evaluate_model.py ppo_epuck_2M
"""

import sys
import os

if len(sys.argv) < 2:
    print("Usage: python evaluate_model.py <model_name>")
    print("\nAvailable models:")
    print("  - ppo_epuck_heavy (GUI, 100k)")
    print("  - ppo_epuck_FAST (FAST, 200k)")
    print("  - ppo_epuck_HEADLESS (HEADLESS, 300k)")
    print("  - ppo_epuck_2M (2M steps, batch=4096)")
    print("  - ppo_2048 (2M steps, batch=2048)")
    print("  - ppo_1024 (2M steps, batch=1024)")
    sys.exit(1)

model_name = sys.argv[1]

# Map model names to their locations and controllers
model_map = {
    "ppo_epuck_heavy": ("epuck_foraging_supervisor", "epuck_foraging.wbt"),
    "ppo_epuck_FAST": ("epuck_foraging_supervisor_fast", "epuck_foraging_fast.wbt"),
    "ppo_epuck_HEADLESS": ("epuck_foraging_supervisor_headless", "epuck_foraging_headless.wbt"),
    "ppo_epuck_2M": ("epuck_foraging_supervisor_2M", "epuck_foraging_2M.wbt"),
    "ppo_2048": ("epuck_supervisor_2048", "epuck_2048.wbt"),
    "ppo_1024": ("epuck_supervisor_1024", "epuck_1024.wbt")
}

if model_name not in model_map:
    print(f"[ERROR] Unknown model: {model_name}")
    print("Use one of:", list(model_map.keys()))
    sys.exit(1)

controller_name, world_file = model_map[model_name]

print("="*60)
print(f"EVALUATING MODEL: {model_name}")
print("="*60)
print(f"Controller: {controller_name}")
print(f"World: {world_file}")
print("\nLaunching Webots...")
print("Watch the trained robots perform!\n")

# Launch Webots
import subprocess
world_path = f"/home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/{world_file}"
subprocess.run(["webots", world_path])
