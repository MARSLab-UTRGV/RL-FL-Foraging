#!/usr/bin/env python3
"""
Simple Parallel Training Launcher
Launches 8 independent Webots training instances
They don't communicate but train 8x faster by running in parallel
"""

import subprocess
import time
import os

# Base directory
BASE_DIR = "/home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project"

# Number of parallel instances
NUM_INSTANCES = 8

print("="*60)
print(f"LAUNCHING {NUM_INSTANCES} PARALLEL TRAINING INSTANCES")
print("Each instance trains independently on GPU 1")
print("="*60)

processes = []

for i in range(NUM_INSTANCES):
    print(f"\nLaunching instance {i+1}/{NUM_INSTANCES}...")
    
    # Each instance uses a unique port
    port = 2000 + i
    
    # Launch Webots in fast mode
    cmd = [
        "webots",
        "--mode=fast",
        "--minimize", 
        "--no-rendering",
        f"--port={port}",
        f"{BASE_DIR}/worlds/epuck_foraging_fast.wbt"
    ]
    
    # Redirect output to log file
    log_file = open(f"{BASE_DIR}/parallel_instance_{i}.log", "w")
    
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=log_file,
        cwd=BASE_DIR
    )
    
    processes.append((proc, log_file))
    
    # Stagger launches to avoid overwhelming the system
    time.sleep(3)

print("\n" + "="*60)
print(f"All {NUM_INSTANCES} instances launched!")
print(f"Check logs: parallel_instance_*.log")
print("="*60)
print("\nNote: Each trains the SAME model type but independently.")
print("After training, you can average their weights for better performance!")
print("\nPress Ctrl+C to stop all instances")
print("="*60)

try:
    # Wait for all processes
    for proc, log_file in processes:
        proc.wait()
        log_file.close()
except KeyboardInterrupt:
    print("\nStopping all instances...")
    for proc, log_file in processes:
        proc.terminate()
        log_file.close()
    print("All instances stopped!")
