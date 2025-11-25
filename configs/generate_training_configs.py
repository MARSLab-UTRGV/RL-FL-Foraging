import os
import shutil

# Base paths
BASE_DIR = "/home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project"
CONTROLLERS_DIR = os.path.join(BASE_DIR, "controllers")
WORLDS_DIR = os.path.join(BASE_DIR, "worlds")

# Base files to copy from
BASE_CONTROLLER_PATH = os.path.join(CONTROLLERS_DIR, "epuck_foraging_supervisor_shaping", "epuck_foraging_supervisor_shaping.py")
BASE_WORLD_PATH = os.path.join(WORLDS_DIR, "epuck_foraging_shaping.wbt")

# Read base content
with open(BASE_CONTROLLER_PATH, 'r') as f:
    base_controller_code = f.read()

with open(BASE_WORLD_PATH, 'r') as f:
    base_world_code = f.read()

# Configurations
configs = [
    {
        "id": "1_baseline",
        "batch_size": 4096, "n_steps": 4096, "lr": "3e-4", "ent_coef": 0.01,
        "desc": "Baseline (Batch 4096)"
    },
    {
        "id": "2_smallbatch",
        "batch_size": 2048, "n_steps": 2048, "lr": "3e-4", "ent_coef": 0.01,
        "desc": "Small Batch (2048)"
    },
    {
        "id": "3_largebatch",
        "batch_size": 8192, "n_steps": 8192, "lr": "3e-4", "ent_coef": 0.01,
        "desc": "Large Batch (8192)"
    },
    {
        "id": "4_highent",
        "batch_size": 4096, "n_steps": 4096, "lr": "3e-4", "ent_coef": 0.05,
        "desc": "High Entropy (0.05)"
    },
    {
        "id": "5_lowlr",
        "batch_size": 4096, "n_steps": 4096, "lr": "1e-4", "ent_coef": 0.01,
        "desc": "Low LR (1e-4)"
    },
    {
        "id": "6_highlr",
        "batch_size": 4096, "n_steps": 4096, "lr": "5e-4", "ent_coef": 0.01,
        "desc": "High LR (5e-4)"
    },
    {
        "id": "7_aggressive",
        "batch_size": 2048, "n_steps": 2048, "lr": "5e-4", "ent_coef": 0.03,
        "desc": "Aggressive (Small Batch + High LR + Med Ent)"
    }
]

run_script_lines = ["#!/bin/bash", "pkill -f webots", "echo 'Starting 7 Parallel Training Instances...'"]

for conf in configs:
    name = f"epuck_shaping_{conf['id']}"
    
    # 1. Create Controller Directory
    ctl_dir = os.path.join(CONTROLLERS_DIR, name)
    os.makedirs(ctl_dir, exist_ok=True)
    
    # 2. Customize Controller Code
    code = base_controller_code
    code = code.replace("batch_size=4096", f"batch_size={conf['batch_size']}")
    code = code.replace("n_steps=4096", f"n_steps={conf['n_steps']}")
    code = code.replace("learning_rate=3e-4", f"learning_rate={conf['lr']}")
    code = code.replace("ent_coef=0.01", f"ent_coef={conf['ent_coef']}")
    code = code.replace("tensorboard_log=\"./ppo_epuck_shaping_tensorboard/\"", f"tensorboard_log=\"./tensorboard_{name}/\"")
    code = code.replace("save_path='./logs_shaping/'", f"save_path='./logs_{name}/'")
    code = code.replace("name_prefix='ppo_shaping'", f"name_prefix='ppo_{name}'")
    code = code.replace("model.save(\"ppo_epuck_shaping\")", f"model.save(\"ppo_{name}\")")
    
    ctl_path = os.path.join(ctl_dir, f"{name}.py")
    with open(ctl_path, 'w') as f:
        f.write(code)
        
    # 3. Customize World File
    world_name = f"shaping_{conf['id']}.wbt"
    world_path = os.path.join(WORLDS_DIR, world_name)
    
    w_code = base_world_code.replace('controller "epuck_foraging_supervisor_shaping"', f'controller "{name}"')
    
    with open(world_path, 'w') as f:
        f.write(w_code)
        
    # 4. Add to Run Script
    # Use different ports to avoid conflicts (1234 is default, increment)
    # Actually Webots handles ports automatically if we don't specify, but better to be safe?
    # No, the warning said "use --port option".
    port = 1234 + configs.index(conf)
    cmd = f"webots --mode=fast --no-rendering --minimize --port={port} {world_path} > {name}.log 2>&1 &"
    run_script_lines.append(f"echo 'Launching {conf['desc']} on port {port}...'")
    run_script_lines.append(cmd)
    run_script_lines.append("sleep 2") # Stagger starts

run_script_lines.append("echo 'All instances launched! Check logs for progress.'")

with open(os.path.join(BASE_DIR, "run_parallel_shaping.sh"), 'w') as f:
    f.write("\n".join(run_script_lines))

print("Generation Complete!")
