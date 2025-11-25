import os

# Base paths
BASE_DIR = "/home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project"
CONTROLLERS_DIR = os.path.join(BASE_DIR, "controllers")
WORLDS_DIR = os.path.join(BASE_DIR, "worlds")

# Base files
BASE_CONTROLLER_PATH = os.path.join(CONTROLLERS_DIR, "epuck_foraging_supervisor_shaping", "epuck_foraging_supervisor_shaping.py")
BASE_WORLD_PATH = os.path.join(WORLDS_DIR, "epuck_foraging_shaping.wbt")

with open(BASE_CONTROLLER_PATH, 'r') as f:
    base_controller_code = f.read()

with open(BASE_WORLD_PATH, 'r') as f:
    base_world_code = f.read()

# 4 Additional Configurations with varied timesteps
configs = [
    {
        "id": "8_quick100k",
        "timesteps": 100000,
        "batch_size": 2048, "n_steps": 2048, "lr": "5e-4", "ent_coef": 0.02,
        "desc": "Quick 100k (Fast LR, Small Batch)"
    },
    {
        "id": "9_short250k",
        "timesteps": 250000,
        "batch_size": 4096, "n_steps": 4096, "lr": "3e-4", "ent_coef": 0.01,
        "desc": "Short 250k (Baseline Config)"
    },
    {
        "id": "10_med500k",
        "timesteps": 500000,
        "batch_size": 4096, "n_steps": 4096, "lr": "3e-4", "ent_coef": 0.01,
        "desc": "Medium 500k (Baseline Config)"
    },
    {
        "id": "11_long1M",
        "timesteps": 1000000,
        "batch_size": 4096, "n_steps": 4096, "lr": "3e-4", "ent_coef": 0.01,
        "desc": "Long 1M (Baseline Config)"
    }
]

run_script_lines = ["#!/bin/bash", "echo 'Starting 4 Additional Training Instances...'"]

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
    code = code.replace("total_timesteps=2000000", f"total_timesteps={conf['timesteps']}")
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
        
    # 4. Add to Run Script (ports 1241-1244)
    port = 1241 + configs.index(conf)
    cmd = f"webots --mode=fast --no-rendering --minimize --port={port} {world_path} > {name}.log 2>&1 &"
    run_script_lines.append(f"echo 'Launching {conf['desc']} on port {port}...'")
    run_script_lines.append(cmd)
    run_script_lines.append("sleep 2")

run_script_lines.append("echo '4 Additional instances launched!'")

with open(os.path.join(BASE_DIR, "run_4more_shaping.sh"), 'w') as f:
    f.write("\n".join(run_script_lines))

print("4 Additional Configs Generated!")
