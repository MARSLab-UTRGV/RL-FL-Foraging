import os

BASE_DIR = "/home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project"
CONTROLLERS_DIR = os.path.join(BASE_DIR, "controllers")
WORLDS_DIR = os.path.join(BASE_DIR, "worlds")

BASE_CONTROLLER_PATH = os.path.join(CONTROLLERS_DIR, "epuck_foraging_supervisor_shaping", "epuck_foraging_supervisor_shaping.py")
BASE_WORLD_PATH = os.path.join(WORLDS_DIR, "epuck_foraging_shaping.wbt")

with open(BASE_CONTROLLER_PATH, 'r') as f:
    base_controller_code = f.read()

with open(BASE_WORLD_PATH, 'r') as f:
    base_world_code = f.read()

# 5 KEY CONFIGS: 2 Short + 3 Long
configs = [
    {
        "id": "quick100k",
        "timesteps": 100000,
        "batch_size": 2048, "n_steps": 2048, "lr": "5e-4", "ent_coef": 0.02,
        "desc": "Quick 100k Test"
    },
    {
        "id": "short250k",
        "timesteps": 250000,
        "batch_size": 4096, "n_steps": 4096, "lr": "3e-4", "ent_coef": 0.01,
        "desc": "Short 250k Baseline"
    },
    {
        "id": "baseline2M",
        "timesteps": 2000000,
        "batch_size": 4096, "n_steps": 4096, "lr": "3e-4", "ent_coef": 0.01,
        "desc": "Baseline 2M"
    },
    {
        "id": "smallbatch2M",
        "timesteps": 2000000,
        "batch_size": 2048, "n_steps": 2048, "lr": "3e-4", "ent_coef": 0.01,
        "desc": "Small Batch 2M"
    },
    {
        "id": "highent2M",
        "timesteps": 2000000,
        "batch_size": 4096, "n_steps": 4096, "lr": "3e-4", "ent_coef": 0.05,
        "desc": "High Entropy 2M"
    }
]

run_script_lines = ["#!/bin/bash", "echo 'Starting 5 Key Training Runs...'", ""]

for i, conf in enumerate(configs):
    name = f"epuck_top5_{conf['id']}"
    
    # Create Controller
    ctl_dir = os.path.join(CONTROLLERS_DIR, name)
    os.makedirs(ctl_dir, exist_ok=True)
    
    code = base_controller_code
    code = code.replace("batch_size=4096", f"batch_size={conf['batch_size']}")
    code = code.replace("n_steps=4096", f"n_steps={conf['n_steps']}")
    code = code.replace("learning_rate=3e-4", f"learning_rate={conf['lr']}")
    code = code.replace("ent_coef=0.01", f"ent_coef={conf['ent_coef']}")
    code = code.replace("total_timesteps=2000000", f"total_timesteps={conf['timesteps']}")
    code = code.replace("tensorboard_log=\"./ppo_epuck_shaping_tensorboard/\"", f"tensorboard_log=\"./tb_{name}/\"")
    code = code.replace("save_path='./logs_shaping/'", f"save_path='./logs_{name}/'")
    code = code.replace("name_prefix='ppo_shaping'", f"name_prefix='ppo_{name}'")
    code = code.replace("model.save(\"ppo_epuck_shaping\")", f"model.save(\"ppo_{name}\")")
    
    with open(os.path.join(ctl_dir, f"{name}.py"), 'w') as f:
        f.write(code)
    
    # Create World
    world_name = f"top5_{conf['id']}.wbt"
    world_path = os.path.join(WORLDS_DIR, world_name)
    w_code = base_world_code.replace('controller "epuck_foraging_supervisor_shaping"', f'controller "{name}"')
    with open(world_path, 'w') as f:
        f.write(w_code)
    
    # Run command (stagger ports)
    port = 1234 + i
    
    # First instance gets GUI for monitoring
    if i == 0:
        cmd = f"webots --port={port} {world_path} > {name}.log 2>&1 &"
        run_script_lines.append(f"echo '[{i+1}/5] {conf['desc']} (GUI) - Port {port}'")
    else:
        cmd = f"webots --mode=fast --no-rendering --minimize --port={port} {world_path} > {name}.log 2>&1 &"
        run_script_lines.append(f"echo '[{i+1}/5] {conf['desc']} (Background) - Port {port}'")
    
    run_script_lines.append(cmd)
    run_script_lines.append("sleep 5  # Stagger starts to avoid overload")
    run_script_lines.append("")

run_script_lines.append("echo 'All 5 instances launched!'")
run_script_lines.append("echo 'Monitor: tail -f epuck_top5_*.log'")

with open(os.path.join(BASE_DIR, "run_top5.sh"), 'w') as f:
    f.write("\n".join(run_script_lines))

print("[SUCCESS] 5 Key Configs Ready!")
