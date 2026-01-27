import os
import re

log_dir = "RL-FL-Foraging/logs"
logs = [f for f in os.listdir(log_dir) if f.endswith("_training.log")]

results = []

for log_file in logs:
    path = os.path.join(log_dir, log_file)
    model_name = log_file.replace("_training.log", "")
    
    last_reward = None
    max_reward = -float('inf')
    
    with open(path, "r") as f:
        for line in f:
            if "ep_rew_mean" in line:
                # Extract number: | ep_rew_mean | 12.34 |
                match = re.search(r"\|\s*ep_rew_mean\s*\|\s*([\d\.-]+)\s*\|", line)
                if match:
                    val = float(match.group(1))
                    last_reward = val
                    if val > max_reward:
                        max_reward = val
                        
    results.append({
        "name": model_name,
        "final": last_reward if last_reward is not None else 0.0,
        "max": max_reward if max_reward != -float('inf') else 0.0
    })

# Sort by Final Reward
results.sort(key=lambda x: x["final"], reverse=True)

print(f"{'Model Name':<35} | {'Final Reward':<12} | {'Max Reward':<12}")
print("-" * 65)
for res in results:
    print(f"{res['name']:<35} | {res['final']:<12.2f} | {res['max']:<12.2f}")


