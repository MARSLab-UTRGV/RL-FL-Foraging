import sys
import os
import argparse
import torch
from stable_baselines3 import PPO

# Add parent directory to path to import controllers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the environment class used for training
# We need to patch sys.path to find the controller module if it's not installed
try:
    from controllers.epuck_foraging_supervisor_shaping.epuck_foraging_supervisor_shaping import EpuckForagingSupervisor
except ImportError:
    print("Could not import EpuckForagingSupervisor. Checking paths...")
    # Fallback: try adding the project root
    project_root = os.path.dirname(parent_dir)
    sys.path.append(project_root)
    from RL_FL_Foraging.controllers.epuck_foraging_supervisor_shaping.epuck_foraging_supervisor_shaping import EpuckForagingSupervisor

def main():
    parser = argparse.ArgumentParser(description='Visualize Best Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .zip model file')
    args = parser.parse_args()

    print(f"Loading model from: {args.model_path}")
    
    # Initialize Environment (This connects to Webots via WEBOTS_CONTROLLER_URL)
    env = EpuckForagingSupervisor()
    
    # Load Model
    model = PPO.load(args.model_path)
    print("Model loaded successfully!")
    
    # Evaluation Loop
    obs = env.reset()
    total_reward = 0
    steps = 0
    
    print("Starting visualization...")
    while True:
        # Predict action
        action, _states = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if steps % 100 == 0:
            print(f"Step {steps}: Current Reward {reward:.4f} | Total {total_reward:.2f}")
            
        if done:
            print(f"Episode finished. Total Reward: {total_reward}")
            obs = env.reset()
            total_reward = 0
            steps = 0

if __name__ == "__main__":
    main()


