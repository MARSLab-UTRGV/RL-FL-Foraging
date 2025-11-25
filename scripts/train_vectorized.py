#!/usr/bin/env python3
"""
Vectorized PPO Training with 8 Parallel Webots Instances
Uses Stable-Baselines3's SubprocVecEnv for true parallelism
"""

import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import subprocess
import time

def make_env(rank, seed=0):
    """
    Create a single Webots environment.
    Each environment runs in its own Webots instance with a unique port.
    """
    def _init():
        # Launch Webots instance with unique port
        port = 1234 + rank
        world_file = "/home/andres2020/Dev/deepbots_test/deepbots-tutorials/emitterReceiverSchemeTutorial/full_project/worlds/epuck_foraging_fast.wbt"
        
        # Start Webots in background (headless, fast mode)
        webots_cmd = [
            "webots",
            "--mode=fast",
            "--minimize",
            "--no-rendering",
            f"--port={port}",
            world_file
        ]
        
        # This will return after Webots starts
        proc = subprocess.Popen(
            webots_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Give Webots time to initialize
        time.sleep(2)
        
        # Import the supervisor environment
        # Note: This connects to the already-running Webots instance
        from controllers.vectorized_supervisor.vectorized_supervisor import supervisor
        
        # Set seed for reproducibility
        #supervisor.seed(seed + rank)
        return supervisor
    
    set_random_seed(seed)
    return _init

if __name__ == "__main__":
    print("="*60)
    print("VECTORIZED PARALLEL TRAINING - 8 WEBOTS INSTANCES")
    print("="*60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"Using GPU 1: {torch.cuda.get_device_name(1)}")
    print("="*60)
    
    # Number of parallel environments
    num_envs = 8
    
    print(f"\nLaunching {num_envs} parallel Webots instances...")
    print("This may take 20-30 seconds...")
    
    # Create vectorized environment
    env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    
    print("\nAll environments initialized!")
    
    # PPO Configuration for GPU 1
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[512, 512, 512]
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_vectorized_tensorboard/",
        policy_kwargs=policy_kwargs,
        n_steps=2048,  # Per environment
        batch_size=4096,
        ent_coef=0.01,
        learning_rate=3e-4,
        device='cuda:1',  # GPU 1
        n_epochs=10
    )
    
    print("\n" + "="*60)
    print("STARTING VECTORIZED TRAINING")
    print(f"Total environments: {num_envs}")
    print(f"Steps per env: 2048")
    print(f"Total steps per update: {num_envs * 2048}")
    print(f"Training on: GPU 1 (NVIDIA RTX A6000)")
    print("="*60 + "\n")
    
    # Train for 500k timesteps (across all envs)
    model.learn(total_timesteps=500000)
    
    # Save model
    model.save("ppo_vectorized_model")
    
    print("\n" + "="*60)
    print("VECTORIZED TRAINING COMPLETE!")
    print("Model saved to: ppo_vectorized_model.zip")
    print("="*60)
    
    # Cleanup
    env.close()
