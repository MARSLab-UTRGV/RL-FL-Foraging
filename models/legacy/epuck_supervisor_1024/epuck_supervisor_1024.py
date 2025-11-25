import math
import random
import numpy as np
from deepbots.supervisor.controllers.deepbots_supervisor_env import DeepbotsSupervisorEnv
from controller import Supervisor
from stable_baselines3 import PPO
import os
import gym
import torch

class EpuckForagingSupervisor(DeepbotsSupervisorEnv):
    def __init__(self):
        # Initialize Supervisor
        self.num_robots = 4
        self.num_tags = 70
        # Observation Space:
        # 8 Proximity
        # 3 Tag Info (Visible, Dist, Angle)
        # 1 Carrying (0/1)
        # 3 Pheromone (Front, Left, Right)
        # Total = 15 per robot
        self.obs_per_robot = 15
        self.observation_space_dim = self.obs_per_robot * self.num_robots
        self.action_space_dim = 2 * self.num_robots
        
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        
        # Define Gym Spaces
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_space_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_space_dim,), dtype=np.float32)
        
        # Robot Nodes
        self.robot_nodes = []
        for i in range(self.num_robots):
            self.robot_nodes.append(self.getFromDef(f"ROBOT{i+1}"))
            
        # Tag Nodes
        self.tag_nodes = []
        for i in range(self.num_tags):
            self.tag_nodes.append(self.getFromDef(f"APRILTAG_{i+1}"))
            
        # Base Station Node
        self.base_node = self.getFromDef("BASE_STATION")
            
        # Emitters and Receivers
        self.emitters = []
        self.receivers = []
        for i in range(self.num_robots):
            self.emitters.append(self.getDevice(f"emitter{i+1}"))
            self.receivers.append(self.getDevice(f"receiver{i+1}"))
            self.receivers[i].enable(self.timestep)
            
        # State variables
        self.robot_states = [None] * self.num_robots # [prox_sensors]
        self.carrying_state = [False] * self.num_robots
        self.prev_base_dists = [None] * self.num_robots
        self.prev_tag_dists = [None] * self.num_robots
        
        # Pheromone Grid
        # Arena is 5x5m (-2.5 to 2.5)
        # Grid 50x50 -> 0.1m resolution
        self.grid_size = 50
        self.grid_res = 5.0 / self.grid_size
        self.pheromone_grid = np.zeros((self.grid_size, self.grid_size))
        
        self.steps_per_episode = 4096 # Matches n_steps
        self.episode_step = 0
        
    def step(self, action):
        self.episode_step += 1
        
        # 1. Send Actions to Robots
        for i in range(self.num_robots):
            robot_action = action[i*2 : (i+1)*2]
            msg = f"{robot_action[0]},{robot_action[1]}".encode('utf-8')
            self.emitters[i].send(msg)
            
        # 2. Step Simulation
        if super(Supervisor, self).step(self.timestep) == -1:
            exit()
            
        # 3. Receive Observations (Proximity)
        for i in range(self.num_robots):
            if self.receivers[i].getQueueLength() > 0:
                msg = self.receivers[i].getString()
                self.receivers[i].nextPacket()
                try:
                    self.robot_states[i] = [float(x) for x in msg.split(',')]
                except ValueError:
                    self.robot_states[i] = [0.0]*8
            else:
                if self.robot_states[i] is None:
                    self.robot_states[i] = [0.0]*8
                    
        # 4. Update Pheromones (Evaporation)
        self.pheromone_grid *= 0.995

        # 5. Calculate Logic & Rewards
        return (
            self.get_observations(),
            self.get_reward(action),
            self.is_done(),
            self.get_info()
        )

    def get_observations(self):
        global_obs = []
        
        for i in range(self.num_robots):
            # A. Proximity
            prox = self.robot_states[i]
            
            # B. Robot Pose
            robot_node = self.robot_nodes[i]
            robot_pos = robot_node.getPosition()
            robot_rot = robot_node.getOrientation()
            forward_vec = [robot_rot[0], robot_rot[3], robot_rot[6]] # X-axis forward
            
            # C. Simulated Vision (Tags)
            tag_visible = 0.0
            tag_dist = 0.0
            tag_angle = 0.0
            
            min_dist = float('inf')
            closest_tag = None
            
            # Only look for tags if NOT carrying
            if not self.carrying_state[i]:
                for tag_node in self.tag_nodes:
                    # Skip hidden tags (z < 0)
                    tag_pos = tag_node.getPosition()
                    if tag_pos[2] < 0: continue
                    
                    dx = tag_pos[0] - robot_pos[0]
                    dy = tag_pos[1] - robot_pos[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    if dist < 2.0: # 2m Range
                        tag_vec_norm = [dx/dist, dy/dist, 0]
                        dot = forward_vec[0]*tag_vec_norm[0] + forward_vec[1]*tag_vec_norm[1]
                        dot = max(min(dot, 1.0), -1.0)
                        angle = math.acos(dot)
                        
                        if angle < 1.0: # ~60 deg FOV
                            if dist < min_dist:
                                min_dist = dist
                                closest_tag = tag_node
                                cross = forward_vec[0]*tag_vec_norm[1] - forward_vec[1]*tag_vec_norm[0]
                                tag_angle = angle if cross > 0 else -angle
                
                if closest_tag:
                    tag_visible = 1.0
                    tag_dist = min_dist
            
            # D. Pheromone Sensors
            gx = int((robot_pos[0] + 2.5) / self.grid_res)
            gy = int((robot_pos[1] + 2.5) / self.grid_res)
            
            def get_grid_val(x, y):
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    return self.pheromone_grid[x, y]
                return 0.0
                
            # Front offset
            fx = robot_pos[0] + forward_vec[0]*0.2
            fy = robot_pos[1] + forward_vec[1]*0.2
            gfx = int((fx + 2.5) / self.grid_res)
            gfy = int((fy + 2.5) / self.grid_res)
            phero_front = get_grid_val(gfx, gfy)
            
            # Left
            lx_vec = forward_vec[0]*0.707 - forward_vec[1]*0.707
            ly_vec = forward_vec[0]*0.707 + forward_vec[1]*0.707
            lx = robot_pos[0] + lx_vec*0.2
            ly = robot_pos[1] + ly_vec*0.2
            glx = int((lx + 2.5) / self.grid_res)
            gly = int((ly + 2.5) / self.grid_res)
            phero_left = get_grid_val(glx, gly)
            
            # Right
            rx_vec = forward_vec[0]*0.707 + forward_vec[1]*0.707
            ry_vec = -forward_vec[0]*0.707 + forward_vec[1]*0.707
            rx = robot_pos[0] + rx_vec*0.2
            ry = robot_pos[1] + ry_vec*0.2
            grx = int((rx + 2.5) / self.grid_res)
            gry = int((ry + 2.5) / self.grid_res)
            phero_right = get_grid_val(grx, gry)
            
            # E. Construct Observation
            obs = []
            obs.extend(prox)
            obs.extend([tag_visible, tag_dist, tag_angle])
            obs.append(1.0 if self.carrying_state[i] else 0.0)
            obs.extend([phero_front, phero_left, phero_right])
            
            global_obs.extend(obs)
            
            # Store state for reward shaping
            self.prev_tag_dists[i] = min_dist if closest_tag else None
            
        return np.array(global_obs, dtype=np.float32)

    def get_reward(self, action):
        total_reward = 0.0
        
        for i in range(self.num_robots):
            robot_node = self.robot_nodes[i]
            robot_pos = robot_node.getPosition()
            
            # 1. Logic: Pick Up
            if not self.carrying_state[i]:
                for tag_node in self.tag_nodes:
                    tag_pos = tag_node.getPosition()
                    if tag_pos[2] < 0: continue
                    
                    dx = tag_pos[0] - robot_pos[0]
                    dy = tag_pos[1] - robot_pos[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    if dist < 0.15:
                        self.carrying_state[i] = True
                        tag_node.getField("translation").setSFVec3f([0, 0, -10])
                        total_reward += 1.0
                        break
                        
            # 2. Logic: Deposit
            else:
                # Deposit Pheromones
                gx = int((robot_pos[0] + 2.5) / self.grid_res)
                gy = int((robot_pos[1] + 2.5) / self.grid_res)
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    self.pheromone_grid[gx, gy] += 1.0
                    self.pheromone_grid[gx, gy] = min(self.pheromone_grid[gx, gy], 10.0)
                
                # Check distance to Base
                base_pos = self.base_node.getPosition()
                dx = base_pos[0] - robot_pos[0]
                dy = base_pos[1] - robot_pos[1]
                dist_to_base = math.sqrt(dx*dx + dy*dy)
                
                if dist_to_base < 0.3:
                    self.carrying_state[i] = False
                    total_reward += 10.0
                    
                    # Respawn tag
                    for tag_node in self.tag_nodes:
                        if tag_node.getPosition()[2] < 0:
                            rx = random.uniform(-1.8, 1.8)
                            ry = random.uniform(-1.8, 1.8)
                            tag_node.getField("translation").setSFVec3f([rx, ry, 0.01375])
                            break
                            
            # 3. Shaping Rewards
            if self.carrying_state[i]:
                base_pos = self.base_node.getPosition()
                dx = base_pos[0] - robot_pos[0]
                dy = base_pos[1] - robot_pos[1]
                dist = math.sqrt(dx*dx + dy*dy)
                if self.prev_base_dists[i] is not None:
                    reward = (self.prev_base_dists[i] - dist) * 2.0
                    total_reward += reward
                self.prev_base_dists[i] = dist
            else:
                self.prev_base_dists[i] = None
            
            # 4. Time Penalty
            total_reward -= 0.001
            
        return total_reward

    def is_done(self):
        return False

    def get_info(self):
        return {}
    
    def reset(self):
        self.respawn_robots()
        self.episode_step = 0
        self.robot_states = [[0.0]*8 for _ in range(self.num_robots)]
        self.carrying_state = [False] * self.num_robots
        self.pheromone_grid.fill(0.0)
        self.prev_base_dists = [None] * self.num_robots
        self.prev_tag_dists = [None] * self.num_robots
        return self.get_observations()

    def respawn_robots(self):
        positions = [
            [-0.5, 0, 0],
            [0.5, 0, 0],
            [0, 0.5, 0],
            [0, -0.5, 0]
        ]
        for i in range(self.num_robots):
            self.robot_nodes[i].getField("translation").setSFVec3f(positions[i])
            self.robot_nodes[i].getField("rotation").setSFRotation([0, 0, 1, 0])
            self.robot_nodes[i].resetPhysics()

if __name__ == "__main__":
    # GPU Check
    print("="*60)
    print("FAST MODE GPU TRAINING")
    print("="*60)
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Count: {torch.cuda.device_count()}")
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print("="*60)
    
    env = EpuckForagingSupervisor()
    
    model_path = "ppo_1024"  # Different path to avoid conflict
    
    # HEAVY Training Config (Optimized for GPU)
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[512, 512, 512]
    )
    
    if os.path.exists(model_path + ".zip"):
        print("Loading existing FAST model...")
        model = PPO.load(model_path, env=env)
    else:
        print("Creating new FAST model...")
        print("Network: [512, 512, 512] neurons")
        print("Batch Size: 1024 (GPU optimized)")
        model = PPO(
            "MlpPolicy", 
            env, 
            verbose=1, 
            tensorboard_log="./ppo_1024_tensorboard/",
            policy_kwargs=policy_kwargs,
            n_steps=1024,  # Batch size
            batch_size=1024,
            ent_coef=0.01,
            learning_rate=3e-4,
            device='cuda'
        )
    
    print("\n" + "="*60)
    print("STARTING FAST GPU TRAINING")
    print("="*60)
    TIMESTEPS = 2000000  # 2M steps
    print(f"Total Timesteps: {TIMESTEPS:,}")
    print(f"Batch size (n_steps): 1024")
    print(f"Estimated Updates: {TIMESTEPS // 1024}")
    print("="*60 + "\n")
    
    model.learn(total_timesteps=TIMESTEPS)
    model.save(model_path)
    
    print("\n" + "="*60)
    print("FAST TRAINING COMPLETE!")
    print("="*60)
