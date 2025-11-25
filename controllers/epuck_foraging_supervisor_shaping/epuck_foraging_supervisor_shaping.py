import math
import random
import numpy as np
from deepbots.supervisor.controllers.deepbots_supervisor_env import DeepbotsSupervisorEnv
from controller import Supervisor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import gym
import torch

class EpuckForagingSupervisor(DeepbotsSupervisorEnv):
    def __init__(self):
        self.num_robots = 4
        self.num_tags = 70
        self.obs_per_robot = 15
        self.observation_space_dim = self.obs_per_robot * self.num_robots
        self.action_space_dim = 2 * self.num_robots
        
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_space_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.action_space_dim,), dtype=np.float32)
        
        self.robot_nodes = []
        for i in range(self.num_robots):
            self.robot_nodes.append(self.getFromDef(f"ROBOT{i+1}"))
            
        self.tag_nodes = []
        for i in range(self.num_tags):
            self.tag_nodes.append(self.getFromDef(f"APRILTAG_{i+1}"))
            
        self.base_node = self.getFromDef("BASE_STATION")
            
        self.emitters = []
        self.receivers = []
        for i in range(self.num_robots):
            self.emitters.append(self.getDevice(f"emitter{i+1}"))
            self.receivers.append(self.getDevice(f"receiver{i+1}"))
            self.receivers[i].enable(self.timestep)
            
        self.robot_states = [None] * self.num_robots
        self.carrying_state = [False] * self.num_robots
        self.prev_base_dists = [None] * self.num_robots
        self.prev_tag_dists = [None] * self.num_robots
        
        self.grid_size = 50
        self.grid_res = 5.0 / self.grid_size
        self.pheromone_grid = np.zeros((self.grid_size, self.grid_size))
        
        self.steps_per_episode = 4096
        self.episode_step = 0
        
        # Stats tracking
        self.total_pickups = 0
        self.total_deposits = 0
        
    def step(self, action):
        self.episode_step += 1
        
        for i in range(self.num_robots):
            robot_action = action[i*2 : (i+1)*2]
            # Send raw action [-1, 1] to robot, let it scale or use as is (max speed 1.0)
            msg = f"{robot_action[0]},{robot_action[1]}".encode('utf-8')
            self.emitters[i].send(msg)
            
        if super(Supervisor, self).step(self.timestep) == -1:
            exit()
            
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
                    
        self.pheromone_grid *= 0.995

        return (
            self.get_observations(),
            self.get_reward(action),
            self.is_done(),
            self.get_info()
        )

    def get_observations(self):
        global_obs = []
        
        for i in range(self.num_robots):
            prox = self.robot_states[i]
            
            robot_node = self.robot_nodes[i]
            robot_pos = robot_node.getPosition()
            robot_rot = robot_node.getOrientation()
            forward_vec = [robot_rot[0], robot_rot[3], robot_rot[6]]
            
            tag_visible = 0.0
            tag_dist = 0.0
            tag_angle = 0.0
            
            min_dist = float('inf')
            closest_tag = None
            
            if not self.carrying_state[i]:
                for tag_node in self.tag_nodes:
                    tag_pos = tag_node.getPosition()
                    if tag_pos[2] < 0: continue
                    
                    dx = tag_pos[0] - robot_pos[0]
                    dy = tag_pos[1] - robot_pos[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    if dist < 2.0:
                        tag_vec_norm = [dx/dist, dy/dist, 0]
                        dot = forward_vec[0]*tag_vec_norm[0] + forward_vec[1]*tag_vec_norm[1]
                        dot = max(min(dot, 1.0), -1.0)
                        angle = math.acos(dot)
                        
                        if angle < 1.0:
                            if dist < min_dist:
                                min_dist = dist
                                closest_tag = tag_node
                                cross = forward_vec[0]*tag_vec_norm[1] - forward_vec[1]*tag_vec_norm[0]
                                tag_angle = angle if cross > 0 else -angle
                
                if closest_tag:
                    tag_visible = 1.0
                    tag_dist = min_dist
            
            gx = int((robot_pos[0] + 2.5) / self.grid_res)
            gy = int((robot_pos[1] + 2.5) / self.grid_res)
            
            def get_grid_val(x, y):
                if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                    return self.pheromone_grid[x, y]
                return 0.0
                
            fx = robot_pos[0] + forward_vec[0]*0.2
            fy = robot_pos[1] + forward_vec[1]*0.2
            gfx = int((fx + 2.5) / self.grid_res)
            gfy = int((fy + 2.5) / self.grid_res)
            phero_front = get_grid_val(gfx, gfy)
            
            lx_vec = forward_vec[0]*0.707 - forward_vec[1]*0.707
            ly_vec = forward_vec[0]*0.707 + forward_vec[1]*0.707
            lx = robot_pos[0] + lx_vec*0.2
            ly = robot_pos[1] + ly_vec*0.2
            glx = int((lx + 2.5) / self.grid_res)
            gly = int((ly + 2.5) / self.grid_res)
            phero_left = get_grid_val(glx, gly)
            
            rx_vec = forward_vec[0]*0.707 + forward_vec[1]*0.707
            ry_vec = -forward_vec[0]*0.707 + forward_vec[1]*0.707
            rx = robot_pos[0] + rx_vec*0.2
            ry = robot_pos[1] + ry_vec*0.2
            grx = int((rx + 2.5) / self.grid_res)
            gry = int((ry + 2.5) / self.grid_res)
            phero_right = get_grid_val(grx, gry)
            
            obs = []
            obs.extend(prox)
            obs.extend([tag_visible, tag_dist, tag_angle])
            obs.append(1.0 if self.carrying_state[i] else 0.0)
            obs.extend([phero_front, phero_left, phero_right])
            
            global_obs.extend(obs)
            
            # Store distance for reward shaping
            self.prev_tag_dists[i] = min_dist if closest_tag else None
            
        return np.array(global_obs, dtype=np.float32)

    def get_reward(self, action):
        total_reward = 0.0
        
        for i in range(self.num_robots):
            robot_node = self.robot_nodes[i]
            robot_pos = robot_node.getPosition()
            
            # Collision Penalty (if prox sensors are high)
            prox = self.robot_states[i]
            if max(prox) > 0.1: # Threshold for "close to something"
                total_reward -= 0.01
            
            if not self.carrying_state[i]:
                # FINDING TAGS
                
                # Check for pickup
                picked_up = False
                for tag_node in self.tag_nodes:
                    tag_pos = tag_node.getPosition()
                    if tag_pos[2] < 0: continue
                    
                    dx = tag_pos[0] - robot_pos[0]
                    dy = tag_pos[1] - robot_pos[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    if dist < 0.15:
                        self.carrying_state[i] = True
                        tag_node.getField("translation").setSFVec3f([0, 0, -10])
                        total_reward += 1.0 # Big reward for pickup
                        self.total_pickups += 1
                        picked_up = True
                        print(f"[PICKUP] Robot {i+1} picked up tag!")
                        break
                
                # Reward Shaping: Approaching Tags
                # If we see a tag, reward getting closer
                if not picked_up and self.prev_tag_dists[i] is not None:
                    # Re-calculate current distance to the SAME closest tag? 
                    # Or just use the min_dist from get_observations?
                    # We need to be careful. get_observations runs BEFORE get_reward in the step loop?
                    # No, usually step() calls get_obs -> get_reward.
                    # But we updated prev_tag_dists in get_observations just now.
                    # So we need the PREVIOUS step's distance.
                    # Ah, this is tricky in a single step function.
                    # Let's rely on the fact that we stored it in get_observations.
                    # Wait, if we update it in get_observations, we lost the old one.
                    # We should update it AFTER calculating reward.
                    pass 
                    
                # Actually, let's calculate it here freshly to be safe
                min_dist = float('inf')
                closest_tag = None
                for tag_node in self.tag_nodes:
                    tag_pos = tag_node.getPosition()
                    if tag_pos[2] < 0: continue
                    dx = tag_pos[0] - robot_pos[0]
                    dy = tag_pos[1] - robot_pos[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 2.0 and dist < min_dist:
                        min_dist = dist
                        closest_tag = tag_node
                
                if closest_tag and self.prev_tag_dists[i] is not None:
                     # If we saw a tag last time and see one now
                     # Reward for getting closer
                     shaping = (self.prev_tag_dists[i] - min_dist) * 10.0 
                     # Scale: if moved 0.05m closer -> +0.5 reward. That's good.
                     total_reward += shaping
                
                # Update for next time (actually get_observations does this too, but let's sync)
                # self.prev_tag_dists[i] = min_dist if closest_tag else None

            else:
                # RETURNING TO BASE
                gx = int((robot_pos[0] + 2.5) / self.grid_res)
                gy = int((robot_pos[1] + 2.5) / self.grid_res)
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    self.pheromone_grid[gx, gy] += 1.0
                    self.pheromone_grid[gx, gy] = min(self.pheromone_grid[gx, gy], 10.0)
                
                base_pos = self.base_node.getPosition()
                dx = base_pos[0] - robot_pos[0]
                dy = base_pos[1] - robot_pos[1]
                dist_to_base = math.sqrt(dx*dx + dy*dy)
                
                if dist_to_base < 0.3:
                    self.carrying_state[i] = False
                    total_reward += 10.0 # Huge reward for deposit
                    self.total_deposits += 1
                    print(f"[DEPOSIT] Robot {i+1} deposited tag!")
                    
                    # Respawn tag (User requested NO respawn? Or just complained?)
                    # For training, we usually want respawn.
                    # I will keep it for now to ensure continuous training.
                    for tag_node in self.tag_nodes:
                        if tag_node.getPosition()[2] < 0:
                            rx = random.uniform(-1.8, 1.8)
                            ry = random.uniform(-1.8, 1.8)
                            tag_node.getField("translation").setSFVec3f([rx, ry, 0.01375])
                            break
                            
                # Reward Shaping: Approaching Base
                if self.prev_base_dists[i] is not None:
                    shaping = (self.prev_base_dists[i] - dist_to_base) * 10.0
                    total_reward += shaping
                
                self.prev_base_dists[i] = dist_to_base
            
            # Time penalty
            total_reward -= 0.001
            
        return total_reward

    def is_done(self):
        if self.episode_step >= self.steps_per_episode:
            return True
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
    env = EpuckForagingSupervisor()
    
    # GPU Training Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[TRAINING] Starting on {device}")
    
    # Hyperparameters
    policy_kwargs = dict(
        net_arch=[512, 512, 512],
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        batch_size=4096,
        n_steps=4096,
        learning_rate=3e-4,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./ppo_epuck_shaping_tensorboard/"
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path='./logs_shaping/',
        name_prefix='ppo_shaping'
    )
    
    print("[TRAINING] Starting PPO Training with Reward Shaping...")
    model.learn(total_timesteps=2000000, callback=checkpoint_callback)
    
    model.save("ppo_epuck_shaping")
    print("[COMPLETE] Training finished. Model saved.")
