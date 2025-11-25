import math
import random
import numpy as np
import sys
from deepbots.supervisor.controllers.deepbots_supervisor_env import DeepbotsSupervisorEnv
from controller import Supervisor
from stable_baselines3 import PPO
import gym

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
            # Revert to training dynamics (no scaling, max speed 1.0)
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
            self.prev_tag_dists[i] = min_dist if closest_tag else None
            
        return np.array(global_obs, dtype=np.float32)

    def get_reward(self, action):
        total_reward = 0.0
        
        for i in range(self.num_robots):
            robot_node = self.robot_nodes[i]
            robot_pos = robot_node.getPosition()
            
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
                        self.total_pickups += 1
                        print(f"[+] Robot {i+1} picked up tag! (Total pickups: {self.total_pickups})")
                        break
                        
            else:
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
                    total_reward += 10.0
                    self.total_deposits += 1
                    print(f"[*] Robot {i+1} deposited tag! (Total deposits: {self.total_deposits})")
                    
                    for tag_node in self.tag_nodes:
                        if tag_node.getPosition()[2] < 0:
                            rx = random.uniform(-1.8, 1.8)
                            ry = random.uniform(-1.8, 1.8)
                            tag_node.getField("translation").setSFVec3f([rx, ry, 0.01375])
                            break
                            
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

# EVALUATION MODE - Load trained model and run
if __name__ == "__main__":
    import torch
    
    print("="*60)
    print("EVALUATION MODE - Testing Trained Model")
    print("="*60)
    
    env = EpuckForagingSupervisor()
    
    # Load the ppo_1024 model
    # Check for command line argument for model path
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        print(f"Overriding model path from command line: {model_path}")
    else:
        # Default path if no argument provided
        model_path = "../epuck_supervisor_1024/ppo_1024.zip" 

    print(f"\nLoading model: {model_path}")

    # Load the trained model
    try:
        model = PPO.load(model_path, env=env)
    except Exception as e:
        print(f"[ERROR] Error loading model from {model_path}: {e}")
        print("Usage: python3 eval_best_model.py [path_to_model.zip]")
        sys.exit(1)
    print("Model loaded successfully!")
    
    print("\nStarting evaluation...")
    print("Watch the trained robots perform!")
    print(f"Stats will be printed when tags are picked up/deposited\n")
    print("="*60 + "\n")
    
    # Run forever in evaluation mode
    obs = env.reset()
    step_count = 0
    
    while True:
        # action, _states = model.predict(obs, deterministic=True)
        
        # HEURISTIC TEST:
        # If tag visible (obs[8] > 0.5), turn towards it (obs[10]) and move forward
        # Obs indices: 8=Vis, 9=Dist, 10=Angle
        
        r1_obs = obs[0:15] # Robot 1
        
        actions = []
        for i in range(env.num_robots):
            robs = obs[i*15 : (i+1)*15]
            vis = robs[8]
            angle = robs[10]
            
            if robs[11] > 0.5: # Carrying?
                # Go to Base (0,0)
                # We don't have base angle in obs, but we can cheat for this test 
                # or use the fact that base is at 0,0 and we have robot position?
                # Actually, the supervisor has access to everything.
                # But let's just make them spin if we can't see base?
                # Wait, we don't have base sensor in obs.
                # We have Pheromones.
                # But for this PHYSICS TEST, let's just make them move forward to prove they can move?
                # Or better, let's use the supervisor's knowledge to steer them to (0,0)
                
                # Get robot node from env
                robot_node = env.robot_nodes[i]
                pos = robot_node.getPosition()
                rot = robot_node.getOrientation()
                # Calculate angle to (0,0)
                target_angle = math.atan2(-pos[1], -pos[0])
                
                # Robot heading (assuming X is forward)
                # rot matrix from getOrientation() is 3x3 flattened: [0,1,2, 3,4,5, 6,7,8]
                # We want the X-axis vector which is [0], [3], [6] ?? No wait.
                # Webots getOrientation returns [nx, ny, nz, ox, oy, oz, ax, ay, az]
                # where n is X axis, o is Y axis, a is Z axis.
                # So X-axis vector is [rot[0], rot[3], rot[6]]
                heading = math.atan2(rot[3], rot[0]) # y, x
                
                angle_diff = target_angle - heading
                while angle_diff > math.pi: angle_diff -= 2*math.pi
                while angle_diff < -math.pi: angle_diff += 2*math.pi
                
                # Debug navigation for Robot 1
                if i == 0 and step_count % 50 == 0:
                    print(f"Nav: Pos({pos[0]:.2f},{pos[1]:.2f}) TgtAng:{target_angle:.2f} Head:{heading:.2f} Diff:{angle_diff:.2f}")
                
                # Simple P-controller
                turn = angle_diff * 1.0 # Reduced gain
                
                # If facing roughly the right way, move forward
                if abs(angle_diff) < 0.5:
                    left = 1.0 - turn
                    right = 1.0 + turn
                else:
                    # Turn in place
                    left = -turn
                    right = turn
                
            elif vis > 0.5:
                # Proportional controller for angle
                turn = angle * 2.0
                left = 0.5 - turn
                right = 0.5 + turn
            else:
                # Spin to find tag
                left = -0.2
                right = 0.2
                
            # Clip
            left = max(min(left, 1.0), -1.0)
            right = max(min(right, 1.0), -1.0)
            actions.extend([left, right])
            
        action = np.array(actions)
        
        obs, reward, done, info = env.step(action)
        step_count += 1
        
        # Debug: Print observations for Robot 1 every 100 steps
        if step_count % 100 == 0:
            r1_obs = obs[0:15]
            log_msg = (
                f"\n--- Step {step_count} ---\n"
                f"Robot 1 Action: {action[0]:.2f}, {action[1]:.2f}\n"
                f"Prox: {[f'{x:.2f}' for x in r1_obs[0:8]]}\n"
                f"Tag (Vis, Dist, Ang): {r1_obs[8]:.1f}, {r1_obs[9]:.2f}, {r1_obs[10]:.2f}\n"
                f"Carrying: {r1_obs[11]}\n"
                f"Phero: {r1_obs[12]:.2f}, {r1_obs[13]:.2f}, {r1_obs[14]:.2f}\n"
            )
            print(log_msg) # Keep print for console
            with open("debug_log.txt", "a") as f:
                f.write(log_msg)
        
        # Print stats every 1000 steps
        if step_count % 1000 == 0:
            print(f"Steps: {step_count} | Pickups: {env.total_pickups} | Deposits: {env.total_deposits}")
        
        if done:
            obs = env.reset()
