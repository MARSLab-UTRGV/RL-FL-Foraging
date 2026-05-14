import math
import numpy as np
import sys
from deepbots.supervisor.controllers.deepbots_supervisor_env import DeepbotsSupervisorEnv
from controller import Supervisor
from stable_baselines3 import PPO
import gymnasium as gym

# =============================================================================
# EVALUATION SUPERVISOR
# Observation space matches the training supervisor exactly:
#   20 values per robot × 4 robots = 80 total
#
#  [0:8]  Proximity sensors
#  [8]    tag_visible
#  [9]    tag_dist_norm       (/ 0.5)
#  [10]   tag_angle_norm      (/ pi)
#  [11]   carrying
#  [12]   dist_to_base_norm   (/ 5.7)   NEW
#  [13]   angle_to_base_norm  (/ pi)    NEW
#  [14]   cluster_known                 NEW
#  [15]   cluster_dist_norm   (/ 5.7)   NEW
#  [16]   cluster_angle_norm  (/ pi)    NEW
#  [17]   phero_front_norm    (/ 10)
#  [18]   phero_left_norm     (/ 10)
#  [19]   phero_right_norm    (/ 10)
# =============================================================================

class EpuckForagingSupervisor(DeepbotsSupervisorEnv):
    def __init__(self):
        self.num_robots   = 4
        self.num_tags     = 64
        self.obs_per_robot = 20
        self.observation_space_dim = self.obs_per_robot * self.num_robots
        self.action_space_dim = 2 * self.num_robots

        super().__init__()
        self.timestep = int(self.getBasicTimeStep())

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.observation_space_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1,
            shape=(self.action_space_dim,), dtype=np.float32
        )

        self.robot_nodes = []
        for i in range(self.num_robots):
            self.robot_nodes.append(self.getFromDef(f"ROBOT{i+1}"))

        self.tag_nodes = []
        for i in range(self.num_tags):
            self.tag_nodes.append(self.getFromDef(f"APRILTAG_{i+1}"))

        self.base_node = self.getFromDef("BASE_STATION")

        self.emitters  = []
        self.receivers = []
        for i in range(self.num_robots):
            self.emitters.append(self.getDevice(f"emitter{i+1}"))
            self.receivers.append(self.getDevice(f"receiver{i+1}"))
            self.receivers[i].enable(self.timestep)

        self.robot_states    = [None] * self.num_robots
        self.carrying_state  = [False] * self.num_robots
        self.prev_base_dists = [None] * self.num_robots
        self.prev_tag_dists  = [None] * self.num_robots

        # (v11: CPFA state and stuck detection removed — matches training overrides exactly)

        self.grid_size = 80
        self.grid_res  = 8.0 / self.grid_size
        self.pheromone_grid = np.zeros((self.grid_size, self.grid_size))

        self.steps_per_episode = 4096
        self.episode_step  = 0
        self.total_pickups  = 0
        self.total_deposits = 0

    def step(self, action):
        self.episode_step += 1

        # Apply hard-coded overrides (wall escape, RTB, base avoid, tag-seek — v13)
        action = self._apply_overrides(action)

        for i in range(self.num_robots):
            robot_action = action[i*2 : (i+1)*2]
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
                    self.robot_states[i] = [0.0] * 8
            else:
                if self.robot_states[i] is None:
                    self.robot_states[i] = [0.0] * 8

        self.pheromone_grid *= 0.9995

        # Same order as training: reward first, then observations
        reward = self.get_reward(action)
        obs    = self.get_observations()

        return obs, reward, self.is_done(), self.get_info()

    def get_observations(self):
        global_obs = []
        base_pos   = self.base_node.getPosition()

        for i in range(self.num_robots):
            prox      = (self.robot_states[i] or [0.0]*8)[:8]
            robot_pos = self.robot_nodes[i].getPosition()
            robot_rot = self.robot_nodes[i].getOrientation()
            forward_vec = [robot_rot[0], robot_rot[3], robot_rot[6]]

            # Tag sensing
            tag_visible   = 0.0
            tag_dist      = 0.0
            tag_angle     = 0.0
            min_dist      = float('inf')
            omni_min_dist = float('inf')   # omnidirectional nearest (matches get_reward curr_min)
            closest_tag   = None

            if not self.carrying_state[i]:
                for tag_node in self.tag_nodes:
                    tag_pos = tag_node.getPosition()
                    if tag_pos[2] < 0:
                        continue
                    dx   = tag_pos[0] - robot_pos[0]
                    dy   = tag_pos[1] - robot_pos[1]
                    dist = math.sqrt(dx*dx + dy*dy)

                    # Track omnidirectional nearest (no FOV restriction)
                    if dist < 0.5 and dist < omni_min_dist:
                        omni_min_dist = dist

                    if dist < 0.5 and dist > 0.001:
                        tag_vec_norm = [dx/dist, dy/dist]
                        dot   = forward_vec[0]*tag_vec_norm[0] + forward_vec[1]*tag_vec_norm[1]
                        dot   = max(min(dot, 1.0), -1.0)
                        angle = math.acos(dot)
                        if angle < 1.2:
                            if dist < min_dist:
                                min_dist    = dist
                                closest_tag = tag_node
                                cross       = forward_vec[0]*tag_vec_norm[1] - forward_vec[1]*tag_vec_norm[0]
                                tag_angle   = angle if cross > 0 else -angle
                if closest_tag:
                    tag_visible = 1.0
                    tag_dist    = min_dist

            # Base navigation
            bdx          = base_pos[0] - robot_pos[0]
            bdy          = base_pos[1] - robot_pos[1]
            dist_to_base = math.sqrt(bdx*bdx + bdy*bdy)
            if dist_to_base > 0.001:
                b_norm  = [bdx/dist_to_base, bdy/dist_to_base]
                b_dot   = max(min(forward_vec[0]*b_norm[0] + forward_vec[1]*b_norm[1], 1.0), -1.0)
                b_angle = math.acos(b_dot)
                b_cross = forward_vec[0]*b_norm[1] - forward_vec[1]*b_norm[0]
                angle_to_base = b_angle if b_cross > 0 else -b_angle
            else:
                angle_to_base = 0.0

            # Cluster signal (inter-robot communication via pheromone)
            cluster_known = 0.0
            cluster_dist_norm  = 0.0
            cluster_angle_norm = 0.0
            if not self.carrying_state[i] and self.pheromone_grid.max() > 0.1:
                peak_idx  = np.unravel_index(self.pheromone_grid.argmax(), self.pheromone_grid.shape)
                cluster_x = peak_idx[0] * self.grid_res - 4.0
                cluster_y = peak_idx[1] * self.grid_res - 4.0
                cdx       = cluster_x - robot_pos[0]
                cdy       = cluster_y - robot_pos[1]
                c_dist    = math.sqrt(cdx*cdx + cdy*cdy)
                if c_dist > 0.001:
                    c_norm  = [cdx/c_dist, cdy/c_dist]
                    c_dot   = max(min(forward_vec[0]*c_norm[0] + forward_vec[1]*c_norm[1], 1.0), -1.0)
                    c_angle = math.acos(c_dot)
                    c_cross = forward_vec[0]*c_norm[1] - forward_vec[1]*c_norm[0]
                    c_angle = c_angle if c_cross > 0 else -c_angle
                    cluster_known      = 1.0
                    cluster_dist_norm  = min(c_dist / 5.7, 1.0)
                    cluster_angle_norm = c_angle / math.pi

            # Local pheromone gradient
            def grid_val(wx, wy):
                gx = int((wx + 4.0) / self.grid_res)
                gy = int((wy + 4.0) / self.grid_res)
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    return self.pheromone_grid[gx, gy]
                return 0.0

            fx = robot_pos[0] + forward_vec[0] * 0.3
            fy = robot_pos[1] + forward_vec[1] * 0.3
            lx = robot_pos[0] + (forward_vec[0]*0.707 - forward_vec[1]*0.707) * 0.3
            ly = robot_pos[1] + (forward_vec[0]*0.707 + forward_vec[1]*0.707) * 0.3
            rx = robot_pos[0] + (forward_vec[0]*0.707 + forward_vec[1]*0.707) * 0.3
            ry = robot_pos[1] + (-forward_vec[0]*0.707 + forward_vec[1]*0.707) * 0.3

            obs = []
            obs.extend(prox)
            obs.extend([tag_visible, tag_dist/0.5, tag_angle/math.pi])
            obs.append(1.0 if self.carrying_state[i] else 0.0)
            obs.append(dist_to_base / 5.7)
            obs.append(angle_to_base / math.pi)
            obs.extend([cluster_known, cluster_dist_norm, cluster_angle_norm])
            obs.extend([grid_val(fx,fy)/10.0, grid_val(lx,ly)/10.0, grid_val(rx,ry)/10.0])

            global_obs.extend(obs)

            # Use omnidirectional nearest so it matches get_reward()'s curr_min metric
            self.prev_tag_dists[i] = omni_min_dist if omni_min_dist < float('inf') else None

        return np.array(global_obs, dtype=np.float32)

    def get_reward(self, action):
        total_reward = 0.0
        base_pos     = self.base_node.getPosition()

        for i in range(self.num_robots):
            robot_pos = self.robot_nodes[i].getPosition()
            prox      = (self.robot_states[i] or [0.0]*8)[:8]

            max_prox = max(prox)
            if max_prox > 0.1:
                total_reward -= max_prox * 0.5

            wall_dist = 4.0 - max(abs(robot_pos[0]), abs(robot_pos[1]))
            if wall_dist < 0.35:
                total_reward -= (0.35 - wall_dist) * 0.5

            if not self.carrying_state[i]:
                picked_up = False
                for tag_node in self.tag_nodes:
                    tag_pos = tag_node.getPosition()
                    if tag_pos[2] < 0:
                        continue
                    dx   = tag_pos[0] - robot_pos[0]
                    dy   = tag_pos[1] - robot_pos[1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 0.15:
                        self.carrying_state[i] = True
                        tag_node.getField("translation").setSFVec3f([0, 0, -10])
                        total_reward += 5.0
                        self.total_pickups += 1
                        picked_up = True
                        print(f"[+] Robot {i+1} picked up tag! Total pickups: {self.total_pickups}")
                        # Mark pickup location for other robots
                        pgx = int((robot_pos[0] + 4.0) / self.grid_res)
                        pgy = int((robot_pos[1] + 4.0) / self.grid_res)
                        if 0 <= pgx < self.grid_size and 0 <= pgy < self.grid_size:
                            self.pheromone_grid[pgx, pgy] = min(
                                self.pheromone_grid[pgx, pgy] + 8.0, 10.0
                            )
                        break

                if not picked_up:
                    curr_min = float('inf')
                    for tag_node in self.tag_nodes:
                        tag_pos = tag_node.getPosition()
                        if tag_pos[2] < 0: continue
                        dx   = tag_pos[0] - robot_pos[0]
                        dy   = tag_pos[1] - robot_pos[1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 0.5 and dist < curr_min:
                            curr_min = dist
                    if self.prev_tag_dists[i] is not None and curr_min < float('inf'):
                        total_reward += (self.prev_tag_dists[i] - curr_min) * 8.0

                    # Tag-visible reward: matches training supervisor exactly
                    robot_rot = self.robot_nodes[i].getOrientation()
                    fwd = [robot_rot[0], robot_rot[3], robot_rot[6]]
                    for tag_node in self.tag_nodes:
                        tag_pos = tag_node.getPosition()
                        if tag_pos[2] < 0:
                            continue
                        tdx = tag_pos[0] - robot_pos[0]
                        tdy = tag_pos[1] - robot_pos[1]
                        td  = math.sqrt(tdx*tdx + tdy*tdy)
                        if td < 0.5 and td > 0.001:
                            dot = fwd[0]*(tdx/td) + fwd[1]*(tdy/td)
                            if math.acos(max(min(dot, 1.0), -1.0)) < 1.2:
                                total_reward += 0.2
                                break

                    # Zone reward: matches training supervisor exactly
                    dist_from_base = math.sqrt(robot_pos[0]**2 + robot_pos[1]**2)
                    if 0.8 < dist_from_base < 3.5:
                        total_reward += 0.05

            else:
                dx           = base_pos[0] - robot_pos[0]
                dy           = base_pos[1] - robot_pos[1]
                dist_to_base = math.sqrt(dx*dx + dy*dy)

                if dist_to_base < 0.25:
                    self.carrying_state[i] = False
                    total_reward          += 20.0
                    self.total_deposits   += 1
                    print(f"[*] Robot {i+1} deposited tag! Total deposits: {self.total_deposits}")

                if self.prev_base_dists[i] is not None:
                    total_reward += (self.prev_base_dists[i] - dist_to_base) * 8.0
                self.prev_base_dists[i] = dist_to_base

                # Face-toward-base reward — matches training supervisor exactly
                if dist_to_base > 0.001:
                    robot_rot = self.robot_nodes[i].getOrientation()
                    fwd       = [robot_rot[0], robot_rot[3], robot_rot[6]]
                    b_norm    = [dx / dist_to_base, dy / dist_to_base]
                    b_dot     = max(min(fwd[0]*b_norm[0] + fwd[1]*b_norm[1], 1.0), -1.0)
                    total_reward += b_dot * 0.5

            total_reward -= 0.005

        return total_reward

    # =========================================================================
    # HARD-CODED BEHAVIOR OVERRIDES  (Problems 1, 2, 3)
    # =========================================================================

    def _steer_to(self, robot_pos, fwd, target, gain=2.5):
        """Proportional steering controller: returns [left, right] in [-1,1]."""
        dx = target[0] - robot_pos[0]
        dy = target[1] - robot_pos[1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.01:
            return [0.0, 0.0]
        t_norm = [dx / dist, dy / dist]
        dot   = fwd[0]*t_norm[0] + fwd[1]*t_norm[1]
        cross = fwd[0]*t_norm[1] - fwd[1]*t_norm[0]  # >0 = target left of fwd
        angle = math.atan2(cross, dot)
        turn  = max(-1.0, min(1.0, gain * angle / math.pi))
        left  = max(-1.0, min(1.0, 1.0 - turn))
        right = max(-1.0, min(1.0, 1.0 + turn))
        m = max(abs(left), abs(right))
        if m > 1.0:
            left /= m
            right /= m
        return [left, right]

    def _apply_overrides(self, action):
        """Matches v18 training exactly: P1 wall/collision, P2 RTB, P3 base avoid (0.8m), P4 tag-seek (0.5m).
           PPO handles global exploration + pheromone-following."""
        final = list(action)
        base_pos = self.base_node.getPosition()
        for i in range(self.num_robots):
            robot_pos = self.robot_nodes[i].getPosition()
            robot_rot = self.robot_nodes[i].getOrientation()
            fwd  = [robot_rot[0], robot_rot[3], robot_rot[6]]
            prox = (self.robot_states[i] or [0.0]*8)[:8]
            wall_dist = 4.0 - max(abs(robot_pos[0]), abs(robot_pos[1]))

            # P1: Wall / collision escape
            if wall_dist < 0.35 or max(prox) > 0.55:
                ov = self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=4.0)
                final[i*2], final[i*2+1] = ov[0], ov[1]
                continue

            # P2: Return to base when carrying
            if self.carrying_state[i]:
                ov = self._steer_to(robot_pos, fwd, base_pos, gain=2.5)
                final[i*2], final[i*2+1] = ov[0], ov[1]
                continue

            # P3: Base avoidance when not carrying (steer radially away)
            dist_to_base = math.sqrt(robot_pos[0]**2 + robot_pos[1]**2)
            if dist_to_base < 0.8:
                away = [robot_pos[0] - base_pos[0], robot_pos[1] - base_pos[1]]
                target = [robot_pos[0] + away[0] * 2.0,
                          robot_pos[1] + away[1] * 2.0]
                ov = self._steer_to(robot_pos, fwd, target, gain=3.0)
                final[i*2], final[i*2+1] = ov[0], ov[1]
                continue

            # P4: Tag-seek — steer toward nearest visible tag in FOV
            if not self.carrying_state[i]:
                best_tag_pos = None
                best_dist    = float('inf')
                for tag_node in self.tag_nodes:
                    tag_pos = tag_node.getPosition()
                    if tag_pos[2] < 0:
                        continue
                    tdx = tag_pos[0] - robot_pos[0]
                    tdy = tag_pos[1] - robot_pos[1]
                    td  = math.sqrt(tdx*tdx + tdy*tdy)
                    if td < 0.5 and td > 0.001 and td < best_dist:
                        dot   = fwd[0]*(tdx/td) + fwd[1]*(tdy/td)
                        angle = math.acos(max(min(dot, 1.0), -1.0))
                        if angle < 1.2:
                            best_dist    = td
                            best_tag_pos = tag_pos
                if best_tag_pos is not None:
                    ov = self._steer_to(robot_pos, fwd, best_tag_pos, gain=3.0)
                    final[i*2], final[i*2+1] = ov[0], ov[1]
                    continue

            # PPO controls global exploration (pheromone-following, site search)

        return np.array(final, dtype=np.float32)

    def is_done(self):
        return False   # run forever in eval mode

    def get_info(self):
        return {}

    def reset(self):
        self.respawn_robots()
        self.episode_step    = 0
        self.robot_states    = [[0.0]*8 for _ in range(self.num_robots)]
        self.carrying_state  = [False] * self.num_robots
        self.pheromone_grid.fill(0.0)
        self.prev_base_dists = [None] * self.num_robots
        self.prev_tag_dists  = [None] * self.num_robots
        return self.get_observations()

    def respawn_robots(self):
        positions = [[-0.5,0,0],[0.5,0,0],[0,0.5,0],[0,-0.5,0]]
        for i in range(self.num_robots):
            self.robot_nodes[i].getField("translation").setSFVec3f(positions[i])
            self.robot_nodes[i].getField("rotation").setSFRotation([0, 0, 1, 0])
            self.robot_nodes[i].resetPhysics()


# =============================================================================
# EVALUATION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import torch

    print("="*60)
    print("EVALUATION MODE — Testing Trained Model")
    print("="*60)

    env = EpuckForagingSupervisor()

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "ppo_v17_phero2.zip"
    # SB3 appends .zip automatically — strip it if already present
    if model_path.endswith('.zip'):
        model_path = model_path[:-4]

    print(f"\nLoading model: {model_path}")
    print(f"Obs space: {env.observation_space_dim} ({env.obs_per_robot} per robot)\n")

    try:
        custom_objects = {
            "lr_schedule": lambda _: 3e-4,
            "clip_range": lambda _: 0.2,
        }
        model = PPO.load(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Make sure you trained with the new supervisor (obs_per_robot=20).")
        print("Old models (obs_per_robot=15) are not compatible.")
        sys.exit(1)

    print("Model loaded. Starting evaluation...\n")
    print("="*60 + "\n")

    obs        = env.reset()
    step_count = 0

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        step_count += 1

        # Debug print every 500 steps — all 4 robots
        if step_count % 500 == 0:
            log_msg = f"\n{'='*60}\nStep {step_count} | Pickups: {env.total_pickups} | Deposits: {env.total_deposits}\n{'='*60}\n"
            for ri in range(4):
                ro  = obs[ri*20:(ri+1)*20]
                ra_l = action[ri*2]
                ra_r = action[ri*2+1]
                rpos = env.robot_nodes[ri].getPosition()
                wall_d = 4.0 - max(abs(rpos[0]), abs(rpos[1]))
                # Show which mode is active
                base_p = env.base_node.getPosition()
                d2base = math.sqrt((rpos[0]-base_p[0])**2 + (rpos[1]-base_p[1])**2)
                if wall_d < 0.35:
                    mode = "WALL_ESC"
                elif ro[11] > 0.5:
                    mode = "RTB"
                elif d2base < 0.5 and ro[11] < 0.5:
                    mode = "BASE_AVOID"
                elif ro[8] > 0.5:
                    mode = "TAG_SEEK"
                else:
                    mode = "PPO"
                log_msg += (
                    f"R{ri+1}[{mode}]: L={ra_l:.2f} R={ra_r:.2f} | "
                    f"carry={ro[11]:.0f} | "
                    f"base_dist={ro[12]:.2f} base_ang={ro[13]:.2f} | "
                    f"tag_vis={ro[8]:.0f} tag_dist={ro[9]:.2f} | "
                    f"wall={wall_d:.2f}\n"
                )
            print(log_msg)
            with open("debug_log.txt", "a") as f:
                f.write(log_msg)

        if done:
            obs = env.reset()
