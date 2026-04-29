import math
import random
import numpy as np
import argparse
from deepbots.supervisor.controllers.deepbots_supervisor_env import DeepbotsSupervisorEnv
from controller import Supervisor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import gym
import torch

# =============================================================================
# OBSERVATION SPACE: 20 values per robot, 80 total (was 15 / 60)
#
#  [0:8]  Proximity sensors (8)   - obstacle avoidance
#  [8]    tag_visible              - is a tag in the FOV?
#  [9]    tag_dist_norm            - distance to nearest visible tag (/ 2.5)
#  [10]   tag_angle_norm           - angle to nearest visible tag (/ pi)
#  [11]   carrying                 - 1 if holding a tag, 0 if not
#  [12]   dist_to_base_norm        - distance to nest (/ 3.5)  *** NEW ***
#  [13]   angle_to_base_norm       - angle to nest relative to heading (/ pi) *** NEW ***
#  [14]   cluster_known            - 1 if pheromone hotspot exists (communication) *** NEW ***
#  [15]   cluster_dist_norm        - distance to hotspot (/ 3.5) *** NEW ***
#  [16]   cluster_angle_norm       - angle to hotspot (/ pi) *** NEW ***
#  [17]   phero_front_norm         - local pheromone ahead (/ 10)
#  [18]   phero_left_norm          - local pheromone left (/ 10)
#  [19]   phero_right_norm         - local pheromone right (/ 10)
# =============================================================================

class EpuckForagingSupervisor(DeepbotsSupervisorEnv):
    def __init__(self):
        self.num_robots = 4
        self.num_tags = 70
        self.obs_per_robot = 20          # FIX B: was 15
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

        # --- Robot nodes (with timeout) ---
        self.robot_nodes = []
        for i in range(self.num_robots):
            node_name = f"ROBOT{i+1}"
            node = self.getFromDef(node_name)
            timeout = 0
            while node is None and timeout < 20:
                print(f"[WAITING] Waiting for {node_name}...")
                Supervisor.step(self, self.timestep)
                node = self.getFromDef(node_name)
                timeout += 1
            if node is None:
                print(f"[ERROR] Could not find {node_name}!")
                exit(1)
            self.robot_nodes.append(node)

        # --- Tag and base nodes ---
        self.tag_nodes = []
        for i in range(self.num_tags):
            self.tag_nodes.append(self.getFromDef(f"APRILTAG_{i+1}"))
        self.base_node = self.getFromDef("BASE_STATION")

        # --- Emitters / Receivers ---
        self.emitters = []
        self.receivers = []
        for i in range(self.num_robots):
            self.emitters.append(self.getDevice(f"emitter{i+1}"))
            self.receivers.append(self.getDevice(f"receiver{i+1}"))
            self.receivers[i].enable(self.timestep)

        # --- State ---
        self.robot_states    = [None] * self.num_robots
        self.carrying_state  = [False] * self.num_robots
        self.prev_base_dists = [None] * self.num_robots
        self.prev_tag_dists  = [None] * self.num_robots

        # --- Pheromone-following reward state ---
        # PPO observes cluster_known/dist/angle (obs[14-16]) and phero_front/left/right (obs[17-19]).
        # prev_cluster_dists gives approach shaping so PPO has a direct gradient to learn
        # to navigate toward the pheromone peak (learned CPFA replacement).
        self.prev_cluster_dists = [None] * self.num_robots

        # --- Pheromone grid (marks pickup locations for communication) ---
        self.grid_size = 50
        self.grid_res  = 5.0 / self.grid_size
        self.pheromone_grid = np.zeros((self.grid_size, self.grid_size))

        self.steps_per_episode = 4096
        self.episode_step  = 0
        self.total_episodes = 0

        # Stats
        self.total_pickups  = 0
        self.total_deposits = 0

    # =========================================================================
    # STEP
    # FIX A: reward is calculated FIRST (uses prev_tag_dists from last step),
    #         then observations are gathered (updates prev_tag_dists for next step).
    #         Previously the tuple was (get_obs(), get_reward(), ...) which meant
    #         get_obs ran first, overwrote prev_tag_dists, and the shaping was
    #         always (current - current) = 0.
    # =========================================================================
    def step(self, action):
        self.episode_step += 1

        # Apply hard-coded behavior overrides BEFORE sending to robots:
        #   P1 Wall escape  → steer toward arena center
        #   P2 Return-to-base when carrying (proportional controller)
        #   CPFA navigation removed — PPO learns pheromone-following via reward shaping
        action = self._apply_overrides(action)

        # Send motor commands to each robot
        for i in range(self.num_robots):
            robot_action = action[i*2 : (i+1)*2]
            msg = f"{robot_action[0]},{robot_action[1]}".encode('utf-8')
            self.emitters[i].send(msg)

        if super(Supervisor, self).step(self.timestep) == -1:
            exit()

        # Read sensor data sent back by robot controllers
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

        # Decay pheromone (persists ~4600 steps = nearly full episode)
        self.pheromone_grid *= 0.999

        # FIX A: reward BEFORE observations
        reward = self.get_reward(action)

        obs    = self.get_observations()

        return obs, reward, self.is_done(), self.get_info()

    # =========================================================================
    # OBSERVATIONS (20 per robot)
    # FIX B: added dist_to_base, angle_to_base, cluster signal (5 new values).
    #         prev_tag_dists is updated HERE (after reward has already used it).
    # =========================================================================
    def get_observations(self):
        global_obs = []
        base_pos   = self.base_node.getPosition()

        for i in range(self.num_robots):
            # Ensure exactly 8 prox values
            prox = (self.robot_states[i] or [0.0]*8)[:8]

            robot_pos = self.robot_nodes[i].getPosition()
            robot_rot = self.robot_nodes[i].getOrientation()
            # Forward vector = local +X axis in world space
            # getOrientation() returns row-major 3x3: [R00,R01,R02, R10,R11,R12, R20,R21,R22]
            # Column 0 = [R00, R10, R20] = where local X points in world space
            forward_vec = [robot_rot[0], robot_rot[3], robot_rot[6]]

            # ------------------------------------------------------------------
            # TAG SENSING
            # ------------------------------------------------------------------
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
                        continue  # tag is hidden below ground (picked up)

                    dx   = tag_pos[0] - robot_pos[0]
                    dy   = tag_pos[1] - robot_pos[1]
                    dist = math.sqrt(dx*dx + dy*dy)

                    # Track omnidirectional nearest (no FOV restriction)
                    if dist < 1.0 and dist < omni_min_dist:
                        omni_min_dist = dist

                    if dist < 1.0 and dist > 0.001:
                        tag_vec_norm = [dx / dist, dy / dist]
                        dot   = forward_vec[0]*tag_vec_norm[0] + forward_vec[1]*tag_vec_norm[1]
                        dot   = max(min(dot, 1.0), -1.0)
                        angle = math.acos(dot)

                        if angle < 1.2:  # ~69 deg half-FOV
                            if dist < min_dist:
                                min_dist    = dist
                                closest_tag = tag_node
                                cross       = forward_vec[0]*tag_vec_norm[1] - forward_vec[1]*tag_vec_norm[0]
                                tag_angle   = angle if cross > 0 else -angle

                if closest_tag:
                    tag_visible = 1.0
                    tag_dist    = min_dist

            # ------------------------------------------------------------------
            # BASE NAVIGATION  *** NEW ***
            # Tells the robot how far away the base is and which way to turn.
            # Without this, a carrying robot has no signal to find home.
            # ------------------------------------------------------------------
            bdx          = base_pos[0] - robot_pos[0]
            bdy          = base_pos[1] - robot_pos[1]
            dist_to_base = math.sqrt(bdx*bdx + bdy*bdy)

            if dist_to_base > 0.001:
                b_norm     = [bdx / dist_to_base, bdy / dist_to_base]
                b_dot      = forward_vec[0]*b_norm[0] + forward_vec[1]*b_norm[1]
                b_dot      = max(min(b_dot, 1.0), -1.0)
                b_angle    = math.acos(b_dot)
                b_cross    = forward_vec[0]*b_norm[1] - forward_vec[1]*b_norm[0]
                angle_to_base = b_angle if b_cross > 0 else -b_angle
            else:
                angle_to_base = 0.0

            # ------------------------------------------------------------------
            # CLUSTER SIGNAL  *** NEW — inter-robot communication ***
            # When robot A picks up a tag it deposits pheromone at that location.
            # Robots B, C, D see this signal and know which direction to search.
            # Only shown when NOT carrying (exploring robots need it).
            # ------------------------------------------------------------------
            cluster_known      = 0.0
            cluster_dist_norm  = 0.0
            cluster_angle_norm = 0.0

            if not self.carrying_state[i] and self.pheromone_grid.max() > 0.1:
                peak_idx   = np.unravel_index(self.pheromone_grid.argmax(), self.pheromone_grid.shape)
                cluster_x  = peak_idx[0] * self.grid_res - 2.5
                cluster_y  = peak_idx[1] * self.grid_res - 2.5
                cdx        = cluster_x - robot_pos[0]
                cdy        = cluster_y - robot_pos[1]
                c_dist     = math.sqrt(cdx*cdx + cdy*cdy)

                if c_dist > 0.001:
                    c_norm   = [cdx / c_dist, cdy / c_dist]
                    c_dot    = forward_vec[0]*c_norm[0] + forward_vec[1]*c_norm[1]
                    c_dot    = max(min(c_dot, 1.0), -1.0)
                    c_angle  = math.acos(c_dot)
                    c_cross  = forward_vec[0]*c_norm[1] - forward_vec[1]*c_norm[0]
                    c_angle  = c_angle if c_cross > 0 else -c_angle

                    cluster_known      = 1.0
                    cluster_dist_norm  = min(c_dist / 3.5, 1.0)
                    cluster_angle_norm = c_angle / math.pi

            # ------------------------------------------------------------------
            # LOCAL PHEROMONE GRADIENT
            # Probe pheromone 0.3m ahead/left/right of the robot.
            # Now marks cluster locations (pickup spots), not return paths.
            # ------------------------------------------------------------------
            def grid_val(wx, wy):
                gx = int((wx + 2.5) / self.grid_res)
                gy = int((wy + 2.5) / self.grid_res)
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    return self.pheromone_grid[gx, gy]
                return 0.0

            fx = robot_pos[0] + forward_vec[0] * 0.3
            fy = robot_pos[1] + forward_vec[1] * 0.3

            lx = robot_pos[0] + (forward_vec[0]*0.707 - forward_vec[1]*0.707) * 0.3
            ly = robot_pos[1] + (forward_vec[0]*0.707 + forward_vec[1]*0.707) * 0.3

            rx = robot_pos[0] + (forward_vec[0]*0.707 + forward_vec[1]*0.707) * 0.3
            ry = robot_pos[1] + (-forward_vec[0]*0.707 + forward_vec[1]*0.707) * 0.3

            phero_front = grid_val(fx, fy)
            phero_left  = grid_val(lx, ly)
            phero_right = grid_val(rx, ry)

            # ------------------------------------------------------------------
            # ASSEMBLE 20-VALUE OBSERVATION FOR THIS ROBOT
            # ------------------------------------------------------------------
            obs = []
            obs.extend(prox)                                    # [0:8]
            obs.extend([
                tag_visible,                                    # [8]
                tag_dist / 1.0,                                 # [9]  normalized (max range 1.0m)
                tag_angle / math.pi,                            # [10] normalized
            ])
            obs.append(1.0 if self.carrying_state[i] else 0.0) # [11]
            obs.append(dist_to_base / 3.5)                     # [12] NEW
            obs.append(angle_to_base / math.pi)                # [13] NEW
            obs.extend([
                cluster_known,                                  # [14] NEW
                cluster_dist_norm,                              # [15] NEW
                cluster_angle_norm,                             # [16] NEW
            ])
            obs.extend([
                phero_front / 10.0,                            # [17]
                phero_left  / 10.0,                            # [18]
                phero_right / 10.0,                            # [19]
            ])

            global_obs.extend(obs)

            # Use omnidirectional nearest so it matches get_reward()'s curr_min metric
            self.prev_tag_dists[i] = omni_min_dist if omni_min_dist < float('inf') else None

        return np.array(global_obs, dtype=np.float32)

    # =========================================================================
    # REWARD
    # FIX C:
    #   1. Approach shaping now works (prev_tag_dists holds last step's distance)
    #   2. Pheromone deposited at PICKUP location (communication to other robots)
    #   3. Exploration reward (encourages leaving base when not carrying)
    #   4. Stronger reward magnitudes so signal is not drowned by noise
    # =========================================================================
    def get_reward(self, action):
        total_reward = 0.0
        base_pos     = self.base_node.getPosition()

        for i in range(self.num_robots):
            robot_pos = self.robot_nodes[i].getPosition()
            prox      = (self.robot_states[i] or [0.0]*8)[:8]

            # Proximity penalty — sensor-based (catches close obstacles/robots)
            # REDUCED ×0.5 (was ×3.0) so rewards of -9750/ep can't recur
            max_prox = max(prox)
            if max_prox > 0.1:
                total_reward -= max_prox * 0.5

            # Wall penalty — position-based gradient.
            # REDUCED ×0.5 (was ×5.0): max -0.175/step vs old -1.75/step.
            # Small enough that positive rewards dominate, still a clear signal.
            wall_dist = 2.5 - max(abs(robot_pos[0]), abs(robot_pos[1]))
            if wall_dist < 0.35:
                total_reward -= (0.35 - wall_dist) * 0.5

            if not self.carrying_state[i]:
                # ---- EXPLORING: looking for tags ----

                # Check for pickup
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
                        total_reward += 5.0  # was +1.0
                        self.total_pickups += 1
                        picked_up = True
                        self.prev_cluster_dists[i] = None  # reset on pickup
                        print(f"[PICKUP] Robot {i+1} picked up tag! Total: {self.total_pickups}")

                        # FIX C: Mark THIS location as a resource cluster.
                        # Other robots read this via cluster_known / cluster_angle obs.
                        pgx = int((robot_pos[0] + 2.5) / self.grid_res)
                        pgy = int((robot_pos[1] + 2.5) / self.grid_res)
                        if 0 <= pgx < self.grid_size and 0 <= pgy < self.grid_size:
                            self.pheromone_grid[pgx, pgy] = min(
                                self.pheromone_grid[pgx, pgy] + 5.0, 10.0
                            )
                        break

                # FIX A: Approach shaping NOW WORKS because prev_tag_dists[i]
                # was set during the PREVIOUS step's get_observations() call,
                # not overwritten yet this step.
                if not picked_up:
                    # Recalculate distance to nearest tag (no FOV restriction here)
                    curr_min = float('inf')
                    for tag_node in self.tag_nodes:
                        tag_pos = tag_node.getPosition()
                        if tag_pos[2] < 0:
                            continue
                        dx   = tag_pos[0] - robot_pos[0]
                        dy   = tag_pos[1] - robot_pos[1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist < 1.0 and dist < curr_min:
                            curr_min = dist

                    if self.prev_tag_dists[i] is not None and curr_min < float('inf'):
                        # Positive = got closer, negative = moved away
                        shaping = (self.prev_tag_dists[i] - curr_min) * 8.0
                        total_reward += shaping

                    # Tag-visible reward: +0.2 per step when robot is actively facing a tag.
                    # Forces active searching — robot must orient toward a tag to get reward.
                    robot_rot = self.robot_nodes[i].getOrientation()
                    fwd = [robot_rot[0], robot_rot[3], robot_rot[6]]
                    for tag_node in self.tag_nodes:
                        tag_pos = tag_node.getPosition()
                        if tag_pos[2] < 0:
                            continue
                        tdx = tag_pos[0] - robot_pos[0]
                        tdy = tag_pos[1] - robot_pos[1]
                        td  = math.sqrt(tdx*tdx + tdy*tdy)
                        if td < 1.0 and td > 0.001:
                            dot = fwd[0]*(tdx/td) + fwd[1]*(tdy/td)
                            if math.acos(max(min(dot, 1.0), -1.0)) < 1.2:
                                total_reward += 0.2
                                break

                    # Zone reward: bonus for being in the cluster zone (0.8–2.4m from base).
                    # Corner clusters are at ~2.33m from base; walls are at 2.5m.
                    dist_from_base = math.sqrt(robot_pos[0]**2 + robot_pos[1]**2)
                    if 0.8 < dist_from_base < 2.4:
                        total_reward += 0.10

                    # Near-base penalty: extended to cover the 0.35–0.8m orbit dead zone.
                    # v13 only penalised < 0.6m so PPO could park at ~0.4m with no cost.
                    # Now: max -1.6/step at base centre, -0 at 0.8m boundary.
                    if dist_from_base < 0.8:
                        total_reward -= (0.8 - dist_from_base) * 2.0

                    # Forward motion bias: small reward for moving forward when exploring.
                    # Prevents the degenerate backward policy (L=-1, R=-1) that v11 learned.
                    # Only applies when PPO is in control (not wall_esc, not carrying).
                    if wall_dist >= 0.35:
                        action_i   = action[i*2 : i*2+2]
                        avg_speed  = (action_i[0] + action_i[1]) / 2.0
                        if avg_speed > 0:
                            total_reward += avg_speed * 0.01  # max +0.01/step

                    # Pheromone-approach reward: approach shaping toward pheromone peak.
                    # Fix 3: ×15 (was ×10) — stronger gradient so PPO prioritises cluster nav.
                    # Fix 2: cluster-facing reward — trains PPO to orient toward cluster,
                    #         not just approach it. Solves random wandering when signal exists.
                    if self.pheromone_grid.max() > 0.1:
                        peak = np.unravel_index(self.pheromone_grid.argmax(), self.pheromone_grid.shape)
                        cx = peak[0] * self.grid_res - 2.5
                        cy = peak[1] * self.grid_res - 2.5
                        curr_cd = math.sqrt((cx - robot_pos[0])**2 + (cy - robot_pos[1])**2)
                        if self.prev_cluster_dists[i] is not None:
                            total_reward += (self.prev_cluster_dists[i] - curr_cd) * 15.0
                        self.prev_cluster_dists[i] = curr_cd
                        # Fix 2: reward facing toward cluster (dot=+1) / penalise facing away (dot=-1)
                        if curr_cd > 0.001:
                            robot_rot = self.robot_nodes[i].getOrientation()
                            fwd = [robot_rot[0], robot_rot[3], robot_rot[6]]
                            c_norm = [(cx - robot_pos[0]) / curr_cd, (cy - robot_pos[1]) / curr_cd]
                            c_dot = max(min(fwd[0]*c_norm[0] + fwd[1]*c_norm[1], 1.0), -1.0)
                            total_reward += c_dot * 0.3
                    else:
                        # No pheromone signal — teach PPO to spread outward and search.
                        # When obs[14]=cluster_known=0, the right action is to cover the arena.
                        # Reward is proportional to distance from base (capped at arena edge).
                        if dist_from_base > 0.8:
                            total_reward += min(dist_from_base / 2.3, 1.0) * 0.15
                        self.prev_cluster_dists[i] = None

            else:
                # ---- CARRYING: returning to nest ----
                dx           = base_pos[0] - robot_pos[0]
                dy           = base_pos[1] - robot_pos[1]
                dist_to_base = math.sqrt(dx*dx + dy*dy)

                # Check deposit
                if dist_to_base < 0.25:
                    self.carrying_state[i] = False
                    total_reward          += 20.0
                    self.total_deposits   += 1
                    print(f"[DEPOSIT] Robot {i+1} deposited! Total: {self.total_deposits}")
                    # Fix 1: pre-initialize cluster dist at the deposit moment so the
                    # pheromone approach gradient is active from step 1 post-deposit.
                    # Without this, prev_cluster_dists is None on the first exploring step
                    # and PPO gets zero gradient exactly when it most needs direction.
                    if self.pheromone_grid.max() > 0.1:
                        peak = np.unravel_index(self.pheromone_grid.argmax(), self.pheromone_grid.shape)
                        cx = peak[0] * self.grid_res - 2.5
                        cy = peak[1] * self.grid_res - 2.5
                        self.prev_cluster_dists[i] = math.sqrt(
                            (cx - robot_pos[0])**2 + (cy - robot_pos[1])**2)

                # Approach-base shaping
                if self.prev_base_dists[i] is not None:
                    total_reward += (self.prev_base_dists[i] - dist_to_base) * 8.0

                self.prev_base_dists[i] = dist_to_base

                # Face-toward-base reward — REDUCED back to 0.5.
                # At 2.0 it caused the std explosion by dominating negative rewards.
                if dist_to_base > 0.001:
                    robot_rot = self.robot_nodes[i].getOrientation()
                    fwd       = [robot_rot[0], robot_rot[3], robot_rot[6]]
                    b_norm    = [dx / dist_to_base, dy / dist_to_base]
                    b_dot     = max(min(fwd[0]*b_norm[0] + fwd[1]*b_norm[1], 1.0), -1.0)
                    total_reward += b_dot * 0.5

                # Fix 1: only reset to None if STILL carrying after this step.
                # If deposit happened above, carrying_state is now False and
                # prev_cluster_dists was pre-initialized — must NOT overwrite with None.
                if self.carrying_state[i]:
                    self.prev_cluster_dists[i] = None

            # Time penalty
            total_reward -= 0.005

        # Inter-robot separation penalty: discourages robots from clustering
        # near the BASE only. Robots should converge on a tag cluster freely —
        # penalising that convergence was the root cause of failed pheromone-following.
        # Only fires when BOTH robots are within 1.5m of base (nest-orbit prevention).
        robot_positions = [self.robot_nodes[k].getPosition() for k in range(self.num_robots)]
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                di = math.sqrt(robot_positions[i][0]**2 + robot_positions[i][1]**2)
                dj = math.sqrt(robot_positions[j][0]**2 + robot_positions[j][1]**2)
                if di < 1.5 and dj < 1.5:  # both near base — penalise nest-clustering
                    sep = math.sqrt(
                        (robot_positions[i][0] - robot_positions[j][0])**2 +
                        (robot_positions[i][1] - robot_positions[j][1])**2
                    )
                    if sep < 1.0:
                        total_reward -= (1.0 - sep) * 0.5

        return total_reward

    # =========================================================================
    # HELPERS
    # =========================================================================
    def _generate_cluster_centers(self):
        """
        Rescue mission: generate 2–4 random cluster locations every episode.
        Clusters vary in position each reset so PPO learns general pheromone-
        following rather than memorising fixed locations.
        Curriculum controls how far from base clusters can appear.
        """
        if self.total_episodes < 100:
            max_dist, n_clusters = 1.0, 2
        elif self.total_episodes < 300:
            max_dist, n_clusters = 1.8, random.randint(2, 3)
        else:
            max_dist, n_clusters = 2.3, random.randint(2, 4)

        centers = []
        for _ in range(n_clusters):
            for _ in range(60):   # max placement attempts
                angle = random.uniform(0, 2 * math.pi)
                dist  = random.uniform(0.7, max_dist)
                cx    = dist * math.cos(angle)
                cy    = dist * math.sin(angle)
                # Keep clusters at least 0.8 m apart from each other
                if all(math.sqrt((cx - c[0])**2 + (cy - c[1])**2) > 0.8
                       for c in centers):
                    centers.append((cx, cy))
                    break

        return centers if centers else [(1.5, 0.0)]  # safe fallback

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
        """Hard-coded overrides — PPO learns global exploration + pheromone-following.
           P1: Wall / obstacle escape (wall < 0.35m or prox > 0.55)
           P2: Return-to-base when carrying
           P3: Base avoidance when not carrying (prevents clustering at nest)
               Threshold raised to 0.5m (was 0.35m) to match the 0.8m near-base
               penalty zone — eliminates boundary oscillation where PPO handed off
               at exactly 0.35m and learned to spin rather than move outward.
           P4: Tag-seek — steer toward nearest visible tag in FOV (< 1.2 rad)
               Reactive local behaviour; frees PPO to focus on global navigation.
        """
        final = list(action)
        base_pos = self.base_node.getPosition()
        for i in range(self.num_robots):
            robot_pos = self.robot_nodes[i].getPosition()
            robot_rot = self.robot_nodes[i].getOrientation()
            fwd  = [robot_rot[0], robot_rot[3], robot_rot[6]]
            prox = (self.robot_states[i] or [0.0]*8)[:8]
            wall_dist = 2.5 - max(abs(robot_pos[0]), abs(robot_pos[1]))

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
            if dist_to_base < 0.5:
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
                    if td < 1.0 and td > 0.001 and td < best_dist:
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
        return self.episode_step >= self.steps_per_episode

    def get_info(self):
        return {}

    # =========================================================================
    # RESET WITH CURRICULUM  (FIX D)
    # Early episodes: tags within 1m so robots reliably find them and learn
    # the pickup/deposit loop. Later episodes: tags at real cluster positions
    # matching the eval world. This bootstraps learning before tackling the
    # hard full-arena task.
    # =========================================================================
    def reset(self):
        self.total_episodes += 1
        self.episode_step    = 0
        self.robot_states    = [[0.0]*8 for _ in range(self.num_robots)]
        self.carrying_state  = [False] * self.num_robots
        self.pheromone_grid.fill(0.0)
        self.prev_base_dists    = [None] * self.num_robots
        self.prev_tag_dists     = [None] * self.num_robots
        self.prev_cluster_dists = [None] * self.num_robots

        self.respawn_robots()
        self.respawn_all_tags()

        return self.get_observations()

    def respawn_robots(self):
        positions = [
            [-0.5, 0, 0],
            [ 0.5, 0, 0],
            [ 0,  0.5, 0],
            [ 0, -0.5, 0],
        ]
        for i in range(self.num_robots):
            self.robot_nodes[i].getField("translation").setSFVec3f(positions[i])
            # Random heading so model learns to navigate from any direction
            rand_yaw = random.uniform(0, 2 * math.pi)
            self.robot_nodes[i].getField("rotation").setSFRotation([0, 0, 1, rand_yaw])
            self.robot_nodes[i].resetPhysics()

    def respawn_all_tags(self):
        """
        Rescue mission spawning: tags always in clusters, cluster locations
        randomised every episode (different rescue site each time).
        Curriculum controls how far clusters can be from the base.
          Episode   0-100:  clusters within 1.0m — robots reliably find tags,
                             learn the pickup/deposit loop
          Episode 100-300:  clusters within 1.8m — medium range exploration
          Episode   300+:   clusters anywhere in arena (up to 2.3m) — full task
        """
        centers = self._generate_cluster_centers()
        for idx, tag_node in enumerate(self.tag_nodes):
            cx, cy = centers[idx % len(centers)]
            tx = cx + random.gauss(0, 0.20)
            ty = cy + random.gauss(0, 0.20)
            tx = max(-2.3, min(2.3, tx))
            ty = max(-2.3, min(2.3, ty))
            tag_node.getField("translation").setSFVec3f([tx, ty, 0.01375])


# =============================================================================
# TRAINING ENTRY POINT
# FIX E: Better hyperparameters
#   - ent_coef 0.01 → 0.05  (5x more exploration, avoids spinning local min)
#   - batch_size 4096 → 1024 (more frequent gradient updates)
#   - net_arch [512,512,512] → [256,256] (smaller = faster convergence here)
#   - activation Tanh (better for bounded obs than ReLU)
#   - total_timesteps recommendation: 5M-10M (2M was not enough)
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name',        type=str,   default='ppo_v17_phero2')
    parser.add_argument('--lr',              type=float, default=3e-4)
    parser.add_argument('--ent_coef',        type=float, default=0.10)
    parser.add_argument('--batch_size',      type=int,   default=1024)
    parser.add_argument('--total_timesteps', type=int,   default=7000000)
    parser.add_argument('--resume',          type=str,   default=None,
                        help='Path to checkpoint .zip to resume from')
    args = parser.parse_args()

    env    = EpuckForagingSupervisor()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[TRAINING] Device: {device}")
    print(f"[TRAINING] Run: {args.run_name} | LR: {args.lr} | Ent: {args.ent_coef} "
          f"| Batch: {args.batch_size} | Steps: {args.total_timesteps}")
    print(f"[INFO] Obs space: {env.observation_space_dim} ({env.obs_per_robot} per robot)")

    policy_kwargs = dict(
        net_arch=[256, 256],
        activation_fn=torch.nn.Tanh
    )

    if args.resume:
        print(f"[TRAINING] Resuming from: {args.resume}")
        model = PPO.load(
            args.resume,
            env=env,
            device=device,
            learning_rate=args.lr,
            ent_coef=args.ent_coef,
            batch_size=args.batch_size,
        )
        # Calculate remaining steps from checkpoint name
        import re
        match = re.search(r'_(\d+)_steps', args.resume)
        steps_done = int(match.group(1)) if match else 0
        remaining  = max(args.total_timesteps - steps_done, 0)
        print(f"[TRAINING] Steps done: {steps_done} | Remaining: {remaining}")
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=device,
            n_steps=4096,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            ent_coef=args.ent_coef,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
        )
        remaining = args.total_timesteps

    checkpoint_callback = CheckpointCallback(
        save_freq=200000,
        save_path=f'./logs/{args.run_name}/',
        name_prefix=args.run_name
    )

    print(f"[TRAINING] Starting: {args.run_name}")
    model.learn(
        total_timesteps=remaining,
        callback=checkpoint_callback,
        reset_num_timesteps=args.resume is None
    )
    model.save(args.run_name)
    print(f"[COMPLETE] Model saved as {args.run_name}.zip")
