import math
import random
import numpy as np
import argparse
from controller import Supervisor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import gymnasium as gym
import torch

# =============================================================================
# CPFA-RL SUPERVISOR — 5×5m arena
#
# Pheromone system follows CPFA exactly:
#   - Pheromone list (not grid): each entry = {x, y, weight, resource_density}
#   - Created at nest deposit (NOT at pickup) via Poisson CDF gate
#   - Site fidelity: robot remembers its own last pickup location
#   - Roulette-wheel selection weighted by pheromone weight
#   - Exponential time-based decay, pruned when weight < PHEROMONE_MIN
#   - Pheromone info assigned ONLY at nest return (like CPFA)
#
# PPO replaces CPFA's hand-coded 4-state machine:
#   DEPARTING  → PPO learns when/where to travel toward target
#   SEARCHING  → PPO learns local search (no tag sensing — same as baseline CRW)
#   SURVEYING  → removed (no rotation pause needed with RL)
#   RETURNING  → P2 hard override (identical to CPFA RETURNING state)
#
# Hard-coded overrides:
#   P1: Wall / obstacle escape
#   BASE_ESC: steer away from nest when not carrying and dist_to_base < 0.25m
#   P2: Return-to-base when carrying  (= CPFA RETURNING state)
#   PPO controls everything else: DEPARTING, local search, give-up
#
# OBSERVATION SPACE: 18 values per robot, 72 total
#  [0:8]  proximity sensors (8)
#  [8]    carrying               1 if holding food
#  [9]    dist_to_base_norm      distance to nest / 3.5m
#  [10]   angle_to_base_norm     signed angle to nest / pi
#  [11]   site_known             1 if site fidelity target assigned at nest
#  [12]   site_dist_norm         distance to site fidelity target / 3.5m
#  [13]   site_angle_norm        signed angle to site fidelity target / pi
#  [14]   phero_known            1 if pheromone target assigned at nest
#  [15]   phero_dist_norm        distance to pheromone target / 3.5m
#  [16]   phero_angle_norm       signed angle to pheromone target / pi
#  [17]   search_duration_norm   steps since last pickup / 4000 (0→1)
#                                signal for PPO: high value = long search = cluster depleted
#                                give-up is probabilistic like CPFA (P=0.0189 per 5s check)
#
#  obs[11-17] zeroed when carrying=True (CPFA: pheromone not used during RTB)
# =============================================================================

class EpuckForagingSupervisor(Supervisor, gym.Env):
    def __init__(self):
        self.num_robots           = 4
        self.num_tags             = 64
        self.obs_per_robot        = 18
        self.observation_space_dim = self.obs_per_robot * self.num_robots
        self.action_space_dim     = 2 * self.num_robots

        Supervisor.__init__(self)
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
            node      = self.getFromDef(node_name)
            timeout   = 0
            while node is None and timeout < 20:
                print(f"[WAITING] Waiting for {node_name}...")
                Supervisor.step(self, self.timestep)
                node    = self.getFromDef(node_name)
                timeout += 1
            if node is None:
                print(f"[ERROR] Could not find {node_name}!")
                exit(1)
            self.robot_nodes.append(node)

        # --- Tag and base nodes ---
        self.tag_nodes = [self.getFromDef(f"APRILTAG_{i+1}") for i in range(self.num_tags)]
        self.base_node = self.getFromDef("BASE_STATION")

        # --- Emitters / Receivers ---
        self.emitters  = []
        self.receivers = []
        for i in range(self.num_robots):
            self.emitters.append(self.getDevice(f"emitter{i+1}"))
            self.receivers.append(self.getDevice(f"receiver{i+1}"))
            self.receivers[i].enable(self.timestep)

        # --- Basic robot state ---
        self.robot_states    = [None]  * self.num_robots
        self.carrying_state  = [False] * self.num_robots
        self.prev_base_dists = [None]  * self.num_robots

        # --- CPFA pheromone list ---
        # Each entry: {x, y, weight, resource_density}
        # Created at nest deposit, decayed each step, pruned when weight < PHEROMONE_MIN
        self.pheromone_list = []

        # --- CPFA parameters (5×5m arena tuned) ---
        self.RATE_OF_LAYING_PHEROMONE = 3.0    # Poisson lambda: higher density → more likely to lay
        self.RATE_OF_SITE_FIDELITY    = 1.376  # ARGoS evolved value — matches cpfa_baseline
        self.RATE_OF_PHEROMONE_DECAY  = 0.05   # matches cpfa_baseline (τ≈20s); ARGoS=0.337/s scaled for 4 robots
        self.PHEROMONE_MIN            = 0.001  # prune entries below this weight

        # --- Per-robot CPFA state ---
        self.carried_from      = [None] * self.num_robots  # (x,y) where food was picked up
        self.resource_density  = [0]    * self.num_robots  # tag count within 0.5m at pickup
        self.site_fidelity_pos = [None] * self.num_robots  # robot's own last pickup location
        # nest_target: None | ('site', x, y) | ('phero', x, y)
        # Assigned at nest deposit, cleared when robot picks up food
        self.nest_target       = [None] * self.num_robots

        # --- Reward shaping state ---
        self.prev_site_dists  = [None] * self.num_robots
        self.prev_phero_dists = [None] * self.num_robots

        # --- Search duration counter ---
        # Incremented every non-pickup step; exposed as obs[17] = search_duration_norm.
        self.steps_without_pickup = [0]    * self.num_robots
        self.SEARCH_DURATION_NORM = 4000.0 # obs[17] saturates at 1.0 near E[give-up]

        # Give-up: identical probabilistic mechanism as baseline CPFA.
        # Baseline: check every 156 steps (5s at 32ms), P=0.0189 → E[give-up]=264s.
        # RL:       check every  78 steps (5s at 64ms), P=0.0189 → E[give-up]=264s.
        # P2 override handles navigation home — no shaping reward needed.
        self.PROB_RETURN_TO_NEST  = 0.0189  # identical to ARGoS/baseline
        self.GIVE_UP_CHECK_STEPS  = 78      # 5s at 64ms/step — matches baseline's 156 at 32ms
        self.give_up_timer        = [0]     * self.num_robots

        # gave_up[i]: set True when give-up fires, cleared only at next pickup.
        # Passed to _assign_target() to suppress stale site fidelity after a
        # failed search trip — mirrors ARGoS CPFA (updateFidelity=False on give-up).
        self.gave_up              = [False] * self.num_robots

        # --- Episode control ---
        self.steps_per_episode = 16384  # 16384 × 64ms ≈ 17.5 min sim time
        self.episode_step      = 0
        self.total_episodes    = 0

        # --- Stats ---
        self.total_pickups  = 0
        self.total_deposits = 0
        self.ep_pickups     = 0
        self.ep_deposits    = 0

        # --- Logging ---
        self.LOG_STEP_EVERY = 500
        self.log_file       = "training_log.txt"
        with open(self.log_file, 'w') as f:
            f.write("=== CPFA-RL Training Log ===\n")
            f.write(f"RateOfLayingPheromone={self.RATE_OF_LAYING_PHEROMONE} "
                    f"RateOfSiteFidelity={self.RATE_OF_SITE_FIDELITY} "
                    f"RateOfPheromoneDecay={self.RATE_OF_PHEROMONE_DECAY} "
                    f"(synced with cpfa_baseline)\n\n")

    # =========================================================================
    # CPFA HELPERS
    # =========================================================================

    def _poisson_cdf(self, n, rate):
        """P(X <= n) where X ~ Poisson(rate).
        Matches CPFA's GetPoissonCDF(n, lambda).
        Higher n (resource density) → higher probability → more likely to act."""
        if n < 0:
            return 0.0
        cdf  = 0.0
        term = math.exp(-rate)
        cdf += term
        for i in range(1, n + 1):
            term *= rate / i
            cdf  += term
        return min(cdf, 1.0)

    def _roulette_select(self):
        """CPFA roulette-wheel selection from pheromone_list, weighted by weight.
        Returns (x, y) of selected entry, or None if list is empty."""
        active = [p for p in self.pheromone_list if p['weight'] > self.PHEROMONE_MIN]
        if not active:
            return None
        total      = sum(p['weight'] for p in active)
        r          = random.random() * total
        cumulative = 0.0
        for p in active:
            cumulative += p['weight']
            if r <= cumulative:
                return (p['x'], p['y'])
        return (active[-1]['x'], active[-1]['y'])  # numerical fallback

    def _assign_target(self, i):
        """CPFA target assignment at nest return.
        Priority 1: site fidelity  (Poisson CDF test on resource density)
                    Skipped if gave_up[i]=True — robot failed last trip, stale
                    fidelity suppressed (ARGoS CPFA: updateFidelity=False on give-up).
        Priority 2: pheromone      (roulette-wheel weighted by weight)
        Priority 3: random search  (None — PPO explores freely)"""
        # Priority 1: site fidelity (only if last trip was a successful pickup)
        if self.site_fidelity_pos[i] is not None and not self.gave_up[i]:
            sf_prob = self._poisson_cdf(self.resource_density[i], self.RATE_OF_SITE_FIDELITY)
            if random.random() < sf_prob:
                sx, sy = self.site_fidelity_pos[i]
                return ('site', sx, sy)

        # Priority 2: pheromone following
        selected = self._roulette_select()
        if selected is not None:
            return ('phero', selected[0], selected[1])

        # Priority 3: random search
        return None

    # =========================================================================
    # STEP
    # =========================================================================
    def step(self, action):
        self.episode_step += 1

        action = self._apply_overrides(action)

        # Send motor commands
        for i in range(self.num_robots):
            robot_action = action[i*2 : (i+1)*2]
            msg = f"{robot_action[0]},{robot_action[1]}".encode('utf-8')
            self.emitters[i].send(msg)

        if Supervisor.step(self, self.timestep) == -1:
            exit()

        # Read proximity sensor data from robots
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

        # CPFA pheromone decay — time-based exponential (matches CPFA model)
        dt    = self.timestep / 1000.0  # ms → seconds
        decay = math.exp(-self.RATE_OF_PHEROMONE_DECAY * dt)
        for p in self.pheromone_list:
            p['weight'] *= decay
        self.pheromone_list = [p for p in self.pheromone_list
                               if p['weight'] > self.PHEROMONE_MIN]

        # Reward BEFORE observations (state transitions must precede obs snapshot)
        reward = self.get_reward(action)
        obs    = self.get_observations()

        if self.episode_step % self.LOG_STEP_EVERY == 0:
            self._log_step(obs, action)

        done = self.is_done()
        return obs, reward, done, False, self.get_info()

    # =========================================================================
    # OBSERVATIONS (18 per robot)
    # =========================================================================
    def get_observations(self):
        global_obs = []
        base_pos   = self.base_node.getPosition()

        for i in range(self.num_robots):
            prox = (self.robot_states[i] or [0.0]*8)[:8]

            robot_pos   = self.robot_nodes[i].getPosition()
            robot_rot   = self.robot_nodes[i].getOrientation()
            forward_vec = [robot_rot[0], robot_rot[3], robot_rot[6]]

            # ------------------------------------------------------------------
            # BASE NAVIGATION
            # ------------------------------------------------------------------
            bdx          = base_pos[0] - robot_pos[0]
            bdy          = base_pos[1] - robot_pos[1]
            dist_to_base = math.sqrt(bdx*bdx + bdy*bdy)

            if dist_to_base > 0.001:
                b_norm        = [bdx / dist_to_base, bdy / dist_to_base]
                b_dot         = max(min(forward_vec[0]*b_norm[0]
                                       + forward_vec[1]*b_norm[1], 1.0), -1.0)
                b_angle       = math.acos(b_dot)
                b_cross       = forward_vec[0]*b_norm[1] - forward_vec[1]*b_norm[0]
                angle_to_base = b_angle if b_cross > 0 else -b_angle
            else:
                angle_to_base = 0.0

            # ------------------------------------------------------------------
            # CPFA SITE FIDELITY SIGNAL  [11-13]
            # Assigned at nest deposit — robot's own last pickup location.
            # Zeroed when carrying (CPFA: info used only during outbound trip).
            # ------------------------------------------------------------------
            site_known      = 0.0
            site_dist_norm  = 0.0
            site_angle_norm = 0.0

            if (not self.carrying_state[i]
                    and self.nest_target[i] is not None
                    and self.nest_target[i][0] == 'site'):
                sx  = self.nest_target[i][1]
                sy  = self.nest_target[i][2]
                sdx = sx - robot_pos[0]
                sdy = sy - robot_pos[1]
                s_dist = math.sqrt(sdx*sdx + sdy*sdy)
                if s_dist > 0.001:
                    s_norm        = [sdx / s_dist, sdy / s_dist]
                    s_dot         = max(min(forward_vec[0]*s_norm[0]
                                           + forward_vec[1]*s_norm[1], 1.0), -1.0)
                    s_angle       = math.acos(s_dot)
                    s_cross       = forward_vec[0]*s_norm[1] - forward_vec[1]*s_norm[0]
                    s_angle       = s_angle if s_cross > 0 else -s_angle
                    site_known      = 1.0
                    site_dist_norm  = min(s_dist / 3.5, 1.0)
                    site_angle_norm = s_angle / math.pi

            # ------------------------------------------------------------------
            # CPFA PHEROMONE SIGNAL  [14-16]
            # Assigned at nest deposit via roulette-wheel selection.
            # Zeroed when carrying (CPFA: info used only during outbound trip).
            # ------------------------------------------------------------------
            phero_known      = 0.0
            phero_dist_norm  = 0.0
            phero_angle_norm = 0.0

            if (not self.carrying_state[i]
                    and self.nest_target[i] is not None
                    and self.nest_target[i][0] == 'phero'):
                px  = self.nest_target[i][1]
                py  = self.nest_target[i][2]
                pdx = px - robot_pos[0]
                pdy = py - robot_pos[1]
                p_dist = math.sqrt(pdx*pdx + pdy*pdy)
                if p_dist > 0.001:
                    p_norm        = [pdx / p_dist, pdy / p_dist]
                    p_dot         = max(min(forward_vec[0]*p_norm[0]
                                           + forward_vec[1]*p_norm[1], 1.0), -1.0)
                    p_angle       = math.acos(p_dot)
                    p_cross       = forward_vec[0]*p_norm[1] - forward_vec[1]*p_norm[0]
                    p_angle       = p_angle if p_cross > 0 else -p_angle
                    phero_known      = 1.0
                    phero_dist_norm  = min(p_dist / 3.5, 1.0)
                    phero_angle_norm = p_angle / math.pi

            # ------------------------------------------------------------------
            # SEARCH DURATION  [17]
            # Zeroed when carrying — only meaningful during exploration.
            # Saturates at 1.0 after SEARCH_DURATION_NORM steps so PPO
            # learns: high value → too long without food → head home.
            # ------------------------------------------------------------------
            if self.carrying_state[i]:
                search_duration_norm = 0.0
            else:
                search_duration_norm = min(
                    self.steps_without_pickup[i] / self.SEARCH_DURATION_NORM, 1.0
                )

            # ------------------------------------------------------------------
            # ASSEMBLE 18-VALUE OBSERVATION
            # ------------------------------------------------------------------
            obs = []
            obs.extend(prox)                                     # [0:8]
            obs.append(1.0 if self.carrying_state[i] else 0.0)  # [8]
            obs.append(dist_to_base / 3.5)                      # [9]
            obs.append(angle_to_base / math.pi)                  # [10]
            obs.extend([
                site_known,                                      # [11]
                site_dist_norm,                                  # [12]
                site_angle_norm,                                 # [13]
            ])
            obs.extend([
                phero_known,                                     # [14]
                phero_dist_norm,                                 # [15]
                phero_angle_norm,                                # [16]
            ])
            obs.append(search_duration_norm)                     # [17]

            global_obs.extend(obs)

        return np.array(global_obs, dtype=np.float32)

    # =========================================================================
    # REWARD
    # =========================================================================
    def get_reward(self, action):
        total_reward = 0.0
        base_pos     = self.base_node.getPosition()

        for i in range(self.num_robots):
            robot_pos = self.robot_nodes[i].getPosition()
            prox      = (self.robot_states[i] or [0.0]*8)[:8]

            # Proximity penalty
            max_prox = max(prox)
            if max_prox > 0.1:
                total_reward -= max_prox * 0.5

            # Wall penalty
            wall_dist = 2.5 - max(abs(robot_pos[0]), abs(robot_pos[1]))
            if wall_dist < 0.35:
                total_reward -= (0.35 - wall_dist) * 0.5

            if not self.carrying_state[i]:
                # ---- EXPLORING ----
                dist_from_base = math.sqrt(robot_pos[0]**2 + robot_pos[1]**2)

                # Check for pickup — count resource density BEFORE hiding tag
                picked_up = False
                for tag_node in self.tag_nodes:
                    tag_pos = tag_node.getPosition()
                    if tag_pos[2] < 0:
                        continue
                    dx   = tag_pos[0] - robot_pos[0]
                    dy   = tag_pos[1] - robot_pos[1]
                    dist = math.sqrt(dx*dx + dy*dy)

                    if dist < 0.15:
                        # Count tags within 0.5m (including this one) for Poisson CDF
                        density = sum(
                            1 for tn in self.tag_nodes
                            if tn.getPosition()[2] >= 0 and
                            math.sqrt((tn.getPosition()[0] - robot_pos[0])**2 +
                                      (tn.getPosition()[1] - robot_pos[1])**2) < 0.5
                        )

                        # Hide tag and update state
                        tag_node.getField("translation").setSFVec3f([0, 0, -10])
                        total_reward           += 5.0
                        self.total_pickups     += 1
                        self.ep_pickups        += 1
                        self.carrying_state[i]  = True
                        picked_up               = True

                        # Store CPFA pickup info for use at nest deposit
                        self.carried_from[i]      = (robot_pos[0], robot_pos[1])
                        self.resource_density[i]  = density
                        self.site_fidelity_pos[i] = (robot_pos[0], robot_pos[1])

                        # Clear nav target — robot reached the cluster
                        self.nest_target[i]         = None
                        self.prev_site_dists[i]     = None
                        self.prev_phero_dists[i]    = None
                        self.steps_without_pickup[i] = 0
                        self.give_up_timer[i]        = 0
                        self.gave_up[i]              = False  # successful pickup — site fidelity re-enabled
                        # Seed prev_base_dists so first carry step has no spike.
                        # Without this, prev_base_dists is stale from deposit (~0.25m)
                        # → first carry step reward = (0.25 − cluster_dist) × 8 ≈ −10.
                        self.prev_base_dists[i]      = dist_from_base

                        print(f"[PICKUP] Robot {i+1} picked up tag "
                              f"(density={density}) | Total: {self.total_pickups}")
                        break

                if not picked_up:
                    # Fix 2: increment search duration counter
                    self.steps_without_pickup[i] += 1

                    # Fix 1: clear stale target when robot has arrived at the location
                    # (0.18m > 0.15m pickup radius — if food were there it would be picked up)
                    target = self.nest_target[i]
                    if target is not None:
                        tx, ty       = target[1], target[2]
                        dist_to_tgt  = math.sqrt((tx - robot_pos[0])**2
                                                 + (ty - robot_pos[1])**2)
                        if dist_to_tgt < 0.05:
                            # Arrived at cluster — matches ARGoS TargetDistanceTolerance=0.05m
                            # PPO now does local search; approach shaping cleared
                            self.nest_target[i]      = None
                            self.prev_site_dists[i]  = None
                            self.prev_phero_dists[i] = None
                            target                   = None

                    # Empty return: gave-up robot arrived at nest — assign new target
                    # Mirrors baseline _handle_nest_arrival for empty returns.
                    # _assign_target called while gave_up=True → skips stale site fidelity.
                    if (self.gave_up[i] and dist_from_base < 0.25
                            and self.nest_target[i] is None):
                        self.steps_without_pickup[i] = 0
                        self.give_up_timer[i]        = 0
                        self.nest_target[i]          = self._assign_target(i)
                        self.gave_up[i]              = False
                        t = self.nest_target[i]
                        if t is not None:
                            tx, ty = t[1], t[2]
                            d = math.sqrt((tx - robot_pos[0])**2
                                          + (ty - robot_pos[1])**2)
                            if t[0] == 'site':
                                self.prev_site_dists[i]  = d
                                self.prev_phero_dists[i] = None
                            else:
                                self.prev_phero_dists[i] = d
                                self.prev_site_dists[i]  = None
                        print(f"[EMPTY_RTN] R{i+1} → "
                              f"{t[0]+'('+f'{t[1]:.2f},{t[2]:.2f}'+')' if t else 'EXPLORE'}")

                    # Forward motion bias — only reward for actual movement, no positional bonus
                    if wall_dist >= 0.35:
                        action_i  = action[i*2 : i*2+2]
                        avg_speed = (action_i[0] + action_i[1]) / 2.0
                        if avg_speed > 0:
                            total_reward += avg_speed * 0.15

                    # CPFA-style navigation reward
                    target = self.nest_target[i]

                    if target is not None and target[0] == 'site':
                        # Site fidelity approach shaping
                        sx      = target[1]; sy = target[2]
                        curr_sd = math.sqrt((sx - robot_pos[0])**2 + (sy - robot_pos[1])**2)
                        if self.prev_site_dists[i] is not None:
                            total_reward += (self.prev_site_dists[i] - curr_sd) * 15.0
                        self.prev_site_dists[i]  = curr_sd
                        self.prev_phero_dists[i] = None
                        # Orientation reward — dense per-step gradient toward site target
                        if curr_sd > 0.001:
                            robot_rot_i = self.robot_nodes[i].getOrientation()
                            fwd_i  = [robot_rot_i[0], robot_rot_i[3], robot_rot_i[6]]
                            t_norm = [(sx - robot_pos[0]) / curr_sd,
                                      (sy - robot_pos[1]) / curr_sd]
                            dot_s  = max(min(fwd_i[0]*t_norm[0] + fwd_i[1]*t_norm[1], 1.0), -1.0)
                            total_reward += dot_s * 0.5

                    elif target is not None and target[0] == 'phero':
                        # Pheromone approach shaping
                        px      = target[1]; py = target[2]
                        curr_pd = math.sqrt((px - robot_pos[0])**2 + (py - robot_pos[1])**2)
                        if self.prev_phero_dists[i] is not None:
                            total_reward += (self.prev_phero_dists[i] - curr_pd) * 15.0
                        self.prev_phero_dists[i] = curr_pd
                        self.prev_site_dists[i]  = None
                        # Orientation reward — dense per-step gradient toward pheromone target
                        if curr_pd > 0.001:
                            robot_rot_i = self.robot_nodes[i].getOrientation()
                            fwd_i  = [robot_rot_i[0], robot_rot_i[3], robot_rot_i[6]]
                            t_norm = [(px - robot_pos[0]) / curr_pd,
                                      (py - robot_pos[1]) / curr_pd]
                            dot_p  = max(min(fwd_i[0]*t_norm[0] + fwd_i[1]*t_norm[1], 1.0), -1.0)
                            total_reward += dot_p * 0.5

                    else:
                        # No target — free exploration
                        # Positional bonus removed: standing still earns nothing anywhere
                        self.prev_site_dists[i]  = None
                        self.prev_phero_dists[i] = None

                        # Give-up: same probabilistic mechanism as baseline CPFA.
                        # Every 78 steps (~5s at 64ms), check P=0.0189 — identical
                        # to baseline's 156 steps at 32ms with same probability.
                        # P2 override steers robot home when gave_up=True.
                        self.give_up_timer[i] += 1
                        if self.give_up_timer[i] >= self.GIVE_UP_CHECK_STEPS:
                            self.give_up_timer[i] = 0
                            if random.random() < self.PROB_RETURN_TO_NEST:
                                self.gave_up[i] = True
                                print(f"[GIVE-UP] R{i+1} giving up after "
                                      f"{self.steps_without_pickup[i]} steps")

                        # Always update prev_base_dists in explore mode so the give-up
                        # reward computes a correct step-delta (not a stale baseline from
                        # deposit time, which caused -10.25/step spike in Phase 3).
                        self.prev_base_dists[i] = dist_from_base

            else:
                # ---- CARRYING: returning to nest (P2 handles movement) ----
                dx           = base_pos[0] - robot_pos[0]
                dy           = base_pos[1] - robot_pos[1]
                dist_to_base = math.sqrt(dx*dx + dy*dy)

                if dist_to_base < 0.25:
                    self.carrying_state[i] = False
                    total_reward          += 20.0
                    self.total_deposits   += 1
                    self.ep_deposits      += 1
                    print(f"[DEPOSIT] Robot {i+1} deposited! Total: {self.total_deposits}")

                    # CPFA pheromone laying — Poisson CDF gate
                    if self.carried_from[i] is not None:
                        density  = self.resource_density[i]
                        lay_prob = self._poisson_cdf(density, self.RATE_OF_LAYING_PHEROMONE)
                        if random.random() < lay_prob:
                            self.pheromone_list.append({
                                'x':                self.carried_from[i][0],
                                'y':                self.carried_from[i][1],
                                'weight':           1.0,
                                'resource_density': density
                            })
                            print(f"  [PHERO] Laid at "
                                  f"({self.carried_from[i][0]:.2f}, "
                                  f"{self.carried_from[i][1]:.2f}) "
                                  f"density={density} prob={lay_prob:.2f} "
                                  f"total_entries={len(self.pheromone_list)}")

                    # CPFA target assignment at nest return
                    self.nest_target[i] = self._assign_target(i)
                    t = self.nest_target[i]
                    print(f"  [TARGET] Robot {i+1} → "
                          f"{t[0].upper() + ' (' + f'{t[1]:.2f},{t[2]:.2f}' + ')' if t else 'EXPLORE'}")

                    # Pre-seed shaping distances so reward fires from step 1
                    if t is not None:
                        tx = t[1]; ty = t[2]
                        d  = math.sqrt((tx - robot_pos[0])**2 + (ty - robot_pos[1])**2)
                        if t[0] == 'site':
                            self.prev_site_dists[i]  = d
                            self.prev_phero_dists[i] = None
                        else:
                            self.prev_phero_dists[i] = d
                            self.prev_site_dists[i]  = None
                    else:
                        self.prev_site_dists[i]  = None
                        self.prev_phero_dists[i] = None

                # Approach-base shaping
                if self.prev_base_dists[i] is not None:
                    total_reward += (self.prev_base_dists[i] - dist_to_base) * 8.0
                self.prev_base_dists[i] = dist_to_base

                # Face-toward-base reward
                if dist_to_base > 0.001:
                    robot_rot = self.robot_nodes[i].getOrientation()
                    fwd       = [robot_rot[0], robot_rot[3], robot_rot[6]]
                    b_norm    = [dx / dist_to_base, dy / dist_to_base]
                    b_dot     = max(min(fwd[0]*b_norm[0] + fwd[1]*b_norm[1], 1.0), -1.0)
                    total_reward += b_dot * 0.5

            # Time penalty
            total_reward -= 0.005

        # Inter-robot separation penalty near base only
        robot_positions = [self.robot_nodes[k].getPosition() for k in range(self.num_robots)]
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                di = math.sqrt(robot_positions[i][0]**2 + robot_positions[i][1]**2)
                dj = math.sqrt(robot_positions[j][0]**2 + robot_positions[j][1]**2)
                if di < 1.5 and dj < 1.5:
                    sep = math.sqrt(
                        (robot_positions[i][0] - robot_positions[j][0])**2 +
                        (robot_positions[i][1] - robot_positions[j][1])**2
                    )
                    if sep < 1.0:
                        total_reward -= (1.0 - sep) * 0.5

        return total_reward

    # =========================================================================
    # HARD-CODED OVERRIDES
    # =========================================================================

    def _steer_to(self, robot_pos, fwd, target, gain=2.5):
        """Proportional steering controller — returns [left, right] in [-1, 1]."""
        dx   = target[0] - robot_pos[0]
        dy   = target[1] - robot_pos[1]
        dist = math.sqrt(dx*dx + dy*dy)
        if dist < 0.01:
            return [0.0, 0.0]
        t_norm = [dx / dist, dy / dist]
        dot    = fwd[0]*t_norm[0] + fwd[1]*t_norm[1]
        cross  = fwd[0]*t_norm[1] - fwd[1]*t_norm[0]
        angle  = math.atan2(cross, dot)
        turn   = max(-1.0, min(1.0, gain * angle / math.pi))
        left   = max(-1.0, min(1.0, 1.0 - turn))
        right  = max(-1.0, min(1.0, 1.0 + turn))
        m = max(abs(left), abs(right))
        if m > 1.0:
            left /= m; right /= m
        return [left, right]

    def _apply_overrides(self, action):
        """P1: Wall escape  |  BASE_ESC: leave nest when not carrying
        P2: RTB when carrying OR gave_up (= CPFA RETURNING state)
        PPO controls everything else: DEPARTING, local search, give-up."""
        final    = list(action)
        base_pos = self.base_node.getPosition()
        for i in range(self.num_robots):
            robot_pos = self.robot_nodes[i].getPosition()
            robot_rot = self.robot_nodes[i].getOrientation()
            fwd       = [robot_rot[0], robot_rot[3], robot_rot[6]]
            prox      = (self.robot_states[i] or [0.0]*8)[:8]
            wall_dist = 2.5 - max(abs(robot_pos[0]), abs(robot_pos[1]))

            bdx          = base_pos[0] - robot_pos[0]
            bdy          = base_pos[1] - robot_pos[1]
            dist_to_base = math.sqrt(bdx*bdx + bdy*bdy)

            # P1: Wall / obstacle escape
            if wall_dist < 0.35 or max(prox) > 0.55:
                ov = self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=4.0)
                final[i*2], final[i*2+1] = ov[0], ov[1]
                continue

            # BASE_ESC: steer away from nest when not carrying, not gave_up, within 0.25m.
            # Excluded when gave_up=True — mirrors baseline where RETURNING state
            # suppresses BASE_ESC (robot must reach nest to deposit/get new target).
            if not self.carrying_state[i] and not self.gave_up[i] and dist_to_base < 0.25:
                if dist_to_base > 0.001:
                    esc_x = robot_pos[0] + (robot_pos[0] / dist_to_base) * 0.5
                    esc_y = robot_pos[1] + (robot_pos[1] / dist_to_base) * 0.5
                else:
                    esc_x, esc_y = 0.5, 0.0
                ov = self._steer_to(robot_pos, fwd, [esc_x, esc_y], gain=4.0)
                final[i*2], final[i*2+1] = ov[0], ov[1]
                continue

            # P2: Return to base when carrying OR gave up empty-handed.
            # gave_up mirrors baseline RETURNING state — hard steer to nest,
            # no RTB shaping reward needed, no PPO learning required.
            if self.carrying_state[i] or self.gave_up[i]:
                ov = self._steer_to(robot_pos, fwd, base_pos, gain=2.5)
                final[i*2], final[i*2+1] = ov[0], ov[1]
                continue

            # PPO controls everything else

        return np.array(final, dtype=np.float32)

    def is_done(self):
        return self.episode_step >= self.steps_per_episode

    def get_info(self):
        return {}

    # =========================================================================
    # LOGGING
    # =========================================================================

    def _get_mode(self, i, wall_dist):
        base_pos = self.base_node.getPosition()
        rpos     = self.robot_nodes[i].getPosition()
        d2base   = math.sqrt((rpos[0]-base_pos[0])**2 + (rpos[1]-base_pos[1])**2)
        if wall_dist < 0.35:
            return "WALL_ESC"
        elif self.carrying_state[i]:
            return "RTB"
        elif self.gave_up[i]:
            return "GIVE_UP"  # P2 steering robot home empty-handed
        elif not self.carrying_state[i] and d2base < 0.25:
            return "BASE_ESC"
        elif self.nest_target[i] is not None:
            return self.nest_target[i][0].upper()  # SITE or PHERO — PPO navigating
        else:
            return "EXPLORE"

    def _log_step(self, obs, action):
        phero_count = len(self.pheromone_list)
        phero_max_w = max((p['weight'] for p in self.pheromone_list), default=0.0)

        lines = [
            f"\n[EP {self.total_episodes} | Step {self.episode_step:4d}] "
            f"Picks={self.ep_pickups} Deps={self.ep_deposits} | "
            f"phero_entries={phero_count} max_w={phero_max_w:.3f}"
        ]

        for ri in range(self.num_robots):
            rpos   = self.robot_nodes[ri].getPosition()
            wall_d = 2.5 - max(abs(rpos[0]), abs(rpos[1]))
            obs_i  = obs[ri * self.obs_per_robot : (ri + 1) * self.obs_per_robot]
            ra_l   = float(action[ri * 2])
            ra_r   = float(action[ri * 2 + 1])
            mode   = self._get_mode(ri, wall_d)

            lines.append(
                f"  R{ri+1}[{mode:8s}]: L={ra_l:+.2f} R={ra_r:+.2f} | "
                f"carry={obs_i[8]:.0f} | "
                f"base={obs_i[9]:.2f} ang={obs_i[10]:+.2f} | "
                f"site={obs_i[11]:.0f} sd={obs_i[12]:.2f} sa={obs_i[13]:+.2f} | "
                f"phero={obs_i[14]:.0f} pd={obs_i[15]:.2f} pa={obs_i[16]:+.2f} | "
                f"search={obs_i[17]:.2f} | wall={wall_d:.2f}"
            )

        with open(self.log_file, 'a') as f:
            f.write('\n'.join(lines) + '\n')

    def _log_episode_summary(self):
        sim_minutes = (self.episode_step * self.timestep) / 60000.0
        rate        = self.ep_deposits / sim_minutes if sim_minutes > 0 else 0.0

        if self.total_episodes < 60:
            phase = "NEAR    (11 clusters, max_dist=1.2m)"
        elif self.total_episodes < 150:
            phase = "MEDIUM  (11 clusters, max_dist=1.6m)"
        elif self.total_episodes < 300:
            phase = "FAR     (11 clusters, max_dist=2.0m)"
        else:
            phase = "FULL    (11 clusters, max_dist=2.3m)"

        phero_count = len(self.pheromone_list)
        phero_max_w = max((p['weight'] for p in self.pheromone_list), default=0.0)

        header = (
            f"\n{'='*65}\n"
            f"[EP {self.total_episodes}] Picks: {self.ep_pickups} | "
            f"Deps: {self.ep_deposits} | "
            f"Rate: {rate:.2f} tags/min (sim) | TotalDeps: {self.total_deposits}\n"
            f"  Curriculum: {phase}\n"
            f"  Pheromone:  entries={phero_count} | max_weight={phero_max_w:.3f}"
        )

        robot_lines = []
        for ri in range(self.num_robots):
            rpos   = self.robot_nodes[ri].getPosition()
            wall_d = 2.5 - max(abs(rpos[0]), abs(rpos[1]))
            d2base = math.sqrt(rpos[0]**2 + rpos[1]**2)
            mode   = self._get_mode(ri, wall_d)
            t      = self.nest_target[ri]
            t_str  = f"{t[0]}({t[1]:.2f},{t[2]:.2f})" if t else "explore"

            robot_lines.append(
                f"  R{ri+1}[{mode:8s}]: "
                f"carry={int(self.carrying_state[ri])} | "
                f"base={d2base:.2f} | "
                f"target={t_str} | "
                f"density={self.resource_density[ri]}"
            )

        summary = header + '\n' + '\n'.join(robot_lines) + f"\n{'='*65}"
        print(summary)
        with open(self.log_file, 'a') as f:
            f.write(summary + '\n')

    # =========================================================================
    # RESET WITH CURRICULUM
    # =========================================================================
    def reset(self, seed=None, options=None):
        if self.episode_step > 0:
            self._log_episode_summary()

        self.total_episodes += 1
        self.episode_step    = 0
        self.ep_pickups      = 0
        self.ep_deposits     = 0
        self.robot_states    = [[0.0]*8 for _ in range(self.num_robots)]
        self.carrying_state  = [False] * self.num_robots

        # Reset CPFA pheromone state
        self.pheromone_list    = []
        self.carried_from      = [None] * self.num_robots
        self.resource_density  = [0]    * self.num_robots
        self.site_fidelity_pos = [None] * self.num_robots
        self.nest_target       = [None] * self.num_robots

        # Reset reward shaping state
        self.prev_base_dists     = [None] * self.num_robots
        self.prev_site_dists      = [None]  * self.num_robots
        self.prev_phero_dists     = [None]  * self.num_robots
        self.steps_without_pickup = [0]     * self.num_robots
        self.give_up_timer        = [0]     * self.num_robots
        self.gave_up              = [False] * self.num_robots

        self.respawn_robots()
        self.respawn_all_tags()

        return self.get_observations(), {}

    def respawn_robots(self):
        positions = [[-0.5, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0]]
        for i in range(self.num_robots):
            self.robot_nodes[i].getField("translation").setSFVec3f(positions[i])
            rand_yaw = random.uniform(0, 2 * math.pi)
            self.robot_nodes[i].getField("rotation").setSFRotation([0, 0, 1, rand_yaw])
            self.robot_nodes[i].resetPhysics()

    def respawn_all_tags(self):
        """Fixed cluster layout: 5 clusters × 8 tags + 6 clusters × 4 tags = 64 tags.
        Cluster count and tag count per cluster are fixed across all episodes.
        Only cluster distance from nest changes across curriculum phases.
        Tags arranged on a 0.0775m grid (5cm edge gap), randomly rotated each episode.
        """
        centers = self._generate_cluster_centers()  # always 11 centers

        # Fixed: first 5 clusters get 8 tags, last 6 clusters get 4 tags
        cluster_sizes = [8] * 5 + [4] * 6  # 40 + 24 = 64 tags

        GRID_SPACING = 0.0775  # tag_size(0.0275) + edge_gap(0.05) — matches eval world
        tag_idx = 0
        for ci, (cx, cy) in enumerate(centers):
            if tag_idx >= self.num_tags:
                break
            count = cluster_sizes[ci]
            cols  = math.ceil(math.sqrt(count))
            rows  = math.ceil(count / cols)
            rot   = random.uniform(0, math.pi / 2)
            cos_r = math.cos(rot)
            sin_r = math.sin(rot)

            placed = 0
            for row in range(rows):
                for col in range(cols):
                    if placed >= count or tag_idx >= self.num_tags:
                        break
                    dx = (col - (cols - 1) / 2.0) * GRID_SPACING
                    dy = (row - (rows - 1) / 2.0) * GRID_SPACING
                    tx = cx + dx * cos_r - dy * sin_r
                    ty = cy + dx * sin_r + dy * cos_r
                    tx = max(-2.3, min(2.3, tx))
                    ty = max(-2.3, min(2.3, ty))
                    self.tag_nodes[tag_idx].getField("translation").setSFVec3f(
                        [tx, ty, 0.01375])
                    tag_idx += 1
                    placed  += 1

    def _generate_cluster_centers(self):
        """Curriculum: 11 clusters fixed throughout, only max distance changes.
        With 10M timesteps / 16384 steps per episode ≈ 610 total episodes:
          Phase 1 (ep   1- 59):  59 eps (10%) — max_dist=1.2m, bootstrap pickup/deposit
          Phase 2 (ep  60-149):  90 eps (15%) — max_dist=1.6m, develop pheromone use
          Phase 3 (ep 150-299): 150 eps (25%) — max_dist=2.0m, mid-to-far range
          Phase 4 (ep 300+   ): 310 eps (51%) — max_dist=2.3m, full arena consolidation
        Wall at 2.5m; WALL_ESC fires at 2.15m on-axis → 2.3m is practical cluster max.
        Cluster layout: 5 × 8 tags + 6 × 4 tags = 64 tags total.
        """
        if self.total_episodes < 60:
            max_dist = 1.2
            min_sep  = 0.40  # tighter packing in small zone
        elif self.total_episodes < 150:
            max_dist = 1.6
            min_sep  = 0.45
        elif self.total_episodes < 300:
            max_dist = 2.0
            min_sep  = 0.50
        else:
            max_dist = 2.3
            min_sep  = 0.50

        centers = []
        for _ in range(11):
            for _ in range(200):
                angle = random.uniform(0, 2 * math.pi)
                dist  = random.uniform(0.4, max_dist)
                cx    = dist * math.cos(angle)
                cy    = dist * math.sin(angle)
                if all(math.sqrt((cx - c[0])**2 + (cy - c[1])**2) > min_sep
                       for c in centers):
                    centers.append((cx, cy))
                    break

        return centers


# =============================================================================
# TRAINING ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name',        type=str,   default='ppo_cpfa_5x5')
    parser.add_argument('--lr',              type=float, default=3e-4)
    parser.add_argument('--ent_coef',        type=float, default=0.03)
    parser.add_argument('--batch_size',      type=int,   default=1024)
    parser.add_argument('--total_timesteps', type=int,   default=10000000)
    parser.add_argument('--resume',          type=str,   default=None,
                        help='Path to checkpoint .zip to resume from')
    args = parser.parse_args()

    env    = EpuckForagingSupervisor()
    device = 'cpu'
    print(f"[TRAINING] Run: {args.run_name} | LR: {args.lr} | "
          f"Ent: {args.ent_coef} | Batch: {args.batch_size} | "
          f"Steps: {args.total_timesteps}")
    print(f"[INFO] Obs space: {env.observation_space_dim} "
          f"({env.obs_per_robot} per robot)")
    print(f"[CPFA] RateOfLayingPheromone={env.RATE_OF_LAYING_PHEROMONE} | "
          f"RateOfSiteFidelity={env.RATE_OF_SITE_FIDELITY} | "
          f"RateOfPheromoneDecay={env.RATE_OF_PHEROMONE_DECAY}")

    policy_kwargs = dict(net_arch=[256, 256], activation_fn=torch.nn.Tanh)

    if args.resume:
        print(f"[TRAINING] Resuming from: {args.resume}")
        model = PPO.load(
            args.resume, env=env, device=device,
            learning_rate=args.lr, ent_coef=args.ent_coef,
            batch_size=args.batch_size
        )
        import re
        match      = re.search(r'_(\d+)_steps', args.resume)
        steps_done = int(match.group(1)) if match else 0
        remaining  = max(args.total_timesteps - steps_done, 0)
        print(f"[TRAINING] Steps done: {steps_done} | Remaining: {remaining}")
    else:
        model = PPO(
            "MlpPolicy", env, verbose=1, device=device,
            n_steps=16384, batch_size=args.batch_size,
            learning_rate=args.lr, ent_coef=args.ent_coef,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=policy_kwargs
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
