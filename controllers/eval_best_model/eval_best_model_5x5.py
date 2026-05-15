import math
import random
import numpy as np
import sys
from controller import Supervisor
from stable_baselines3 import PPO
import gymnasium as gym

# =============================================================================
# EVALUATION SUPERVISOR — CPFA-RL  (5×5m arena)
#
# Observation space matches epuck_foraging_supervisor_cpfa.py exactly:
#   21 values per robot × 4 robots = 84 total
#
#  [0:8]  Proximity sensors
#  [8]    tag_visible
#  [9]    tag_dist_norm          (/ 1.0m)
#  [10]   tag_angle_norm         (/ pi)
#  [11]   carrying
#  [12]   dist_to_base_norm      (/ 3.5m)
#  [13]   angle_to_base_norm     (/ pi)
#  [14]   site_known             CPFA site fidelity target (assigned at nest)
#  [15]   site_dist_norm         (/ 3.5m)
#  [16]   site_angle_norm        (/ pi)
#  [17]   phero_known            CPFA pheromone target (roulette-wheel, at nest)
#  [18]   phero_dist_norm        (/ 3.5m)
#  [19]   phero_angle_norm       (/ pi)
#  [20]   search_duration_norm   steps_without_pickup / 700  (give-up signal)
#
#  obs[14-20] zeroed when carrying=True
#
# Hard-coded overrides (identical to training supervisor):
#   P1: Wall / obstacle escape
#   BASE_ESC: steer away from nest when not carrying and dist_to_base < 0.25m
#   P2: Return-to-base when carrying  (= CPFA RETURNING state)
#   PPO controls everything else: DEPARTING, local search, tag approach,
#                                 give-up, exploration
# =============================================================================

class EpuckForagingSupervisor(Supervisor, gym.Env):
    def __init__(self):
        self.num_robots            = 4
        self.num_tags              = 64
        self.obs_per_robot         = 21
        self.observation_space_dim = self.obs_per_robot * self.num_robots
        self.action_space_dim      = 2 * self.num_robots

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

        # --- Robot nodes ---
        self.robot_nodes = []
        for i in range(self.num_robots):
            node = self.getFromDef(f"ROBOT{i+1}")
            if node is None:
                print(f"[ERROR] Could not find ROBOT{i+1}!")
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
        self.prev_tag_dists  = [None]  * self.num_robots

        # --- CPFA pheromone list ---
        # Each entry: {x, y, weight, resource_density}
        # Created at nest deposit, decayed each step, pruned when weight < PHEROMONE_MIN
        self.pheromone_list = []

        # --- CPFA parameters (must match training supervisor exactly) ---
        self.RATE_OF_LAYING_PHEROMONE = 3.0
        self.RATE_OF_SITE_FIDELITY    = 3.0
        self.RATE_OF_PHEROMONE_DECAY  = 0.01  # exponential decay per second
        self.PHEROMONE_MIN            = 0.001

        # --- Per-robot CPFA state ---
        self.carried_from      = [None] * self.num_robots  # (x,y) where food was picked up
        self.resource_density  = [0]    * self.num_robots  # tag count within 0.5m at pickup
        self.site_fidelity_pos = [None] * self.num_robots  # robot's own last pickup location
        # nest_target: None | ('site', x, y) | ('phero', x, y)
        # Assigned at nest deposit, cleared when robot picks up food
        self.nest_target       = [None] * self.num_robots

        # --- Reward shaping state (needed for deposit logic) ---
        self.prev_site_dists  = [None] * self.num_robots
        self.prev_phero_dists = [None] * self.num_robots

        # --- Search duration counter (matches training) ---
        self.steps_without_pickup = [0] * self.num_robots
        self.SEARCH_GIVE_UP_STEPS = 500
        self.SEARCH_DURATION_NORM = 700.0

        # --- Stats ---
        self.step_count     = 0
        self.total_pickups  = 0
        self.total_deposits = 0

    # =========================================================================
    # CPFA HELPERS (identical to training supervisor)
    # =========================================================================

    def _poisson_cdf(self, n, rate):
        """P(X <= n) where X ~ Poisson(rate). Matches CPFA GetPoissonCDF."""
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
        """CPFA roulette-wheel selection from pheromone_list, weighted by weight."""
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
        return (active[-1]['x'], active[-1]['y'])

    def _assign_target(self, i):
        """CPFA target assignment at nest return.
        Priority 1: site fidelity  (Poisson CDF gate)
        Priority 2: pheromone      (roulette-wheel)
        Priority 3: random search  (None)"""
        if self.site_fidelity_pos[i] is not None:
            sf_prob = self._poisson_cdf(self.resource_density[i], self.RATE_OF_SITE_FIDELITY)
            if random.random() < sf_prob:
                sx, sy = self.site_fidelity_pos[i]
                return ('site', sx, sy)
        selected = self._roulette_select()
        if selected is not None:
            return ('phero', selected[0], selected[1])
        return None

    # =========================================================================
    # STEP
    # =========================================================================
    def step(self, action):
        self.step_count += 1

        action = self._apply_overrides(action)

        # Send motor commands
        for i in range(self.num_robots):
            robot_action = action[i*2 : (i+1)*2]
            msg = f"{robot_action[0]},{robot_action[1]}".encode('utf-8')
            self.emitters[i].send(msg)

        if Supervisor.step(self, self.timestep) == -1:
            exit()

        # Read proximity sensor data
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

        # CPFA pheromone decay — time-based exponential (matches training)
        dt    = self.timestep / 1000.0
        decay = math.exp(-self.RATE_OF_PHEROMONE_DECAY * dt)
        for p in self.pheromone_list:
            p['weight'] *= decay
        self.pheromone_list = [p for p in self.pheromone_list
                               if p['weight'] > self.PHEROMONE_MIN]

        # Process pickups and deposits (reward logic drives state transitions)
        self._process_events()
        obs = self.get_observations()

        return obs, 0.0, False, {}

    def _process_events(self):
        """Handle pickup and deposit state transitions (mirrors get_reward logic)."""
        base_pos = self.base_node.getPosition()

        for i in range(self.num_robots):
            robot_pos = self.robot_nodes[i].getPosition()

            if not self.carrying_state[i]:
                # Fix 1: clear stale target on arrival
                target = self.nest_target[i]
                if target is not None:
                    tx, ty      = target[1], target[2]
                    dist_to_tgt = math.sqrt((tx - robot_pos[0])**2 + (ty - robot_pos[1])**2)
                    if dist_to_tgt < 0.18:
                        self.nest_target[i]      = None
                        self.prev_site_dists[i]  = None
                        self.prev_phero_dists[i] = None

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
                        density = sum(
                            1 for tn in self.tag_nodes
                            if tn.getPosition()[2] >= 0 and
                            math.sqrt((tn.getPosition()[0] - robot_pos[0])**2 +
                                      (tn.getPosition()[1] - robot_pos[1])**2) < 0.5
                        )
                        tag_node.getField("translation").setSFVec3f([0, 0, -10])
                        self.carrying_state[i]       = True
                        self.carried_from[i]         = (robot_pos[0], robot_pos[1])
                        self.resource_density[i]     = density
                        self.site_fidelity_pos[i]    = (robot_pos[0], robot_pos[1])
                        self.nest_target[i]          = None
                        self.prev_site_dists[i]      = None
                        self.prev_phero_dists[i]     = None
                        self.steps_without_pickup[i] = 0
                        picked_up                    = True
                        self.total_pickups          += 1
                        print(f"[PICKUP] Robot {i+1} picked up tag "
                              f"(density={density}) | Total: {self.total_pickups}")
                        break

                if not picked_up:
                    self.steps_without_pickup[i] += 1

            else:
                # Carrying — check for deposit
                dx           = base_pos[0] - robot_pos[0]
                dy           = base_pos[1] - robot_pos[1]
                dist_to_base = math.sqrt(dx*dx + dy*dy)
                self.prev_base_dists[i] = dist_to_base

                if dist_to_base < 0.25:
                    self.carrying_state[i]  = False
                    self.total_deposits    += 1
                    print(f"[DEPOSIT] Robot {i+1} deposited! Total: {self.total_deposits}")

                    # CPFA pheromone laying at nest — Poisson CDF gate
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

                    # CPFA target assignment — site fidelity or pheromone roulette
                    self.nest_target[i] = self._assign_target(i)
                    t = self.nest_target[i]
                    print(f"  [TARGET] Robot {i+1} → "
                          f"{t[0].upper() + ' (' + f'{t[1]:.2f},{t[2]:.2f}' + ')' if t else 'EXPLORE'}")

    # =========================================================================
    # OBSERVATIONS (21 per robot — identical to training supervisor)
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
            # TAG SENSING
            # ------------------------------------------------------------------
            tag_visible   = 0.0
            tag_dist      = 0.0
            tag_angle     = 0.0
            min_dist      = float('inf')
            omni_min_dist = float('inf')

            if not self.carrying_state[i]:
                for tag_node in self.tag_nodes:
                    tag_pos = tag_node.getPosition()
                    if tag_pos[2] < 0:
                        continue
                    dx   = tag_pos[0] - robot_pos[0]
                    dy   = tag_pos[1] - robot_pos[1]
                    dist = math.sqrt(dx*dx + dy*dy)

                    if dist < 1.0 and dist < omni_min_dist:
                        omni_min_dist = dist

                    if dist < 1.0 and dist > 0.001:
                        tag_vec_norm = [dx / dist, dy / dist]
                        dot          = max(min(forward_vec[0]*tag_vec_norm[0]
                                              + forward_vec[1]*tag_vec_norm[1], 1.0), -1.0)
                        angle        = math.acos(dot)
                        if angle < 1.2 and dist < min_dist:
                            min_dist    = dist
                            cross       = forward_vec[0]*tag_vec_norm[1] - forward_vec[1]*tag_vec_norm[0]
                            tag_angle   = angle if cross > 0 else -angle
                            tag_visible = 1.0
                            tag_dist    = dist

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
            # CPFA SITE FIDELITY SIGNAL  [14-16]
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
            # CPFA PHEROMONE SIGNAL  [17-19]
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
            # SEARCH DURATION  [20]
            # ------------------------------------------------------------------
            if self.carrying_state[i]:
                search_duration_norm = 0.0
            else:
                search_duration_norm = min(
                    self.steps_without_pickup[i] / self.SEARCH_DURATION_NORM, 1.0
                )

            # ------------------------------------------------------------------
            # ASSEMBLE 21-VALUE OBSERVATION
            # ------------------------------------------------------------------
            obs = []
            obs.extend(prox)                                     # [0:8]
            obs.extend([
                tag_visible,                                     # [8]
                tag_dist / 1.0,                                  # [9]
                tag_angle / math.pi,                             # [10]
            ])
            obs.append(1.0 if self.carrying_state[i] else 0.0)  # [11]
            obs.append(dist_to_base / 3.5)                      # [12]
            obs.append(angle_to_base / math.pi)                  # [13]
            obs.extend([
                site_known,                                      # [14]
                site_dist_norm,                                  # [15]
                site_angle_norm,                                 # [16]
            ])
            obs.extend([
                phero_known,                                     # [17]
                phero_dist_norm,                                 # [18]
                phero_angle_norm,                                # [19]
            ])
            obs.append(search_duration_norm)                     # [20]

            global_obs.extend(obs)
            self.prev_tag_dists[i] = omni_min_dist if omni_min_dist < float('inf') else None

        return np.array(global_obs, dtype=np.float32)

    # =========================================================================
    # HARD-CODED OVERRIDES (identical to training supervisor)
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
        P2: RTB when carrying (= CPFA RETURNING state)
        PPO controls everything else: DEPARTING, local search, tag seek, give-up."""
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

            # BASE_ESC: steer away from nest when not carrying and within 0.25m.
            # Base station is a 0.1m radius cylinder — robots get stuck against it
            # after deposit. Same 0.25m threshold used in cpfa_baseline for a
            # fair comparison.
            if not self.carrying_state[i] and dist_to_base < 0.25:
                if dist_to_base > 0.001:
                    esc_x = robot_pos[0] + (robot_pos[0] / dist_to_base) * 0.5
                    esc_y = robot_pos[1] + (robot_pos[1] / dist_to_base) * 0.5
                else:
                    esc_x, esc_y = 0.5, 0.0
                ov = self._steer_to(robot_pos, fwd, [esc_x, esc_y], gain=4.0)
                final[i*2], final[i*2+1] = ov[0], ov[1]
                continue

            # P2: Return to base when carrying
            if self.carrying_state[i]:
                ov = self._steer_to(robot_pos, fwd, base_pos, gain=2.5)
                final[i*2], final[i*2+1] = ov[0], ov[1]
                continue

            # PPO controls everything else

        return np.array(final, dtype=np.float32)

    def _get_mode(self, i, wall_dist):
        base_pos = self.base_node.getPosition()
        rpos     = self.robot_nodes[i].getPosition()
        d2base   = math.sqrt((rpos[0]-base_pos[0])**2 + (rpos[1]-base_pos[1])**2)
        if wall_dist < 0.35:
            return "WALL_ESC"
        elif not self.carrying_state[i] and d2base < 0.25:
            return "BASE_ESC"
        elif self.carrying_state[i]:
            return "RTB"
        elif self.nest_target[i] is not None:
            return self.nest_target[i][0].upper()  # SITE or PHERO — PPO navigating
        else:
            if self.steps_without_pickup[i] >= self.SEARCH_GIVE_UP_STEPS:
                return "GIVE_UP"
            return "EXPLORE"

    def reset(self):
        positions = [[-0.5, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0]]
        for i in range(self.num_robots):
            self.robot_nodes[i].getField("translation").setSFVec3f(positions[i])
            self.robot_nodes[i].getField("rotation").setSFRotation([0, 0, 1, 0])
            self.robot_nodes[i].resetPhysics()

        self.robot_states         = [None]  * self.num_robots
        self.carrying_state       = [False] * self.num_robots
        self.prev_base_dists      = [None]  * self.num_robots
        self.prev_tag_dists       = [None]  * self.num_robots
        self.pheromone_list       = []
        self.carried_from         = [None]  * self.num_robots
        self.resource_density     = [0]     * self.num_robots
        self.site_fidelity_pos    = [None]  * self.num_robots
        self.nest_target          = [None]  * self.num_robots
        self.prev_site_dists      = [None]  * self.num_robots
        self.prev_phero_dists     = [None]  * self.num_robots
        self.steps_without_pickup = [0]     * self.num_robots

        return self.get_observations()


# =============================================================================
# EVALUATION ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("EVALUATION MODE — CPFA-RL Trained Model")
    print("Obs space: 21 per robot × 4 robots = 84 total")
    print("=" * 60)

    env = EpuckForagingSupervisor()

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "ppo_cpfa_5x5"
    if model_path.endswith('.zip'):
        model_path = model_path[:-4]

    print(f"\nLoading model: {model_path}")

    try:
        custom_objects = {
            "lr_schedule": lambda _: 3e-4,
            "clip_range":  lambda _: 0.2,
        }
        model = PPO.load(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Make sure you trained with epuck_foraging_supervisor_cpfa.py "
              "(obs_per_robot=21, CPFA pheromone list).")
        sys.exit(1)

    print("Model loaded. Starting evaluation...\n")
    print("=" * 60 + "\n")

    obs        = env.reset()
    step_count = 0

    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, _, done, _ = env.step(action)
        step_count += 1

        # Debug print every 500 steps
        if step_count % 500 == 0:
            phero_active = len([p for p in env.pheromone_list
                                if p['weight'] > env.PHEROMONE_MIN])
            phero_max    = max((p['weight'] for p in env.pheromone_list), default=0.0)
            elapsed_min  = step_count * env.timestep / 1000.0 / 60.0
            rate         = env.total_deposits / elapsed_min if elapsed_min > 0 else 0.0

            log_msg = (
                f"\n{'='*70}\n"
                f"Step {step_count} ({elapsed_min:.1f} min) | "
                f"Pickups: {env.total_pickups} | Deposits: {env.total_deposits} | "
                f"Rate: {rate:.2f} tags/min | "
                f"phero_entries={phero_active} phero_max={phero_max:.3f}\n"
                f"{'='*70}\n"
            )

            base_pos = env.base_node.getPosition()
            for ri in range(4):
                ro     = obs[ri*21 : (ri+1)*21]
                ra_l   = action[ri*2]
                ra_r   = action[ri*2+1]
                rpos   = env.robot_nodes[ri].getPosition()
                wall_d = 2.5 - max(abs(rpos[0]), abs(rpos[1]))
                d2base = math.sqrt((rpos[0]-base_pos[0])**2 + (rpos[1]-base_pos[1])**2)
                mode   = env._get_mode(ri, wall_d)

                log_msg += (
                    f"R{ri+1}[{mode:10s}]: L={ra_l:+.2f} R={ra_r:+.2f} | "
                    f"carry={ro[11]:.0f} | "
                    f"tag_vis={ro[8]:.0f} td={ro[9]:.2f} ta={ro[10]:+.2f} | "
                    f"base={ro[12]:.2f} ba={ro[13]:+.2f} | "
                    f"site={ro[14]:.0f} sd={ro[15]:.2f} sa={ro[16]:+.2f} | "
                    f"phero={ro[17]:.0f} pd={ro[18]:.2f} pa={ro[19]:+.2f} | "
                    f"srch={ro[20]:.2f} | wall={wall_d:.2f}\n"
                )

            print(log_msg)
            with open("eval_cpfa_log.txt", "a") as f:
                f.write(log_msg)