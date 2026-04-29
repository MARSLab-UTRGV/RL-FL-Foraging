import math
import random
import numpy as np
import argparse
import torch
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from controller import Supervisor
import gym

# =============================================================================
# DECENTRALIZED MULTI-AGENT FORAGING — Supervisor + RLlib
#
# Architecture:
#   - 4 independent PPO policies (one per robot) via RLlib multi-agent
#   - No shared brain, no central pheromone grid
#   - Pheromone: per-robot waypoint memory, peer-to-peer sharing within 2m
#   - Same hard-coded overrides as centralized (P1–P4)
#   - Same curriculum and reward structure as centralized v17
#
# Observation per robot (18D):
#   [0:8]  proximity sensors
#   [8]    tag_visible
#   [9]    tag_dist / 1.0
#   [10]   tag_angle / π
#   [11]   carrying
#   [12]   dist_to_base / 3.5
#   [13]   angle_to_base / π
#   [14]   comm_known       — active waypoint in memory?
#   [15]   comm_dist / 3.5  — distance to waypoint
#   [16]   comm_angle / π   — signed angle to waypoint
#   [17]   comm_strength    — tag density signal (0–1)
#
# Pheromone waypoint per robot:
#   { "hotspot": (x, y), "strength": 0–1, "ttl": 0–400, "from": agent_id }
#   pickup_strength = 0.2 + 0.8 × min(nearby_tags / 5.0, 1.0)
#   decay: strength *= exp(−0.003) per step; deleted when strength < 0.01 or ttl ≤ 0
# =============================================================================

# ---------- Constants ----------
PHEROMONE_MIN   = 0.01
PHEROMONE_DECAY = math.exp(-0.003)   # half-life ~230 steps (~7 s)
INITIAL_TTL     = 400
COMM_RANGE      = 2.0                # metres — peer-to-peer radius
TAG_SEEK_RANGE  = 1.0                # metres — same as centralized v16/v17
FOV_HALF_ANGLE  = 1.2                # radians (~69°)
DENSITY_RADIUS  = 0.5                # metres — nearby-tag count for strength
DENSITY_MAX     = 5                  # tags at max density


class DecentralizedForagingEnv(MultiAgentEnv):
    """
    RLlib MultiAgentEnv wrapping the Webots supervisor.
    4 robots, each with an independent PPO policy.
    Pheromone communication is simulated as peer-to-peer with 2 m range limit.
    """

    def __init__(self, config=None):
        super().__init__()

        self.supervisor = Supervisor()
        self.timestep   = int(self.supervisor.getBasicTimeStep())

        self.num_robots        = 4
        self.num_tags          = 70
        self.obs_per_robot     = 18
        self.steps_per_episode = 4096

        # Agent IDs (strings used as dict keys throughout)
        self.agents      = [f"robot_{i+1}" for i in range(self.num_robots)]
        self._agent_ids  = set(self.agents)

        # RLlib spaces (per agent)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.obs_per_robot,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(2,), dtype=np.float32
        )

        # ---------- Webots nodes ----------
        self.robot_nodes = [
            self.supervisor.getFromDef(f"ROBOT{i+1}") for i in range(self.num_robots)
        ]
        self.tag_nodes = [
            self.supervisor.getFromDef(f"APRILTAG_{i+1}") for i in range(self.num_tags)
        ]
        self.base_node = self.supervisor.getFromDef("BASE_STATION")

        # Supervisor emitters/receivers (supervisor ↔ each robot)
        self.emitters  = []
        self.receivers = []
        for i in range(self.num_robots):
            self.emitters.append(self.supervisor.getDevice(f"emitter{i+1}"))
            recv = self.supervisor.getDevice(f"receiver{i+1}")
            recv.enable(self.timestep)
            self.receivers.append(recv)

        # ---------- Per-robot runtime state ----------
        self.robot_states       = {aid: [0.0] * 8 for aid in self.agents}
        self.carrying_state     = {aid: False      for aid in self.agents}
        self.prev_tag_dists     = {aid: None       for aid in self.agents}
        self.prev_base_dists    = {aid: None       for aid in self.agents}
        self.prev_hotspot_dists = {aid: None       for aid in self.agents}

        # Per-robot independent pheromone memory (decentralized — no shared grid)
        self.pheromone_memory = {
            aid: {"hotspot": None, "strength": 0.0, "ttl": 0, "from": None}
            for aid in self.agents
        }

        # Episode counters
        self.episode_step   = 0
        self.total_episodes = 0
        self.total_pickups  = 0
        self.total_deposits = 0

    # =========================================================================
    # RLlib interface
    # =========================================================================

    def reset(self):
        self.total_episodes += 1
        self.episode_step = 0

        self.carrying_state     = {aid: False for aid in self.agents}
        self.robot_states       = {aid: [0.0] * 8 for aid in self.agents}
        self.prev_tag_dists     = {aid: None  for aid in self.agents}
        self.prev_base_dists    = {aid: None  for aid in self.agents}
        self.prev_hotspot_dists = {aid: None  for aid in self.agents}
        self.pheromone_memory   = {
            aid: {"hotspot": None, "strength": 0.0, "ttl": 0, "from": None}
            for aid in self.agents
        }

        self._respawn_robots()
        self._respawn_all_tags()

        # Advance one step to populate sensor readings
        self.supervisor.step(self.timestep)
        self._collect_robot_states()

        return self._get_observations()

    def step(self, action_dict):
        self.episode_step += 1

        # Apply overrides and send motor commands to each robot
        for i, aid in enumerate(self.agents):
            raw_action   = action_dict.get(aid, [0.0, 0.0])
            final_action = self._apply_overrides(aid, raw_action)
            msg = f"{final_action[0]},{final_action[1]}".encode('utf-8')
            self.emitters[i].send(msg)

        # Advance Webots simulation one step
        if self.supervisor.step(self.timestep) == -1:
            exit()

        # Collect sensor readings from robots
        self._collect_robot_states()

        # Decay all pheromone waypoints
        self._decay_pheromone()

        # Compute per-agent rewards (includes pickup/deposit detection)
        rewards = self._compute_rewards(action_dict)

        # Share pheromone between robots within COMM_RANGE (peer-to-peer)
        self._share_pheromone()

        # Build observations
        obs = self._get_observations()

        # Episode termination
        done_flag = self.episode_step >= self.steps_per_episode
        dones = {aid: done_flag for aid in self.agents}
        dones["__all__"] = done_flag

        infos = {aid: {} for aid in self.agents}

        return obs, rewards, dones, infos

    # =========================================================================
    # Observations  (18D per robot)
    # =========================================================================

    def _get_observations(self):
        obs_dict = {}
        base_pos = self.base_node.getPosition()

        for i, aid in enumerate(self.agents):
            robot_pos = self.robot_nodes[i].getPosition()
            robot_rot = self.robot_nodes[i].getOrientation()
            fwd  = [robot_rot[0], robot_rot[3], robot_rot[6]]
            prox = self.robot_states[aid]

            # ------ Tag sensing (same as centralized) ------
            tag_visible = 0.0
            tag_dist    = 0.0
            tag_angle   = 0.0
            min_dist    = float('inf')

            if not self.carrying_state[aid]:
                for tag_node in self.tag_nodes:
                    tag_pos = tag_node.getPosition()
                    if tag_pos[2] < 0:
                        continue
                    dx   = tag_pos[0] - robot_pos[0]
                    dy   = tag_pos[1] - robot_pos[1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < TAG_SEEK_RANGE and dist > 0.001:
                        t_norm = [dx / dist, dy / dist]
                        dot    = max(min(fwd[0]*t_norm[0] + fwd[1]*t_norm[1], 1.0), -1.0)
                        angle  = math.acos(dot)
                        if angle < FOV_HALF_ANGLE and dist < min_dist:
                            min_dist  = dist
                            cross     = fwd[0]*t_norm[1] - fwd[1]*t_norm[0]
                            tag_angle = angle if cross > 0 else -angle
                if min_dist < float('inf'):
                    tag_visible = 1.0
                    tag_dist    = min_dist

            # ------ Base navigation ------
            bdx          = base_pos[0] - robot_pos[0]
            bdy          = base_pos[1] - robot_pos[1]
            dist_to_base = math.sqrt(bdx * bdx + bdy * bdy)
            if dist_to_base > 0.001:
                b_norm        = [bdx / dist_to_base, bdy / dist_to_base]
                b_dot         = max(min(fwd[0]*b_norm[0] + fwd[1]*b_norm[1], 1.0), -1.0)
                b_angle       = math.acos(b_dot)
                b_cross       = fwd[0]*b_norm[1] - fwd[1]*b_norm[0]
                angle_to_base = b_angle if b_cross > 0 else -b_angle
            else:
                angle_to_base = 0.0

            # ------ Pheromone waypoint (comm signal) ------
            # Only populated when NOT carrying — carrying robot already knows to RTB
            comm_known    = 0.0
            comm_dist     = 0.0
            comm_angle    = 0.0
            comm_strength = 0.0

            if not self.carrying_state[aid]:
                mem = self.pheromone_memory[aid]
                if mem["hotspot"] is not None and mem["strength"] >= PHEROMONE_MIN:
                    hx, hy  = mem["hotspot"]
                    hdx     = hx - robot_pos[0]
                    hdy     = hy - robot_pos[1]
                    h_dist  = math.sqrt(hdx * hdx + hdy * hdy)
                    if h_dist > 0.001:
                        h_norm    = [hdx / h_dist, hdy / h_dist]
                        h_dot     = max(min(fwd[0]*h_norm[0] + fwd[1]*h_norm[1], 1.0), -1.0)
                        h_angle   = math.acos(h_dot)
                        h_cross   = fwd[0]*h_norm[1] - fwd[1]*h_norm[0]
                        comm_known    = 1.0
                        comm_dist     = min(h_dist / 3.5, 1.0)
                        comm_angle    = h_angle if h_cross > 0 else -h_angle
                        comm_strength = mem["strength"]

            # ------ Assemble 18D observation ------
            robot_obs = list(prox)          # [0:8]
            robot_obs += [
                tag_visible,                # [8]
                tag_dist / TAG_SEEK_RANGE,  # [9]
                tag_angle / math.pi,        # [10]
                1.0 if self.carrying_state[aid] else 0.0,  # [11]
                dist_to_base / 3.5,         # [12]
                angle_to_base / math.pi,    # [13]
                comm_known,                 # [14]
                comm_dist,                  # [15]
                comm_angle / math.pi,       # [16]
                comm_strength,              # [17]
            ]
            obs_dict[aid] = np.array(robot_obs, dtype=np.float32)

        return obs_dict

    # =========================================================================
    # Rewards  (per robot, matching centralized v17)
    # =========================================================================

    def _compute_rewards(self, action_dict):
        rewards  = {aid: 0.0 for aid in self.agents}
        base_pos = self.base_node.getPosition()

        for i, aid in enumerate(self.agents):
            robot_pos = self.robot_nodes[i].getPosition()
            prox      = self.robot_states[aid]
            wall_dist = 2.5 - max(abs(robot_pos[0]), abs(robot_pos[1]))

            # Proximity penalty
            max_prox = max(prox)
            if max_prox > 0.1:
                rewards[aid] -= max_prox * 0.5

            # Wall penalty
            if wall_dist < 0.35:
                rewards[aid] -= (0.35 - wall_dist) * 0.5

            if not self.carrying_state[aid]:
                # ---- EXPLORING ----
                picked_up = False
                for tag_node in self.tag_nodes:
                    tag_pos = tag_node.getPosition()
                    if tag_pos[2] < 0:
                        continue
                    dx   = tag_pos[0] - robot_pos[0]
                    dy   = tag_pos[1] - robot_pos[1]
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < 0.15:
                        self.carrying_state[aid] = True
                        tag_node.getField("translation").setSFVec3f([0, 0, -10])
                        rewards[aid]      += 5.0
                        self.total_pickups += 1
                        picked_up          = True
                        print(f"[PICKUP] {aid} | Total: {self.total_pickups}")

                        # Create pheromone waypoint encoding cluster density
                        strength = self._pickup_strength(robot_pos)
                        self.pheromone_memory[aid] = {
                            "hotspot":  (robot_pos[0], robot_pos[1]),
                            "strength": strength,
                            "ttl":      INITIAL_TTL,
                            "from":     aid
                        }
                        self.prev_hotspot_dists[aid] = None
                        self.prev_tag_dists[aid]     = None
                        break

                if not picked_up:
                    # Tag approach shaping (within TAG_SEEK_RANGE)
                    curr_min = float('inf')
                    for tag_node in self.tag_nodes:
                        tag_pos = tag_node.getPosition()
                        if tag_pos[2] < 0:
                            continue
                        dx   = tag_pos[0] - robot_pos[0]
                        dy   = tag_pos[1] - robot_pos[1]
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist < TAG_SEEK_RANGE and dist < curr_min:
                            curr_min = dist
                    if self.prev_tag_dists[aid] is not None and curr_min < float('inf'):
                        rewards[aid] += (self.prev_tag_dists[aid] - curr_min) * 8.0
                    self.prev_tag_dists[aid] = curr_min if curr_min < float('inf') else None

                    # Tag-visible reward
                    robot_rot = self.robot_nodes[i].getOrientation()
                    fwd = [robot_rot[0], robot_rot[3], robot_rot[6]]
                    for tag_node in self.tag_nodes:
                        tag_pos = tag_node.getPosition()
                        if tag_pos[2] < 0:
                            continue
                        tdx = tag_pos[0] - robot_pos[0]
                        tdy = tag_pos[1] - robot_pos[1]
                        td  = math.sqrt(tdx * tdx + tdy * tdy)
                        if td < TAG_SEEK_RANGE and td > 0.001:
                            dot = fwd[0]*(tdx/td) + fwd[1]*(tdy/td)
                            if math.acos(max(min(dot, 1.0), -1.0)) < FOV_HALF_ANGLE:
                                rewards[aid] += 0.2
                                break

                    # Zone reward
                    dist_from_base = math.sqrt(robot_pos[0]**2 + robot_pos[1]**2)
                    if 0.8 < dist_from_base < 2.4:
                        rewards[aid] += 0.10

                    # Near-base penalty
                    if dist_from_base < 0.8:
                        rewards[aid] -= (0.8 - dist_from_base) * 2.0

                    # Forward motion bias (prevents backward degenerate policy)
                    if wall_dist >= 0.35:
                        action    = action_dict.get(aid, [0.0, 0.0])
                        avg_speed = (action[0] + action[1]) / 2.0
                        if avg_speed > 0:
                            rewards[aid] += avg_speed * 0.01

                    # Pheromone approach + facing rewards (v17 fixes 2 & 3)
                    mem = self.pheromone_memory[aid]
                    if mem["hotspot"] is not None and mem["strength"] >= PHEROMONE_MIN:
                        hx, hy  = mem["hotspot"]
                        curr_hd = math.sqrt((hx - robot_pos[0])**2 + (hy - robot_pos[1])**2)

                        # Approach shaping ×3.0 (user spec) + extra ×15 gradient
                        if self.prev_hotspot_dists[aid] is not None:
                            rewards[aid] += (self.prev_hotspot_dists[aid] - curr_hd) * 3.0
                        self.prev_hotspot_dists[aid] = curr_hd

                        # Cluster-facing reward: trains PPO to orient toward waypoint
                        if curr_hd > 0.001:
                            robot_rot = self.robot_nodes[i].getOrientation()
                            fwd       = [robot_rot[0], robot_rot[3], robot_rot[6]]
                            h_norm    = [(hx - robot_pos[0]) / curr_hd,
                                         (hy - robot_pos[1]) / curr_hd]
                            h_dot = max(min(fwd[0]*h_norm[0] + fwd[1]*h_norm[1], 1.0), -1.0)
                            rewards[aid] += h_dot * 0.3
                    else:
                        # No signal: reward outward exploration
                        dist_from_base = math.sqrt(robot_pos[0]**2 + robot_pos[1]**2)
                        if dist_from_base > 0.8:
                            rewards[aid] += min(dist_from_base / 2.3, 1.0) * 0.15
                        self.prev_hotspot_dists[aid] = None

            else:
                # ---- CARRYING: RTB ----
                dx           = base_pos[0] - robot_pos[0]
                dy           = base_pos[1] - robot_pos[1]
                dist_to_base = math.sqrt(dx * dx + dy * dy)

                if dist_to_base < 0.25:
                    self.carrying_state[aid]  = False
                    rewards[aid]             += 20.0
                    self.total_deposits      += 1
                    print(f"[DEPOSIT] {aid} | Total: {self.total_deposits}")

                    # Fix 1: pre-initialize hotspot dist at deposit moment
                    # so pheromone gradient is active from step 1 post-deposit
                    mem = self.pheromone_memory[aid]
                    if mem["hotspot"] is not None and mem["strength"] >= PHEROMONE_MIN:
                        hx, hy = mem["hotspot"]
                        self.prev_hotspot_dists[aid] = math.sqrt(
                            (hx - robot_pos[0])**2 + (hy - robot_pos[1])**2)

                # Approach-base shaping
                if self.prev_base_dists[aid] is not None:
                    rewards[aid] += (self.prev_base_dists[aid] - dist_to_base) * 8.0
                self.prev_base_dists[aid] = dist_to_base

                # Face-toward-base reward
                if dist_to_base > 0.001:
                    robot_rot = self.robot_nodes[i].getOrientation()
                    fwd       = [robot_rot[0], robot_rot[3], robot_rot[6]]
                    b_norm    = [dx / dist_to_base, dy / dist_to_base]
                    b_dot     = max(min(fwd[0]*b_norm[0] + fwd[1]*b_norm[1], 1.0), -1.0)
                    rewards[aid] += b_dot * 0.5

                # Only reset hotspot dist if still carrying (not just deposited)
                if self.carrying_state[aid]:
                    self.prev_hotspot_dists[aid] = None

            # Time penalty
            rewards[aid] -= 0.005

        # Near-base separation penalty (base-zone only, same as centralized v16)
        robot_positions = [self.robot_nodes[k].getPosition() for k in range(self.num_robots)]
        for i, aid_i in enumerate(self.agents):
            for j, aid_j in enumerate(self.agents):
                if j <= i:
                    continue
                di = math.sqrt(robot_positions[i][0]**2 + robot_positions[i][1]**2)
                dj = math.sqrt(robot_positions[j][0]**2 + robot_positions[j][1]**2)
                if di < 1.5 and dj < 1.5:
                    sep = math.sqrt(
                        (robot_positions[i][0] - robot_positions[j][0])**2 +
                        (robot_positions[i][1] - robot_positions[j][1])**2
                    )
                    if sep < 1.0:
                        penalty = (1.0 - sep) * 0.5
                        rewards[aid_i] -= penalty
                        rewards[aid_j] -= penalty

        return rewards

    # =========================================================================
    # Pheromone system  (decentralized, peer-to-peer)
    # =========================================================================

    def _pickup_strength(self, pickup_pos):
        """
        Encode cluster density at pickup location.
        Denser clusters produce stronger pheromones, attracting more robots.
            nearby=0  → 0.20  (isolated tag)
            nearby=3  → 0.68
            nearby=5+ → 1.00  (dense cluster)
        """
        nearby = sum(
            1 for tag_node in self.tag_nodes
            if tag_node.getPosition()[2] >= 0 and
               math.sqrt((tag_node.getPosition()[0] - pickup_pos[0])**2 +
                         (tag_node.getPosition()[1] - pickup_pos[1])**2) <= DENSITY_RADIUS
        )
        return 0.2 + 0.8 * min(nearby / DENSITY_MAX, 1.0)

    def _decay_pheromone(self):
        """Exponential decay per step; delete waypoint when below threshold."""
        for aid in self.agents:
            mem = self.pheromone_memory[aid]
            if mem["hotspot"] is not None:
                mem["strength"] *= PHEROMONE_DECAY
                mem["ttl"]      -= 1
                if mem["strength"] < PHEROMONE_MIN or mem["ttl"] <= 0:
                    self.pheromone_memory[aid] = {
                        "hotspot": None, "strength": 0.0, "ttl": 0, "from": None}

    def _share_pheromone(self):
        """
        Simulate peer-to-peer pheromone sharing within COMM_RANGE (2 m).
        Carrying robots broadcast their fresh pickup hotspot.
        Non-carrying robots with active memory relay it at current (decayed) strength.
        Receiving robot accepts signal only if it is stronger than its current memory.
        Multi-hop: A→B→C even if A and C are >2 m apart.
        """
        robot_positions = [self.robot_nodes[k].getPosition() for k in range(self.num_robots)]

        for i, aid_sender in enumerate(self.agents):
            mem = self.pheromone_memory[aid_sender]
            if mem["hotspot"] is None:
                continue

            # Carrying: broadcast at current pickup strength (fresh signal)
            # Not carrying: relay at current decayed strength
            broadcast_strength = mem["strength"]
            if broadcast_strength < PHEROMONE_MIN:
                continue

            for j, aid_recv in enumerate(self.agents):
                if aid_recv == aid_sender:
                    continue
                dx   = robot_positions[i][0] - robot_positions[j][0]
                dy   = robot_positions[i][1] - robot_positions[j][1]
                dist = math.sqrt(dx * dx + dy * dy)

                if dist <= COMM_RANGE:
                    # Accept only if incoming signal is stronger (fresher or denser)
                    if broadcast_strength > self.pheromone_memory[aid_recv]["strength"]:
                        self.pheromone_memory[aid_recv] = {
                            "hotspot":  mem["hotspot"],
                            "strength": broadcast_strength,
                            "ttl":      mem["ttl"],
                            "from":     aid_sender
                        }

    # =========================================================================
    # Hard-coded overrides  (P1–P4, identical to centralized v17)
    # =========================================================================

    def _apply_overrides(self, aid, action):
        i         = self._idx(aid)
        robot_pos = self.robot_nodes[i].getPosition()
        robot_rot = self.robot_nodes[i].getOrientation()
        fwd       = [robot_rot[0], robot_rot[3], robot_rot[6]]
        prox      = self.robot_states[aid]
        base_pos  = self.base_node.getPosition()
        wall_dist = 2.5 - max(abs(robot_pos[0]), abs(robot_pos[1]))

        # P1: Wall / collision escape
        if wall_dist < 0.35 or max(prox) > 0.55:
            return self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=4.0)

        # P2: Return to base when carrying
        if self.carrying_state[aid]:
            return self._steer_to(robot_pos, fwd, base_pos, gain=2.5)

        # P3: Base avoidance when not carrying
        dist_to_base = math.sqrt(robot_pos[0]**2 + robot_pos[1]**2)
        if dist_to_base < 0.5:
            away   = [robot_pos[0] - base_pos[0], robot_pos[1] - base_pos[1]]
            target = [robot_pos[0] + away[0] * 2.0, robot_pos[1] + away[1] * 2.0]
            return self._steer_to(robot_pos, fwd, target, gain=3.0)

        # P4: Tag-seek within 1.0 m and FOV
        if not self.carrying_state[aid]:
            best_pos  = None
            best_dist = float('inf')
            for tag_node in self.tag_nodes:
                tag_pos = tag_node.getPosition()
                if tag_pos[2] < 0:
                    continue
                tdx = tag_pos[0] - robot_pos[0]
                tdy = tag_pos[1] - robot_pos[1]
                td  = math.sqrt(tdx * tdx + tdy * tdy)
                if td < TAG_SEEK_RANGE and td > 0.001 and td < best_dist:
                    dot   = fwd[0]*(tdx/td) + fwd[1]*(tdy/td)
                    angle = math.acos(max(min(dot, 1.0), -1.0))
                    if angle < FOV_HALF_ANGLE:
                        best_dist = td
                        best_pos  = tag_pos
            if best_pos is not None:
                return self._steer_to(robot_pos, fwd, best_pos, gain=3.0)

        # PPO controls global exploration and pheromone-following
        return list(action)

    def _steer_to(self, robot_pos, fwd, target, gain=2.5):
        """Proportional steering controller — returns [left, right] in [−1, 1]."""
        dx   = target[0] - robot_pos[0]
        dy   = target[1] - robot_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
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

    # =========================================================================
    # Environment reset helpers  (same curriculum as centralized)
    # =========================================================================

    def _respawn_robots(self):
        positions = [[-0.5, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0]]
        for i in range(self.num_robots):
            self.robot_nodes[i].getField("translation").setSFVec3f(positions[i])
            rand_yaw = random.uniform(0, 2 * math.pi)
            self.robot_nodes[i].getField("rotation").setSFRotation([0, 0, 1, rand_yaw])
            self.robot_nodes[i].resetPhysics()

    def _respawn_all_tags(self):
        centers = self._generate_cluster_centers()
        tags_per = self.num_tags // len(centers)
        tag_idx  = 0

        for cx, cy in centers:
            count = tags_per if tag_idx + tags_per <= self.num_tags else self.num_tags - tag_idx
            for _ in range(count):
                if tag_idx >= self.num_tags:
                    break
                angle = random.uniform(0, 2 * math.pi)
                r     = random.uniform(0, 0.3)
                tx    = max(-2.3, min(2.3, cx + math.cos(angle) * r))
                ty    = max(-2.3, min(2.3, cy + math.sin(angle) * r))
                self.tag_nodes[tag_idx].getField("translation").setSFVec3f([tx, ty, 0.01375])
                tag_idx += 1

        while tag_idx < self.num_tags:
            cx, cy = random.choice(centers)
            angle  = random.uniform(0, 2 * math.pi)
            r      = random.uniform(0, 0.3)
            tx     = max(-2.3, min(2.3, cx + math.cos(angle) * r))
            ty     = max(-2.3, min(2.3, cy + math.sin(angle) * r))
            self.tag_nodes[tag_idx].getField("translation").setSFVec3f([tx, ty, 0.01375])
            tag_idx += 1

    def _generate_cluster_centers(self):
        """Same curriculum as centralized: gradually expand cluster range."""
        if self.total_episodes < 100:
            max_dist, n_clusters = 1.0, 2
        elif self.total_episodes < 300:
            max_dist, n_clusters = 1.8, random.randint(2, 3)
        else:
            max_dist, n_clusters = 2.3, random.randint(2, 4)

        centers = []
        for _ in range(n_clusters):
            for _ in range(50):
                angle = random.uniform(0, 2 * math.pi)
                dist  = random.uniform(0.8, max_dist)
                cx    = math.cos(angle) * dist
                cy    = math.sin(angle) * dist
                if all(math.sqrt((cx-ox)**2 + (cy-oy)**2) > 0.8 for ox, oy in centers):
                    centers.append((cx, cy))
                    break
        return centers if centers else [(1.0, 0.0)]

    def _idx(self, aid):
        return int(aid.split("_")[1]) - 1

    def _collect_robot_states(self):
        for i, aid in enumerate(self.agents):
            if self.receivers[i].getQueueLength() > 0:
                msg = self.receivers[i].getString()
                self.receivers[i].nextPacket()
                try:
                    vals = [float(x) for x in msg.split(',')]
                    self.robot_states[aid] = vals[:8]
                except (ValueError, IndexError):
                    pass


# =============================================================================
# TRAINING ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name',        type=str,   default='decentralized_v1')
    parser.add_argument('--total_timesteps', type=int,   default=7_000_000)
    parser.add_argument('--lr',              type=float, default=3e-4)
    parser.add_argument('--ent_coef',        type=float, default=0.10)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[TRAINING] Device: {device}")
    print(f"[TRAINING] Run: {args.run_name} | LR: {args.lr} | Ent: {args.ent_coef} "
          f"| Steps: {args.total_timesteps}")

    # Ray must run in local mode — Webots allows only ONE supervisor instance.
    # num_rollout_workers=0  → main process collects rollouts (no worker forks).
    # disable_env_checking=True → prevents RLlib from creating a second env
    #                             instance for space validation.
    ray.init(
        ignore_reinit_error=True,
        num_gpus=1 if device == 'cuda' else 0,
        num_cpus=2,
    )

    obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    config = (
        PPOConfig()
        .environment(
            DecentralizedForagingEnv,
            disable_env_checking=True,   # prevents second Supervisor() instantiation
        )
        .rollouts(num_rollout_workers=0)  # main process only — required for Webots
        .multi_agent(
            policies={
                f"robot_{i+1}": (None, obs_space, act_space, {})
                for i in range(4)
            },
            policy_mapping_fn=lambda agent_id, **kwargs: agent_id,
            policies_to_train=[f"robot_{i+1}" for i in range(4)],
        )
        .training(
            gamma=0.99,
            lambda_=0.95,
            clip_param=0.2,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            entropy_coeff=args.ent_coef,
            lr=args.lr,
            train_batch_size=4096,
            sgd_minibatch_size=1024,
            num_sgd_iter=10,
            model={
                "fcnet_hiddens":    [256, 256],
                "fcnet_activation": "tanh",
            },
        )
        .framework("torch")
        .resources(num_gpus=1 if device == 'cuda' else 0)
    )

    algo = config.build()

    total_steps = 0
    iteration   = 0
    print(f"[TRAINING] Starting: {args.run_name}")

    while total_steps < args.total_timesteps:
        result      = algo.train()
        total_steps = result.get("timesteps_total", 0)
        mean_reward = result.get("episode_reward_mean", 0.0)
        iteration  += 1

        print(f"[ITER {iteration:4d}] Steps: {total_steps:>8}/{args.total_timesteps} "
              f"| Mean Reward: {mean_reward:+.2f}")

        # Checkpoint every 10 iterations (~40k steps)
        if iteration % 10 == 0:
            ckpt = algo.save(f"./logs/{args.run_name}/checkpoint_{iteration:04d}")
            print(f"[CHECKPOINT] {ckpt}")

    # Save final policy for each robot (for standalone decentralized execution)
    import os
    os.makedirs(f"./models/{args.run_name}", exist_ok=True)
    for i in range(4):
        pid    = f"robot_{i+1}"
        policy = algo.get_policy(pid)
        torch.save(
            policy.get_weights(),
            f"./models/{args.run_name}/{pid}.pt"
        )
        print(f"[SAVED] Policy: {pid} → ./models/{args.run_name}/{pid}.pt")

    algo.save(f"./logs/{args.run_name}/final")
    ray.shutdown()
    print(f"[COMPLETE] Run: {args.run_name}")
