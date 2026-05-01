import math
import random
import numpy as np
import argparse
import torch
import torch.nn as nn
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from controller import Supervisor

# =============================================================================
# HYBRID OPTION-A DECENTRALIZED SUPERVISOR  (CoRL 2026)
#
# Architecture — Centralized Training, Decentralized Execution (CTDE):
#   TRAINING:  supervisor assembles 18D obs from robot-reported components
#              (GPS/IMU computed by robot) + tag obs (supervisor camera sim).
#              Runs shared PPO policy via SB3 with parameter sharing (n_envs=4).
#   EXECUTION: each robot runs its own policy locally, computes full 18D obs
#              autonomously (GPS + IMU + phero_receiver). Supervisor only
#              needed for tag pickup confirmation (camera simulation).
#
# Pheromone: fully peer-to-peer via Webots Emitter/Receiver on each robot
#   (channel 10, emitter range=2.0 m enforced by Webots physics).
#   Supervisor has NO pheromone grid — grid-free by design.
#
# Message layout:
#   Robot → Supervisor (17 floats):
#     [0:8]  prox sensors (÷4096)
#     [8]    carrying (0/1)          — robot tracks own state
#     [9]    base_dist_norm (GPS)    — dist to origin / 3.5
#     [10]   base_angle_norm (GPS+IMU) — signed angle / π
#     [11]   phero_known             — 1 if active hotspot
#     [12]   phero_dist_norm         — dist to hotspot / 3.5
#     [13]   phero_angle_norm        — signed angle / π
#     [14]   phero_strength
#     [15]   gps_x   (raw — for P1-P4 steering)
#     [16]   gps_y
#
#   Supervisor → Robot (3 floats):
#     [0]    left_motor  ([-1, 1])
#     [1]    right_motor ([-1, 1])
#     [2]    pickup_signal:
#              > 0  → pickup event, value = pheromone strength (0.2–1.0)
#              < 0  → deposit confirmed
#                0  → normal step
#
# Assembled 18D obs:
#   [0:8]  prox          ← robot message [0:8]
#   [8]    tag_visible   ← supervisor (camera sim)
#   [9]    tag_dist_norm ← supervisor
#   [10]   tag_angle_norm← supervisor
#   [11]   carrying      ← robot message [8]
#   [12]   base_dist_norm← robot message [9]
#   [13]   base_angle_norm← robot message [10]
#   [14]   phero_known   ← robot message [11]
#   [15]   phero_dist_norm← robot message [12]
#   [16]   phero_angle_norm← robot message [13]
#   [17]   phero_strength ← robot message [14]
# =============================================================================

# ---------- Constants ----------
TAG_SEEK_RANGE  = 1.0
FOV_HALF_ANGLE  = 1.2
DENSITY_RADIUS  = 0.5
DENSITY_MAX     = 5


class DecentralizedForagingEnv(VecEnv):
    """
    SB3 VecEnv with n_envs=4 for parameter-sharing multi-robot training.
    Each "env" corresponds to one robot. One Webots step advances all 4 simultaneously.
    The shared policy sees each robot's 18D obs independently — true parameter sharing.
    """

    def __init__(self):
        obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)
        act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        super().__init__(num_envs=4, observation_space=obs_space, action_space=act_space)

        self.supervisor = Supervisor()
        self.timestep   = int(self.supervisor.getBasicTimeStep())

        self.num_robots        = 4
        self.num_tags          = 70
        self.steps_per_episode = 4096

        self.agents = [f"robot_{i+1}" for i in range(self.num_robots)]

        # ---------- Webots nodes ----------
        self.robot_nodes = [
            self.supervisor.getFromDef(f"ROBOT{i+1}") for i in range(self.num_robots)
        ]
        self.tag_nodes = [
            self.supervisor.getFromDef(f"APRILTAG_{i+1}") for i in range(self.num_tags)
        ]

        # Supervisor ↔ robot communication (channels 1-4)
        self.emitters  = []
        self.receivers = []
        for i in range(self.num_robots):
            self.emitters.append(self.supervisor.getDevice(f"emitter{i+1}"))
            recv = self.supervisor.getDevice(f"receiver{i+1}")
            recv.enable(self.timestep)
            self.receivers.append(recv)

        # ---------- Per-robot runtime state ----------
        self.robot_states       = {aid: self._empty_state() for aid in self.agents}
        self.carrying_state     = {aid: False for aid in self.agents}
        self.prev_tag_dists     = {aid: None  for aid in self.agents}
        self.prev_base_dists    = {aid: None  for aid in self.agents}
        self.prev_hotspot_dists = {aid: None  for aid in self.agents}
        self.pending_signals    = {aid: 0.0   for aid in self.agents}

        self.episode_step   = 0
        self.total_episodes = 0
        self.total_pickups  = 0
        self.total_deposits = 0

        # Actions stored between step_async and step_wait
        self._actions = np.zeros((self.num_robots, 2), dtype=np.float32)

    def _empty_state(self):
        return {
            "prox":             [0.0] * 8,
            "carrying":         False,
            "base_dist_norm":   0.0,
            "base_angle_norm":  0.0,
            "phero_known":      0.0,
            "phero_dist_norm":  0.0,
            "phero_angle_norm": 0.0,
            "phero_strength":   0.0,
            "gps_x":            0.0,
            "gps_y":            0.0,
        }

    # =========================================================================
    # SB3 VecEnv interface
    # =========================================================================

    def reset(self):
        self.total_episodes += 1
        self.episode_step    = 0

        self.carrying_state     = {aid: False for aid in self.agents}
        self.prev_tag_dists     = {aid: None  for aid in self.agents}
        self.prev_base_dists    = {aid: None  for aid in self.agents}
        self.prev_hotspot_dists = {aid: None  for aid in self.agents}
        self.pending_signals    = {aid: 0.0   for aid in self.agents}
        self.robot_states       = {aid: self._empty_state() for aid in self.agents}

        self._respawn_robots()
        self._respawn_all_tags()
        self.supervisor.step(self.timestep)
        self._collect_robot_states()

        return self._get_obs_array()   # (4, 18)

    def step_async(self, actions):
        """Store actions (4, 2) from SB3 for use in step_wait."""
        self._actions = actions

    def step_wait(self):
        self.episode_step += 1

        # Send motor commands + pending pickup/deposit signals to each robot
        for i, aid in enumerate(self.agents):
            final_action  = self._apply_overrides(aid, self._actions[i])
            pickup_signal = self.pending_signals[aid]
            msg = f"{final_action[0]},{final_action[1]},{pickup_signal}".encode('utf-8')
            self.emitters[i].send(msg)
            self.pending_signals[aid] = 0.0

        if self.supervisor.step(self.timestep) == -1:
            exit()

        self._collect_robot_states()

        rewards = self._compute_rewards()         # np.ndarray (4,)
        obs     = self._get_obs_array()           # (4, 18)

        done_flag = self.episode_step >= self.steps_per_episode
        dones     = np.array([done_flag] * self.num_robots, dtype=bool)
        infos     = [{} for _ in range(self.num_robots)]

        if done_flag:
            for i in range(self.num_robots):
                infos[i]["terminal_observation"] = obs[i]
            obs = self.reset()

        return obs, rewards, dones, infos

    def close(self):
        pass

    def seed(self, _seed=None):
        return [None] * self.num_robots

    def get_attr(self, attr_name, _indices=None):
        return [getattr(self, attr_name)] * self.num_robots

    def set_attr(self, attr_name, value, _indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *method_args, _indices=None, **method_kwargs):
        return [getattr(self, method_name)(*method_args, **method_kwargs)] * self.num_robots

    def env_is_wrapped(self, _wrapper_class, _indices=None):
        return [False] * self.num_robots

    def _get_obs_array(self):
        obs_dict = self._get_observations()
        return np.array([obs_dict[aid] for aid in self.agents], dtype=np.float32)

    # =========================================================================
    # Observations  (18D — supervisor adds tag obs, rest from robot message)
    # =========================================================================

    def _get_observations(self):
        obs_dict = {}

        for i, aid in enumerate(self.agents):
            rs        = self.robot_states[aid]
            robot_pos = [rs["gps_x"], rs["gps_y"], 0.0]
            robot_rot = self.robot_nodes[i].getOrientation()
            fwd       = [robot_rot[0], robot_rot[3], robot_rot[6]]

            # ── Tag sensing (supervisor camera simulation) ─────────────
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

            # ── Assemble 18D obs ──────────────────────────────────────
            robot_obs = list(rs["prox"]) + [
                tag_visible,                       # [8]
                tag_dist / TAG_SEEK_RANGE,         # [9]
                tag_angle / math.pi,               # [10]
                1.0 if rs["carrying"] else 0.0,    # [11]
                rs["base_dist_norm"],               # [12]
                rs["base_angle_norm"],              # [13]
                rs["phero_known"],                  # [14]
                rs["phero_dist_norm"],              # [15]
                rs["phero_angle_norm"],             # [16]  already ÷ π (robot-side)
                rs["phero_strength"],               # [17]
            ]
            obs_dict[aid] = np.array(robot_obs, dtype=np.float32)

        return obs_dict

    # =========================================================================
    # Rewards
    # =========================================================================

    def _compute_rewards(self):
        reward_arr = np.zeros(self.num_robots, dtype=np.float32)

        for i, aid in enumerate(self.agents):
            rs        = self.robot_states[aid]
            gps_x     = rs["gps_x"]
            gps_y     = rs["gps_y"]
            prox      = rs["prox"]
            wall_dist = 2.5 - max(abs(gps_x), abs(gps_y))

            # Proximity penalty
            max_prox = max(prox)
            if max_prox > 0.1:
                reward_arr[i] -= max_prox * 0.5

            # Wall penalty
            if wall_dist < 0.35:
                reward_arr[i] -= (0.35 - wall_dist) * 0.5

            if not self.carrying_state[aid]:
                # ── EXPLORING ──────────────────────────────────────────
                picked_up = False
                for tag_node in self.tag_nodes:
                    tag_pos = tag_node.getPosition()
                    if tag_pos[2] < 0:
                        continue
                    dx   = tag_pos[0] - gps_x
                    dy   = tag_pos[1] - gps_y
                    dist = math.sqrt(dx * dx + dy * dy)
                    if dist < 0.15:
                        self.carrying_state[aid] = True
                        tag_node.getField("translation").setSFVec3f([0, 0, -10])
                        reward_arr[i]           += 5.0
                        self.total_pickups       += 1
                        picked_up                = True
                        print(f"[PICKUP] {aid} | Total: {self.total_pickups}")

                        strength = self._pickup_strength([gps_x, gps_y])
                        self.pending_signals[aid] = strength

                        self.prev_tag_dists[aid] = None
                        break

                if not picked_up:
                    # Tag approach shaping
                    curr_min = float('inf')
                    for tag_node in self.tag_nodes:
                        tag_pos = tag_node.getPosition()
                        if tag_pos[2] < 0:
                            continue
                        dx   = tag_pos[0] - gps_x
                        dy   = tag_pos[1] - gps_y
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist < TAG_SEEK_RANGE and dist < curr_min:
                            curr_min = dist
                    if self.prev_tag_dists[aid] is not None and curr_min < float('inf'):
                        reward_arr[i] += (self.prev_tag_dists[aid] - curr_min) * 8.0
                    self.prev_tag_dists[aid] = curr_min if curr_min < float('inf') else None

                    # Tag-visible reward
                    robot_rot = self.robot_nodes[i].getOrientation()
                    fwd       = [robot_rot[0], robot_rot[3], robot_rot[6]]
                    for tag_node in self.tag_nodes:
                        tag_pos = tag_node.getPosition()
                        if tag_pos[2] < 0:
                            continue
                        tdx = tag_pos[0] - gps_x
                        tdy = tag_pos[1] - gps_y
                        td  = math.sqrt(tdx * tdx + tdy * tdy)
                        if td < TAG_SEEK_RANGE and td > 0.001:
                            dot = fwd[0]*(tdx/td) + fwd[1]*(tdy/td)
                            if math.acos(max(min(dot, 1.0), -1.0)) < FOV_HALF_ANGLE:
                                reward_arr[i] += 0.2
                                break

                    # Zone reward
                    dist_from_base = math.sqrt(gps_x**2 + gps_y**2)
                    if 0.8 < dist_from_base < 2.4:
                        reward_arr[i] += 0.10

                    # Near-base penalty
                    if dist_from_base < 0.8:
                        reward_arr[i] -= (0.8 - dist_from_base) * 2.0

                    # Forward motion bias
                    if wall_dist >= 0.35:
                        avg_speed = (self._actions[i][0] + self._actions[i][1]) / 2.0
                        if avg_speed > 0:
                            reward_arr[i] += avg_speed * 0.01

                    # Pheromone approach reward
                    if rs["phero_known"] > 0.5:
                        curr_hd = rs["phero_dist_norm"] * 3.5
                        if self.prev_hotspot_dists[aid] is not None:
                            reward_arr[i] += (self.prev_hotspot_dists[aid] - curr_hd) * 3.0
                        self.prev_hotspot_dists[aid] = curr_hd

                        # Facing toward pheromone reward
                        h_dot = math.cos(rs["phero_angle_norm"] * math.pi)
                        reward_arr[i] += h_dot * 0.3
                    else:
                        # No pheromone: reward outward exploration
                        dist_from_base = math.sqrt(gps_x**2 + gps_y**2)
                        if dist_from_base > 0.8:
                            reward_arr[i] += min(dist_from_base / 2.3, 1.0) * 0.15
                        self.prev_hotspot_dists[aid] = None

            else:
                # ── CARRYING: RTB ──────────────────────────────────────
                dist_to_base = rs["base_dist_norm"] * 3.5

                if dist_to_base < 0.25:
                    self.carrying_state[aid] = False
                    reward_arr[i]           += 20.0
                    self.total_deposits      += 1
                    print(f"[DEPOSIT] {aid} | Total: {self.total_deposits}")

                    self.pending_signals[aid] = -1.0

                    # Fix 1: pre-seed hotspot dist so approach reward fires immediately
                    if rs["phero_known"] > 0.5:
                        self.prev_hotspot_dists[aid] = rs["phero_dist_norm"] * 3.5

                # RTB approach shaping
                if self.prev_base_dists[aid] is not None:
                    reward_arr[i] += (self.prev_base_dists[aid] - dist_to_base) * 8.0
                self.prev_base_dists[aid] = dist_to_base

                # Face-toward-base reward
                b_dot = math.cos(rs["base_angle_norm"] * math.pi)
                reward_arr[i] += b_dot * 0.5

                # Only reset hotspot dist if still carrying (not just deposited)
                if self.carrying_state[aid]:
                    self.prev_hotspot_dists[aid] = None

            # Time penalty
            reward_arr[i] -= 0.005

        # Near-base separation penalty (base-zone only)
        robot_gpss = [(self.robot_states[aid]["gps_x"], self.robot_states[aid]["gps_y"])
                      for aid in self.agents]
        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                di = math.sqrt(robot_gpss[i][0]**2 + robot_gpss[i][1]**2)
                dj = math.sqrt(robot_gpss[j][0]**2 + robot_gpss[j][1]**2)
                if di < 1.5 and dj < 1.5:
                    sep = math.sqrt(
                        (robot_gpss[i][0] - robot_gpss[j][0])**2 +
                        (robot_gpss[i][1] - robot_gpss[j][1])**2
                    )
                    if sep < 1.0:
                        penalty = (1.0 - sep) * 0.5
                        reward_arr[i] -= penalty
                        reward_arr[j] -= penalty

        return reward_arr

    # =========================================================================
    # Pheromone density  (supervisor computes; robots cannot count tag nodes)
    # =========================================================================

    def _pickup_strength(self, pickup_pos):
        nearby = sum(
            1 for tag_node in self.tag_nodes
            if tag_node.getPosition()[2] >= 0 and
               math.sqrt((tag_node.getPosition()[0] - pickup_pos[0])**2 +
                         (tag_node.getPosition()[1] - pickup_pos[1])**2) <= DENSITY_RADIUS
        )
        return 0.2 + 0.8 * min(nearby / DENSITY_MAX, 1.0)

    # =========================================================================
    # Hard-coded overrides  (P1-P4 — uses robot's GPS + supervisor orientation)
    # =========================================================================

    def _apply_overrides(self, aid, action):
        i         = self._idx(aid)
        rs        = self.robot_states[aid]
        robot_pos = [rs["gps_x"], rs["gps_y"], 0.0]
        robot_rot = self.robot_nodes[i].getOrientation()
        fwd       = [robot_rot[0], robot_rot[3], robot_rot[6]]
        prox      = rs["prox"]
        wall_dist = 2.5 - max(abs(robot_pos[0]), abs(robot_pos[1]))

        # P1: wall / collision escape
        if wall_dist < 0.35 or max(prox) > 0.55:
            return self._steer_to(robot_pos, fwd, [0.0, 0.0, 0.0], gain=4.0)

        # P2: return to base when carrying
        if self.carrying_state[aid]:
            return self._steer_to(robot_pos, fwd, [0.0, 0.0, 0.0], gain=2.5)

        # P3: base avoidance when not carrying
        dist_to_base = math.sqrt(robot_pos[0]**2 + robot_pos[1]**2)
        if dist_to_base < 0.5:
            target = [robot_pos[0] * 3.0, robot_pos[1] * 3.0, 0.0]
            return self._steer_to(robot_pos, fwd, target, gain=3.0)

        # P4: tag-seek within TAG_SEEK_RANGE and FOV
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

        return list(action)

    def _steer_to(self, robot_pos, fwd, target, gain=2.5):
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
    # Episode reset helpers  (same curriculum as centralized)
    # =========================================================================

    def _respawn_robots(self):
        positions = [[-0.5, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0]]
        for i in range(self.num_robots):
            self.robot_nodes[i].getField("translation").setSFVec3f(positions[i])
            rand_yaw = random.uniform(0, 2 * math.pi)
            self.robot_nodes[i].getField("rotation").setSFRotation([0, 0, 1, rand_yaw])
            self.robot_nodes[i].resetPhysics()

    def _respawn_all_tags(self):
        centers  = self._generate_cluster_centers()
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
        """Parse 17-value message from each robot."""
        for i, aid in enumerate(self.agents):
            if self.receivers[i].getQueueLength() > 0:
                msg = self.receivers[i].getString()
                self.receivers[i].nextPacket()
                try:
                    vals = [float(x) for x in msg.split(',')]
                    if len(vals) >= 17:
                        rs = self.robot_states[aid]
                        rs["prox"]             = vals[0:8]
                        rs["carrying"]         = vals[8] > 0.5
                        rs["base_dist_norm"]   = vals[9]
                        rs["base_angle_norm"]  = vals[10]
                        rs["phero_known"]      = vals[11]
                        rs["phero_dist_norm"]  = vals[12]
                        rs["phero_angle_norm"] = vals[13]
                        rs["phero_strength"]   = vals[14]
                        rs["gps_x"]            = vals[15]
                        rs["gps_y"]            = vals[16]
                except (ValueError, IndexError):
                    pass


# =============================================================================
# TRAINING ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name',        type=str,   default='decentralized_optA_v1')
    parser.add_argument('--total_timesteps', type=int,   default=7_000_000)
    parser.add_argument('--lr',              type=float, default=3e-4)
    parser.add_argument('--ent_coef',        type=float, default=0.10)
    parser.add_argument('--resume',          type=str,   default=None,
                        help='Path to .zip model to resume from')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[TRAINING] Device: {device}")
    print(f"[TRAINING] Run: {args.run_name} | LR: {args.lr} | Ent: {args.ent_coef} "
          f"| Steps: {args.total_timesteps}")
    print(f"[TRAINING] Parameter sharing: 1 policy × 4 robots (n_envs=4)")
    print(f"[TRAINING] Option A — robot computes GPS/IMU/phero obs autonomously")

    env = DecentralizedForagingEnv()

    if args.resume:
        print(f"[TRAINING] Resuming from: {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)
        model.learning_rate = args.lr
        model.ent_coef      = args.ent_coef
        remaining = args.total_timesteps - model.num_timesteps
    else:
        model = PPO(
            "MlpPolicy",
            env,
            # Effective batch = n_steps × n_envs = 1024 × 4 = 4096
            n_steps=1024,
            batch_size=1024,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            ent_coef=args.ent_coef,
            learning_rate=args.lr,
            policy_kwargs=dict(
                net_arch=[256, 256],
                activation_fn=nn.Tanh,
            ),
            device=device,
            verbose=1,
        )
        remaining = args.total_timesteps

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,   # every 50000 VecEnv steps = 200000 total env steps
        save_path=f'./logs/{args.run_name}/',
        name_prefix=args.run_name,
    )

    print(f"[TRAINING] Starting: {args.run_name}")
    model.learn(
        total_timesteps=remaining,
        callback=checkpoint_callback,
        reset_num_timesteps=args.resume is None,
    )

    model.save(args.run_name)
    print(f"[COMPLETE] Model saved as {args.run_name}.zip")
