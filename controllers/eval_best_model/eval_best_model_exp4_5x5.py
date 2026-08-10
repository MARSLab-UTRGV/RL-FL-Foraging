import math
import random
import numpy as np
import sys
import argparse
from controller import Supervisor
from stable_baselines3 import PPO
import gymnasium as gym

MAX_DIST = 2.5 * math.sqrt(2)

# =============================================================================
# EXPERIMENT 4 — Central Server Disconnect Robustness (5×5m arena)
#
# Identical to eval_best_model_5x5.py except during the disconnect window
# [disconnect_start_min, disconnect_start_min + disconnect_duration_min):
#   - PPO inference is skipped
#   - All hard-coded overrides (P1, BASE_ESC, P2) are suppressed
#   - Robots receive [0.0, 0.0] — full stop
#   - Pheromone decay and event processing still run every step
#
# Scenarios:
#   1-min  disconnect: --disconnect-start-min 5.0 --disconnect-duration-min 1.0
#   2-min  disconnect: --disconnect-start-min 4.0 --disconnect-duration-min 2.0
#   3-min  disconnect: --disconnect-start-min 4.0 --disconnect-duration-min 3.0
#   5-min  disconnect: --disconnect-start-min 3.0 --disconnect-duration-min 5.0
#   full   disconnect: --disconnect-start-min 0.0 --disconnect-duration-min 10.0
# =============================================================================

class EpuckForagingSupervisor(Supervisor, gym.Env):
    def __init__(self):
        self.num_robots            = 4
        self.num_tags              = 64
        self.obs_per_robot         = 18
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

        self.robot_nodes = []
        for i in range(self.num_robots):
            node = self.getFromDef(f"ROBOT{i+1}")
            if node is None:
                print(f"[ERROR] Could not find ROBOT{i+1}!")
                exit(1)
            self.robot_nodes.append(node)

        self.tag_nodes = [self.getFromDef(f"APRILTAG_{i+1}") for i in range(self.num_tags)]
        self.base_node = self.getFromDef("BASE_STATION")

        self.emitters  = []
        self.receivers = []
        for i in range(self.num_robots):
            self.emitters.append(self.getDevice(f"emitter{i+1}"))
            self.receivers.append(self.getDevice(f"receiver{i+1}"))
            self.receivers[i].enable(self.timestep)

        self.robot_states    = [None]  * self.num_robots
        self.carrying_state  = [False] * self.num_robots
        self.prev_base_dists = [None]  * self.num_robots

        self.pheromone_list = []

        self.RATE_OF_LAYING_PHEROMONE = 3.0
        self.RATE_OF_SITE_FIDELITY    = 1.376
        self.RATE_OF_PHEROMONE_DECAY  = 0.05
        self.PHEROMONE_MIN            = 0.001

        self.carried_from      = [None]  * self.num_robots
        self.resource_density  = [0]     * self.num_robots
        self.site_fidelity_pos = [None]  * self.num_robots
        self.nest_target       = [None]  * self.num_robots
        self.prev_site_dists   = [None]  * self.num_robots
        self.prev_phero_dists  = [None]  * self.num_robots

        self.steps_without_pickup = [0]     * self.num_robots
        self.give_up_timer        = [0]     * self.num_robots
        self.PROB_RETURN_TO_NEST  = 0.0189
        self.GIVE_UP_CHECK_STEPS  = 78
        self.SEARCH_DURATION_NORM = 4000.0
        self.gave_up              = [False] * self.num_robots

        # Disconnect flag — set each step by the main loop
        self.disconnected = False

        self.step_count     = 0
        self.total_pickups  = 0
        self.total_deposits = 0
        self.last_action    = np.zeros(self.action_space_dim, dtype=np.float32)

    # =========================================================================
    # CPFA HELPERS
    # =========================================================================

    def _poisson_cdf(self, n, rate):
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
        if self.site_fidelity_pos[i] is not None and not self.gave_up[i]:
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

        if self.disconnected:
            # Supervisor policy is offline — no PPO, no coordination.
            # Safety-only: wall escape and base-stuck escape still run so robots
            # don't get permanently pinned. Pre-disconnect velocity is replayed
            # for robots that are neither near a wall nor near the base.
            safety_action = self._apply_safety_overrides(self.last_action)
            for i in range(self.num_robots):
                robot_action = safety_action[i*2 : (i+1)*2]
                msg = f"{robot_action[0]},{robot_action[1]}".encode('utf-8')
                self.emitters[i].send(msg)
        else:
            final_action = self._apply_overrides(action)
            self.last_action = final_action
            for i in range(self.num_robots):
                robot_action = final_action[i*2 : (i+1)*2]
                msg = f"{robot_action[0]},{robot_action[1]}".encode('utf-8')
                self.emitters[i].send(msg)

        if Supervisor.step(self, self.timestep) == -1:
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

        # Pheromone decay runs regardless of disconnect (time still passes)
        dt    = self.timestep / 1000.0
        decay = math.exp(-self.RATE_OF_PHEROMONE_DECAY * dt)
        for p in self.pheromone_list:
            p['weight'] *= decay
        self.pheromone_list = [p for p in self.pheromone_list
                               if p['weight'] > self.PHEROMONE_MIN]

        # Event processing runs regardless — robots may still be at nest/tag
        self._process_events()
        obs = self.get_observations()
        return obs, 0.0, False, {}

    def _process_events(self):
        base_pos = self.base_node.getPosition()

        for i in range(self.num_robots):
            robot_pos = self.robot_nodes[i].getPosition()

            if not self.carrying_state[i]:
                target = self.nest_target[i]
                if target is not None:
                    tx, ty      = target[1], target[2]
                    dist_to_tgt = math.sqrt((tx - robot_pos[0])**2 + (ty - robot_pos[1])**2)
                    if dist_to_tgt < 0.05:
                        self.nest_target[i]      = None
                        self.prev_site_dists[i]  = None
                        self.prev_phero_dists[i] = None

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
                        self.give_up_timer[i]        = 0
                        self.gave_up[i]              = False
                        picked_up                    = True
                        self.total_pickups          += 1
                        print(f"[PICKUP] Robot {i+1} picked up tag "
                              f"(density={density}) | Total: {self.total_pickups}")
                        break

                if not picked_up:
                    self.steps_without_pickup[i] += 1
                    if self.nest_target[i] is None:
                        self.give_up_timer[i] += 1
                    if self.give_up_timer[i] >= self.GIVE_UP_CHECK_STEPS:
                        self.give_up_timer[i] = 0
                        if random.random() < self.PROB_RETURN_TO_NEST:
                            self.gave_up[i] = True
                            print(f"[GIVE-UP] R{i+1} giving up after "
                                  f"{self.steps_without_pickup[i]} steps")

                    dist_from_base = math.sqrt(
                        (robot_pos[0] - base_pos[0])**2
                        + (robot_pos[1] - base_pos[1])**2
                    )
                    if (self.gave_up[i] and dist_from_base < 0.25
                            and self.nest_target[i] is None):
                        self.steps_without_pickup[i] = 0
                        self.give_up_timer[i]        = 0
                        self.nest_target[i]          = self._assign_target(i)
                        self.gave_up[i]              = False
                        t = self.nest_target[i]
                        print(f"[EMPTY_RTN] R{i+1} → "
                              f"{t[0]+'('+f'{t[1]:.2f},{t[2]:.2f}'+')' if t else 'EXPLORE'}")

            else:
                dx           = base_pos[0] - robot_pos[0]
                dy           = base_pos[1] - robot_pos[1]
                dist_to_base = math.sqrt(dx*dx + dy*dy)
                self.prev_base_dists[i] = dist_to_base

                if dist_to_base < 0.25:
                    self.carrying_state[i]  = False
                    self.total_deposits    += 1
                    print(f"[DEPOSIT] Robot {i+1} deposited! Total: {self.total_deposits}")

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

                    self.nest_target[i] = self._assign_target(i)
                    t = self.nest_target[i]
                    print(f"  [TARGET] Robot {i+1} → "
                          f"{t[0].upper() + ' (' + f'{t[1]:.2f},{t[2]:.2f}' + ')' if t else 'EXPLORE'}")

    # =========================================================================
    # OBSERVATIONS
    # =========================================================================

    def get_observations(self):
        global_obs = []
        base_pos   = self.base_node.getPosition()

        for i in range(self.num_robots):
            prox = (self.robot_states[i] or [0.0]*8)[:8]

            robot_pos   = self.robot_nodes[i].getPosition()
            robot_rot   = self.robot_nodes[i].getOrientation()
            forward_vec = [robot_rot[0], robot_rot[3], robot_rot[6]]

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

            site_known = site_dist_norm = site_angle_norm = 0.0
            if (not self.carrying_state[i]
                    and self.nest_target[i] is not None
                    and self.nest_target[i][0] == 'site'):
                sx, sy = self.nest_target[i][1], self.nest_target[i][2]
                sdx, sdy = sx - robot_pos[0], sy - robot_pos[1]
                s_dist = math.sqrt(sdx*sdx + sdy*sdy)
                if s_dist > 0.001:
                    s_norm  = [sdx / s_dist, sdy / s_dist]
                    s_dot   = max(min(forward_vec[0]*s_norm[0] + forward_vec[1]*s_norm[1], 1.0), -1.0)
                    s_angle = math.acos(s_dot)
                    s_cross = forward_vec[0]*s_norm[1] - forward_vec[1]*s_norm[0]
                    s_angle = s_angle if s_cross > 0 else -s_angle
                    site_known      = 1.0
                    site_dist_norm  = min(s_dist / MAX_DIST, 1.0)
                    site_angle_norm = s_angle / math.pi

            phero_known = phero_dist_norm = phero_angle_norm = 0.0
            if (not self.carrying_state[i]
                    and self.nest_target[i] is not None
                    and self.nest_target[i][0] == 'phero'):
                px, py = self.nest_target[i][1], self.nest_target[i][2]
                pdx, pdy = px - robot_pos[0], py - robot_pos[1]
                p_dist = math.sqrt(pdx*pdx + pdy*pdy)
                if p_dist > 0.001:
                    p_norm  = [pdx / p_dist, pdy / p_dist]
                    p_dot   = max(min(forward_vec[0]*p_norm[0] + forward_vec[1]*p_norm[1], 1.0), -1.0)
                    p_angle = math.acos(p_dot)
                    p_cross = forward_vec[0]*p_norm[1] - forward_vec[1]*p_norm[0]
                    p_angle = p_angle if p_cross > 0 else -p_angle
                    phero_known      = 1.0
                    phero_dist_norm  = min(p_dist / MAX_DIST, 1.0)
                    phero_angle_norm = p_angle / math.pi

            if self.carrying_state[i]:
                search_duration_norm = 0.0
            else:
                search_duration_norm = min(
                    self.steps_without_pickup[i] / self.SEARCH_DURATION_NORM, 1.0
                )

            obs = []
            obs.extend(prox)
            obs.append(1.0 if self.carrying_state[i] else 0.0)
            obs.append(dist_to_base / MAX_DIST)
            obs.append(angle_to_base / math.pi)
            obs.extend([site_known, site_dist_norm, site_angle_norm])
            obs.extend([phero_known, phero_dist_norm, phero_angle_norm])
            obs.append(search_duration_norm)
            global_obs.extend(obs)

        return np.array(global_obs, dtype=np.float32)

    # =========================================================================
    # OVERRIDES  (only active when not disconnected)
    # =========================================================================

    def _steer_to(self, robot_pos, fwd, target, gain=2.5):
        dx, dy = target[0] - robot_pos[0], target[1] - robot_pos[1]
        dist   = math.sqrt(dx*dx + dy*dy)
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

            if wall_dist < 0.35 or max(prox) > 0.55:
                ov = self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=4.0)
                final[i*2], final[i*2+1] = ov[0], ov[1]
                continue

            if not self.carrying_state[i] and not self.gave_up[i] and dist_to_base < 0.25:
                if dist_to_base > 0.001:
                    esc_x = robot_pos[0] + (robot_pos[0] / dist_to_base) * 0.5
                    esc_y = robot_pos[1] + (robot_pos[1] / dist_to_base) * 0.5
                else:
                    esc_x, esc_y = 0.5, 0.0
                ov = self._steer_to(robot_pos, fwd, [esc_x, esc_y], gain=4.0)
                final[i*2], final[i*2+1] = ov[0], ov[1]
                continue

            if self.carrying_state[i] or self.gave_up[i]:
                ov = self._steer_to(robot_pos, fwd, base_pos, gain=2.5)
                final[i*2], final[i*2+1] = ov[0], ov[1]
                continue

        return np.array(final, dtype=np.float32)

    def _apply_safety_overrides(self, base_action):
        """Wall-escape and base-stuck escape only — no policy, no coordination.
        Used during disconnect to keep robots from getting permanently stuck."""
        final    = list(base_action)
        base_pos = self.base_node.getPosition()
        for i in range(self.num_robots):
            robot_pos = self.robot_nodes[i].getPosition()
            robot_rot = self.robot_nodes[i].getOrientation()
            fwd       = [robot_rot[0], robot_rot[3], robot_rot[6]]
            prox      = (self.robot_states[i] or [0.0]*8)[:8]
            wall_dist = 2.5 - max(abs(robot_pos[0]), abs(robot_pos[1]))

            # P1: wall escape
            if wall_dist < 0.35 or max(prox) > 0.55:
                ov = self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=4.0)
                final[i*2], final[i*2+1] = ov[0], ov[1]
                continue

            # P2: base stuck escape (robot loitering at nest without a tag)
            bdx          = base_pos[0] - robot_pos[0]
            bdy          = base_pos[1] - robot_pos[1]
            dist_to_base = math.sqrt(bdx*bdx + bdy*bdy)
            if not self.carrying_state[i] and not self.gave_up[i] and dist_to_base < 0.25:
                if dist_to_base > 0.001:
                    esc_x = robot_pos[0] + (robot_pos[0] / dist_to_base) * 0.5
                    esc_y = robot_pos[1] + (robot_pos[1] / dist_to_base) * 0.5
                else:
                    esc_x, esc_y = 0.5, 0.0
                ov = self._steer_to(robot_pos, fwd, [esc_x, esc_y], gain=4.0)
                final[i*2], final[i*2+1] = ov[0], ov[1]
                continue

            # No policy during disconnect — robot continues at pre-disconnect velocity

        return np.array(final, dtype=np.float32)

    def _get_mode(self, i, wall_dist):
        base_pos = self.base_node.getPosition()
        rpos     = self.robot_nodes[i].getPosition()
        d2base   = math.sqrt((rpos[0]-base_pos[0])**2 + (rpos[1]-base_pos[1])**2)
        if self.disconnected:
            return "DISCONNECTED"
        if wall_dist < 0.35:
            return "WALL_ESC"
        elif self.carrying_state[i]:
            return "RTB"
        elif self.gave_up[i]:
            return "GIVE_UP"
        elif not self.carrying_state[i] and d2base < 0.25:
            return "BASE_ESC"
        elif self.nest_target[i] is not None:
            return self.nest_target[i][0].upper()
        else:
            return "EXPLORE"

    def _positions_finite(self):
        for node in [self.base_node] + self.robot_nodes:
            if node is None:
                return False
            pos = node.getPosition()
            if pos is None or not all(math.isfinite(v) for v in pos[:2]):
                return False
        return True

    def wait_until_ready(self, max_steps=300):
        for _ in range(max_steps):
            if self._positions_finite():
                return True
            if Supervisor.step(self, self.timestep) == -1:
                return False
        return self._positions_finite()

    def reset(self):
        positions = [[-0.5, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0]]
        for i in range(self.num_robots):
            translation = self.robot_nodes[i].getField("translation")
            rotation    = self.robot_nodes[i].getField("rotation")
            if translation is not None:
                translation.setSFVec3f(positions[i])
            if rotation is not None:
                rotation.setSFRotation([0, 0, 1, 0])
            self.robot_nodes[i].resetPhysics()

        if not self.wait_until_ready():
            print("[ERROR] Webots scene did not produce finite positions after reset.")
            sys.exit(1)

        self.robot_states         = [None]  * self.num_robots
        self.carrying_state       = [False] * self.num_robots
        self.prev_base_dists      = [None]  * self.num_robots
        self.pheromone_list       = []
        self.carried_from         = [None]  * self.num_robots
        self.resource_density     = [0]     * self.num_robots
        self.site_fidelity_pos    = [None]  * self.num_robots
        self.nest_target          = [None]  * self.num_robots
        self.prev_site_dists      = [None]  * self.num_robots
        self.prev_phero_dists     = [None]  * self.num_robots
        self.steps_without_pickup = [0]     * self.num_robots
        self.give_up_timer        = [0]     * self.num_robots
        self.gave_up              = [False] * self.num_robots
        self.disconnected         = False

        return self.get_observations()


# =============================================================================
# EVALUATION ENTRY POINT
# =============================================================================

def _machine_result(label, env, step_count, disconnect_start, disconnect_duration):
    elapsed_min = step_count * env.timestep / 1000.0 / 60.0
    return (
        f"{label} pickups={env.total_pickups} "
        f"deposits={env.total_deposits} "
        f"steps={step_count} "
        f"elapsed_min={elapsed_min:.6f} "
        f"disconnect_start={disconnect_start:.1f} "
        f"disconnect_duration={disconnect_duration:.1f}"
    )


def _batch_result(env):
    return f"BATCH_RESULT pickups={env.total_pickups} deposits={env.total_deposits}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?",
                        help="PPO model path (.zip suffix optional)")
    parser.add_argument("--duration-sim-min", type=float, default=10.0,
                        help="Total foraging time in simulated minutes (default 10)")
    parser.add_argument("--disconnect-start-min", type=float, default=-1.0,
                        help="Sim minute at which central server disconnects (-1 = no disconnect)")
    parser.add_argument("--disconnect-duration-min", type=float, default=0.0,
                        help="Duration of disconnect in simulated minutes")
    parser.add_argument("--stop-on-completion", action="store_true")
    args = parser.parse_args()

    disconnect_start    = args.disconnect_start_min
    disconnect_duration = args.disconnect_duration_min
    has_disconnect      = (disconnect_start >= 0.0 and disconnect_duration > 0.0)

    print("=" * 65)
    print("EXPERIMENT 4 — Central Server Disconnect Robustness")
    print(f"  Arena: 5x5m | Tags: 64 | Robots: 4 | Time: {args.duration_sim_min} min")
    if has_disconnect:
        disconnect_end = disconnect_start + disconnect_duration
        print(f"  Disconnect: {disconnect_start:.1f} → {disconnect_end:.1f} min "
              f"({disconnect_duration:.1f} min offline)")
    else:
        print("  Disconnect: none (control run)")
    print("=" * 65)

    env = EpuckForagingSupervisor()

    model_path = args.model or "logs/ppo_cpfa_v8/ppo_cpfa_v8_2000000_steps"
    if model_path.endswith('.zip'):
        model_path = model_path[:-4]

    print(f"\nLoading model: {model_path}")
    try:
        custom_objects = {
            "lr_schedule": lambda _: 3e-4,
            "clip_range":  lambda _: 0.2,
        }
        model = PPO.load(model_path, custom_objects=custom_objects, device="cpu")
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print("Model loaded. Starting evaluation...\n")

    obs          = env.reset()
    step_count   = 0
    target_steps = math.ceil(args.duration_sim_min * 60.0 * 1000.0 / env.timestep)

    was_disconnected  = False
    reconnect_logged  = False

    while True:
        elapsed_min  = step_count * env.timestep / 1000.0 / 60.0
        in_disconnect = (
            has_disconnect and
            disconnect_start <= elapsed_min < disconnect_start + disconnect_duration
        )

        # Log transition events once
        if in_disconnect and not was_disconnected:
            print(f"\n[DISCONNECT] Central server OFFLINE at {elapsed_min:.2f} min "
                  f"(will reconnect at {disconnect_start + disconnect_duration:.1f} min)\n")
            was_disconnected = True

        if not in_disconnect and was_disconnected and not reconnect_logged:
            print(f"\n[RECONNECT] Central server ONLINE at {elapsed_min:.2f} min\n")
            reconnect_logged = True

        env.disconnected = in_disconnect

        if in_disconnect:
            action = np.zeros(env.action_space_dim, dtype=np.float32)
        else:
            if not np.isfinite(obs).all():
                if env.wait_until_ready():
                    obs = env.get_observations()
                if not np.isfinite(obs).all():
                    print(f"[ERROR] Non-finite obs at step {step_count}")
                    sys.exit(1)
            action, _ = model.predict(obs, deterministic=True)

        obs, _, done, _ = env.step(action)
        step_count += 1

        if step_count % 500 == 0:
            phero_active = len([p for p in env.pheromone_list
                                if p['weight'] > env.PHEROMONE_MIN])
            elapsed_min  = step_count * env.timestep / 1000.0 / 60.0
            rate         = env.total_deposits / elapsed_min if elapsed_min > 0 else 0.0
            status       = "DISCONNECTED" if in_disconnect else "ONLINE"

            log_msg = (
                f"\n{'='*70}\n"
                f"Step {step_count} ({elapsed_min:.1f} min) [{status}] | "
                f"Pickups: {env.total_pickups} | Deposits: {env.total_deposits} | "
                f"Rate: {rate:.2f} tags/min | phero={phero_active}\n"
                f"{'='*70}\n"
            )
            base_pos = env.base_node.getPosition()
            for ri in range(env.num_robots):
                rpos   = env.robot_nodes[ri].getPosition()
                wall_d = 2.5 - max(abs(rpos[0]), abs(rpos[1]))
                mode   = env._get_mode(ri, wall_d)
                log_msg += (
                    f"R{ri+1}[{mode:12s}]: "
                    f"carry={int(env.carrying_state[ri])} | "
                    f"gave_up={int(env.gave_up[ri])}\n"
                )
            print(log_msg)

        completion_reached = env.total_deposits >= env.num_tags
        if completion_reached and args.stop_on_completion:
            print(_machine_result("COMPLETION_RESULT", env, step_count,
                                  disconnect_start, disconnect_duration), flush=True)
            print(_batch_result(env), flush=True)
            env.simulationQuit(0)
            break

        if step_count >= target_steps:
            label = "COMPLETION_RESULT" if completion_reached else "TIMEOUT_RESULT"
            print(_machine_result(label, env, step_count,
                                  disconnect_start, disconnect_duration), flush=True)
            print(_batch_result(env), flush=True)
            env.simulationQuit(0)
            break
