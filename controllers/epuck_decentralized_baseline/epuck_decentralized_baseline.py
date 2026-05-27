import sys
import os
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'epuck_decentralized'))
from epuck_decentralized_v4 import (
    EpuckDecentralizedV4,
    TARGET_ARRIVAL_DIST, PHEROMONE_MIN,
    RATE_OF_LAYING_PHEROMONE,
)

# =============================================================================
# CPFA BASELINE ROBOT CONTROLLER  (no PPO — logic validation only)
#
# Identical to v7 training robot in every respect EXCEPT movement:
#   - No PPO, no SB3, no model loading
#   - Navigation: explicit _steer_to() toward assigned site/phero target
#   - Exploration: correlated random walk when no target assigned
#
# All CPFA mechanics are unchanged vs v7:
#   - Pheromone laid at PICKUP (Poisson CDF gate)
#   - Site fidelity assigned at DEPOSIT (Poisson CDF gate)
#   - Give-up timer during free exploration
#   - Target depletion: halve pheromone weight on arrival with no food
#   - Pheromone used at deposit/give-up via _assign_target() (base class)
#   - P1 (wall escape) + BASE_ESC + P2 (carrying only) overrides
#
# Purpose: verify CPFA logic produces correct pickups/deposits/pheromone
# flow before attributing performance differences to PPO.
#
# Run in eval world with eval supervisor — same message protocol:
#   Supervisor → Robot: [tag_visible, tag_dist_norm, tag_angle_norm, pickup_signal]
#   Robot → Supervisor: 17 floats [prox×8, carrying, base_dist, base_angle,
#                                   site_known, phero_known, 0, 0, gps_x, gps_y]
# =============================================================================


class EpuckDecentralizedBaseline(EpuckDecentralizedV4):

    def __init__(self):
        super().__init__()
        self._explore_turn = 0.0   # persistent heading bias for correlated walk
        self._open_robot_log('baseline')

    # =========================================================================
    # deepbots interface
    # =========================================================================

    def create_message(self):
        """Delegates entirely to base class (pheromone decay/broadcast/give-up)."""
        return super().create_message()

    def use_message_data(self, message):
        if not message or len(message) < 4:
            return

        try:
            pickup_signal = float(message[3])
        except (ValueError, IndexError):
            return

        # ── Pickup ────────────────────────────────────────────────────
        if pickup_signal > 0.0 and not self.carrying:
            self.carrying = True
            gps           = self.gps.getValues()
            px, py        = gps[0], gps[1]
            density       = max(0, round(pickup_signal))
            self._site_fidelity_pos = (px, py)
            self._resource_density  = density
            lay_prob = self._poisson_cdf(density, RATE_OF_LAYING_PHEROMONE)
            laid     = False
            if random.random() < lay_prob:
                self._add_pheromone(px, py, 1.0)
                laid = True
            self._current_target  = None
            self._gave_up         = False
            self._give_up_timer   = 0
            self._log(f"[PICKUP]   {self.getName()} at ({px:.2f},{py:.2f}) | density={density}")
            if laid:
                self._log(f"  [PHERO]  Laid at ({px:.2f},{py:.2f}) "
                          f"density={density} prob={lay_prob:.2f} "
                          f"total_entries={len(self.pheromone_list)}")

        # ── Deposit ───────────────────────────────────────────────────
        elif pickup_signal < -0.5 and self.carrying:
            self.carrying = False
            sf = self._site_fidelity_pos
            sf_str = f"({sf[0]:.2f},{sf[1]:.2f}) density={self._resource_density}" if sf else "None"
            self._log(f"[DEPOSIT]  {self.getName()} | site={sf_str}")
            self._assign_target()   # prints [TARGET] / [EMPTY_RTN] via _log in base class

        # ── Target depletion: arrived at cluster but no food ──────────
        if not self.carrying and self._current_target is not None:
            gps    = self.gps.getValues()
            tx, ty = self._current_target[1], self._current_target[2]
            if math.sqrt((gps[0] - tx) ** 2 + (gps[1] - ty) ** 2) < TARGET_ARRIVAL_DIST:
                if self._current_target[0] == 'phero':
                    for entry in self.pheromone_list:
                        if math.sqrt((entry['x'] - tx) ** 2 +
                                     (entry['y'] - ty) ** 2) < 0.3:
                            entry['weight'] = max(entry['weight'] * 0.5, PHEROMONE_MIN)
                            break
                self._current_target = None


        # ── Compute obs components (needed for overrides) ─────────────
        obs_comp = self._obs_components()
        gps      = self.gps.getValues()
        yaw      = self.imu.getRollPitchYaw()[2]
        fwd      = [math.cos(yaw), math.sin(yaw)]
        robot_pos = [gps[0], gps[1]]

        # ── Base action: steer to target or random walk ───────────────
        left, right = self._base_action(robot_pos, fwd)

        # ── P1 / BASE_ESC / P2 overrides ─────────────────────────────
        left, right = self._apply_overrides(
            left, right,
            obs_comp["pos_x"], obs_comp["pos_y"],
        )

        # ── Drive motors ──────────────────────────────────────────────
        max_speed    = 6.28
        scale_factor = 6.0
        self.left_motor.setVelocity(
            max(min(left  * scale_factor, max_speed), -max_speed))
        self.right_motor.setVelocity(
            max(min(right * scale_factor, max_speed), -max_speed))

    # =========================================================================
    # Navigation
    # =========================================================================

    def _base_action(self, robot_pos, fwd):
        """Steer toward assigned target; correlated random walk when exploring."""
        if self._current_target is not None:
            tx, ty = self._current_target[1], self._current_target[2]
            return self._steer_to(robot_pos, fwd, [tx, ty], gain=3.0)

        # Correlated random walk — smooth direction changes, occasional resets
        self._explore_turn += random.gauss(0, 0.08)
        self._explore_turn  = max(-1.0, min(1.0, self._explore_turn))
        if random.random() < 0.003:   # ~every 330 steps (~21s) pick new heading
            self._explore_turn = random.uniform(-1.0, 1.0)

        left  = max(-1.0, min(1.0, 1.0 - self._explore_turn * 0.5))
        right = max(-1.0, min(1.0, 1.0 + self._explore_turn * 0.5))
        return [left, right]

    def _apply_overrides(self, left, right, pos_x, pos_y):
        gps       = self.gps.getValues()
        yaw       = self.imu.getRollPitchYaw()[2]
        fwd       = [math.cos(yaw), math.sin(yaw)]
        robot_pos = [gps[0], gps[1]]
        wall_dist = 2.5 - max(abs(pos_x), abs(pos_y))
        prox      = [s.getValue() / 4096.0 for s in self.ps]

        # P1: wall / obstacle escape
        if wall_dist < 0.35 or max(prox) > 0.55:
            return self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=4.0)

        dist_to_base = math.sqrt(pos_x ** 2 + pos_y ** 2)

        # BASE_ESC: nudge away from nest when not carrying
        if not self.carrying and dist_to_base < 0.25:
            if dist_to_base > 0.001:
                esc_x = pos_x + (pos_x / dist_to_base) * 0.5
                esc_y = pos_y + (pos_y / dist_to_base) * 0.5
            else:
                esc_x, esc_y = 0.5, 0.0
            return self._steer_to(robot_pos, fwd, [esc_x, esc_y], gain=4.0)

        # P2: return to base when carrying a tag
        if self.carrying:
            return self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=2.5)

        return [left, right]

    def _steer_to(self, robot_pos, fwd, target, gain=2.5):
        dx   = target[0] - robot_pos[0]
        dy   = target[1] - robot_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.01:
            return [0.0, 0.0]
        t_norm = [dx / dist, dy / dist]
        dot    = fwd[0] * t_norm[0] + fwd[1] * t_norm[1]
        cross  = fwd[0] * t_norm[1] - fwd[1] * t_norm[0]
        angle  = math.atan2(cross, dot)
        turn   = max(-1.0, min(1.0, gain * angle / math.pi))
        l      = max(-1.0, min(1.0, 1.0 - turn))
        r      = max(-1.0, min(1.0, 1.0 + turn))
        m      = max(abs(l), abs(r))
        if m > 1.0:
            l /= m
            r /= m
        return [l, r]


if __name__ == '__main__':
    robot = EpuckDecentralizedBaseline()
    robot.run()
