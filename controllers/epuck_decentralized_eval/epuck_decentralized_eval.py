import sys
import os
import math
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'epuck_decentralized'))
from epuck_decentralized_v4 import (
    EpuckDecentralizedV4,
    TARGET_ARRIVAL_DIST, PHEROMONE_MIN, MERGE_RADIUS,
    RATE_OF_LAYING_PHEROMONE,
)

from stable_baselines3 import PPO

# =============================================================================
# DECENTRALIZED EVAL ROBOT CONTROLLER  (CoRL 2026 — synced to v8)
#
# Mirrors training robot (epuck_decentralized_train_v8) exactly:
#   - Same base class: EpuckDecentralizedV4 (CPFA list pheromone, site fidelity)
#   - Same 19D obs layout (no tag sensing — matches centralized; adds phero_density_norm at [17])
#   - Same overrides: P1 (wall escape) + BASE_ESC + P2 (carrying only, no gave_up RTB)
#   - Same pickup/deposit/target-depletion handling (Poisson CDF gate at pickup)
#   - PPO inference: deterministic=True (no exploration noise)
#
# Message protocol (identical to training):
#   Supervisor → Robot: [tag_visible, tag_dist_norm, tag_angle_norm, pickup_signal]
#   Robot → Supervisor: 17 floats [prox×8, carrying, base_dist, base_angle,
#                                   site_known, phero_known, reserved×2, gps_x, gps_y]
#
# Model loading priority:
#   1. current_eval_run.txt  → loads {robot_name}_{run_name}.zip  (per-robot, v8)
#   2. current_eval_model.txt → loads shared model path            (CTDE / fallback)
#   3. Hard-coded fallback: decentralized_optA_v1
# =============================================================================

OBS_DIM             = 19
SEARCH_DURATION_MAX = 4000
TARGET_LINGER_STEPS = 3      # steps to stay at cluster before declaring depletion


class EpuckDecentralizedEval(EpuckDecentralizedV4):

    # Arena half-sizes for wall-escape override (matches supervisor ARENA_CONFIGS)
    _ARENA_HALF = {'5x5': 2.5, '7x7': 3.5, '9x9': 4.5, '12x12': 6.0}

    def __init__(self):
        super().__init__()
        self._ppo                  = None
        self._steps_without_pickup = 0
        self._arrival_steps        = 0   # steps spent within TARGET_ARRIVAL_DIST of current target
        self._arena_half           = 2.5  # default 5x5; overridden in _load_ppo

    def _load_ppo(self):
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
        )
        robot_name = self.getName()

        # Read arena size written by supervisor before Webots started
        arena_cfg = os.path.join(project_root, 'current_eval_arena.txt')
        if os.path.exists(arena_cfg):
            arena_size = open(arena_cfg).read().strip()
            self._arena_half = self._ARENA_HALF.get(arena_size, 2.5)
            self._max_dist   = self._arena_half * math.sqrt(2)
        self._log(f"[{robot_name}] Arena: {self._arena_half*2:.0f}×"
                  f"{self._arena_half*2:.0f} m "
                  f"(half={self._arena_half} m, max_dist={self._max_dist:.3f} m)")

        # Priority 1: per-robot model from independent training
        run_name = None   # initialise before use — guards cycling code below
        run_cfg  = os.path.join(project_root, 'current_eval_run.txt')
        if os.path.exists(run_cfg):
            run_name   = open(run_cfg).read().strip()
            model_path = os.path.join(project_root, f"{robot_name}_{run_name}")
            if os.path.exists(model_path + '.zip'):
                self._open_robot_log(f"eval_{run_name}")
                self._log(f"[{robot_name}] Loading per-robot model: {model_path}.zip")
                self._ppo = PPO.load(model_path, device='cpu')
                self._log(f"[{robot_name}] Model loaded (19D obs, independent PPO v8, cpu).")
                return

        # Cycle robots 5-16 → trained robots 1-4 (robot5→1, robot6→2, …)
        if run_name is not None:
            try:
                robot_num  = int(''.join(filter(str.isdigit, robot_name)))
                mapped_num = ((robot_num - 1) % 4) + 1
                if mapped_num != robot_num:
                    mapped_name = f"robot{mapped_num}"
                    cycle_path  = os.path.join(project_root, f"{mapped_name}_{run_name}")
                    if os.path.exists(cycle_path + '.zip'):
                        self._open_robot_log(f"eval_{run_name}")
                        self._log(f"[{robot_name}] No own model — cycling to {mapped_name}'s model")
                        self._ppo = PPO.load(cycle_path, device='cpu')
                        self._log(f"[{robot_name}] Model loaded (cycled from {mapped_name}, cpu).")
                        return
            except ValueError:
                pass

        # Priority 2: shared model from current_eval_model.txt (CTDE / fallback)
        model_cfg = os.path.join(project_root, 'current_eval_model.txt')
        if os.path.exists(model_cfg):
            model_path = open(model_cfg).read().strip()
        else:
            model_path = os.path.join(project_root, 'decentralized_optA_v1')
        self._open_robot_log("eval_shared")
        self._log(f"[{robot_name}] Loading shared model: {model_path}")
        self._ppo = PPO.load(model_path, device='cpu')
        self._log(f"[{robot_name}] Model loaded (cpu).")

    # =========================================================================
    # deepbots interface
    # =========================================================================

    def create_message(self):
        """17-float message to supervisor (pheromone decay/broadcast/receive runs here)."""
        return super().create_message()

    def use_message_data(self, message):
        if self._ppo is None:
            self._load_ppo()

        if not message or len(message) < 4:
            return

        try:
            pickup_signal = float(message[3])
        except (ValueError, IndexError):
            return

        # ── Pickup ────────────────────────────────────────────────────
        if pickup_signal > 0.0 and not self.carrying:
            self.carrying = True
            density = max(0, round(pickup_signal))
            # Supervisor sends exact tag position in message[0:1] on pickup.
            # Use tag position (not robot GPS) for pheromone — eliminates the
            # up-to-0.15m offset between robot GPS and actual tag location.
            try:
                px = float(message[0])
                py = float(message[1])
            except (ValueError, IndexError):
                gps = self.gps.getValues()
                px, py = gps[0], gps[1]
            self._site_fidelity_pos = (px, py)
            self._resource_density  = density
            # Update any existing pheromone entry at this cluster to current
            # actual density — corrects stale density from earlier denser pickups
            for entry in self.pheromone_list:
                if math.sqrt((entry['x'] - px)**2 +
                             (entry['y'] - py)**2) < MERGE_RADIUS:
                    entry['density'] = density
                    break
            # CPFA pheromone laying at PICKUP — Poisson CDF gate (P2P advantage)
            lay_prob = self._poisson_cdf(density, RATE_OF_LAYING_PHEROMONE)
            laid = False
            if random.random() < lay_prob:
                self._add_pheromone(px, py, 1.0, density)
                laid = True
            self._current_target       = None
            self._gave_up              = False
            self._give_up_timer        = 0
            self._steps_without_pickup = 0
            self._arrival_steps        = 0
            self._log(f"[PICKUP]   {self.getName()} at tag=({px:.2f},{py:.2f}) | density={density}")
            if laid:
                self._log(f"  [PHERO]  Laid at ({px:.2f},{py:.2f}) "
                          f"density={density} prob={lay_prob:.2f} "
                          f"total_entries={len(self.pheromone_list)}")

        # ── Deposit ───────────────────────────────────────────────────
        elif pickup_signal < -0.5 and self.carrying:
            self.carrying = False
            self._arrival_steps = 0
            sf = self._site_fidelity_pos
            sf_str = f"({sf[0]:.2f},{sf[1]:.2f}) density={self._resource_density}" if sf else "None"
            self._log(f"[DEPOSIT]  {self.getName()} | site={sf_str}")
            self._assign_target()   # prints [TARGET] / [EMPTY_RTN] via _log in base class

        # ── Target depletion: arrived at cluster, no food found ───────
        # Linger TARGET_LINGER_STEPS steps before declaring depleted so the
        # supervisor has time to detect adjacent tags (which may be up to
        # 0.0779 m from the stored pheromone point) and send a pickup signal.
        # P2.5 keeps steering to (tx,ty) while _current_target is set, so the
        # robot stays in the cluster area during the wait.
        if not self.carrying and self._current_target is not None:
            gps = self.gps.getValues()
            tx, ty = self._current_target[1], self._current_target[2]
            if math.sqrt((gps[0]-tx)**2 + (gps[1]-ty)**2) < TARGET_ARRIVAL_DIST:
                self._arrival_steps += 1
                if self._arrival_steps >= TARGET_LINGER_STEPS:
                    self._arrival_steps = 0
                    if self._current_target[0] == 'phero':
                        # Mark depleted: density=0 removes from roulette; expire quickly
                        for entry in self.pheromone_list:
                            if math.sqrt((entry['x'] - tx)**2 +
                                         (entry['y'] - ty)**2) < MERGE_RADIUS:
                                entry['density'] = 0
                                entry['weight']  = PHEROMONE_MIN
                                break
                    elif self._current_target[0] == 'site':
                        # Clear own site fidelity — don't revisit a depleted personal cluster
                        self._site_fidelity_pos = None
                        self._resource_density  = 0
                    # Cluster empty: assign next best (skip give-up wait)
                    self._gave_up = True   # suppress stale site fidelity re-assignment
                    self._current_target = None
                    self._assign_target()
                    self._gave_up = False
            else:
                self._arrival_steps = 0   # reset if robot moves away from target

        # ── Increment search counter (only while not carrying) ───────
        if not self.carrying:
            self._steps_without_pickup += 1

        # ── Assemble 19D obs (identical layout to training robot) ─────
        prox     = [s.getValue() / 4096.0 for s in self.ps]
        obs_comp = self._obs_components()
        search_norm = (0.0 if self.carrying else
                       min(self._steps_without_pickup / SEARCH_DURATION_MAX, 1.0))
        obs_arr  = np.array(
            prox + [
                1.0 if self.carrying else 0.0,        # [8]
                obs_comp["base_dist_norm"],            # [9]
                obs_comp["base_angle_norm"],           # [10]
                obs_comp["site_known"],                # [11]
                obs_comp["site_dist_norm"],            # [12]
                obs_comp["site_angle_norm"],           # [13]
                obs_comp["phero_known"],               # [14]
                obs_comp["phero_dist_norm"],           # [15]
                obs_comp["phero_angle_norm"],          # [16]
                obs_comp["phero_density_norm"],        # [17] density of best pheromone target
                search_norm,                           # [18] informational only
            ],
            dtype=np.float32
        ).reshape(1, OBS_DIM)

        # ── PPO inference (deterministic — no exploration noise) ──────
        action, _ = self._ppo.predict(obs_arr, deterministic=True)
        left_cmd  = float(action[0][0])
        right_cmd = float(action[0][1])

        # ── P1 / BASE_ESC / P2 overrides (matches training exactly) ──
        left_cmd, right_cmd = self._apply_overrides(
            left_cmd, right_cmd,
            obs_comp["pos_x"], obs_comp["pos_y"],
        )

        # ── Drive motors ──────────────────────────────────────────────
        max_speed    = 6.28
        scale_factor = 6.0
        self.left_motor.setVelocity(
            max(min(left_cmd  * scale_factor, max_speed), -max_speed))
        self.right_motor.setVelocity(
            max(min(right_cmd * scale_factor, max_speed), -max_speed))

    # =========================================================================
    # Overrides — identical to training robot
    # =========================================================================

    def _apply_overrides(self, left, right, pos_x, pos_y):
        gps       = self.gps.getValues()
        yaw       = self.imu.getRollPitchYaw()[2]
        fwd       = [math.cos(yaw), math.sin(yaw)]
        robot_pos = [gps[0], gps[1]]
        wall_dist = self._arena_half - max(abs(pos_x), abs(pos_y))
        prox      = [s.getValue() / 4096.0 for s in self.ps]

        # P1: wall / obstacle escape
        if wall_dist < 0.35 or max(prox) > 0.55:
            return self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=4.0)

        dist_to_base = math.sqrt(pos_x ** 2 + pos_y ** 2)

        # BASE_ESC: nudge away from nest after deposit
        if not self.carrying and dist_to_base < 0.25:
            if dist_to_base > 0.001:
                esc_x = pos_x + (pos_x / dist_to_base) * 0.5
                esc_y = pos_y + (pos_y / dist_to_base) * 0.5
            else:
                esc_x, esc_y = 0.5, 0.0
            return self._steer_to(robot_pos, fwd, [esc_x, esc_y], gain=4.0)

        # P2: return to base when carrying
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
    robot = EpuckDecentralizedEval()
    robot.run()
