import sys
import os
import math
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'epuck_decentralized'))
from epuck_decentralized_v4 import EpuckDecentralizedV4, TARGET_ARRIVAL_DIST, PHEROMONE_MIN

from stable_baselines3 import PPO

# =============================================================================
# DECENTRALIZED EVAL ROBOT CONTROLLER  (CoRL 2026)
#
# Mirrors training robot (epuck_decentralized_train_v4) exactly:
#   - Same base class: EpuckDecentralizedV4 (CPFA list pheromone, site fidelity)
#   - Same 20D obs layout
#   - Same P1 / BASE_ESC / P2 override logic
#   - Same pickup/deposit/target-depletion handling
#   - PPO inference: deterministic=True (no exploration noise)
#
# Message protocol (identical to training):
#   Supervisor → Robot: [tag_visible, tag_dist_norm, tag_angle_norm, pickup_signal]
#   Robot → Supervisor: 17 floats [prox×8, carrying, base_dist, base_angle,
#                                   site_known, phero_known, reserved×2, gps_x, gps_y]
#
# Model loading priority:
#   1. current_eval_run.txt  → loads {robot_name}_{run_name}.zip  (per-robot independent)
#   2. current_eval_model.txt → loads shared model path            (CTDE / fallback)
#   3. Hard-coded fallback: decentralized_optA_v1
# =============================================================================

TAG_SEEK_RANGE = 1.0
OBS_DIM        = 20


class EpuckDecentralizedEval(EpuckDecentralizedV4):

    def __init__(self):
        super().__init__()
        self._ppo = None

        self._tag_visible    = 0.0
        self._tag_dist_norm  = 0.0
        self._tag_angle_norm = 0.0

    def _load_ppo(self):
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
        )
        robot_name = self.getName()

        # Priority 1: per-robot model from independent training
        run_cfg = os.path.join(project_root, 'current_eval_run.txt')
        if os.path.exists(run_cfg):
            run_name   = open(run_cfg).read().strip()
            model_path = os.path.join(project_root, f"{robot_name}_{run_name}")
            if os.path.exists(model_path + '.zip'):
                print(f"[{robot_name}] Loading per-robot model: {model_path}.zip")
                self._ppo = PPO.load(model_path)
                print(f"[{robot_name}] Model loaded (20D obs, independent PPO).")
                return

        # Priority 2: shared model from current_eval_model.txt (CTDE / fallback)
        model_cfg = os.path.join(project_root, 'current_eval_model.txt')
        if os.path.exists(model_cfg):
            model_path = open(model_cfg).read().strip()
        else:
            model_path = os.path.join(project_root, 'decentralized_optA_v1')
        print(f"[{robot_name}] Loading shared model: {model_path}")
        self._ppo = PPO.load(model_path)
        print(f"[{robot_name}] Model loaded.")

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
            self._tag_visible    = float(message[0])
            self._tag_dist_norm  = float(message[1])
            self._tag_angle_norm = float(message[2])
            pickup_signal        = float(message[3])
        except (ValueError, IndexError):
            return

        # ── Pickup ────────────────────────────────────────────────────
        if pickup_signal > 0.0 and not self.carrying:
            self.carrying = True
            gps = self.gps.getValues()
            px, py = gps[0], gps[1]
            self._add_pheromone(px, py, pickup_signal)
            self._site_fidelity_pos  = (px, py)
            self._last_pickup_weight = pickup_signal
            self._current_target     = None
            print(f"[PICKUP] {self.getName()} at ({px:.2f},{py:.2f}) | "
                  f"strength={pickup_signal:.2f}")

        # ── Deposit ───────────────────────────────────────────────────
        elif pickup_signal < -0.5 and self.carrying:
            self.carrying = False
            self._assign_target()
            t = self._current_target
            t_str = f"{t[0].upper()} ({t[1]:.2f},{t[2]:.2f})" if t else "EXPLORE"
            sf = self._site_fidelity_pos
            sf_str = f"({sf[0]:.2f},{sf[1]:.2f}) w={self._last_pickup_weight:.2f}" if sf else "None"
            print(f"[DEPOSIT] {self.getName()}")
            print(f"  [SITE_FID] {sf_str}")
            print(f"  [TARGET]   → {t_str}")

        # ── Target depletion: arrived at cluster, no food found ───────
        if not self.carrying and self._current_target is not None:
            gps = self.gps.getValues()
            tx, ty = self._current_target[1], self._current_target[2]
            if math.sqrt((gps[0]-tx)**2 + (gps[1]-ty)**2) < TARGET_ARRIVAL_DIST:
                if self._current_target[0] == 'site':
                    self._last_pickup_weight = max(self._last_pickup_weight * 0.5, 0.05)
                elif self._current_target[0] == 'phero':
                    for entry in self.pheromone_list:
                        if math.sqrt((entry['x'] - tx)**2 +
                                     (entry['y'] - ty)**2) < 0.3:
                            entry['weight'] = max(entry['weight'] * 0.5, PHEROMONE_MIN)
                            break
                self._assign_target()

        # ── Assemble 20D obs (identical layout to training robot) ─────
        prox     = [s.getValue() / 4096.0 for s in self.ps]
        obs_comp = self._obs_components()
        obs_arr  = np.array(
            prox + [
                self._tag_visible,                    # [8]
                self._tag_dist_norm,                  # [9]
                self._tag_angle_norm,                 # [10]
                1.0 if self.carrying else 0.0,        # [11]
                obs_comp["base_dist_norm"],            # [12]
                obs_comp["base_angle_norm"],           # [13]
                obs_comp["site_known"],                # [14]
                obs_comp["site_dist_norm"],            # [15]
                obs_comp["site_angle_norm"],           # [16]
                obs_comp["phero_known"],               # [17]
                obs_comp["phero_dist_norm"],           # [18]
                obs_comp["phero_angle_norm"],          # [19]
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
        wall_dist = 2.5 - max(abs(pos_x), abs(pos_y))
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
