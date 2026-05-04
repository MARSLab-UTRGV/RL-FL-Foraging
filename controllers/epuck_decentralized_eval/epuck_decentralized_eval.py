import sys
import os
import math
import numpy as np

# Import EpuckDecentralized for sensor/pheromone infrastructure
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'epuck_decentralized'))
from epuck_decentralized import EpuckDecentralized, INITIAL_TTL

from stable_baselines3 import PPO

# =============================================================================
# DECENTRALIZED EVAL ROBOT CONTROLLER  (True CTDE — robot runs PPO locally)
#
# Each robot:
#   1. Receives from supervisor (4 floats per step):
#        [tag_visible, tag_dist_norm, tag_angle_norm, pickup_signal]
#   2. Assembles 18D obs from onboard sensors + supervisor tag obs
#   3. Runs PPO inference locally (no supervisor involved in action selection)
#   4. Applies P1-P4 overrides fully onboard using GPS + IMU + prox
#        P1 (0.6m): wall escape — catches backward-driving before wall contact
#        P2: return to base when carrying
#        P3 (0.3m): base avoidance after deposit
#        P4: tag-seek steering toward detected tag
#   5. Drives own motors
#
# Supervisor → Robot: [tag_visible, tag_dist_norm, tag_angle_norm, pickup_signal]
# Robot → Supervisor: 17 floats (same as training — for tag pickup detection)
#
# This is proper CTDE: 14/18 obs dims fully onboard; tag obs (3 dims) maps to
# real onboard camera at deployment; supervisor is only a thin tag-mechanics shim.
# =============================================================================

TAG_SEEK_RANGE = 1.0


class EpuckDecentralizedEval(EpuckDecentralized):
    """
    Eval robot controller: loads PPO model and runs inference locally.
    Inherits all sensor/pheromone infrastructure from EpuckDecentralized.
    Overrides use_message_data to receive tag obs instead of motor commands.
    """

    def __init__(self):
        super().__init__()

        # PPO model loaded lazily on first message (supervisor writes model
        # path to current_eval_model.txt before connecting to Webots).
        self.ppo = None

        # Tag obs received from supervisor each step
        self.tag_visible    = 0.0
        self.tag_dist_norm  = 0.0
        self.tag_angle_norm = 0.0

    def _load_ppo(self):
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
        )
        config_path = os.path.join(project_root, 'current_eval_model.txt')
        if os.path.exists(config_path):
            with open(config_path) as f:
                model_path = f.read().strip()
        else:
            model_path = os.path.join(project_root, 'decentralized_optA_v1')
        print(f"[ROBOT] Loading model: {model_path}")
        self.ppo = PPO.load(model_path)
        print("[ROBOT] Model loaded. Running inference locally.")

    # =========================================================================
    # deepbots interface
    # =========================================================================

    def create_message(self):
        """
        Same 17-float message as training (sent to supervisor for tag mechanics).
        Also handles pheromone decay/receive/broadcast each step.
        """
        return super().create_message()

    def use_message_data(self, message):
        """
        Receive [tag_visible, tag_dist_norm, tag_angle_norm, pickup_signal].
        Assemble 18D obs, run PPO, apply P1-P4 overrides onboard, drive motors.
        All overrides (P1 wall-escape included) run locally using GPS + IMU.
        """
        if self.ppo is None:
            self._load_ppo()

        if not message or len(message) < 4:
            return

        try:
            self.tag_visible    = float(message[0])
            self.tag_dist_norm  = float(message[1])
            self.tag_angle_norm = float(message[2])
            pickup_signal       = float(message[3])
        except (ValueError, IndexError):
            return

        # Handle pickup/deposit events (mirrors base class logic)
        if pickup_signal > 0.0 and not self.carrying:
            self.carrying = True
            gps = self.gps.getValues()
            self.phero_memory = {
                "hotspot":  (gps[0], gps[1]),
                "strength": pickup_signal,
                "ttl":      INITIAL_TTL,
            }
        elif pickup_signal < 0.0 and self.carrying:
            self.carrying = False

        # Assemble 18D obs (same layout as training)
        prox = [s.getValue() / 4096.0 for s in self.ps]
        obs  = self._obs_components()

        obs_arr = np.array(
            prox + [
                self.tag_visible,               # [8]
                self.tag_dist_norm,             # [9]
                self.tag_angle_norm,            # [10]
                1.0 if self.carrying else 0.0,  # [11]
                obs["base_dist_norm"],          # [12]
                obs["base_angle_norm"],         # [13]
                obs["phero_known"],             # [14]
                obs["phero_dist_norm"],         # [15]
                obs["phero_angle_norm"],        # [16]
                obs["phero_strength"],          # [17]
            ],
            dtype=np.float32
        ).reshape(1, 18)

        # PPO inference (deterministic — no exploration noise)
        action, _ = self.ppo.predict(obs_arr, deterministic=True)
        left_cmd  = float(action[0][0])
        right_cmd = float(action[0][1])

        # Apply P1-P4 overrides fully onboard (GPS + IMU — no supervisor dependency)
        left_cmd, right_cmd = self._apply_overrides(
            left_cmd, right_cmd,
            obs["pos_x"], obs["pos_y"]
        )

        # Drive motors
        max_speed    = 6.28
        scale_factor = 6.0
        self.left_motor.setVelocity(
            max(min(left_cmd  * scale_factor, max_speed), -max_speed))
        self.right_motor.setVelocity(
            max(min(right_cmd * scale_factor, max_speed), -max_speed))

    # =========================================================================
    # P1-P4 overrides (fully onboard — GPS + IMU, no supervisor dependency)
    # =========================================================================

    def _apply_overrides(self, left, right, pos_x, pos_y):
        gps       = self.gps.getValues()
        yaw       = self.imu.getRollPitchYaw()[2]
        fwd       = [math.cos(yaw), math.sin(yaw)]
        robot_pos = [gps[0], gps[1]]
        wall_dist = 2.5 - max(abs(pos_x), abs(pos_y))

        # P1: wall escape — 0.6m threshold catches backward-driving robots before
        # they physically hit the wall (the Phase-2 failure mode in eval logs).
        if wall_dist < 0.6:
            return self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=4.0)

        # P2: return to base when carrying
        if self.carrying:
            return self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=2.5)

        # P3: base avoidance when not carrying
        # 0.3m threshold — just above deposit zone (0.25m).
        # Multiplier 5.0 pushes robot to 1.2m (inside active exploration zone 0.8–2.4m),
        # matching the training supervisor so eval behaviour is consistent.
        dist_to_base = math.sqrt(pos_x ** 2 + pos_y ** 2)
        if dist_to_base < 0.3:
            target = [pos_x * 5.0, pos_y * 5.0]
            return self._steer_to(robot_pos, fwd, target, gain=3.0)

        # P4: tag-seek — steer toward tag using supervisor-provided angle
        if self.tag_visible > 0.5:
            angle = self.tag_angle_norm * math.pi
            turn  = max(-1.0, min(1.0, 3.0 * angle / math.pi))
            l     = max(-1.0, min(1.0, 1.0 - turn))
            r     = max(-1.0, min(1.0, 1.0 + turn))
            m     = max(abs(l), abs(r))
            if m > 1.0:
                l /= m
                r /= m
            return [l, r]

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
