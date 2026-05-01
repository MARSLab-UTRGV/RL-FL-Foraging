from deepbots.robots.controllers.csv_robot import CSVRobot
import math

# =============================================================================
# HYBRID OPTION-A ROBOT CONTROLLER  (CoRL decentralized execution)
#
# Each robot computes its own observation components autonomously:
#   GPS          → base distance + angle (no supervisor position query)
#   InertialUnit → heading for angle computation
#   phero_receiver (channel 10, range-limited by Webots emitter) → comm signal
#
# Message to supervisor  (17 floats):
#   [0:8]  prox sensors (normalized /4096)
#   [8]    carrying flag (0/1)
#   [9]    base_dist_norm       (GPS dist to origin / 3.5)
#   [10]   base_angle_norm      (signed angle to base / π)
#   [11]   phero_known          (1 if active hotspot, else 0)
#   [12]   phero_dist_norm      (dist to hotspot / 3.5, clamped 0–1)
#   [13]   phero_angle_norm     (signed angle to hotspot / π)
#   [14]   phero_strength       (0–1)
#   [15]   gps_x                (raw, for supervisor P1-P4 steering)
#   [16]   gps_y
#
# Message from supervisor  (3 floats):
#   [0]  left_motor  ([-1, 1])
#   [1]  right_motor ([-1, 1])
#   [2]  pickup_signal:
#          > 0  → pickup event; value = pheromone strength (0.2–1.0)
#          < 0  → deposit confirmed at base
#            0  → normal step, no event
#
# Pheromone P2P broadcast (channel 10, range enforced by Webots emitter to 2 m):
#   "hotspot_x,hotspot_y,strength"
# =============================================================================

PHEROMONE_MIN   = 0.01
PHEROMONE_DECAY = math.exp(-0.003)   # half-life ~230 steps
INITIAL_TTL     = 400
BASE_X          = 0.0
BASE_Y          = 0.0                # base station at arena origin


class EpuckDecentralized(CSVRobot):

    def __init__(self):
        super().__init__()
        self.time_step = int(self.getBasicTimeStep())

        # ── Proximity sensors ─────────────────────────────────────────
        self.ps = []
        for i in range(8):
            s = self.getDevice(f'ps{i}')
            s.enable(self.time_step)
            self.ps.append(s)

        # ── Camera (supervisor uses it for P4 simulation) ─────────────
        self.camera = self.getDevice('camera')
        if self.camera:
            self.camera.enable(self.time_step)

        # ── Differential-drive motors ─────────────────────────────────
        self.left_motor  = self.getDevice('left wheel motor')
        self.right_motor = self.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # ── GPS for absolute position (accuracy=0 → perfect in sim) ──
        self.gps = self.getDevice('gps')
        self.gps.enable(self.time_step)

        # ── InertialUnit for heading ──────────────────────────────────
        self.imu = self.getDevice('inertial_unit')
        self.imu.enable(self.time_step)

        # ── Pheromone peer-to-peer (channel 10, Webots enforces 2 m) ─
        self.phero_emitter  = self.getDevice('phero_emitter')
        self.phero_receiver = self.getDevice('phero_receiver')
        self.phero_receiver.enable(self.time_step)

        # ── Internal state ────────────────────────────────────────────
        self.carrying     = False
        self.phero_memory = {"hotspot": None, "strength": 0.0, "ttl": 0}

    # =========================================================================
    # Pheromone management  (fully onboard — no supervisor involvement)
    # =========================================================================

    def _decay_pheromone(self):
        mem = self.phero_memory
        if mem["hotspot"] is not None:
            mem["strength"] *= PHEROMONE_DECAY
            mem["ttl"]      -= 1
            if mem["strength"] < PHEROMONE_MIN or mem["ttl"] <= 0:
                self.phero_memory = {"hotspot": None, "strength": 0.0, "ttl": 0}

    def _receive_pheromone(self):
        """
        Read P2P pheromone from nearby robots.
        Webots Emitter range=2.0 m enforces the 2 m limit — no distance check needed here.
        Accept-if-stronger rule prevents stale signals from overwriting fresh ones.
        """
        while self.phero_receiver.getQueueLength() > 0:
            try:
                msg   = self.phero_receiver.getString()
                parts = [float(x) for x in msg.split(',')]
                if len(parts) >= 3:
                    hx, hy, strength = parts[0], parts[1], parts[2]
                    if strength >= PHEROMONE_MIN and strength > self.phero_memory["strength"]:
                        self.phero_memory = {
                            "hotspot":  (hx, hy),
                            "strength": strength,
                            "ttl":      INITIAL_TTL,
                        }
            except (ValueError, IndexError):
                pass
            finally:
                self.phero_receiver.nextPacket()

    def _broadcast_pheromone(self):
        """Broadcast active hotspot. Webots emitter range=2.0 limits delivery to 2 m."""
        mem = self.phero_memory
        if mem["hotspot"] is not None and mem["strength"] >= PHEROMONE_MIN:
            hx, hy = mem["hotspot"]
            self.phero_emitter.send(
                f"{hx},{hy},{mem['strength']}".encode('utf-8'))

    # =========================================================================
    # Self-observed obs components  (GPS + IMU, no supervisor API)
    # =========================================================================

    def _obs_components(self):
        """
        Compute base-navigation and pheromone obs from onboard sensors only.
        At execution time this runs entirely on the robot — supervisor not needed.
        """
        gps   = self.gps.getValues()
        pos_x = gps[0]
        pos_y = gps[1]
        yaw   = self.imu.getRollPitchYaw()[2]   # rotation around Z (up axis)
        fwd   = [math.cos(yaw), math.sin(yaw)]  # forward unit vector, world frame

        # ── Base direction ────────────────────────────────────────────
        bdx          = BASE_X - pos_x
        bdy          = BASE_Y - pos_y
        dist_to_base = math.sqrt(bdx * bdx + bdy * bdy)
        if dist_to_base > 0.001:
            b_norm        = [bdx / dist_to_base, bdy / dist_to_base]
            dot           = max(min(fwd[0]*b_norm[0] + fwd[1]*b_norm[1], 1.0), -1.0)
            cross         = fwd[0]*b_norm[1] - fwd[1]*b_norm[0]
            angle_to_base = math.copysign(math.acos(dot), cross)
        else:
            angle_to_base = 0.0

        # ── Pheromone waypoint ────────────────────────────────────────
        phero_known = 0.0
        phero_dist  = 0.0
        phero_angle = 0.0
        phero_str   = 0.0

        if not self.carrying:
            mem = self.phero_memory
            if mem["hotspot"] is not None and mem["strength"] >= PHEROMONE_MIN:
                hx, hy = mem["hotspot"]
                hdx    = hx - pos_x
                hdy    = hy - pos_y
                h_dist = math.sqrt(hdx * hdx + hdy * hdy)
                if h_dist > 0.001:
                    h_norm  = [hdx / h_dist, hdy / h_dist]
                    h_dot   = max(min(fwd[0]*h_norm[0] + fwd[1]*h_norm[1], 1.0), -1.0)
                    h_cross = fwd[0]*h_norm[1] - fwd[1]*h_norm[0]
                    phero_known = 1.0
                    phero_dist  = min(h_dist / 3.5, 1.0)
                    phero_angle = math.copysign(math.acos(h_dot), h_cross) / math.pi
                    phero_str   = mem["strength"]

        return {
            "pos_x":            pos_x,
            "pos_y":            pos_y,
            "base_dist_norm":   min(dist_to_base / 3.5, 1.0),
            "base_angle_norm":  angle_to_base / math.pi,
            "phero_known":      phero_known,
            "phero_dist_norm":  phero_dist,
            "phero_angle_norm": phero_angle,   # already ÷ π
            "phero_strength":   phero_str,
        }

    # =========================================================================
    # deepbots interface
    # =========================================================================

    def create_message(self):
        """
        Each step (before supervisor processes):
          1. Decay own pheromone memory
          2. Receive P2P pheromone from nearby robots
          3. Broadcast own hotspot to nearby robots
          4. Compute self-observed obs components
          5. Return 17-value list to supervisor
        """
        self._decay_pheromone()
        self._receive_pheromone()
        self._broadcast_pheromone()

        prox = [s.getValue() / 4096.0 for s in self.ps]
        obs  = self._obs_components()

        return prox + [
            1.0 if self.carrying else 0.0,  # [8]
            obs["base_dist_norm"],           # [9]
            obs["base_angle_norm"],          # [10]
            obs["phero_known"],              # [11]
            obs["phero_dist_norm"],          # [12]
            obs["phero_angle_norm"],         # [13]
            obs["phero_strength"],           # [14]
            obs["pos_x"],                    # [15]  raw — supervisor uses for P1-P4
            obs["pos_y"],                    # [16]
        ]

    def use_message_data(self, message):
        """
        Receive [left, right, pickup_signal] from supervisor each step.

        pickup_signal > 0  → tag pickup; value = pheromone strength → create hotspot
        pickup_signal < 0  → deposit confirmed → clear carrying flag
        pickup_signal = 0  → normal step
        """
        if not message:
            return
        try:
            left          = float(message[0])
            right         = float(message[1])
            pickup_signal = float(message[2]) if len(message) > 2 else 0.0

            # Motor commands
            max_speed    = 6.28
            scale_factor = 6.0
            self.left_motor.setVelocity(
                max(min(left  * scale_factor, max_speed), -max_speed))
            self.right_motor.setVelocity(
                max(min(right * scale_factor, max_speed), -max_speed))

            # Pickup event: create pheromone hotspot at current GPS position
            if pickup_signal > 0.0 and not self.carrying:
                self.carrying = True
                gps = self.gps.getValues()
                self.phero_memory = {
                    "hotspot":  (gps[0], gps[1]),
                    "strength": pickup_signal,
                    "ttl":      INITIAL_TTL,
                }

            # Deposit confirmed: clear carrying; keep pheromone to navigate back
            elif pickup_signal < 0.0 and self.carrying:
                self.carrying = False

        except (ValueError, IndexError):
            pass


if __name__ == '__main__':
    robot = EpuckDecentralized()
    robot.run()
