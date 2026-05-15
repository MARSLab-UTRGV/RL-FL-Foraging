import os
import math
import time
from controller import Supervisor

# =============================================================================
# LIGHTWEIGHT DECENTRALIZED EVAL SUPERVISOR  (CoRL 2026)
#
# This supervisor does NOT run PPO. Each robot runs PPO locally.
# The supervisor's only roles are:
#   1. Camera simulation: detect which tags are in each robot's FOV
#   2. Tag mechanics: detect pickups (robot near tag), detect deposits (robot
#      near base while carrying), hide picked-up tags
#   3. Send per-step message to each robot (4 floats):
#        [tag_visible, tag_dist_norm, tag_angle_norm, pickup_signal]
#   4. Log per-robot statistics (mode, carry, base_dist, tag_vis, wall_dist)
#
# Supervisor → Robot: [tag_visible, tag_dist_norm, tag_angle_norm, pickup_signal]
#   pickup_signal > 0  → pickup event; value = pheromone strength (0.2–1.0)
#   pickup_signal < 0  → deposit confirmed
#   pickup_signal = 0  → normal step
#
# Robot → Supervisor: 17 floats (prox×8, carrying, base_dist, base_angle,
#                                 phero×4, gps_x, gps_y)
#
# P1-P4 overrides run fully onboard in each robot (GPS + IMU — no supervisor
# dependency). This is proper CTDE: supervisor is only a thin tag-mechanics shim.
#
# Usage (from project root, Webots already open on eval_decentralized.wbt):
#   cd controllers/eval_decentralized
#   python3 eval_decentralized.py
# =============================================================================

TAG_SEEK_RANGE = 1.0
FOV_HALF_ANGLE = 1.2
DENSITY_RADIUS = 0.5
DENSITY_MAX    = 5
NUM_ROBOTS     = 4
NUM_TAGS       = 64


class DecentralizedEvalSupervisor:

    def __init__(self):
        self.supervisor = Supervisor()
        self.timestep   = int(self.supervisor.getBasicTimeStep())

        # Webots nodes
        self.robot_nodes = [
            self.supervisor.getFromDef(f"ROBOT{i+1}") for i in range(NUM_ROBOTS)
        ]
        self.tag_nodes = [
            self.supervisor.getFromDef(f"APRILTAG_{i+1}") for i in range(NUM_TAGS)
        ]

        # Supervisor ↔ robot communication (channels 1-4)
        self.emitters  = []
        self.receivers = []
        for i in range(NUM_ROBOTS):
            self.emitters.append(self.supervisor.getDevice(f"emitter{i+1}"))
            recv = self.supervisor.getDevice(f"receiver{i+1}")
            recv.enable(self.timestep)
            self.receivers.append(recv)

        # Robot GPS and pheromone state parsed from messages
        self.robot_gps   = [[0.0, 0.0] for _ in range(NUM_ROBOTS)]
        self._robot_phero = [[0.0, 0.0] for _ in range(NUM_ROBOTS)]  # [phero_known, phero_strength]

        # Carrying state (supervisor is authoritative for pickup/deposit logic)
        self.carrying = [False] * NUM_ROBOTS

        # Stats
        self.total_pickups  = 0
        self.total_deposits = 0
        self.step           = 0
        self.start_time     = time.time()

    # =========================================================================
    # Communication
    # =========================================================================

    def _collect_robot_states(self):
        """Parse GPS from 17-float robot message."""
        for i in range(NUM_ROBOTS):
            if self.receivers[i].getQueueLength() > 0:
                msg = self.receivers[i].getString()
                self.receivers[i].nextPacket()
                try:
                    vals = [float(x) for x in msg.split(',')]
                    if len(vals) >= 17:
                        self.robot_gps[i]    = [vals[15], vals[16]]
                        self._robot_phero[i] = [vals[11], vals[12]]  # site_known, phero_known
                except (ValueError, IndexError):
                    pass

    def _tag_obs(self, robot_idx):
        """
        Simulate camera: find nearest visible tag within TAG_SEEK_RANGE and FOV.
        Returns (tag_visible, tag_dist_norm, tag_angle_norm).
        """
        pos       = self.robot_gps[robot_idx]
        robot_rot = self.robot_nodes[robot_idx].getOrientation()
        fwd       = [robot_rot[0], robot_rot[3]]

        tag_visible = 0.0
        tag_dist    = 0.0
        tag_angle   = 0.0
        min_dist    = float('inf')

        if not self.carrying[robot_idx]:
            for tag_node in self.tag_nodes:
                tag_pos = tag_node.getPosition()
                if tag_pos[2] < 0:
                    continue
                dx   = tag_pos[0] - pos[0]
                dy   = tag_pos[1] - pos[1]
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

        return tag_visible, tag_dist / TAG_SEEK_RANGE, tag_angle / math.pi

    def _pickup_strength(self, pos):
        """Pheromone strength based on tag density near pickup location."""
        nearby = sum(
            1 for t in self.tag_nodes
            if t.getPosition()[2] >= 0 and
               math.sqrt((t.getPosition()[0] - pos[0])**2 +
                         (t.getPosition()[1] - pos[1])**2) <= DENSITY_RADIUS
        )
        return 0.2 + 0.8 * min(nearby / DENSITY_MAX, 1.0)

    # =========================================================================
    # Main loop
    # =========================================================================

    def run_step(self):
        self._collect_robot_states()

        for i in range(NUM_ROBOTS):
            gps_x, gps_y  = self.robot_gps[i]
            pickup_signal = 0.0

            if not self.carrying[i]:
                # Check for tag pickup (robot within 0.15 m of a tag)
                for tag_node in self.tag_nodes:
                    tag_pos = tag_node.getPosition()
                    if tag_pos[2] < 0:
                        continue
                    dx = tag_pos[0] - gps_x
                    dy = tag_pos[1] - gps_y
                    if math.sqrt(dx * dx + dy * dy) < 0.15:
                        self.carrying[i] = True
                        tag_node.getField("translation").setSFVec3f([0, 0, -10])
                        pickup_signal      = self._pickup_strength([gps_x, gps_y])
                        self.total_pickups += 1
                        print(f"[PICKUP]  robot_{i+1} | Total: {self.total_pickups}")
                        break
            else:
                # Check for deposit (robot within 0.25 m of base while carrying)
                if math.sqrt(gps_x**2 + gps_y**2) < 0.25:
                    self.carrying[i] = False
                    pickup_signal      = -1.0
                    self.total_deposits += 1
                    print(f"[DEPOSIT] robot_{i+1} | Total: {self.total_deposits}")

            # Compute tag obs (camera simulation)
            tag_vis, tag_dist_n, tag_angle_n = self._tag_obs(i)

            # Send [tag_visible, tag_dist_norm, tag_angle_norm, pickup_signal]
            msg = f"{tag_vis},{tag_dist_n},{tag_angle_n},{pickup_signal}".encode('utf-8')
            self.emitters[i].send(msg)

        if self.supervisor.step(self.timestep) == -1:
            return False
        return True

    def run(self):
        print("=" * 60)
        print("DECENTRALIZED EVAL SUPERVISOR (lightweight)")
        print("Robots run PPO locally — supervisor handles tag mechanics only")
        print("=" * 60 + "\n")

        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "eval_decentralized_log.txt")

        while True:
            if not self.run_step():
                break
            self.step += 1

            if self.step % 500 == 0:
                elapsed_min      = (time.time() - self.start_time) / 60.0
                deposits_per_min = (self.total_deposits / elapsed_min
                                    if elapsed_min > 0.01 else 0.0)
                sim_time_min     = (self.step * self.timestep / 1000.0) / 60.0

                log_msg = (
                    f"\n{'='*60}\n"
                    f"Step {self.step} | Pickups: {self.total_pickups} | "
                    f"Deposits: {self.total_deposits}\n"
                    f"Sim: {sim_time_min:.1f} min | Wall: {elapsed_min:.1f} min | "
                    f"Rate: {deposits_per_min:.2f} tags/min\n"
                    f"{'='*60}\n"
                )
                for ri in range(NUM_ROBOTS):
                    rpos   = self.robot_nodes[ri].getPosition()
                    wall_d = 2.5 - max(abs(rpos[0]), abs(rpos[1]))
                    d2base = math.sqrt(rpos[0] ** 2 + rpos[1] ** 2)
                    carry  = self.carrying[ri]
                    tag_vis, tag_dist_n, _ = self._tag_obs(ri)

                    if wall_d < 0.35:              mode = "WALL_ESC"
                    elif not carry and d2base < 0.25: mode = "BASE_ESC"
                    elif carry:                    mode = "RTB"
                    else:                          mode = "PPO"

                    site_k  = self._robot_phero[ri][0]
                    phero_k = self._robot_phero[ri][1]
                    log_msg += (
                        f"R{ri+1}[{mode:8s}]: "
                        f"carry={1 if carry else 0} | "
                        f"base={d2base:.2f} | "
                        f"tag_vis={tag_vis:.0f} td={tag_dist_n:.2f} | "
                        f"site={site_k:.0f} phero={phero_k:.0f} | "
                        f"wall={wall_d:.2f}\n"
                    )
                print(log_msg)
                with open(log_path, "a") as f:
                    f.write(log_msg)


if __name__ == "__main__":
    import sys

    # Resolve model path and write config file BEFORE connecting to Webots.
    # Robot controllers load PPO lazily on first message, so the file is
    # guaranteed to exist by the time they read it.
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if len(sys.argv) > 1:
        _model_path = os.path.abspath(sys.argv[1])
        if _model_path.endswith('.zip'):
            _model_path = _model_path[:-4]
    else:
        _model_path = os.path.join(_project_root, 'decentralized_optA_v1')

    _config_path = os.path.join(_project_root, 'current_eval_model.txt')
    with open(_config_path, 'w') as _f:
        _f.write(_model_path)
    print(f"[SUPERVISOR] Model path: {_model_path}")

    sup = DecentralizedEvalSupervisor()
    sup.run()
