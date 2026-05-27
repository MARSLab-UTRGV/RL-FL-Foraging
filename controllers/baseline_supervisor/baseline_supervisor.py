import os
import math
import time
from controller import Supervisor

# =============================================================================
# BASELINE SUPERVISOR
#
# Thin tag-mechanics shim for the hand-coded CPFA baseline.
# No learning, no episode resets, no curriculum — runs indefinitely.
#
# Roles:
#   1. Camera simulation: detect which tags are in each robot's FOV
#   2. Tag mechanics: detect pickups (robot within 0.15m of tag),
#      detect deposits (robot within 0.25m of base while carrying),
#      hide picked-up tags (z=-10)
#   3. Send per-step message to each robot (4 floats):
#        [tag_visible, tag_dist_norm, tag_angle_norm, pickup_signal]
#        pickup_signal > 0  → pickup; encodes pheromone strength (0.2–1.0)
#        pickup_signal = -1 → deposit confirmed
#        pickup_signal = 0  → normal step
#   4. Log stats every 500 steps
#
# Density counted BEFORE hiding tag (matches centralized CPFA).
#
# Usage: open worlds/baseline.wbt in Webots — supervisor runs as <extern>.
# =============================================================================

TAG_SEEK_RANGE = 1.0
FOV_HALF_ANGLE = 1.2
DENSITY_RADIUS = 0.5
NUM_ROBOTS     = 4
NUM_TAGS       = 64


class BaselineSupervisor:

    def __init__(self):
        self.supervisor = Supervisor()
        self.timestep   = int(self.supervisor.getBasicTimeStep())

        self.robot_nodes = [
            self.supervisor.getFromDef(f"ROBOT{i+1}") for i in range(NUM_ROBOTS)
        ]
        self.tag_nodes = [
            self.supervisor.getFromDef(f"APRILTAG_{i+1}") for i in range(NUM_TAGS)
        ]

        # Supervisor ↔ robot communication (channels 1–4)
        self.emitters  = []
        self.receivers = []
        for i in range(NUM_ROBOTS):
            self.emitters.append(self.supervisor.getDevice(f"emitter{i+1}"))
            recv = self.supervisor.getDevice(f"receiver{i+1}")
            recv.enable(self.timestep)
            self.receivers.append(recv)

        self.robot_gps = [[0.0, 0.0] for _ in range(NUM_ROBOTS)]
        self.carrying  = [False] * NUM_ROBOTS

        self.total_pickups  = 0
        self.total_deposits = 0
        self.step           = 0
        self.start_time     = time.time()

        # Supervisor log (step summaries + individual pickup/deposit events)
        _project_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
        _log_dir = os.path.join(_project_root, 'logs', 'baseline')
        os.makedirs(_log_dir, exist_ok=True)
        self._log_path = os.path.join(_log_dir, 'supervisor_log.txt')

    # =========================================================================
    # Visual range circles
    # =========================================================================

    def _create_comm_circles(self):
        """
        Dynamically inject one hollow ring per robot into the scene.
        Each ring marks the 0.25m pheromone-emitter broadcast radius.
        Colors: R1=blue, R2=green, R3=red, R4=yellow.
        """
        N      = 36
        R_out  = 0.29
        R_in   = 0.21
        colors = [
            (0.2, 0.6, 1.0),
            (0.2, 0.9, 0.2),
            (1.0, 0.3, 0.1),
            (1.0, 0.85, 0.0),
        ]
        root_ch = self.supervisor.getRoot().getField('children')
        self.comm_circles = []

        for i, (cr, cg, cb) in enumerate(colors):
            outer = [(R_out * math.cos(2 * math.pi * j / N),
                      R_out * math.sin(2 * math.pi * j / N)) for j in range(N)]
            inner = [(R_in  * math.cos(2 * math.pi * j / N),
                      R_in  * math.sin(2 * math.pi * j / N)) for j in range(N)]
            pts = " ".join(f"{x:.3f} {y:.3f} 0" for x, y in outer + inner)
            idx = " ".join(
                f"{j} {(j+1)%N} {(j+1)%N+N} {j+N} -1" for j in range(N)
            )
            node_str = (
                f'DEF COMM_CIRCLE_{i+1} Solid {{\n'
                f'  translation 0 0 0.005\n'
                f'  children [\n'
                f'    Shape {{\n'
                f'      appearance Appearance {{\n'
                f'        material Material {{\n'
                f'          diffuseColor {cr} {cg} {cb}\n'
                f'          emissiveColor {cr} {cg} {cb}\n'
                f'          transparency 0.35\n'
                f'        }}\n'
                f'      }}\n'
                f'      geometry IndexedFaceSet {{\n'
                f'        coord Coordinate {{\n'
                f'          point [ {pts} ]\n'
                f'        }}\n'
                f'        coordIndex [ {idx} ]\n'
                f'      }}\n'
                f'    }}\n'
                f'  ]\n'
                f'}}'
            )
            root_ch.importMFNodeFromString(-1, node_str)
            self.comm_circles.append(
                self.supervisor.getFromDef(f'COMM_CIRCLE_{i+1}')
            )

    def _update_comm_circles(self):
        """Move each ring to its robot's current physics position."""
        for i in range(NUM_ROBOTS):
            pos = self.robot_nodes[i].getPosition()
            self.comm_circles[i].getField('translation').setSFVec3f(
                [pos[0], pos[1], 0.005]
            )

    # =========================================================================
    # Communication
    # =========================================================================

    def _collect_robot_states(self):
        for i in range(NUM_ROBOTS):
            if self.receivers[i].getQueueLength() > 0:
                msg = self.receivers[i].getString()
                self.receivers[i].nextPacket()
                try:
                    vals = [float(x) for x in msg.split(',')]
                    if len(vals) >= 17:
                        self.robot_gps[i] = [vals[15], vals[16]]
                except (ValueError, IndexError):
                    pass

    def _count_density(self, pos):
        """Raw tag count within 0.5m — counted before hiding, matches centralized exactly."""
        return sum(
            1 for t in self.tag_nodes
            if t.getPosition()[2] >= 0 and
               math.sqrt((t.getPosition()[0] - pos[0])**2 +
                         (t.getPosition()[1] - pos[1])**2) < 0.5
        )

    def _tag_obs(self, robot_idx):
        """Camera simulation: nearest visible tag within FOV."""
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

    # =========================================================================
    # Main loop
    # =========================================================================

    def run_step(self):
        self._collect_robot_states()
        self._update_comm_circles()

        for i in range(NUM_ROBOTS):
            gps_x, gps_y  = self.robot_gps[i]
            pickup_signal = 0.0

            if not self.carrying[i]:
                for tag_node in self.tag_nodes:
                    tag_pos = tag_node.getPosition()
                    if tag_pos[2] < 0:
                        continue
                    dx = tag_pos[0] - gps_x
                    dy = tag_pos[1] - gps_y
                    if math.sqrt(dx * dx + dy * dy) < 0.15:
                        self.carrying[i] = True
                        pickup_signal = float(self._count_density([gps_x, gps_y]))
                        tag_node.getField("translation").setSFVec3f([0, 0, -10])
                        self.total_pickups += 1
                        msg_ev = (f"[PICKUP]  robot{i+1} | "
                                  f"density={pickup_signal:.0f} | "
                                  f"Total: {self.total_pickups}")
                        print(msg_ev)
                        with open(self._log_path, 'a') as f:
                            f.write(msg_ev + '\n')
                        break
            else:
                if math.sqrt(gps_x**2 + gps_y**2) < 0.25:
                    self.carrying[i] = False
                    pickup_signal      = -1.0
                    self.total_deposits += 1
                    msg_ev = f"[DEPOSIT] robot{i+1} | Total: {self.total_deposits}"
                    print(msg_ev)
                    with open(self._log_path, 'a') as f:
                        f.write(msg_ev + '\n')

            tag_vis, tag_dist_n, tag_angle_n = self._tag_obs(i)
            msg = f"{tag_vis},{tag_dist_n},{tag_angle_n},{pickup_signal}".encode('utf-8')
            self.emitters[i].send(msg)

        if self.supervisor.step(self.timestep) == -1:
            return False
        return True

    def run(self):
        print("=" * 60)
        print("BASELINE SUPERVISOR — hand-coded CPFA, no learning")
        print("No episode resets. Runs until Webots is closed.")
        print("=" * 60 + "\n")

        self._create_comm_circles()
        print(f"[SUPERVISOR] Logging to: {self._log_path}\n")

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
                    d2base = math.sqrt(rpos[0]**2 + rpos[1]**2)
                    carry  = self.carrying[ri]
                    tag_vis, _, _ = self._tag_obs(ri)

                    if wall_d < 0.35:                  mode = "WALL_ESC"
                    elif not carry and d2base < 0.25:  mode = "BASE_ESC"
                    elif carry:                        mode = "RTB"
                    else:                              mode = "EXPLORE"

                    log_msg += (
                        f"R{ri+1}[{mode:8s}]: "
                        f"carry={1 if carry else 0} | "
                        f"base={d2base:.2f} | "
                        f"tag_vis={tag_vis:.0f} | "
                        f"wall={wall_d:.2f}\n"
                    )
                print(log_msg)
                with open(self._log_path, "a") as f:
                    f.write(log_msg)


if __name__ == "__main__":
    sup = BaselineSupervisor()
    sup.run()
