import csv
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
#   4. Log per-robot statistics
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
# =============================================================================

TAG_SEEK_RANGE = 1.0
FOV_HALF_ANGLE = 1.2
DENSITY_RADIUS = 0.5

# Arena size → (num_tags, arena_half_m)
ARENA_CONFIGS = {
    '5x5':   (64,  2.5),
    '7x7':   (128, 3.5),
    '9x9':   (208, 4.5),
    '12x12': (368, 6.0),
}


class DecentralizedEvalSupervisor:

    def __init__(self, arena_size='5x5', num_robots=4, num_tags_override=None):
        self.arena_size = arena_size
        self.num_robots = num_robots
        num_tags, self.arena_half = ARENA_CONFIGS.get(arena_size, ARENA_CONFIGS['5x5'])
        if num_tags_override is not None:
            num_tags = num_tags_override

        self.supervisor = Supervisor()
        self.timestep   = int(self.supervisor.getBasicTimeStep())

        # Webots nodes
        self.robot_nodes = [
            self.supervisor.getFromDef(f"ROBOT{i+1}") for i in range(self.num_robots)
        ]
        self.tag_nodes = [
            self.supervisor.getFromDef(f"APRILTAG_{i+1}") for i in range(num_tags)
        ]

        # Hide any tags in the world beyond num_tags (supports partial-tag experiments)
        _i = num_tags
        while True:
            _node = self.supervisor.getFromDef(f"APRILTAG_{_i+1}")
            if _node is None:
                break
            _node.getField("translation").setSFVec3f([0.0, 0.0, -10.0])
            _i += 1

        # Supervisor ↔ robot communication (channels 1-N)
        self.emitters  = []
        self.receivers = []
        for i in range(self.num_robots):
            self.emitters.append(self.supervisor.getDevice(f"emitter{i+1}"))
            recv = self.supervisor.getDevice(f"receiver{i+1}")
            recv.enable(self.timestep)
            self.receivers.append(recv)

        # Robot GPS and pheromone state parsed from messages
        self.robot_gps    = [[0.0, 0.0] for _ in range(self.num_robots)]
        self._robot_phero = [[0.0, 0.0] for _ in range(self.num_robots)]

        # Carrying state (supervisor is authoritative for pickup/deposit logic)
        self.carrying = [False] * self.num_robots

        # Stats
        self.total_pickups  = 0
        self.total_deposits = 0
        self.step           = 0
        self.start_time     = time.time()

        # Supervisor log file
        _project_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
        _run_cfg  = os.path.join(_project_root, 'current_eval_run.txt')
        _run_name = open(_run_cfg).read().strip() if os.path.exists(_run_cfg) else 'eval'
        _r_suffix = f'_{self.num_robots}r' if self.num_robots != 4 else ''
        _log_dir  = os.path.join(_project_root, 'logs',
                                 f'eval_{_run_name}_{arena_size}{_r_suffix}')
        os.makedirs(_log_dir, exist_ok=True)
        self._log_path = os.path.join(_log_dir, 'supervisor_log.txt')

    # =========================================================================
    # Communication
    # =========================================================================

    def _collect_robot_states(self):
        """Parse GPS from 17-float robot message."""
        for i in range(self.num_robots):
            if self.receivers[i].getQueueLength() > 0:
                msg = self.receivers[i].getString()
                self.receivers[i].nextPacket()
                try:
                    vals = [float(x) for x in msg.split(',')]
                    if len(vals) >= 17:
                        self.robot_gps[i]    = [vals[15], vals[16]]
                        self._robot_phero[i] = [vals[11], vals[12]]
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

    def _count_density(self, pos):
        """Raw tag count within 0.5m — counted after hiding, reflects remaining tags."""
        return sum(
            1 for t in self.tag_nodes
            if t.getPosition()[2] >= 0 and
               math.sqrt((t.getPosition()[0] - pos[0])**2 +
                         (t.getPosition()[1] - pos[1])**2) < 0.5
        )

    # =========================================================================
    # Main loop
    # =========================================================================

    def run_step(self):
        self._collect_robot_states()

        for i in range(self.num_robots):
            rpos         = self.robot_nodes[i].getPosition()
            gps_x, gps_y = rpos[0], rpos[1]
            pickup_signal = 0.0
            pickup_tag_x  = 0.0
            pickup_tag_y  = 0.0

            if not self.carrying[i]:
                for tag_node in self.tag_nodes:
                    tag_pos = tag_node.getPosition()
                    if tag_pos[2] < 0:
                        continue
                    dx = tag_pos[0] - gps_x
                    dy = tag_pos[1] - gps_y
                    if math.sqrt(dx * dx + dy * dy) < 0.15:
                        self.carrying[i] = True
                        pickup_tag_x = tag_pos[0]
                        pickup_tag_y = tag_pos[1]
                        tag_node.getField("translation").setSFVec3f([0, 0, -10])
                        pickup_signal      = float(self._count_density([gps_x, gps_y]))
                        self.total_pickups += 1
                        msg_ev = (f"[PICKUP]  robot{i+1} | "
                                  f"tag=({pickup_tag_x:.2f},{pickup_tag_y:.2f}) | "
                                  f"density={pickup_signal:.0f} | "
                                  f"Total: {self.total_pickups}")
                        print(msg_ev)
                        with open(self._log_path, 'a') as _f:
                            _f.write(msg_ev + '\n')
                        break
            else:
                if math.sqrt(gps_x**2 + gps_y**2) < 0.25:
                    self.carrying[i] = False
                    pickup_signal      = -1.0
                    self.total_deposits += 1
                    msg_ev = f"[DEPOSIT] robot{i+1} | Total: {self.total_deposits}"
                    print(msg_ev)
                    with open(self._log_path, 'a') as _f:
                        _f.write(msg_ev + '\n')

            if pickup_signal > 0:
                msg = f"{pickup_tag_x},{pickup_tag_y},0.0,{pickup_signal}".encode('utf-8')
            else:
                tag_vis, tag_dist_n, tag_angle_n = self._tag_obs(i)
                msg = f"{tag_vis},{tag_dist_n},{tag_angle_n},{pickup_signal}".encode('utf-8')
            self.emitters[i].send(msg)

        if self.supervisor.step(self.timestep) == -1:
            return False
        return True

    def run(self, duration_sim_min=0.0, results_csv=None, sample_id=None):
        print("=" * 60)
        print(f"DECENTRALIZED EVAL SUPERVISOR  "
              f"(arena: {self.arena_size} | robots: {self.num_robots} | "
              f"tags: {len(self.tag_nodes)})")
        print(f"Arena half: {self.arena_half} m")
        if duration_sim_min > 0:
            print(f"Duration: {duration_sim_min} sim-min | Fast mode ON")
        print("=" * 60 + "\n")
        print(f"[SUPERVISOR] Logging to: {self._log_path}\n")

        if duration_sim_min > 0:
            self.supervisor.simulationSetMode(self.supervisor.SIMULATION_MODE_FAST)

        while True:
            if not self.run_step():
                break
            self.step += 1

            sim_time_min = (self.step * self.timestep / 1000.0) / 60.0

            if duration_sim_min > 0 and sim_time_min >= duration_sim_min:
                break

            if self.step % 500 == 0:
                elapsed_min  = (time.time() - self.start_time) / 60.0
                sim_rate     = (self.total_deposits / sim_time_min
                                if sim_time_min > 0.01 else 0.0)
                wall_rate    = (self.total_deposits / elapsed_min
                                if elapsed_min > 0.01 else 0.0)

                log_msg = (
                    f"\n{'='*60}\n"
                    f"Step {self.step} | Pickups: {self.total_pickups} | "
                    f"Deposits: {self.total_deposits}\n"
                    f"Sim: {sim_time_min:.1f} min | Wall: {elapsed_min:.1f} min | "
                    f"SimRate: {sim_rate:.2f} tags/sim-min | "
                    f"WallRate: {wall_rate:.2f} tags/wall-min\n"
                    f"{'='*60}\n"
                )
                for ri in range(self.num_robots):
                    rpos   = self.robot_nodes[ri].getPosition()
                    wall_d = self.arena_half - max(abs(rpos[0]), abs(rpos[1]))
                    d2base = math.sqrt(rpos[0] ** 2 + rpos[1] ** 2)
                    carry  = self.carrying[ri]
                    tag_vis, tag_dist_n, _ = self._tag_obs(ri)

                    if wall_d < 0.35:                 mode = "WALL_ESC"
                    elif not carry and d2base < 0.25: mode = "BASE_ESC"
                    elif carry:                       mode = "RTB"
                    else:                             mode = "PPO"

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
                with open(self._log_path, "a") as f:
                    f.write(log_msg)

        # ── Final stats ───────────────────────────────────────────────
        elapsed_min  = (time.time() - self.start_time) / 60.0
        sim_time_min = (self.step * self.timestep / 1000.0) / 60.0
        sim_rate     = self.total_deposits / sim_time_min if sim_time_min > 0.01 else 0.0
        wall_rate    = self.total_deposits / elapsed_min  if elapsed_min  > 0.01 else 0.0

        final_msg = (
            f"\n[FINAL] Sample={sample_id} | Arena={self.arena_size} | "
            f"Robots={self.num_robots} | "
            f"Pickups={self.total_pickups} | Deposits={self.total_deposits} | "
            f"Sim={sim_time_min:.2f} min | SimRate={sim_rate:.4f} | "
            f"Wall={elapsed_min:.2f} min | WallRate={wall_rate:.4f}\n"
        )
        print(final_msg)
        with open(self._log_path, "a") as f:
            f.write(final_msg)

        # ── Write single CSV row ──────────────────────────────────────
        if results_csv:
            write_header = not os.path.exists(results_csv)
            with open(results_csv, 'a', newline='') as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(['sample', 'arena', 'num_robots', 'deposits',
                                'sim_time_min', 'sim_rate', 'wall_time_min'])
                w.writerow([sample_id, self.arena_size, self.num_robots,
                            self.total_deposits, round(sim_time_min, 2),
                            round(sim_rate, 4), round(elapsed_min, 2)])

        if duration_sim_min > 0:
            self.supervisor.simulationQuit(0)


if __name__ == "__main__":
    import argparse

    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=None,
                        help='Per-robot run name (e.g. decentralized_indep_v9)')
    parser.add_argument('--model', type=str, default=None,
                        help='Shared model path .zip (CTDE / fallback)')
    parser.add_argument('--arena_size', type=str, default='5x5',
                        choices=list(ARENA_CONFIGS.keys()))
    parser.add_argument('--num_robots', type=int, default=4,
                        help='Number of robots in the world (default: 4)')
    parser.add_argument('--num_tags', type=int, default=None,
                        help='Override active tag count (hides excess tags; default: arena default)')
    parser.add_argument('--duration_sim_min', type=float, default=0.0,
                        help='Stop after N simulation minutes (0 = run forever)')
    parser.add_argument('--results_csv', type=str, default=None,
                        help='Append per-minute rows to this CSV')
    parser.add_argument('--sample_id', type=str, default=None,
                        help='Sample label written into the CSV (e.g. "1")')
    args = parser.parse_args()

    # Write arena config BEFORE connecting to Webots — robots read this on startup.
    _arena_cfg = os.path.join(_project_root, 'current_eval_arena.txt')
    with open(_arena_cfg, 'w') as _f:
        _f.write(args.arena_size)
    _active_tags = args.num_tags if args.num_tags else ARENA_CONFIGS[args.arena_size][0]
    print(f"[SUPERVISOR] Arena: {args.arena_size} "
          f"(active_tags={_active_tags}, half={ARENA_CONFIGS[args.arena_size][1]} m) "
          f"| Robots: {args.num_robots}")

    if args.run_name:
        _cfg = os.path.join(_project_root, 'current_eval_run.txt')
        with open(_cfg, 'w') as _f:
            _f.write(args.run_name)
        print(f"[SUPERVISOR] Per-robot eval | run_name: {args.run_name}")
    else:
        if args.model:
            _model_path = os.path.abspath(args.model)
            if _model_path.endswith('.zip'):
                _model_path = _model_path[:-4]
        else:
            _model_path = os.path.join(_project_root, 'decentralized_optA_v1')
        _cfg = os.path.join(_project_root, 'current_eval_model.txt')
        with open(_cfg, 'w') as _f:
            _f.write(_model_path)
        print(f"[SUPERVISOR] Shared model eval | path: {_model_path}")

    sup = DecentralizedEvalSupervisor(
        arena_size=args.arena_size,
        num_robots=args.num_robots,
        num_tags_override=args.num_tags,
    )
    sup.run(
        duration_sim_min=args.duration_sim_min,
        results_csv=args.results_csv,
        sample_id=args.sample_id,
    )
