import csv
import os
import math
import time
from controller import Supervisor

# =============================================================================
# DECENTRALIZED EVAL SUPERVISOR — ROBOT FAULT INJECTION  (EX4b / EX4b-2fault)
#
# Supports 1 or 2 robots being halted at pre-scheduled simulation times.
# Fault protocol:
#   - Failure schedule read from CSV (project root)
#   - At each robot's failure_time_min: supervisor sends HALT_SIGNAL every step
#   - Robot controller detects HALT_SIGNAL (pickup_signal < -50.0) and zeros motors
#   - Supervisor stops processing pickups/deposits for the halted robot
#   - The robot's physical body remains in the arena
#   - Remaining active robots continue with full onboard PPO, unaffected
#
# HALT_SIGNAL = -99.0
# Normal eval is NOT affected: eval_decentralized.py never sends values < -1.0
# =============================================================================

TAG_SEEK_RANGE = 1.0
FOV_HALF_ANGLE = 1.2
DENSITY_RADIUS = 0.5
HALT_SIGNAL    = -99.0

ARENA_CONFIGS = {
    '5x5':   (64,  2.5),
    '7x7':   (128, 3.5),
    '9x9':   (208, 4.5),
    '12x12': (368, 6.0),
}


def _load_failure_schedule(schedule_path, sample_id):
    """
    Read schedule CSV and return a list of fault dicts for the given sample:
      [{'robot_idx': int (0-based), 'time_min': float, 'halted': False}, ...]

    Supports both 1-fault and 2-fault CSV formats:
      1-fault: columns  sample, failed_robot, failure_time_min
      2-fault: columns  sample, failed_robot_1, failure_time_min_1,
                                failed_robot_2, failure_time_min_2
    """
    with open(schedule_path, newline='') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            if int(row['sample']) != sample_id:
                continue
            faults = []
            if 'failed_robot_1' in fieldnames:
                for i in ('1', '2'):
                    r = int(row[f'failed_robot_{i}'])
                    t = float(row[f'failure_time_min_{i}'])
                    faults.append({'robot_idx': r - 1, 'time_min': t, 'halted': False})
            else:
                r = int(row['failed_robot'])
                t = float(row['failure_time_min'])
                faults.append({'robot_idx': r - 1, 'time_min': t, 'halted': False})
            return faults
    return []


class DecentralizedFaultSupervisor:

    def __init__(self, arena_size='5x5', num_robots=4, faults=None):
        """
        faults: list of {'robot_idx': int (0-based), 'time_min': float, 'halted': False}
                None or empty list → run without any fault (baseline mode)
        """
        self.arena_size = arena_size
        self.num_robots = num_robots
        num_tags, self.arena_half = ARENA_CONFIGS.get(arena_size, ARENA_CONFIGS['5x5'])

        self.faults = faults if faults else []

        self.supervisor = Supervisor()
        self.timestep   = int(self.supervisor.getBasicTimeStep())

        self.robot_nodes = [
            self.supervisor.getFromDef(f"ROBOT{i+1}") for i in range(self.num_robots)
        ]
        self.tag_nodes = [
            self.supervisor.getFromDef(f"APRILTAG_{i+1}") for i in range(num_tags)
        ]

        self.emitters  = []
        self.receivers = []
        for i in range(self.num_robots):
            self.emitters.append(self.supervisor.getDevice(f"emitter{i+1}"))
            recv = self.supervisor.getDevice(f"receiver{i+1}")
            recv.enable(self.timestep)
            self.receivers.append(recv)

        self.robot_gps    = [[0.0, 0.0] for _ in range(self.num_robots)]
        self._robot_phero = [[0.0, 0.0] for _ in range(self.num_robots)]
        self.carrying     = [False] * self.num_robots

        self.total_pickups  = 0
        self.total_deposits = 0
        self.step_count     = 0
        self.start_time     = time.time()

        _project_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
        _run_cfg  = os.path.join(_project_root, 'current_eval_run.txt')
        _run_name = open(_run_cfg).read().strip() if os.path.exists(_run_cfg) else 'eval'
        n_faults  = len(self.faults)
        _log_dir  = os.path.join(_project_root, 'logs',
                                 f'eval_{_run_name}_{arena_size}_fault{n_faults}')
        os.makedirs(_log_dir, exist_ok=True)
        self._log_path = os.path.join(_log_dir, 'supervisor_log.txt')

    # =========================================================================
    # Helpers
    # =========================================================================

    def _is_halted(self, robot_idx):
        return any(f['halted'] and f['robot_idx'] == robot_idx for f in self.faults)

    def _collect_robot_states(self):
        for i in range(self.num_robots):
            if self._is_halted(i):
                while self.receivers[i].getQueueLength() > 0:
                    self.receivers[i].nextPacket()
                continue
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
        pos       = self.robot_gps[robot_idx]
        robot_rot = self.robot_nodes[robot_idx].getOrientation()
        fwd       = [robot_rot[0], robot_rot[3]]
        tag_visible = 0.0; tag_dist = 0.0; tag_angle = 0.0
        min_dist = float('inf')
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
        return sum(
            1 for t in self.tag_nodes
            if t.getPosition()[2] >= 0 and
               math.sqrt((t.getPosition()[0] - pos[0])**2 +
                         (t.getPosition()[1] - pos[1])**2) < DENSITY_RADIUS
        )

    # =========================================================================
    # Main loop
    # =========================================================================

    def run_step(self):
        self._collect_robot_states()

        sim_time_min = (self.step_count * self.timestep / 1000.0) / 60.0

        # Check each fault entry independently
        for fault in self.faults:
            if not fault['halted'] and sim_time_min >= fault['time_min']:
                fault['halted'] = True
                msg = (f"[FAULT] robot{fault['robot_idx']+1} HALTED at "
                       f"t={sim_time_min:.3f} min (scheduled: {fault['time_min']:.4f} min)")
                print(msg)
                with open(self._log_path, 'a') as f:
                    f.write(msg + '\n')

        for i in range(self.num_robots):
            # ── Halted robot ──────────────────────────────────────────────────
            if self._is_halted(i):
                self.emitters[i].send(f"0.0,0.0,0.0,{HALT_SIGNAL}".encode('utf-8'))
                continue

            # ── Active robot ──────────────────────────────────────────────────
            rpos          = self.robot_nodes[i].getPosition()
            gps_x, gps_y  = rpos[0], rpos[1]
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
                        self.carrying[i]   = True
                        pickup_tag_x       = tag_pos[0]
                        pickup_tag_y       = tag_pos[1]
                        pickup_signal      = float(self._count_density([gps_x, gps_y]))
                        tag_node.getField("translation").setSFVec3f([0, 0, -10])
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
                    self.carrying[i]     = False
                    pickup_signal        = -1.0
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

    def run(self, duration_sim_min=10.0, results_csv=None, sample_id=None):
        num_active_tags = len(self.tag_nodes)
        fault_labels = [f"robot{f['robot_idx']+1}@{f['time_min']:.4f}min"
                        for f in self.faults] or ['none']
        print("=" * 60)
        print(f"DECENTRALIZED FAULT EVAL (EX4b — {len(self.faults)}-robot fault)")
        print(f"Arena: {self.arena_size} | Robots: {self.num_robots} | Tags: {num_active_tags}")
        print(f"Faults: {', '.join(fault_labels)} | Duration: {duration_sim_min} sim-min")
        print("=" * 60 + "\n")

        self.supervisor.simulationSetMode(self.supervisor.SIMULATION_MODE_FAST)

        while True:
            if not self.run_step():
                break
            self.step_count += 1
            sim_time_min = (self.step_count * self.timestep / 1000.0) / 60.0
            if sim_time_min >= duration_sim_min:
                break

            if self.step_count % 500 == 0:
                elapsed_min = (time.time() - self.start_time) / 60.0
                sim_rate    = (self.total_deposits / sim_time_min
                               if sim_time_min > 0.01 else 0.0)
                halted_ids  = [f['robot_idx']+1 for f in self.faults if f['halted']]
                halted_str  = (f" | HALTED: {halted_ids}" if halted_ids else " | all active")
                log_msg = (
                    f"\n{'='*60}\n"
                    f"Step {self.step_count} | Deps: {self.total_deposits} | "
                    f"Sim: {sim_time_min:.1f} min | Wall: {elapsed_min:.1f} min | "
                    f"Rate: {sim_rate:.2f} tags/min{halted_str}\n"
                    f"{'='*60}\n"
                )
                print(log_msg)
                with open(self._log_path, 'a') as f:
                    f.write(log_msg)

        # ── Final stats ───────────────────────────────────────────────────────
        elapsed_min  = (time.time() - self.start_time) / 60.0
        sim_time_min = (self.step_count * self.timestep / 1000.0) / 60.0
        sim_rate     = self.total_deposits / sim_time_min if sim_time_min > 0.01 else 0.0

        final_msg = (
            f"\n[FINAL] Sample={sample_id} | Arena={self.arena_size} | "
            f"Faults={fault_labels} | "
            f"Deposits={self.total_deposits} | "
            f"Sim={sim_time_min:.2f} min | SimRate={sim_rate:.4f} | "
            f"Wall={elapsed_min:.2f} min\n"
        )
        print(final_msg)
        with open(self._log_path, 'a') as f:
            f.write(final_msg)

        if results_csv:
            write_header = not os.path.exists(results_csv)
            pct = round(100.0 * self.total_deposits / num_active_tags, 1)
            # Build per-fault columns (up to 2)
            fault_cols = {}
            for k, fault in enumerate(self.faults, 1):
                fault_cols[f'failed_robot_{k}']     = fault['robot_idx'] + 1
                fault_cols[f'failure_time_min_{k}'] = fault['time_min']

            base_row = {
                'sample':       sample_id,
                'arena':        self.arena_size,
                'num_robots':   self.num_robots,
                'num_faults':   len(self.faults),
                'deposits':     self.total_deposits,
                'sim_time_min': round(sim_time_min, 2),
                'sim_rate':     round(sim_rate, 4),
                'wall_time_min': round(elapsed_min, 2),
                'pct_collected': pct,
            }
            row = {**base_row, **fault_cols}

            with open(results_csv, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(row)

        self.supervisor.simulationQuit(0)


if __name__ == "__main__":
    import argparse

    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name',          type=str,   default='decentralized_indep_v10')
    parser.add_argument('--arena_size',        type=str,   default='5x5',
                        choices=list(ARENA_CONFIGS.keys()))
    parser.add_argument('--duration_sim_min',  type=float, default=10.0)
    parser.add_argument('--results_csv',       type=str,   default=None)
    parser.add_argument('--sample_id',         type=str,   default=None)
    parser.add_argument('--failure_schedule',  type=str,
                        default=os.path.join(_project_root, 'failure_schedule_exp4b.csv'))
    args = parser.parse_args()

    for fname, val in [('current_eval_run.txt', args.run_name),
                       ('current_eval_arena.txt', args.arena_size)]:
        with open(os.path.join(_project_root, fname), 'w') as f:
            f.write(val)

    faults = []
    if args.sample_id and os.path.exists(args.failure_schedule):
        faults = _load_failure_schedule(args.failure_schedule, int(args.sample_id))
        for fault in faults:
            print(f"[SUPERVISOR] Sample {args.sample_id}: "
                  f"robot{fault['robot_idx']+1} halted at t={fault['time_min']} min")
    else:
        print("[SUPERVISOR] WARNING: no failure schedule found — running without fault")

    sup = DecentralizedFaultSupervisor(
        arena_size=args.arena_size,
        num_robots=4,
        faults=faults,
    )
    sup.run(
        duration_sim_min=args.duration_sim_min,
        results_csv=args.results_csv,
        sample_id=args.sample_id,
    )
