import math
import random
import numpy as np
from controller import Supervisor

# =============================================================================
# CENTRALIZED HAND-CODED BASELINE — No PPO, pure rule-based controller
#
# Purpose:
#   Establishes how well the hand-coded rules alone perform, so that the gap
#   between this baseline and ppo_v16_phero.zip isolates exactly what RL
#   training contributes.
#
# Design principle:
#   P1–P4 are IDENTICAL to eval_best_model.py (same thresholds, same gains).
#   The only difference is what happens when none of P1–P4 apply — this is
#   where PPO normally acts. Here we replace PPO with two hand-coded rules:
#
#   P5  Pheromone following — if a cluster peak is known (pheromone_grid.max
#       > 0.1), steer directly toward the strongest grid cell.
#       PPO learns to do this from reward shaping; here we hard-code it.
#
#   P6  Lévy walk exploration — when no pheromone signal exists, pick a
#       random target point in the arena and drive toward it. When reached,
#       pick a new random target. Distance sampled uniformly from [0.8, 2.3m]
#       to bias search toward the cluster zone (PPO learns this implicitly via
#       the zone reward; here we encode it explicitly).
#
# P1–P4 thresholds (must exactly match eval_best_model.py _apply_overrides):
#   P1  wall_dist < 0.35 OR max(prox) > 0.55  →  steer to centre (gain 4.0)
#   P2  carrying                               →  steer to base   (gain 2.5)
#   P3  dist_to_base < 0.5, not carrying       →  steer away      (gain 3.0)
#   P4  tag in FOV (1.0 m, 1.2 rad)           →  steer to tag    (gain 3.0)
#
# Pheromone grid:
#   Same 50×50 grid over 5×5 m arena, decay ×0.999/step, deposited at pickup.
#   Matches training supervisor exactly so pheromone dynamics are identical.
#
# Usage:
#   webots worlds/eval_best.wbt &
#   sleep 10
#   python3 controllers/centralized_baseline/centralized_baseline.py
# =============================================================================

class CentralizedBaseline(Supervisor):
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())

        self.num_robots = 4
        self.num_tags   = 64

        # --- World nodes ---
        self.robot_nodes = [self.getFromDef(f"ROBOT{i+1}") for i in range(self.num_robots)]
        self.tag_nodes   = [self.getFromDef(f"APRILTAG_{i+1}") for i in range(self.num_tags)]
        self.base_node   = self.getFromDef("BASE_STATION")

        # --- Communication ---
        self.emitters  = [self.getDevice(f"emitter{i+1}") for i in range(self.num_robots)]
        self.receivers = [self.getDevice(f"receiver{i+1}") for i in range(self.num_robots)]
        for r in self.receivers:
            r.enable(self.timestep)

        # --- Per-robot state ---
        self.robot_states   = [None]  * self.num_robots  # proximity sensor array (8 values)
        self.carrying_state = [False] * self.num_robots
        self.explore_target = [None]  * self.num_robots  # current Lévy walk target [x, y]

        # --- Pheromone grid (identical to training supervisor) ---
        self.grid_size      = 80
        self.grid_res       = 8.0 / self.grid_size        # 0.1 m per cell
        self.pheromone_grid = np.zeros((self.grid_size, self.grid_size))

        # --- Metrics ---
        self.total_pickups  = 0
        self.total_deposits = 0
        self.step_count     = 0

        print("=" * 60)
        print("CENTRALIZED HAND-CODED BASELINE")
        print("P1-P4: identical to eval_best_model.py overrides")
        print("P5:    greedy pheromone following")
        print("P6:    Lévy walk exploration")
        print("=" * 60 + "\n")

    # =========================================================================
    # Proportional steering (identical to training/eval supervisor)
    # =========================================================================
    def _steer_to(self, robot_pos, fwd, target, gain=2.5):
        dx   = target[0] - robot_pos[0]
        dy   = target[1] - robot_pos[1]
        dist = math.sqrt(dx*dx + dy*dy)
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
            left /= m
            right /= m
        return [left, right]

    # =========================================================================
    # P6 helper: pick a new Lévy walk target
    #   Distance biased toward cluster zone (0.8–3.5 m from base) so the
    #   robot covers the 8×8m arena rather than hovering near the centre.
    # =========================================================================
    def _new_explore_target(self, robot_pos):
        angle = random.uniform(0.0, 2.0 * math.pi)
        dist  = random.uniform(0.8, 3.5)
        tx    = robot_pos[0] + math.cos(angle) * dist
        ty    = robot_pos[1] + math.sin(angle) * dist
        # Clamp to arena bounds (walls at ±4.0 m, keep 0.3 m margin)
        tx = max(-3.7, min(3.7, tx))
        ty = max(-3.7, min(3.7, ty))
        return [tx, ty]

    # =========================================================================
    # Main control loop
    # =========================================================================
    def run(self):
        while self.step(self.timestep) != -1:
            self.step_count += 1

            # --- Read proximity sensors from robot receivers ---
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

            # --- Pheromone decay (matches v17 training supervisor) ---
            self.pheromone_grid *= 0.9995

            base_pos = self.base_node.getPosition()
            modes    = [""] * self.num_robots

            # --- Per-robot pickup and deposit detection ---
            for i in range(self.num_robots):
                robot_pos = self.robot_nodes[i].getPosition()

                if not self.carrying_state[i]:
                    for tag_node in self.tag_nodes:
                        tag_pos = tag_node.getPosition()
                        if tag_pos[2] < 0:
                            continue
                        dx   = tag_pos[0] - robot_pos[0]
                        dy   = tag_pos[1] - robot_pos[1]
                        if math.sqrt(dx*dx + dy*dy) < 0.15:
                            self.carrying_state[i] = True
                            tag_node.getField("translation").setSFVec3f([0, 0, -10])
                            self.total_pickups += 1
                            self.explore_target[i] = None  # clear walk target on pickup
                            # Deposit pheromone at pickup location (identical to training)
                            pgx = int((robot_pos[0] + 4.0) / self.grid_res)
                            pgy = int((robot_pos[1] + 4.0) / self.grid_res)
                            if 0 <= pgx < self.grid_size and 0 <= pgy < self.grid_size:
                                self.pheromone_grid[pgx, pgy] = min(
                                    self.pheromone_grid[pgx, pgy] + 8.0, 10.0
                                )
                            print(f"[+] Robot {i+1} picked up tag! Total pickups: {self.total_pickups}")
                            break

                if self.carrying_state[i]:
                    dx   = base_pos[0] - robot_pos[0]
                    dy   = base_pos[1] - robot_pos[1]
                    if math.sqrt(dx*dx + dy*dy) < 0.25:
                        self.carrying_state[i] = False
                        self.total_deposits   += 1
                        print(f"[*] Robot {i+1} deposited tag! Total deposits: {self.total_deposits}")

            # --- Per-robot action selection ---
            for i in range(self.num_robots):
                robot_pos = self.robot_nodes[i].getPosition()
                robot_rot = self.robot_nodes[i].getOrientation()
                fwd       = [robot_rot[0], robot_rot[3], robot_rot[6]]
                prox      = (self.robot_states[i] or [0.0]*8)[:8]
                wall_dist = 4.0 - max(abs(robot_pos[0]), abs(robot_pos[1]))

                # ----------------------------------------------------------
                # P1: Wall / collision escape  (identical to eval_best_model)
                # ----------------------------------------------------------
                if wall_dist < 0.35 or max(prox) > 0.55:
                    action = self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=4.0)
                    modes[i] = "WALL_ESC"

                # ----------------------------------------------------------
                # P2: Return to base when carrying  (identical)
                # ----------------------------------------------------------
                elif self.carrying_state[i]:
                    action = self._steer_to(robot_pos, fwd, base_pos, gain=2.5)
                    modes[i] = "RTB"

                # ----------------------------------------------------------
                # P3: Base avoidance when not carrying  (identical, 0.8 m)
                # ----------------------------------------------------------
                elif math.sqrt(robot_pos[0]**2 + robot_pos[1]**2) < 0.8:
                    away   = [robot_pos[0] - base_pos[0], robot_pos[1] - base_pos[1]]
                    target = [robot_pos[0] + away[0] * 2.0,
                              robot_pos[1] + away[1] * 2.0]
                    action = self._steer_to(robot_pos, fwd, target, gain=3.0)
                    modes[i] = "BASE_AVOID"

                # ----------------------------------------------------------
                # P4: Tag seek — nearest visible tag in FOV  (identical)
                #     Range 1.0 m, half-FOV 1.2 rad (~69°)
                # ----------------------------------------------------------
                else:
                    best_tag_pos = None
                    best_dist    = float('inf')
                    for tag_node in self.tag_nodes:
                        tag_pos = tag_node.getPosition()
                        if tag_pos[2] < 0:
                            continue
                        tdx = tag_pos[0] - robot_pos[0]
                        tdy = tag_pos[1] - robot_pos[1]
                        td  = math.sqrt(tdx*tdx + tdy*tdy)
                        if td < 0.5 and td > 0.001 and td < best_dist:
                            dot   = fwd[0]*(tdx/td) + fwd[1]*(tdy/td)
                            angle = math.acos(max(min(dot, 1.0), -1.0))
                            if angle < 1.2:
                                best_dist    = td
                                best_tag_pos = tag_pos

                    if best_tag_pos is not None:
                        action = self._steer_to(robot_pos, fwd, best_tag_pos, gain=3.0)
                        modes[i] = "TAG_SEEK"

                    # ----------------------------------------------------------
                    # P5: Pheromone following — steer toward strongest grid cell
                    #     This replaces what PPO learned to do from reward shaping.
                    # ----------------------------------------------------------
                    elif self.pheromone_grid.max() > 0.1:
                        peak      = np.unravel_index(
                            self.pheromone_grid.argmax(), self.pheromone_grid.shape)
                        cluster_x = peak[0] * self.grid_res - 4.0
                        cluster_y = peak[1] * self.grid_res - 4.0

                        # # Phantom-stall fix (commented out): if robot reaches the peak
                        # # but finds no tags within 0.6m, wipe that grid region so all
                        # # robots stop converging on an empty cluster and re-explore.
                        # dist_to_peak = math.sqrt(
                        #     (robot_pos[0] - cluster_x)**2 + (robot_pos[1] - cluster_y)**2)
                        # if dist_to_peak < 0.4:
                        #     tags_nearby = any(
                        #         tag_node.getPosition()[2] >= 0 and
                        #         math.sqrt((tag_node.getPosition()[0] - cluster_x)**2 +
                        #                   (tag_node.getPosition()[1] - cluster_y)**2) < 0.6
                        #         for tag_node in self.tag_nodes
                        #     )
                        #     if not tags_nearby:
                        #         # Cluster is depleted — wipe a 1m radius on the grid
                        #         for gx in range(self.grid_size):
                        #             for gy in range(self.grid_size):
                        #                 wx = gx * self.grid_res - 4.0
                        #                 wy = gy * self.grid_res - 4.0
                        #                 if math.sqrt((wx-cluster_x)**2+(wy-cluster_y)**2) < 1.0:
                        #                     self.pheromone_grid[gx, gy] = 0.0
                        #         self.explore_target[i] = None
                        #         action = [0.8, 0.8]
                        #         modes[i] = "CLUST_DONE"

                        action = self._steer_to(robot_pos, fwd,
                                                [cluster_x, cluster_y], gain=2.5)
                        modes[i] = "PHERO"

                    # ----------------------------------------------------------
                    # P6: Lévy walk exploration — pick random arena targets,
                    #     biased toward cluster zone (0.8–2.3 m from base).
                    #     This replaces what PPO learned for global exploration.
                    # ----------------------------------------------------------
                    else:
                        if self.explore_target[i] is None:
                            self.explore_target[i] = self._new_explore_target(robot_pos)

                        tx, ty = self.explore_target[i]
                        tdx    = tx - robot_pos[0]
                        tdy    = ty - robot_pos[1]
                        if math.sqrt(tdx*tdx + tdy*tdy) < 0.2:
                            # Reached target — pick a new one next step
                            self.explore_target[i] = None
                            action = [0.8, 0.8]   # brief straight drive while choosing next target
                        else:
                            action = self._steer_to(robot_pos, fwd, [tx, ty], gain=2.5)
                        modes[i] = "EXPLORE"

                # --- Send motor command ---
                msg = f"{action[0]},{action[1]}".encode('utf-8')
                self.emitters[i].send(msg)

            # ---- Logging every 500 steps ----
            if self.step_count % 500 == 0:
                elapsed_min = self.step_count * self.timestep / 1000.0 / 60.0
                rate = self.total_deposits / elapsed_min if elapsed_min > 0 else 0.0
                log_msg = (
                    f"\n{'='*60}\n"
                    f"Step {self.step_count} ({elapsed_min:.1f} min sim) | "
                    f"Pickups: {self.total_pickups} | Deposits: {self.total_deposits} | "
                    f"Rate: {rate:.2f} tags/min\n"
                    f"{'='*60}\n"
                )
                for ri in range(self.num_robots):
                    rpos      = self.robot_nodes[ri].getPosition()
                    wall_d    = 4.0 - max(abs(rpos[0]), abs(rpos[1]))
                    d_to_base = math.sqrt(rpos[0]**2 + rpos[1]**2)
                    phero_str = self.pheromone_grid.max()
                    log_msg += (
                        f"R{ri+1}[{modes[ri]:10s}]: "
                        f"carry={int(self.carrying_state[ri])} | "
                        f"base_dist={d_to_base:.2f} | "
                        f"wall={wall_d:.2f} | "
                        f"phero_max={phero_str:.2f}\n"
                    )
                print(log_msg)
                with open("centralized_baseline_log.txt", "a") as f:
                    f.write(log_msg)


# =============================================================================
controller = CentralizedBaseline()
controller.run()
