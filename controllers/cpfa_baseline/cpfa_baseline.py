import math
import random
import numpy as np
from controller import Supervisor

# =============================================================================
# CPFA BASELINE — Pure hard-coded controller, NO PPO model
#
# Behaviour priority (same structure as v10_cpfa training overrides):
#   P1  Wall escape     — steer to centre when wall < 0.35m or prox > 0.55
#   P2  RTB             — steer to base when carrying
#   P3  Base avoid      — steer away from base when too close and not carrying
#   P4  Stuck recovery  — forward-bias drive when no movement for 50 steps
#   P5  Tag seek        — steer toward nearest visible tag in FOV (< 1.2 rad)
#   P6  CPFA explore    — 60% site fidelity / 20% pheromone peak / 20% random walk
#
# Run via:
#   webots-controller --robot-name=supervisor cpfa_baseline.py
# =============================================================================

class CPFABaselineSupervisor(Supervisor):
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())

        self.num_robots = 4
        self.num_tags   = 70

        self.robot_nodes = [self.getFromDef(f"ROBOT{i+1}") for i in range(self.num_robots)]
        self.tag_nodes   = [self.getFromDef(f"APRILTAG_{i+1}") for i in range(self.num_tags)]
        self.base_node   = self.getFromDef("BASE_STATION")

        self.emitters  = [self.getDevice(f"emitter{i+1}") for i in range(self.num_robots)]
        self.receivers = [self.getDevice(f"receiver{i+1}") for i in range(self.num_robots)]
        for r in self.receivers:
            r.enable(self.timestep)

        # Per-robot state
        self.robot_states    = [None] * self.num_robots   # proximity sensor array
        self.carrying_state  = [False] * self.num_robots
        self.was_carrying    = [False] * self.num_robots
        self.last_pickup_pos = [None] * self.num_robots
        self.search_target   = [None] * self.num_robots
        self.stuck_counter   = [0] * self.num_robots
        self.last_tracked_pos = [[0.0, 0.0] for _ in range(self.num_robots)]

        # Pheromone grid (matches training: 50×50 over 5×5 m arena)
        self.grid_size = 50
        self.grid_res  = 5.0 / self.grid_size
        self.pheromone_grid = np.zeros((self.grid_size, self.grid_size))

        # Metrics
        self.total_pickups  = 0
        self.total_deposits = 0
        self.step_count     = 0

        print("="*60)
        print("CPFA BASELINE — No PPO, pure hard-coded behaviour")
        print("="*60 + "\n")

    # -------------------------------------------------------------------------
    # Proportional steering — returns [left, right] in [-1, 1]
    # -------------------------------------------------------------------------
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
            left /= m; right /= m
        return [left, right]

    # -------------------------------------------------------------------------
    # CPFA exploration target selection
    #   60%  site fidelity  (last pickup location)
    #   20%  pheromone peak (highest grid cell)
    #   20%  random walk    (random point within arena)
    # -------------------------------------------------------------------------
    def _pick_search_target(self, i):
        r = random.random()
        if r < 0.60 and self.last_pickup_pos[i] is not None:
            return list(self.last_pickup_pos[i])
        elif r < 0.80 and self.pheromone_grid.max() > 0.1:
            peak = np.unravel_index(self.pheromone_grid.argmax(), self.pheromone_grid.shape)
            return [peak[0] * self.grid_res - 2.5, peak[1] * self.grid_res - 2.5]
        else:
            robot_pos = self.robot_nodes[i].getPosition()
            angle = random.uniform(-math.pi, math.pi)
            dist  = random.uniform(0.5, 2.0)
            tx = max(-2.2, min(2.2, robot_pos[0] + math.cos(angle) * dist))
            ty = max(-2.2, min(2.2, robot_pos[1] + math.sin(angle) * dist))
            return [tx, ty]

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
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

            # --- Pheromone decay ---
            self.pheromone_grid *= 0.999

            base_pos = self.base_node.getPosition()
            modes    = [""] * self.num_robots

            # --- Per-robot behaviour ---
            for i in range(self.num_robots):
                robot_pos = self.robot_nodes[i].getPosition()
                robot_rot = self.robot_nodes[i].getOrientation()
                fwd       = [robot_rot[0], robot_rot[3], robot_rot[6]]
                prox      = (self.robot_states[i] or [0.0]*8)[:8]
                wall_dist = 2.5 - max(abs(robot_pos[0]), abs(robot_pos[1]))

                # ---- Pickup detection ----
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
                            self.last_pickup_pos[i] = [robot_pos[0], robot_pos[1]]
                            self.search_target[i]   = None
                            # Deposit pheromone at pickup site
                            pgx = int((robot_pos[0] + 2.5) / self.grid_res)
                            pgy = int((robot_pos[1] + 2.5) / self.grid_res)
                            if 0 <= pgx < self.grid_size and 0 <= pgy < self.grid_size:
                                self.pheromone_grid[pgx, pgy] = min(
                                    self.pheromone_grid[pgx, pgy] + 5.0, 10.0
                                )
                            print(f"[+] R{i+1} pickup! Total: {self.total_pickups}")
                            break

                # ---- Deposit detection ----
                if self.carrying_state[i]:
                    dx   = base_pos[0] - robot_pos[0]
                    dy   = base_pos[1] - robot_pos[1]
                    if math.sqrt(dx*dx + dy*dy) < 0.25:
                        self.carrying_state[i] = False
                        self.total_deposits    += 1
                        print(f"[*] R{i+1} deposit! Total: {self.total_deposits}")

                # ---- Post-deposit: reset search target ----
                if self.was_carrying[i] and not self.carrying_state[i]:
                    self.search_target[i] = None
                self.was_carrying[i] = self.carrying_state[i]

                # ---- Stuck detection ----
                curr_pos = [robot_pos[0], robot_pos[1]]
                dist_moved = math.sqrt(
                    (curr_pos[0] - self.last_tracked_pos[i][0])**2 +
                    (curr_pos[1] - self.last_tracked_pos[i][1])**2
                )
                if dist_moved < 0.01:
                    self.stuck_counter[i] += 1
                else:
                    self.stuck_counter[i] = 0
                    self.last_tracked_pos[i] = curr_pos

                # ============================================================
                # Behaviour priority cascade
                # ============================================================

                # P1 — Wall escape
                if wall_dist < 0.35 or max(prox) > 0.55:
                    action = self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=4.0)
                    modes[i] = "WALL_ESC"

                # P2 — Return to base
                elif self.carrying_state[i]:
                    action = self._steer_to(robot_pos, fwd, base_pos, gain=2.5)
                    modes[i] = "RTB"

                # P3 — Base avoid (don't cluster at nest)
                elif math.sqrt(robot_pos[0]**2 + robot_pos[1]**2) < 0.35:
                    away_x = robot_pos[0] * 2.0
                    away_y = robot_pos[1] * 2.0
                    action = self._steer_to(robot_pos, fwd,
                                            [robot_pos[0] + away_x,
                                             robot_pos[1] + away_y], gain=3.0)
                    modes[i] = "BASE_AVOID"

                # P4 — Stuck recovery
                elif self.stuck_counter[i] > 50:
                    action = [1.0, 0.2] if i % 2 == 0 else [0.2, 1.0]
                    modes[i] = "STUCK"

                else:
                    # P5 — Tag seek (nearest visible tag within FOV)
                    tag_action = None
                    min_td = float('inf')
                    for tag_node in self.tag_nodes:
                        tag_pos = tag_node.getPosition()
                        if tag_pos[2] < 0:
                            continue
                        tdx = tag_pos[0] - robot_pos[0]
                        tdy = tag_pos[1] - robot_pos[1]
                        td  = math.sqrt(tdx*tdx + tdy*tdy)
                        if td < 2.0 and td > 0.001 and td < min_td:
                            dot   = fwd[0]*(tdx/td) + fwd[1]*(tdy/td)
                            angle = math.acos(max(min(dot, 1.0), -1.0))
                            if angle < 1.2:
                                min_td     = td
                                tag_action = self._steer_to(robot_pos, fwd, tag_pos, gain=3.0)

                    if tag_action is not None:
                        action  = tag_action
                        modes[i] = "TAG_SEEK"

                    else:
                        # P6 — CPFA exploration
                        if self.search_target[i] is None:
                            self.search_target[i] = self._pick_search_target(i)

                        tx, ty = self.search_target[i]
                        tdx    = tx - robot_pos[0]
                        tdy    = ty - robot_pos[1]
                        if math.sqrt(tdx*tdx + tdy*tdy) < 0.2:
                            # Reached target — pick a new one next step
                            self.search_target[i] = None
                            action = [0.5, 0.5]
                        else:
                            action = self._steer_to(robot_pos, fwd,
                                                    [tx, ty], gain=2.5)
                        modes[i] = "CPFA"

                # ---- Send action ----
                msg = f"{action[0]},{action[1]}".encode('utf-8')
                self.emitters[i].send(msg)

            # ---- Logging every 500 steps ----
            if self.step_count % 500 == 0:
                elapsed_min = self.step_count * self.timestep / 1000.0 / 60.0
                log_msg = (
                    f"\n{'='*60}\n"
                    f"Step {self.step_count} ({elapsed_min:.1f} min) | "
                    f"Pickups: {self.total_pickups} | Deposits: {self.total_deposits}\n"
                    f"{'='*60}\n"
                )
                for ri in range(self.num_robots):
                    rpos   = self.robot_nodes[ri].getPosition()
                    wall_d = 2.5 - max(abs(rpos[0]), abs(rpos[1]))
                    log_msg += (
                        f"R{ri+1}[{modes[ri]}]: "
                        f"carry={int(self.carrying_state[ri])} | "
                        f"wall={wall_d:.2f} | "
                        f"stuck={self.stuck_counter[ri]}\n"
                    )
                print(log_msg)
                with open("cpfa_baseline_log.txt", "a") as f:
                    f.write(log_msg)

            # ---- Stop once all 70 tags collected ----
            if self.total_deposits >= self.num_tags:
                elapsed_min = self.step_count * self.timestep / 1000.0 / 60.0
                result = (
                    f"\n{'='*60}\n"
                    f"ALL {self.num_tags} TAGS COLLECTED!\n"
                    f"Total time: {elapsed_min:.2f} min\n"
                    f"Steps: {self.step_count}\n"
                    f"Rate: {self.total_deposits/elapsed_min:.2f} tags/min\n"
                    f"{'='*60}\n"
                )
                print(result)
                with open("cpfa_baseline_log.txt", "a") as f:
                    f.write(result)
                break


# =============================================================================
controller = CPFABaselineSupervisor()
controller.run()
