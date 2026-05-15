import os
import math
import random
import argparse
import time
from controller import Supervisor

# =============================================================================
# THIN SUPERVISOR — Fully Decentralized Independent Training
#
# This supervisor does NOT run PPO and does NOT compute rewards.
# Each robot runs its own independent PPO and computes its own reward.
#
# Supervisor responsibilities (simulation infrastructure only):
#   1. Camera simulation: detect tags in each robot's FOV
#   2. Tag mechanics: pickup detection, tag hiding, deposit detection
#   3. Pheromone strength: density-based value sent on pickup
#   4. Episode management: respawn robots + tags, curriculum
#   5. Per-episode stats logging
#
# Message protocol (same as eval supervisor):
#   Supervisor → Robot: [tag_visible, tag_dist_norm, tag_angle_norm, pickup_signal]
#     pickup_signal > 0  → pickup; value = pheromone strength (0.2–1.0)
#     pickup_signal = -1 → deposit confirmed
#     pickup_signal = 0  → normal step
#   Robot → Supervisor: 17 floats [prox×8, carrying, base_dist, base_angle,
#                                   phero×4, gps_x, gps_y]
#
# Episode sync: supervisor and each robot independently count STEPS_PER_EPISODE
# and reset together — no special reset signal needed.
#
# Usage (from project root, Webots open on epuck_foraging_decentralized.wbt):
#   cd controllers/decentralized_supervisor
#   python3 decentralized_supervisor.py --run_name decentralized_indep_v1
# =============================================================================

TAG_SEEK_RANGE  = 1.0
FOV_HALF_ANGLE  = 1.2
DENSITY_RADIUS  = 0.5
DENSITY_MAX     = 5
NUM_ROBOTS      = 4
NUM_TAGS        = 64
STEPS_PER_EPISODE = 16384


class DecentralizedTrainSupervisor:
    """
    Thin supervisor shim for fully decentralized independent-PPO training.
    Handles Webots-only tasks: tag mechanics, camera sim, episode resets.
    """

    def __init__(self):
        self.supervisor = Supervisor()
        self.timestep   = int(self.supervisor.getBasicTimeStep())

        # ── Webots nodes ──────────────────────────────────────────────────────
        self.robot_nodes = [
            self.supervisor.getFromDef(f"ROBOT{i+1}") for i in range(NUM_ROBOTS)
        ]
        self.tag_nodes = [
            self.supervisor.getFromDef(f"APRILTAG_{i+1}") for i in range(NUM_TAGS)
        ]

        # ── Supervisor ↔ robot communication (channels 1-4) ──────────────────
        self.emitters  = []
        self.receivers = []
        for i in range(NUM_ROBOTS):
            self.emitters.append(self.supervisor.getDevice(f"emitter{i+1}"))
            recv = self.supervisor.getDevice(f"receiver{i+1}")
            recv.enable(self.timestep)
            self.receivers.append(recv)

        # ── Runtime state ─────────────────────────────────────────────────────
        self.robot_gps    = [[0.0, 0.0] for _ in range(NUM_ROBOTS)]
        self.carrying     = [False] * NUM_ROBOTS

        # ── Episode / stats tracking ──────────────────────────────────────────
        self.episode_step    = 0
        self.total_episodes  = 0
        self.total_pickups   = 0
        self.total_deposits  = 0
        self.ep_pickups      = 0
        self.ep_deposits     = 0
        self.ep_start_time   = time.time()
        self.start_time      = time.time()

        # ── Log file ──────────────────────────────────────────────────────────
        _project_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
        _cfg = os.path.join(_project_root, 'current_run_name.txt')
        _run_name = open(_cfg).read().strip() if os.path.exists(_cfg) else 'decentralized_indep'
        _log_dir = os.path.join(_project_root, 'logs', _run_name)
        os.makedirs(_log_dir, exist_ok=True)
        self._log_path = os.path.join(_log_dir, 'supervisor_log.txt')

    # =========================================================================
    # Communication helpers
    # =========================================================================

    def _collect_robot_states(self):
        """Parse GPS position from each robot's 17-float message."""
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

    def _tag_obs(self, robot_idx):
        """
        Simulate camera: find nearest tag in robot's FOV within TAG_SEEK_RANGE.
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
        """Pheromone strength based on tag cluster density near pickup position."""
        nearby = sum(
            1 for t in self.tag_nodes
            if t.getPosition()[2] >= 0 and
               math.sqrt((t.getPosition()[0] - pos[0])**2 +
                         (t.getPosition()[1] - pos[1])**2) <= DENSITY_RADIUS
        )
        return 0.2 + 0.8 * min(nearby / DENSITY_MAX, 1.0)

    # =========================================================================
    # Main step
    # =========================================================================

    def run_step(self):
        self._collect_robot_states()

        for i in range(NUM_ROBOTS):
            gps_x, gps_y  = self.robot_gps[i]
            pickup_signal = 0.0

            if not self.carrying[i]:
                # Pickup detection: robot within 0.15m of any active tag
                for tag_node in self.tag_nodes:
                    tag_pos = tag_node.getPosition()
                    if tag_pos[2] < 0:
                        continue
                    dx = tag_pos[0] - gps_x
                    dy = tag_pos[1] - gps_y
                    if math.sqrt(dx * dx + dy * dy) < 0.15:
                        self.carrying[i] = True
                        tag_node.getField("translation").setSFVec3f([0, 0, -10])
                        pickup_signal        = self._pickup_strength([gps_x, gps_y])
                        self.total_pickups  += 1
                        self.ep_pickups     += 1
                        print(f"[PICKUP]  robot{i+1} | "
                              f"strength={pickup_signal:.2f} | "
                              f"Total: {self.total_pickups}")
                        break
            else:
                # Deposit detection: robot within 0.25m of base while carrying
                if math.sqrt(gps_x**2 + gps_y**2) < 0.25:
                    self.carrying[i]     = False
                    pickup_signal        = -1.0
                    self.total_deposits += 1
                    self.ep_deposits    += 1
                    print(f"[DEPOSIT] robot{i+1} | Total: {self.total_deposits}")

            # Camera simulation → tag obs
            tag_vis, tag_dist_n, tag_angle_n = self._tag_obs(i)

            # Send [tag_visible, tag_dist_norm, tag_angle_norm, pickup_signal]
            msg = f"{tag_vis},{tag_dist_n},{tag_angle_n},{pickup_signal}".encode('utf-8')
            self.emitters[i].send(msg)

        if self.supervisor.step(self.timestep) == -1:
            return False

        self.episode_step += 1

        # ── Episode summary + reset ───────────────────────────────────────────
        if self.episode_step >= STEPS_PER_EPISODE:
            self._log_episode()
            self._episode_reset()

        return True

    # =========================================================================
    # Episode management
    # =========================================================================

    def _log_episode(self):
        sim_min  = STEPS_PER_EPISODE * self.timestep / 1000.0 / 60.0
        rate     = self.ep_deposits / sim_min if sim_min > 0 else 0.0
        wall_min = (time.time() - self.ep_start_time) / 60.0

        ep = self.total_episodes + 1
        if ep < 60:
            phase = "CLOSE   (2 clusters, max_dist=1.0m)"
        elif ep < 201:
            phase = "MEDIUM  (3-5 clusters, max_dist=1.8m)"
        else:
            phase = "FULL    (6-8 clusters, max_dist=2.3m)"

        log = (
            f"\n{'='*65}\n"
            f"[EP {ep}] "
            f"Picks: {self.ep_pickups} | Deps: {self.ep_deposits} | "
            f"Rate: {rate:.2f} tags/min (sim) | "
            f"TotalDeps: {self.total_deposits} | "
            f"Wall: {wall_min:.1f} min\n"
            f"  Curriculum: {phase}\n"
            f"  Pheromone:  P2P per-robot (not tracked by supervisor)\n"
        )
        for i in range(NUM_ROBOTS):
            rpos   = self.robot_nodes[i].getPosition()
            wall_d = 2.5 - max(abs(rpos[0]), abs(rpos[1]))
            d2base = math.sqrt(rpos[0]**2 + rpos[1]**2)
            carry  = self.carrying[i]

            if wall_d < 0.35:              mode = "WALL_ESC"
            elif not carry and d2base < 0.25: mode = "BASE_ESC"
            elif carry:                    mode = "RTB"
            else:                          mode = "PPO"

            log += (f"  R{i+1}[{mode:8s}]: carry={1 if carry else 0} | "
                    f"base={d2base:.2f} | wall={wall_d:.2f}\n")
        log += f"{'='*65}"
        print(log)
        with open(self._log_path, 'a') as f:
            f.write(log + '\n')

    def _episode_reset(self):
        self.total_episodes += 1
        self.episode_step    = 0
        self.ep_pickups      = 0
        self.ep_deposits     = 0
        self.ep_start_time   = time.time()
        self.carrying        = [False] * NUM_ROBOTS

        self._respawn_robots()
        self._respawn_tags()
        self.supervisor.step(self.timestep)

    def _respawn_robots(self):
        positions = [[-0.5, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, -0.5, 0]]
        for i in range(NUM_ROBOTS):
            self.robot_nodes[i].getField("translation").setSFVec3f(positions[i])
            rand_yaw = random.uniform(0, 2 * math.pi)
            self.robot_nodes[i].getField("rotation").setSFRotation([0, 0, 1, rand_yaw])
            self.robot_nodes[i].resetPhysics()

    def _respawn_tags(self):
        """Place tags in rotated grids — uniform 0.10m spacing matches eval world geometry.
        Random rotation each episode keeps geometry consistent but orientation varied."""
        centers      = self._cluster_centers()
        tags_per     = NUM_TAGS // len(centers)
        tag_idx      = 0
        GRID_SPACING = 0.10   # metres between tags — matches eval world

        for cx, cy in centers:
            count = (tags_per if tag_idx + tags_per <= NUM_TAGS
                     else NUM_TAGS - tag_idx)
            cols  = max(1, int(math.ceil(math.sqrt(count))))
            rows  = int(math.ceil(count / cols))
            rot   = random.uniform(0, math.pi / 2)   # random orientation per episode

            k = 0
            for row in range(rows):
                for col in range(cols):
                    if k >= count or tag_idx >= NUM_TAGS:
                        break
                    dx = (col - (cols - 1) / 2.0) * GRID_SPACING
                    dy = (row - (rows - 1) / 2.0) * GRID_SPACING
                    tx = cx + dx * math.cos(rot) - dy * math.sin(rot)
                    ty = cy + dx * math.sin(rot) + dy * math.cos(rot)
                    self.tag_nodes[tag_idx].getField("translation").setSFVec3f(
                        [max(-2.3, min(2.3, tx)), max(-2.3, min(2.3, ty)), 0.01375])
                    tag_idx += 1
                    k += 1

        while tag_idx < NUM_TAGS:
            cx, cy = random.choice(centers)
            rot = random.uniform(0, math.pi * 2)
            tx  = cx + math.cos(rot) * GRID_SPACING
            ty  = cy + math.sin(rot) * GRID_SPACING
            self.tag_nodes[tag_idx].getField("translation").setSFVec3f(
                [max(-2.3, min(2.3, tx)), max(-2.3, min(2.3, ty)), 0.01375])
            tag_idx += 1

    def _cluster_centers(self):
        """Curriculum matching centralized proportions (14%/33%/53%) scaled to ~427 episodes.
        With 7M steps / 16384 per episode ≈ 427 episodes total:
          Phase 1 (ep   1-60):  60 eps = 14% — bootstraps basic pickup/deposit
          Phase 2 (ep  61-201): 141 eps = 33% — medium-range pheromone following
          Phase 3 (ep 202+  ): 226 eps = 53% — full arena, 6-8 clusters
        """
        if self.total_episodes < 60:
            max_dist, n_clusters = 1.0, 2
        elif self.total_episodes < 201:
            max_dist, n_clusters = 1.8, random.randint(3, 5)
        else:
            max_dist, n_clusters = 2.3, random.randint(6, 8)

        centers = []
        for _ in range(n_clusters):
            for _ in range(60):
                angle = random.uniform(0, 2 * math.pi)
                dist  = random.uniform(0.7, max_dist)
                cx    = math.cos(angle) * dist
                cy    = math.sin(angle) * dist
                if all(math.sqrt((cx-ox)**2 + (cy-oy)**2) > 0.8
                       for ox, oy in centers):
                    centers.append((cx, cy))
                    break
        return centers if centers else [(1.5, 0.0)]

    # =========================================================================
    # Main loop
    # =========================================================================

    def run(self):
        print("=" * 60)
        print("DECENTRALIZED INDEPENDENT TRAINING SUPERVISOR")
        print("Each robot runs its own PPO — supervisor handles tag mechanics only")
        print("=" * 60 + "\n")

        # Initial episode setup
        self._respawn_robots()
        self._respawn_tags()
        self.supervisor.step(self.timestep)

        while self.run_step():
            pass


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='decentralized_indep_v1',
                        help='Run name — written to current_run_name.txt for robots to read')
    args = parser.parse_args()

    # Write run name so robot controllers can read it on startup
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    cfg_path     = os.path.join(project_root, 'current_run_name.txt')
    with open(cfg_path, 'w') as f:
        f.write(args.run_name)
    print(f"[SUPERVISOR] Run name: {args.run_name}")
    print(f"[SUPERVISOR] Config written to: {cfg_path}")

    sup = DecentralizedTrainSupervisor()
    sup.run()
