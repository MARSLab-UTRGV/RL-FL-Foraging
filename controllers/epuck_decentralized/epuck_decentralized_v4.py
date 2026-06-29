import math
import os
import random
from deepbots.robots.controllers.csv_robot import CSVRobot

# =============================================================================
# DECENTRALIZED ROBOT BASE CLASS — v4 (synced to centralized CPFA-RL v5)
#
# Pheromone system — CPFA parameters identical to centralized + cpfa_baseline:
#   - pheromone_list: [{x, y, weight}, ...] — one entry per distinct cluster
#   - Created at PICKUP (P2P, immediate broadcast) — decentralized advantage
#   - Poisson CDF gate (RATE_OF_LAYING_PHEROMONE=3.0) at pickup: higher
#     resource density → higher probability of adding to own list
#   - Broadcast: strongest entry every step, channel 10, 0.25m range
#   - Receive:   location-based merge (MERGE_RADIUS=0.3m), accept-if-stronger
#   - Decay:     weight × exp(-0.05 × dt_sec), τ≈20s — matches centralized
#   - Roulette selection from list at deposit → next trip pheromone target
#   - Site fidelity: Poisson CDF gate (RATE_OF_SITE_FIDELITY=1.376) at deposit
#
# Obs dims produced by _obs_components() — 17D (no tag sensing, matches centralized):
#   [0:8]  prox
#   [8]    carrying
#   [9]    base_dist_norm
#   [10]   base_angle_norm
#   [11]   site_known
#   [12]   site_dist_norm
#   [13]   site_angle_norm
#   [14]   phero_known
#   [15]   phero_dist_norm
#   [16]   phero_angle_norm
#   obs[11-16] zeroed when carrying=True
#
# Message to supervisor (17 floats — GPS at [15,16]):
#   [0:8]  prox, [8] carrying, [9] base_dist, [10] base_angle,
#   [11] site_known, [12] phero_known, [13-14] reserved,
#   [15] gps_x, [16] gps_y
# =============================================================================

PHEROMONE_MIN              = 0.001
PHEROMONE_DECAY_RATE       = 0.05    # /sec — τ≈20s, matches centralized + cpfa_baseline
RATE_OF_LAYING_PHEROMONE   = 3.0     # Poisson λ: higher density → more likely to lay at pickup
RATE_OF_SITE_FIDELITY      = 1.376   # ARGoS-evolved — matches centralized + cpfa_baseline
MERGE_RADIUS               = 0.3     # metres — same cluster if within this distance
TARGET_ARRIVAL_DIST        = 0.05    # metres — matches centralized ARGoS TargetDistanceTolerance
PHERO_BROADCAST_INTERVAL   = 30      # steps between pheromone broadcasts (~1.0 s at 32 ms/step)
BASE_X                   = 0.0
BASE_Y                   = 0.0
GIVE_UP_CHECK_STEPS      = 78      # steps between give-up checks (5s at 64ms) — matches centralized
GIVE_UP_PROB             = 0.0189  # P(give-up) per check → E[give-up]≈264s — matches centralized


class EpuckDecentralizedV4(CSVRobot):

    def __init__(self):
        super().__init__()
        self.time_step = int(self.getBasicTimeStep())

        # ── Proximity sensors ─────────────────────────────────────────
        self.ps = []
        for i in range(8):
            s = self.getDevice(f'ps{i}')
            s.enable(self.time_step)
            self.ps.append(s)

        # ── Camera ───────────────────────────────────────────────────
        self.camera = self.getDevice('camera')
        if self.camera:
            self.camera.enable(self.time_step)

        # ── Motors ───────────────────────────────────────────────────
        self.left_motor  = self.getDevice('left wheel motor')
        self.right_motor = self.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # ── GPS + IMU ────────────────────────────────────────────────
        self.gps = self.getDevice('gps')
        self.gps.enable(self.time_step)
        self.imu = self.getDevice('inertial_unit')
        self.imu.enable(self.time_step)

        # ── Pheromone P2P (channel 10, 0.5m range) ──────────────────
        self.phero_emitter  = self.getDevice('phero_emitter')
        self.phero_receiver = self.getDevice('phero_receiver')
        self.phero_receiver.enable(self.time_step)

        # ── Carrying state ────────────────────────────────────────────
        self.carrying = False

        # ── Pheromone list (CPFA-style, one entry per cluster) ────────
        self.pheromone_list      = []   # [{"x": float, "y": float, "weight": float}, ...]
        self._phero_broadcast_step = 0  # broadcast every PHERO_BROADCAST_INTERVAL steps

        # ── Site fidelity (own last pickup — private, never broadcast) ─
        self._site_fidelity_pos = None   # (x, y) of own last pickup
        self._resource_density  = 0      # raw tag count within 0.5m at pickup (matches centralized)

        # ── Current trip target — set at nest deposit ─────────────────
        # None | ('site', x, y) | ('phero', x, y)
        self._current_target = None

        # ── Give-up mechanism (mirrors centralized CPFA) ──────────────
        # gave_up=True: site fidelity suppressed in _assign_target().
        # Unlike centralized, robot does NOT RTB — continues exploring freely.
        # Cleared only at next successful pickup.
        self._gave_up            = False
        self._give_up_timer      = 0
        self._steps_since_pickup = 0   # increments while not carrying; reset at pickup
        self._carrying_prev      = False

        # ── Arena geometry — subclasses override _arena_half for non-5×5 eval ──
        self._arena_half = 2.5                              # default: 5×5 arena
        self._max_dist   = self._arena_half * math.sqrt(2) # max distance from base (corner)

        # ── Per-robot log file (set via _open_robot_log in subclass) ──
        self._robot_log_path = None

    # =========================================================================
    # CPFA helpers
    # =========================================================================

    def _poisson_cdf(self, n, rate):
        """P(X ≤ n) where X ~ Poisson(rate). Matches centralized CPFA + baseline.
        Higher n (resource density) → higher probability → more likely to act."""
        if n < 0:
            return 0.0
        cdf = term = math.exp(-rate)
        for i in range(1, n + 1):
            term *= rate / i
            cdf  += term
        return min(cdf, 1.0)

    def _open_robot_log(self, run_name):
        """Open per-robot log file at logs/<run_name>/<robot_name>_log.txt."""
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..'))
        log_dir = os.path.join(project_root, 'logs', run_name)
        os.makedirs(log_dir, exist_ok=True)
        self._robot_log_path = os.path.join(log_dir, f'{self.getName()}_log.txt')

    def _log(self, msg):
        """Print to Webots robot console and append to per-robot log file."""
        print(msg)
        if self._robot_log_path:
            with open(self._robot_log_path, 'a') as f:
                f.write(msg + '\n')

    # =========================================================================
    # Pheromone list management
    # =========================================================================

    def _print_pheromone_list(self, event):
        entries = " | ".join(
            f"({e['x']:.2f},{e['y']:.2f}) w={e['weight']:.3f} den={e.get('density', 1)}"
            for e in self.pheromone_list
        )
        self._log(f"[PHERO {event}] {self.getName()} "
                  f"[{len(self.pheromone_list)}]: "
                  f"{entries if entries else 'empty'}")

    def _add_pheromone(self, x, y, weight=1.0, density=1, source="LAID"):
        """Add location to pheromone list, or accept-if-stronger merge.
        Weight starts at 1.0 (decay brings it down from there).
        source: 'LAID' = own pickup, 'RECV' = P2P broadcast from neighbor."""
        for entry in self.pheromone_list:
            d = math.sqrt((entry['x'] - x) ** 2 + (entry['y'] - y) ** 2)
            if d < MERGE_RADIUS:
                if weight > entry['weight']:
                    entry['weight']  = weight
                    entry['density'] = density
                    self._print_pheromone_list(f"{source}-UPD")
                return
        self.pheromone_list.append({'x': x, 'y': y, 'weight': weight, 'density': density})
        self._print_pheromone_list(f"{source}-ADD")

    def _decay_pheromone(self):
        """Exponential time-based decay — matches centralized CPFA rate."""
        before = len(self.pheromone_list)
        dt     = self.time_step / 1000.0
        decay  = math.exp(-PHEROMONE_DECAY_RATE * dt)
        for e in self.pheromone_list:
            e['weight'] *= decay
        self.pheromone_list = [e for e in self.pheromone_list
                               if e['weight'] > PHEROMONE_MIN]
        if len(self.pheromone_list) < before:
            self._print_pheromone_list("EXPIRED")

    def _receive_pheromone(self):
        """
        Receive P2P broadcasts from robots within 0.5m.
        Location-based merge: accept-if-stronger per cluster location.
        Logged as RECV-ADD/RECV-UPD to distinguish from own pickup (LAID-ADD).
        """
        while self.phero_receiver.getQueueLength() > 0:
            try:
                msg   = self.phero_receiver.getString()
                parts = [float(x) for x in msg.split(',')]
                if len(parts) >= 3:
                    rx, ry, rw = parts[0], parts[1], parts[2]
                    rd = int(parts[3]) if len(parts) >= 4 else 1
                    if rw >= PHEROMONE_MIN:
                        self._add_pheromone(rx, ry, rw, rd, source="RECV")
            except (ValueError, IndexError):
                pass
            finally:
                self.phero_receiver.nextPacket()

    def _broadcast_pheromone(self):
        """Broadcast strongest known cluster every step within 0.5m."""
        if not self.pheromone_list:
            return
        best = max(self.pheromone_list, key=lambda e: e['weight'])
        if best['weight'] >= PHEROMONE_MIN:
            self.phero_emitter.send(
                f"{best['x']},{best['y']},{best['weight']},{best.get('density', 1)}".encode('utf-8'))

    def _roulette_select(self):
        """
        Density-weighted roulette selection from pheromone list.
        Combined score = weight × density — prefers fresh, high-density clusters.
        Entries with density=0 (marked depleted) are excluded entirely.
        Returns (x, y, density) or None if list is empty.
        """
        active = [e for e in self.pheromone_list
                  if e['weight'] > PHEROMONE_MIN and e.get('density', 1) > 0]
        if not active:
            return None
        scores = [e['weight'] * e.get('density', 1) for e in active]
        total  = sum(scores)
        if total <= 0:
            return None
        r      = random.random() * total
        cumul  = 0.0
        for e, sc in zip(active, scores):
            cumul += sc
            if r <= cumul:
                return (e['x'], e['y'], e.get('density', 1))
        last = active[-1]
        return (last['x'], last['y'], last.get('density', 1))

    def _assign_target(self):
        """CPFA target assignment — called at nest deposit.
        Matches centralized CPFA priority structure exactly:
          Priority 1: site fidelity  — Poisson CDF gate (RATE_OF_SITE_FIDELITY=1.376).
                      Higher resource density → higher P(return to own last pickup).
          Priority 2: pheromone roulette — weighted random from P2P-received list.
          Priority 3: free exploration  — PPO learns efficient search.
        """
        prefix = "[EMPTY_RTN]" if self._gave_up else "[TARGET]  "

        # Priority 1: site fidelity — skipped if gave_up (ARGoS: updateFidelity=False)
        if self._site_fidelity_pos is not None and not self._gave_up:
            sf_prob = self._poisson_cdf(self._resource_density, RATE_OF_SITE_FIDELITY)
            if random.random() < sf_prob:
                sx, sy = self._site_fidelity_pos
                self._current_target = ('site', sx, sy)
                self._log(f"  {prefix} {self.getName()} → SITE ({sx:.2f},{sy:.2f})")
                return

        # Priority 2: pheromone roulette from P2P-received broadcasts
        selected = self._roulette_select()
        if selected is not None:
            self._current_target = ('phero', selected[0], selected[1], selected[2])
            self._log(f"  {prefix} {self.getName()} → PHERO ({selected[0]:.2f},{selected[1]:.2f}) density={selected[2]}")
            return

        # Priority 3: free exploration
        self._current_target = None
        self._log(f"  {prefix} {self.getName()} → EXPLORE")

    # =========================================================================
    # Self-observed obs components (GPS + IMU — fully onboard)
    # =========================================================================

    def _obs_components(self):
        gps   = self.gps.getValues()
        pos_x = gps[0]
        pos_y = gps[1]
        yaw   = self.imu.getRollPitchYaw()[2]
        fwd   = [math.cos(yaw), math.sin(yaw)]

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

        # ── Site fidelity [14-16] — zeroed when carrying ──────────────
        site_known = site_dist = site_angle = 0.0
        if (not self.carrying
                and self._current_target is not None
                and self._current_target[0] == 'site'):
            sx  = self._current_target[1]
            sy  = self._current_target[2]
            sdx = sx - pos_x
            sdy = sy - pos_y
            s_d = math.sqrt(sdx * sdx + sdy * sdy)
            if s_d > 0.001:
                s_norm  = [sdx / s_d, sdy / s_d]
                s_dot   = max(min(fwd[0]*s_norm[0] + fwd[1]*s_norm[1], 1.0), -1.0)
                s_cross = fwd[0]*s_norm[1] - fwd[1]*s_norm[0]
                site_known  = 1.0
                site_dist   = min(s_d / self._max_dist, 1.0)
                site_angle  = math.copysign(math.acos(s_dot), s_cross) / math.pi

        # ── Pheromone target — zeroed when carrying ───────────────────
        phero_known = phero_dist = phero_angle = phero_density_norm = 0.0
        if (not self.carrying
                and self._current_target is not None
                and self._current_target[0] == 'phero'):
            px  = self._current_target[1]
            py  = self._current_target[2]
            pdx = px - pos_x
            pdy = py - pos_y
            p_d = math.sqrt(pdx * pdx + pdy * pdy)
            if p_d > 0.001:
                p_norm  = [pdx / p_d, pdy / p_d]
                p_dot   = max(min(fwd[0]*p_norm[0] + fwd[1]*p_norm[1], 1.0), -1.0)
                p_cross = fwd[0]*p_norm[1] - fwd[1]*p_norm[0]
                phero_known = 1.0
                phero_dist  = min(p_d / self._max_dist, 1.0)
                phero_angle = math.copysign(math.acos(p_dot), p_cross) / math.pi
                if len(self._current_target) > 3:
                    phero_density_norm = min(self._current_target[3] / 12.0, 1.0)

        return {
            "pos_x":              pos_x,
            "pos_y":              pos_y,
            "base_dist_norm":     min(dist_to_base / self._max_dist, 1.0),
            "base_angle_norm":    angle_to_base / math.pi,
            "site_known":         site_known,
            "site_dist_norm":     site_dist,
            "site_angle_norm":    site_angle,
            "phero_known":        phero_known,
            "phero_dist_norm":    phero_dist,
            "phero_angle_norm":   phero_angle,
            "phero_density_norm": phero_density_norm,
        }

    # =========================================================================
    # deepbots interface
    # =========================================================================

    def create_message(self):
        """
        Every step:
          1. Decay own pheromone list
          2. Receive P2P broadcasts → merge into list
          3. Broadcast strongest entry to nearby robots
          4. Compute obs components
          5. Return 17-float message to supervisor (GPS at [15,16])
        """
        self._decay_pheromone()
        self._receive_pheromone()
        self._phero_broadcast_step += 1
        if self._phero_broadcast_step >= PHERO_BROADCAST_INTERVAL:
            self._phero_broadcast_step = 0
            self._broadcast_pheromone()

        # Give-up: mirrors centralized CPFA exactly.
        # Timer runs ONLY during free exploration (no target) — same as centralized
        # which puts the timer in the "else: # No target" branch.
        # When give-up fires: _assign_target(gave_up=True) skips stale site fidelity.
        # Robot does NOT RTB (decentralized advantage over centralized).
        # Cleared only at next successful pickup.
        # Track steps since last pickup (for give-up print)
        if self.carrying and not self._carrying_prev:
            self._steps_since_pickup = 0
        elif not self.carrying:
            self._steps_since_pickup += 1
        self._carrying_prev = self.carrying

        if not self.carrying:
            if self._current_target is None:   # free exploration only — matches centralized
                self._give_up_timer += 1
                if self._give_up_timer >= GIVE_UP_CHECK_STEPS:
                    self._give_up_timer = 0
                    if random.random() < GIVE_UP_PROB:
                        self._log(f"[GIVE-UP]  {self.getName()} giving up after "
                                  f"{self._steps_since_pickup} searching steps")
                        self._gave_up = True
                        self._assign_target()
                        self._gave_up = False  # mirrors centralized: cleared after new target assigned
        else:
            self._give_up_timer = 0

        prox = [s.getValue() / 4096.0 for s in self.ps]
        obs  = self._obs_components()

        return prox + [
            1.0 if self.carrying else 0.0,  # [8]
            obs["base_dist_norm"],           # [9]
            obs["base_angle_norm"],          # [10]
            obs["site_known"],               # [11]
            obs["phero_known"],              # [12]
            0.0,                             # [13] reserved
            0.0,                             # [14] reserved
            obs["pos_x"],                    # [15] GPS — supervisor reads here
            obs["pos_y"],                    # [16] GPS
        ]

    def use_message_data(self, message):
        """
        Receive [tag_visible, tag_dist_norm, tag_angle_norm, pickup_signal] from supervisor.
        pickup_signal > 0  → pickup; value encodes resource density (0.2–1.0)
        pickup_signal < 0  → deposit confirmed
        pickup_signal = 0  → normal step

        Pheromone created at PICKUP with Poisson CDF gate, then broadcast immediately.
        Target assigned at DEPOSIT via CPFA Poisson CDF priority.
        """
        if not message:
            return
        try:
            pickup_signal = float(message[3]) if len(message) > 3 else 0.0

            if pickup_signal > 0.0 and not self.carrying:
                self.carrying = True
                gps = self.gps.getValues()
                px, py = gps[0], gps[1]
                # Decode integer resource density from supervisor's normalised signal
                density = max(0, round(pickup_signal))
                # Store for site fidelity at next deposit
                self._site_fidelity_pos = (px, py)
                self._resource_density  = density
                # CPFA pheromone laying at PICKUP — Poisson CDF gate
                # (P2P: starts broadcasting immediately to robots within 1m)
                lay_prob = self._poisson_cdf(density, RATE_OF_LAYING_PHEROMONE)
                if random.random() < lay_prob:
                    self._add_pheromone(px, py, 1.0, density)
                # Clear current target — arrived at cluster
                self._current_target = None
                # Successful pickup clears give-up — site fidelity re-enabled
                self._gave_up       = False
                self._give_up_timer = 0

            elif pickup_signal < 0.0 and self.carrying:
                self.carrying = False
                # CPFA target assignment at nest deposit (site → phero → explore)
                self._assign_target()

        except (ValueError, IndexError):
            pass


if __name__ == '__main__':
    robot = EpuckDecentralizedV4()
    robot.run()
