import math
import random
from deepbots.robots.controllers.csv_robot import CSVRobot

# =============================================================================
# DECENTRALIZED ROBOT BASE CLASS — v4
#
# Pheromone system upgraded to match centralized CPFA list-based design:
#   - pheromone_list: [{x, y, weight}, ...] — one entry per distinct cluster
#   - Created at pickup (P2P, immediate) vs centralized (at nest deposit)
#   - Broadcast: strongest entry every step, channel 10, 2m range
#   - Receive:   location-based merge (MERGE_RADIUS=0.3m), accept-if-stronger
#   - Decay:     weight × exp(-0.01 × dt_sec), pruned when weight < 0.001
#   - Roulette selection from list at deposit → next trip target
#   - Site fidelity: robot's own last pickup location, priority over roulette
#
# Obs dims produced by _obs_components() — 20D total:
#   [0:8]  prox (in create_message)
#   [8]    tag_visible        (from supervisor)
#   [9]    tag_dist_norm      (from supervisor)
#   [10]   tag_angle_norm     (from supervisor)
#   [11]   carrying
#   [12]   base_dist_norm
#   [13]   base_angle_norm
#   [14]   site_known         ← robot's own last pickup target
#   [15]   site_dist_norm
#   [16]   site_angle_norm
#   [17]   phero_known        ← roulette-selected target from list
#   [18]   phero_dist_norm
#   [19]   phero_angle_norm
#
# Message to supervisor (17 floats — GPS kept at [15,16]):
#   [0:8]  prox, [8] carrying, [9] base_dist, [10] base_angle,
#   [11] site_known, [12] phero_known, [13-14] reserved,
#   [15] gps_x, [16] gps_y
# =============================================================================

PHEROMONE_MIN        = 0.001
PHEROMONE_DECAY_RATE = 0.01   # per second — matches centralized
MERGE_RADIUS         = 0.3    # metres — same cluster if within this distance
TARGET_ARRIVAL_DIST  = 0.18   # metres — target considered reached
BASE_X               = 0.0
BASE_Y               = 0.0


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

        # ── Pheromone P2P (channel 10, Webots enforces 2m range) ─────
        self.phero_emitter  = self.getDevice('phero_emitter')
        self.phero_receiver = self.getDevice('phero_receiver')
        self.phero_receiver.enable(self.time_step)

        # ── Carrying state ────────────────────────────────────────────
        self.carrying = False

        # ── Pheromone list (CPFA-style, one entry per cluster) ────────
        self.pheromone_list = []   # [{"x": float, "y": float, "weight": float}, ...]

        # ── Site fidelity (own last pickup — private, never broadcast) ─
        self._site_fidelity_pos   = None   # (x, y) of own last pickup
        self._last_pickup_weight  = 0.0    # pickup_signal at last pickup (encodes density)

        # ── Current trip target — set at deposit ──────────────────────
        # None | ('site', x, y) | ('phero', x, y)
        self._current_target = None

    # =========================================================================
    # Pheromone list management
    # =========================================================================

    def _add_pheromone(self, x, y, weight):
        """Add pickup location to list, or refresh weight if location already known."""
        for entry in self.pheromone_list:
            d = math.sqrt((entry['x'] - x) ** 2 + (entry['y'] - y) ** 2)
            if d < MERGE_RADIUS:
                if weight > entry['weight']:
                    entry['weight'] = weight
                return
        self.pheromone_list.append({'x': x, 'y': y, 'weight': weight})

    def _decay_pheromone(self):
        """Exponential time-based decay — matches centralized CPFA rate."""
        dt    = self.time_step / 1000.0
        decay = math.exp(-PHEROMONE_DECAY_RATE * dt)
        for e in self.pheromone_list:
            e['weight'] *= decay
        self.pheromone_list = [e for e in self.pheromone_list
                               if e['weight'] > PHEROMONE_MIN]

    def _receive_pheromone(self):
        """
        Receive P2P broadcasts from robots within 2m.
        Location-based merge: accept-if-stronger per cluster location.
        Prevents list bloat from multiple robots broadcasting same cluster.
        """
        while self.phero_receiver.getQueueLength() > 0:
            try:
                msg   = self.phero_receiver.getString()
                parts = [float(x) for x in msg.split(',')]
                if len(parts) >= 3:
                    rx, ry, rw = parts[0], parts[1], parts[2]
                    if rw >= PHEROMONE_MIN:
                        self._add_pheromone(rx, ry, rw)
            except (ValueError, IndexError):
                pass
            finally:
                self.phero_receiver.nextPacket()

    def _broadcast_pheromone(self):
        """Broadcast strongest known cluster every step within 2m."""
        if not self.pheromone_list:
            return
        best = max(self.pheromone_list, key=lambda e: e['weight'])
        if best['weight'] >= PHEROMONE_MIN:
            self.phero_emitter.send(
                f"{best['x']},{best['y']},{best['weight']}".encode('utf-8'))

    def _roulette_select(self):
        """
        Weighted random selection from pheromone list.
        Higher weight = more likely selected. Matches centralized CPFA roulette.
        Returns (x, y) or None if list is empty.
        """
        active = [e for e in self.pheromone_list if e['weight'] > PHEROMONE_MIN]
        if not active:
            return None
        total  = sum(e['weight'] for e in active)
        r      = random.random() * total
        cumul  = 0.0
        for e in active:
            cumul += e['weight']
            if r <= cumul:
                return (e['x'], e['y'])
        return (active[-1]['x'], active[-1]['y'])

    def _assign_target(self):
        """
        CPFA target assignment — called at deposit.
        Matches CPFA's probabilistic structure exactly:
          Priority 1: site fidelity — Poisson-like gate using pickup density as probability.
                      _last_pickup_weight = pickup_signal (0.2–1.0, density-encoded).
                      High density cluster → high probability of return. Decays via
                      depletion feedback (weight halved each failed visit in train robot).
          Priority 2: pheromone roulette — weighted random from list (matches CPFA roulette).
          Priority 3: free exploration — PPO learns efficient search, beating CPFA random walk.
        """
        # Priority 1: site fidelity — probabilistic Poisson gate (mirrors CPFA)
        if self._site_fidelity_pos is not None:
            if random.random() < self._last_pickup_weight:
                sx, sy = self._site_fidelity_pos
                self._current_target = ('site', sx, sy)
                return

        # Priority 2: pheromone roulette from received broadcasts
        selected = self._roulette_select()
        if selected is not None:
            self._current_target = ('phero', selected[0], selected[1])
            return

        # Priority 3: PPO-learned exploration (advantage over CPFA random walk)
        self._current_target = None

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
                site_dist   = min(s_d / 3.5, 1.0)
                site_angle  = math.copysign(math.acos(s_dot), s_cross) / math.pi

        # ── Pheromone target [17-19] — zeroed when carrying ───────────
        phero_known = phero_dist = phero_angle = 0.0
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
                phero_dist  = min(p_d / 3.5, 1.0)
                phero_angle = math.copysign(math.acos(p_dot), p_cross) / math.pi

        return {
            "pos_x":            pos_x,
            "pos_y":            pos_y,
            "base_dist_norm":   min(dist_to_base / 3.5, 1.0),
            "base_angle_norm":  angle_to_base / math.pi,
            "site_known":       site_known,
            "site_dist_norm":   site_dist,
            "site_angle_norm":  site_angle,
            "phero_known":      phero_known,
            "phero_dist_norm":  phero_dist,
            "phero_angle_norm": phero_angle,
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
        self._broadcast_pheromone()

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
        Receive [left, right, pickup_signal] from supervisor.
        pickup_signal > 0  → pickup; value = density-based weight (0.2–1.0)
        pickup_signal < 0  → deposit confirmed
        pickup_signal = 0  → normal step
        """
        if not message:
            return
        try:
            left          = float(message[0])
            right         = float(message[1])
            pickup_signal = float(message[2]) if len(message) > 2 else 0.0

            max_speed    = 6.28
            scale_factor = 6.0
            self.left_motor.setVelocity(
                max(min(left  * scale_factor, max_speed), -max_speed))
            self.right_motor.setVelocity(
                max(min(right * scale_factor, max_speed), -max_speed))

            if pickup_signal > 0.0 and not self.carrying:
                self.carrying = True
                gps = self.gps.getValues()
                px, py = gps[0], gps[1]
                # Add to own pheromone list
                self._add_pheromone(px, py, pickup_signal)
                # Store site fidelity
                self._site_fidelity_pos  = (px, py)
                self._last_pickup_weight = pickup_signal
                # Clear current target — arrived at cluster
                self._current_target = None

            elif pickup_signal < 0.0 and self.carrying:
                self.carrying = False
                # Assign next trip target via CPFA priority
                self._assign_target()

        except (ValueError, IndexError):
            pass


if __name__ == '__main__':
    robot = EpuckDecentralizedV4()
    robot.run()
