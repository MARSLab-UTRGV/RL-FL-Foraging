import math
import random
from controller import Supervisor

# =============================================================================
# CPFA BASELINE — Central Place Foraging Algorithm (hand-coded)
#
# This is the performance baseline for the CoRL 2026 paper.
# It implements CPFA's 4-state machine with the IDENTICAL pheromone model
# used in the RL training supervisor (epuck_foraging_supervisor_cpfa.py):
#
#   Pheromone model (ARGoS-calibrated):
#     - List-based: each entry = {x, y, weight, resource_density}
#     - Created at NEST DEPOSIT (not at pickup) via Poisson CDF gate
#     - Roulette-wheel selection weighted by weight
#     - Site fidelity: robot's own last pickup location (Poisson CDF priority)
#     - Exponential decay: exp(-0.05 * dt_sec) per step — lifetime ~20s (scaled from ARGoS 0.337)
#     - Pruned when weight < 0.001
#     - Parameters: RATE_OF_LAYING=3.0 (scaled from 14.44), RATE_OF_SITE_FIDELITY=1.376
#
#   State machine (replaces PPO — CPFA's original 4-state machine):
#     SEARCHING  → correlated random walk (CRW) toward cluster or arena.
#                  Switch to RETURNING immediately on food pickup.
#                  After probabilistic give-up (P=0.0189 per 5s) → RETURNING empty.
#
#     DEPARTING  → drive directly toward the nest-assigned target
#                  (site fidelity location or pheromone roulette location).
#                  On arrival (< 0.18m) → switch to SEARCHING around that spot.
#
#     RETURNING  → drive to nest (P2 hard override — identical to RL system).
#                  On arrival (< 0.25m):
#                    1. If carrying food: deposit, lay pheromone (Poisson CDF),
#                       assign new target → switch to DEPARTING or SEARCHING.
#                    2. If empty (gave up): assign new target → DEPARTING or SEARCHING.
#
#   Hard overrides (identical to RL training):
#     P1: Wall escape (wall < 0.35m or prox > 0.55) → steer to centre, gain 4.0
#     P2: RTB when carrying → steer to nest, gain 2.5
#     (P3 removed — matching the RL supervisor)
#
#
# Run:
#   webots worlds/eval_sample1_7x7.wbt
#   python3 controllers/cpfa_baseline/cpfa_baseline_7x7.py
# =============================================================================

SEARCHING  = "SEARCHING"
DEPARTING  = "DEPARTING"
RETURNING  = "RETURNING"
SURVEYING  = "SURVEYING"


class CPFABaseline(Supervisor):
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())

        self.num_robots = 4
        self.num_tags   = 128

        # --- World nodes ---
        self.robot_nodes = [self.getFromDef(f"ROBOT{i+1}") for i in range(self.num_robots)]
        self.tag_nodes   = [self.getFromDef(f"APRILTAG_{i+1}") for i in range(self.num_tags)]
        self.base_node   = self.getFromDef("BASE_STATION")

        # --- Communication ---
        self.emitters  = [self.getDevice(f"emitter{i+1}") for i in range(self.num_robots)]
        self.receivers = [self.getDevice(f"receiver{i+1}") for i in range(self.num_robots)]
        for r in self.receivers:
            r.enable(self.timestep)

        # --- Per-robot sensor state ---
        self.robot_states   = [None]  * self.num_robots
        self.carrying_state = [False] * self.num_robots

        # --- CPFA per-robot state machine ---
        self.robot_mode            = [DEPARTING] * self.num_robots  # ARGoS starts in DEPARTING
        # nest_target: None | ('site', x, y) | ('phero', x, y)
        # Assigned ONLY at nest return — identical to training supervisor
        self.nest_target           = [None] * self.num_robots
        # search_wp: current random-walk waypoint during SEARCHING
        self.search_wp             = [None] * self.num_robots
        # steps_without_pickup: give-up counter (matches training obs[20])
        self.steps_without_pickup  = [0]    * self.num_robots  # incremented each searching step
        # survey_count: which of the 5 heading targets (0,90,180,270,360°) we are on
        # ARGoS Surveying(): robot_count * π/2 → 0, π/2, π, 3π/2, 2π → then RETURNING
        self.survey_count          = [0]    * self.num_robots
        # survey_steps: guard against physical obstruction — force RETURNING after timeout
        self.survey_steps          = [0]    * self.num_robots

        # --- CPFA pheromone state ---
        self.carried_from      = [None] * self.num_robots  # (x,y) pickup location
        self.resource_density  = [0]    * self.num_robots  # tag count within 0.5m at pickup
        self.site_fidelity_pos = [None] * self.num_robots  # robot's own last pickup

        # --- CPFA pheromone list (identical to training supervisor) ---
        self.pheromone_list = []

        # --- CPFA parameters — scaled for Webots (4 robots, 5x5m, 16 items/cluster, no respawn)
        # ARGoS source: Clustered_CPFA_r64_tag512_16by16.xml
        # RATE_OF_LAYING_PHEROMONE: ARGoS=14.44 (32 items/cluster → density 8-16 → P(lay)≈1).
        #   Webots: 16 items/cluster → density 1-3 → P(lay|λ=14.44)≈0 always. Scaled to 3.0
        #   so P(lay|density=2, λ=3.0)≈0.42.
        self.RATE_OF_LAYING_PHEROMONE = 3.0
        self.RATE_OF_SITE_FIDELITY    = 1.376   # unchanged — already works in Webots
        # RATE_OF_PHEROMONE_DECAY: ARGoS=0.337/s (τ≈3s) with 64 robots constantly reinforcing.
        #   Webots: 4 robots → trail not reinforced for ~4+ min → fully decayed. Scaled to 0.05/s (τ≈20s).
        self.RATE_OF_PHEROMONE_DECAY  = 0.05
        self.PHEROMONE_MIN            = 0.001

        # Give-up / uninformed-switch: both checked every 5 sim-seconds.
        # At 32ms/step → 5 sec = 156 steps.
        self.PROB_RETURN_TO_NEST      = 0.0189  # per 5-sec check in SEARCHING  → E[give-up] ~265s
        self.PROB_SWITCH_TO_SEARCHING = 0.765   # per 5-sec check in uninformed DEPARTING
        self.GIVE_UP_CHECK_STEPS      = 156     # 5 seconds at 32ms/step
        self.give_up_timer            = [0]     * self.num_robots
        # Per-robot DEPARTING timer — ARGoS each robot checks independently (not global step_count)
        self.departing_timer          = [0]     * self.num_robots

        # uninformed_wall_target[i]: wall-edge position assigned when nest_target=None.
        # ARGoS SetRandomSearchLocation() — robot heads to a random wall, switches
        # to SEARCHING probabilistically before arriving (ProbabilityOfSwitchingToSearching).
        # Initialized at startup because robots start in uninformed DEPARTING (matches ARGoS).
        self.uninformed_wall_target   = [self._random_wall_target() for _ in range(self.num_robots)]

        # gave_up[i]: True after give-up until next successful pickup.
        # Suppresses stale site fidelity at next nest return (ARGoS: updateFidelity=False).
        self.gave_up              = [False] * self.num_robots

        # CRW parameters (CPFA uninformed/informed search)
        # ARGoS UninformedSearchVariation = 3.67 rad (was math.radians(30°) = 0.524 rad)
        self.UNINFORMED_SEARCH_VARIATION   = 3.67    # rad — matches ARGoS evolved value
        # ARGoS RateOfInformedSearchDecay = 0.346/waypoint (formula overhaul in Gap 4)
        self.RATE_OF_INFORMED_SEARCH_DECAY = 0.346   # /waypoint (was 0.0002/step)
        self.search_heading            = [0.0] * self.num_robots
        self.informed_search_steps     = [0]   * self.num_robots  # waypoints since informed

        # --- Metrics ---
        self.total_pickups  = 0
        self.total_deposits = 0
        self.step_count     = 0

        print("=" * 65)
        print("CPFA BASELINE — ARGoS-calibrated parameters  (7×7m arena  128 tags)")
        print(f"  RateOfLayingPheromone        = {self.RATE_OF_LAYING_PHEROMONE}")
        print(f"  RateOfSiteFidelity           = {self.RATE_OF_SITE_FIDELITY}")
        print(f"  RateOfPheromoneDecay         = {self.RATE_OF_PHEROMONE_DECAY} /s  (lifetime ~{-math.log(0.001)/self.RATE_OF_PHEROMONE_DECAY:.0f}s)")
        print(f"  ProbabilityOfReturningToNest = {self.PROB_RETURN_TO_NEST} (every 5 sec → E[give-up] ~{5/self.PROB_RETURN_TO_NEST:.0f}s)")
        print(f"  UninformedSearchVariation    = {self.UNINFORMED_SEARCH_VARIATION:.2f} rad ({math.degrees(self.UNINFORMED_SEARCH_VARIATION):.1f}°)")
        print(f"  RateOfInformedSearchDecay    = {self.RATE_OF_INFORMED_SEARCH_DECAY} /waypoint")
        print("=" * 65 + "\n")

    # =========================================================================
    # CPFA HELPERS (identical to epuck_foraging_supervisor_cpfa.py)
    # =========================================================================

    def _poisson_cdf(self, n, rate):
        """P(X <= n) for X ~ Poisson(rate). Matches CPFA GetPoissonCDF."""
        if n < 0:
            return 0.0
        cdf  = 0.0
        term = math.exp(-rate)
        cdf += term
        for i in range(1, n + 1):
            term *= rate / i
            cdf  += term
        return min(cdf, 1.0)

    def _roulette_select(self):
        """Roulette-wheel selection from pheromone_list weighted by weight."""
        active = [p for p in self.pheromone_list if p['weight'] > self.PHEROMONE_MIN]
        if not active:
            return None
        total      = sum(p['weight'] for p in active)
        r          = random.random() * total
        cumulative = 0.0
        for p in active:
            cumulative += p['weight']
            if r <= cumulative:
                return (p['x'], p['y'])
        return (active[-1]['x'], active[-1]['y'])

    def _assign_target(self, i):
        """CPFA target assignment at nest return.
        Priority 1: site fidelity  (Poisson CDF gate on resource_density)
        Priority 2: pheromone      (roulette-wheel selection)
        Priority 3: None           (random search — SEARCHING mode)"""
        # Priority 1: site fidelity — skipped if gave_up (stale target after failed trip)
        if self.site_fidelity_pos[i] is not None and not self.gave_up[i]:
            sf_prob = self._poisson_cdf(self.resource_density[i], self.RATE_OF_SITE_FIDELITY)
            if random.random() < sf_prob:
                sx, sy = self.site_fidelity_pos[i]
                return ('site', sx, sy)
        selected = self._roulette_select()
        if selected is not None:
            return ('phero', selected[0], selected[1])
        return None

    def _random_wall_target(self):
        """ARGoS SetRandomSearchLocation(): random point on one of the 4 arena walls.
        Robot heads toward the wall; switches to SEARCHING before arriving (prob 0.765)."""
        r      = random.random()
        lo, hi = -3.415, 3.415   # ±(3.5m - 0.085m robot radius)
        if r < 0.25:
            return (random.uniform(lo, hi), hi)   # north wall
        elif r < 0.50:
            return (random.uniform(lo, hi), lo)   # south wall
        elif r < 0.75:
            return (hi,  random.uniform(lo, hi))  # east wall
        else:
            return (lo,  random.uniform(lo, hi))  # west wall

    # =========================================================================
    # NAVIGATION HELPER
    # =========================================================================

    def _steer_to(self, robot_pos, fwd, target, gain=2.5):
        """Proportional steering — returns [left, right] in [-1, 1]."""
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

    def _crw_step(self, i, robot_pos, informed=False):
        """Correlated Random Walk step — ARGoS CPFA search locomotion.

        Uninformed: rotation = Gaussian(0, USV) — constant heading variation.

        Informed (ARGoS formula):
          rand = Gaussian(0, USV)  — fresh draw each waypoint
          rotation = rand + (2π - rand) × exp(-RateOfInformedSearchDecay × t)
          At t=0: rotation≈2π (broad random spin).
          As t→∞: rotation→rand (converges to a fixed bias direction).

        Returns a waypoint 0.08m (SearchStepSize) ahead along the new heading."""
        if informed:
            rand     = random.gauss(0, self.UNINFORMED_SEARCH_VARIATION)
            t        = self.informed_search_steps[i]
            rotation = rand + (2 * math.pi - rand) * math.exp(
                           -self.RATE_OF_INFORMED_SEARCH_DECAY * t)
            # Wrap to [-π, π] so heading change stays in a sensible range
            rotation = (rotation + math.pi) % (2 * math.pi) - math.pi
        else:
            rotation = random.gauss(0, self.UNINFORMED_SEARCH_VARIATION)

        self.search_heading[i] = (self.search_heading[i] + rotation) % (2 * math.pi)
        step_dist = 0.08  # metres — ARGoS SearchStepSize=0.08m scaled to 7×7m arena
        tx = robot_pos[0] + math.cos(self.search_heading[i]) * step_dist
        ty = robot_pos[1] + math.sin(self.search_heading[i]) * step_dist
        # Clamp to safe arena bounds (walls at ±2.5m, keep 0.3m margin)
        tx = max(-3.2, min(3.2, tx))
        ty = max(-3.2, min(3.2, ty))
        return (tx, ty)

    # =========================================================================
    # NEST ARRIVAL — deposit food, lay pheromone, assign next target
    # =========================================================================

    def _handle_nest_arrival(self, i, robot_pos):
        """Called when robot arrives at nest (dist < 0.25m).
        Handles deposit (if carrying), pheromone laying, and target assignment."""
        if self.carrying_state[i]:
            self.carrying_state[i] = False
            self.total_deposits   += 1
            print(f"[DEPOSIT] R{i+1} deposited! Total: {self.total_deposits}")

            # CPFA pheromone laying — Poisson CDF gate
            if self.carried_from[i] is not None:
                density  = self.resource_density[i]
                lay_prob = self._poisson_cdf(density, self.RATE_OF_LAYING_PHEROMONE)
                if random.random() < lay_prob:
                    self.pheromone_list.append({
                        'x':                self.carried_from[i][0],
                        'y':                self.carried_from[i][1],
                        'weight':           1.0,
                        'resource_density': density
                    })
                    print(f"  [PHERO] Laid at "
                          f"({self.carried_from[i][0]:.2f}, {self.carried_from[i][1]:.2f}) "
                          f"density={density} prob={lay_prob:.2f} "
                          f"entries={len(self.pheromone_list)}")
        else:
            # Returned empty (gave up) — still assign new target
            print(f"  [GIVE-UP] R{i+1} returned empty after {self.steps_without_pickup[i]} searching steps")

        # CPFA target assignment
        self.nest_target[i]             = self._assign_target(i)
        self.steps_without_pickup[i]    = 0
        self.give_up_timer[i]           = 0
        self.departing_timer[i]         = 0
        self.informed_search_steps[i]   = 0
        self.search_wp[i]               = None
        self.uninformed_wall_target[i]  = None
        self.search_heading[i] = random.uniform(0, 2 * math.pi)

        t = self.nest_target[i]
        if t is not None:
            # Informed DEPARTING — drive to site fidelity or pheromone cluster
            self.robot_mode[i] = DEPARTING
            print(f"  [TARGET] R{i+1} → {t[0].upper()} ({t[1]:.2f}, {t[2]:.2f})")
        else:
            # Uninformed DEPARTING — head to a random wall position.
            # ARGoS: SetRandomSearchLocation() + ProbabilityOfSwitchingToSearching check.
            self.robot_mode[i]              = DEPARTING
            self.uninformed_wall_target[i]  = self._random_wall_target()
            wx, wy = self.uninformed_wall_target[i]
            print(f"  [TARGET] R{i+1} → WALL ({wx:.2f}, {wy:.2f})")

    # =========================================================================
    # MAIN CONTROL LOOP
    # =========================================================================

    def run(self):
        while self.step(self.timestep) != -1:
            self.step_count += 1

            # --- Read proximity sensors ---
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

            # --- CPFA pheromone decay (time-based exponential) ---
            dt    = self.timestep / 1000.0
            decay = math.exp(-self.RATE_OF_PHEROMONE_DECAY * dt)
            for p in self.pheromone_list:
                p['weight'] *= decay
            self.pheromone_list = [p for p in self.pheromone_list
                                   if p['weight'] > self.PHEROMONE_MIN]

            base_pos = self.base_node.getPosition()
            modes    = [""] * self.num_robots

            # --- Per-robot logic ---
            for i in range(self.num_robots):
                robot_pos = self.robot_nodes[i].getPosition()
                robot_rot = self.robot_nodes[i].getOrientation()
                fwd       = [robot_rot[0], robot_rot[3], robot_rot[6]]
                prox      = (self.robot_states[i] or [0.0]*8)[:8]
                wall_dist = 3.5 - max(abs(robot_pos[0]), abs(robot_pos[1]))

                # ==============================================================
                # PICKUP DETECTION  (not carrying, within 0.15m of a tag)
                # ==============================================================
                if not self.carrying_state[i]:
                    for tag_node in self.tag_nodes:
                        tag_pos = tag_node.getPosition()
                        if tag_pos[2] < 0:
                            continue
                        dx   = tag_pos[0] - robot_pos[0]
                        dy   = tag_pos[1] - robot_pos[1]
                        if math.sqrt(dx*dx + dy*dy) < 0.15:
                            # Count resource density before hiding tag
                            density = sum(
                                1 for tn in self.tag_nodes
                                if tn.getPosition()[2] >= 0 and
                                math.sqrt((tn.getPosition()[0] - robot_pos[0])**2 +
                                          (tn.getPosition()[1] - robot_pos[1])**2) < 0.5
                            )
                            tag_node.getField("translation").setSFVec3f([0, 0, -10])
                            self.carrying_state[i]       = True
                            self.carried_from[i]         = (robot_pos[0], robot_pos[1])
                            self.resource_density[i]     = density
                            self.site_fidelity_pos[i]    = (robot_pos[0], robot_pos[1])
                            self.nest_target[i]           = None
                            self.search_wp[i]             = None
                            self.steps_without_pickup[i]  = 0
                            self.give_up_timer[i]         = 0
                            self.informed_search_steps[i] = 0
                            self.gave_up[i]               = False  # successful pickup — re-enable site fidelity
                            self.robot_mode[i]            = SURVEYING  # ARGoS: rotate 360° before RETURNING
                            self.survey_count[i]          = 0
                            self.survey_steps[i]          = 0
                            self.total_pickups          += 1
                            print(f"[PICKUP] R{i+1} picked up tag "
                                  f"(density={density}) | Total: {self.total_pickups}")
                            break

                # ==============================================================
                # NEST ARRIVAL DETECTION  (carrying or gave-up, within 0.25m)
                # ==============================================================
                dx_base      = base_pos[0] - robot_pos[0]
                dy_base      = base_pos[1] - robot_pos[1]
                dist_to_base = math.sqrt(dx_base*dx_base + dy_base*dy_base)

                if self.robot_mode[i] == RETURNING:
                    if dist_to_base < 0.25:
                        self._handle_nest_arrival(i, robot_pos)

                # ==============================================================
                # DEPARTING: arrived at cluster target → switch to SEARCHING
                # ==============================================================
                if self.robot_mode[i] == DEPARTING and self.nest_target[i] is not None:
                    tx, ty      = self.nest_target[i][1], self.nest_target[i][2]
                    dist_to_tgt = math.sqrt((tx - robot_pos[0])**2 + (ty - robot_pos[1])**2)
                    if dist_to_tgt < 0.05:
                        # Arrived at cluster — switch to local search
                        self.robot_mode[i] = SEARCHING
                        self.search_wp[i]  = None

                # ==============================================================
                # GIVE-UP CHECK — CPFA ProbabilityOfReturningToNest
                # Checked every 5 sim-seconds (GIVE_UP_CHECK_STEPS) during
                # SEARCHING only (DEPARTING excluded — robot is still en route).
                # ==============================================================
                if self.robot_mode[i] == SEARCHING and not self.carrying_state[i]:
                    self.steps_without_pickup[i] += 1   # counts every searching step for logging
                    self.give_up_timer[i] += 1
                    if self.give_up_timer[i] >= self.GIVE_UP_CHECK_STEPS:
                        self.give_up_timer[i] = 0
                        if random.random() < self.PROB_RETURN_TO_NEST:
                            self.robot_mode[i]  = RETURNING
                            self.nest_target[i] = None
                            self.search_wp[i]   = None
                            self.gave_up[i]     = True  # suppress site fidelity at next nest return

                # ==============================================================
                # ACTION SELECTION
                # ==============================================================

                # --- P1: Wall / obstacle escape (overrides everything) ---
                if wall_dist < 0.35 or max(prox) > 0.55:
                    action    = self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=4.0)
                    modes[i]  = "WALL_ESC"

                # --- BASE_ESC: push away from nest when not carrying (not RETURNING)
                # The base station is a 0.1m radius cylinder. Robots can get physically
                # stuck against it after deposit. Both CPFA and RL use this same 0.25m
                # threshold (matches deposit radius) for a fair comparison.
                elif (not self.carrying_state[i]
                        and self.robot_mode[i] != RETURNING
                        and dist_to_base < 0.25):
                    if dist_to_base > 0.001:
                        # Steer to a point 0.5m directly away from the base center
                        esc_x = robot_pos[0] + (robot_pos[0] / dist_to_base) * 0.5
                        esc_y = robot_pos[1] + (robot_pos[1] / dist_to_base) * 0.5
                    else:
                        esc_x, esc_y = 0.5, 0.0  # exact-center fallback
                    action   = self._steer_to(robot_pos, fwd, [esc_x, esc_y], gain=4.0)
                    modes[i] = "BASE_ESC"
                    # Reset stale CRW waypoint — if it was inside the nest zone, the robot
                    # would oscillate forever (BASE_ESC pushes out, steer_to pulls back in).
                    if self.robot_mode[i] == SEARCHING:
                        self.search_wp[i] = None

                # --- P2: Return to nest (RETURNING state) ---
                elif self.robot_mode[i] == RETURNING:
                    action   = self._steer_to(robot_pos, fwd, base_pos, gain=2.5)
                    modes[i] = "RTB" if self.carrying_state[i] else "GIVE_UP"

                # --- DEPARTING: informed → drive to cluster; uninformed → drive to wall ---
                elif self.robot_mode[i] == DEPARTING:
                    if self.nest_target[i] is not None:
                        # Informed: steer to site-fidelity or pheromone location
                        t        = self.nest_target[i]
                        action   = self._steer_to(robot_pos, fwd, [t[1], t[2]], gain=2.5)
                        modes[i] = t[0].upper()  # "SITE" or "PHERO"
                    else:
                        # Uninformed: head toward wall, switch to SEARCHING probabilistically.
                        # ARGoS: every 5 sec check ProbabilityOfSwitchingToSearching=0.765.
                        if self.uninformed_wall_target[i] is None:
                            self.uninformed_wall_target[i] = self._random_wall_target()

                        switched = False
                        self.departing_timer[i] += 1
                        if self.departing_timer[i] >= self.GIVE_UP_CHECK_STEPS:
                            self.departing_timer[i] = 0
                            if random.random() < self.PROB_SWITCH_TO_SEARCHING:
                                # Switch to SEARCHING at current location
                                self.robot_mode[i]             = SEARCHING
                                self.uninformed_wall_target[i] = None
                                # ARGoS: new heading = current + Gaussian(USV)
                                self.search_heading[i] = (
                                    math.atan2(fwd[1], fwd[0])
                                    + random.gauss(0, self.UNINFORMED_SEARCH_VARIATION)
                                ) % (2 * math.pi)
                                self.search_wp[i] = None
                                switched = True

                        if switched:
                            action   = [0.8, 0.8]
                            modes[i] = "SEARCH"
                        else:
                            wx, wy = self.uninformed_wall_target[i]
                            # Near the wall — pick a new wall target before wall-escape fires
                            if wall_dist < 0.4:
                                self.uninformed_wall_target[i] = self._random_wall_target()
                                wx, wy = self.uninformed_wall_target[i]
                            action   = self._steer_to(robot_pos, fwd, [wx, wy], gain=2.5)
                            modes[i] = "DEPART"

                # --- SURVEYING: rotate through 0°,90°,180°,270°,360° then RETURNING ---
                # Mirrors ARGoS Surveying(): survey_count * π/2 target headings.
                # TargetAngleTolerance = 0.1 rad (from ARGoS XML).
                elif self.robot_mode[i] == SURVEYING:
                    self.survey_steps[i] += 1
                    # Timeout: if physically blocked (e.g. by food cluster), force RETURNING
                    # 500 steps = 16s — generous enough for 5 quarter-turns at any speed
                    if self.survey_steps[i] > 500:
                        self.survey_count[i] = 0
                        self.survey_steps[i] = 0
                        self.robot_mode[i]   = RETURNING
                        modes[i] = "SURVEY"
                    else:
                        heading        = math.atan2(fwd[1], fwd[0])
                        target_heading = self.survey_count[i] * math.pi / 2
                        # Wrap error to [-π, π]
                        error = (target_heading - heading + math.pi) % (2 * math.pi) - math.pi
                        if abs(error) < 0.1:
                            self.survey_count[i] += 1
                            if self.survey_count[i] > 4:
                                self.survey_count[i] = 0
                                self.survey_steps[i] = 0
                                self.robot_mode[i]   = RETURNING
                        # Proportional in-place rotation toward target heading
                        turn   = max(-1.0, min(1.0, error * 3.0 / math.pi))
                        action = [-turn, turn]
                        modes[i] = "SURVEY"

                # --- SEARCHING: Correlated Random Walk ---
                else:
                    informed = (self.nest_target[i] is not None)
                    if not informed:
                        self.informed_search_steps[i] = 0

                    if self.search_wp[i] is None:
                        self.search_wp[i] = self._crw_step(i, robot_pos, informed)

                    tx, ty = self.search_wp[i]
                    tdx    = tx - robot_pos[0]
                    tdy    = ty - robot_pos[1]
                    if math.sqrt(tdx*tdx + tdy*tdy) < 0.05:
                        if informed:
                            self.informed_search_steps[i] += 1
                        self.search_wp[i] = self._crw_step(i, robot_pos, informed)
                        action = [0.8, 0.8]
                    else:
                        action = self._steer_to(robot_pos, fwd, [tx, ty], gain=2.5)
                    modes[i] = "SEARCH"

                # --- Send motor command ---
                msg = f"{action[0]},{action[1]}".encode('utf-8')
                self.emitters[i].send(msg)

            # --- Logging every 500 steps ---
            if self.step_count % 500 == 0:
                elapsed_min  = self.step_count * self.timestep / 1000.0 / 60.0
                rate         = self.total_deposits / elapsed_min if elapsed_min > 0 else 0.0
                phero_active = len([p for p in self.pheromone_list
                                    if p['weight'] > self.PHEROMONE_MIN])
                phero_max    = max((p['weight'] for p in self.pheromone_list), default=0.0)

                log_msg = (
                    f"\n{'='*70}\n"
                    f"Step {self.step_count} ({elapsed_min:.1f} min) | "
                    f"Pickups: {self.total_pickups} | Deposits: {self.total_deposits} | "
                    f"Rate: {rate:.2f} tags/min | "
                    f"phero_entries={phero_active} max_w={phero_max:.3f}\n"
                    f"{'='*70}\n"
                )
                for ri in range(self.num_robots):
                    rpos      = self.robot_nodes[ri].getPosition()
                    wall_d    = 3.5 - max(abs(rpos[0]), abs(rpos[1]))
                    d_to_base = math.sqrt(rpos[0]**2 + rpos[1]**2)
                    t         = self.nest_target[ri]
                    t_str     = f"{t[0]}({t[1]:.2f},{t[2]:.2f})" if t else "none"
                    log_msg  += (
                        f"R{ri+1}[{modes[ri]:10s}]: "
                        f"carry={int(self.carrying_state[ri])} | "
                        f"base={d_to_base:.2f} | wall={wall_d:.2f} | "
                        f"search={self.steps_without_pickup[ri]:3d} | "
                        f"target={t_str}\n"
                    )
                print(log_msg)
                with open("cpfa_baseline_log_7x7.txt", "a") as f:
                    f.write(log_msg)


# =============================================================================
controller = CPFABaseline()
controller.run()