import sys
import os
import math
import pickle
import random
import zlib
import base64
import numpy as np
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import Logger as SB3Logger
from stable_baselines3.common.callbacks import BaseCallback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'epuck_decentralized'))
from epuck_decentralized_v4 import (
    EpuckDecentralizedV4,
    TARGET_ARRIVAL_DIST, PHEROMONE_MIN, MERGE_RADIUS,
    RATE_OF_LAYING_PHEROMONE,
)

# =============================================================================
# FULLY DECENTRALIZED TRAINING ROBOT — v8
#
# Changes from v7:
#   1. Density in pheromone: broadcast (x,y,weight,density); roulette weighted
#      by weight×density — robots prefer high-density clusters.
#   2. OBS_DIM = 19: adds phero_density_norm at [17] (search_norm shifts to [18]).
#   3. Performance-aware gossip: each robot broadcasts (weights, ema_reward).
#      Receiver merges only if neighbor_ema >= own_ema × 0.95 — prevents bad
#      policies from contaminating stronger learners.
#   4. Pheromone range: 0.5m (updated in world file).
#
# Overrides: P1 (wall escape) + BASE_ESC + P2 (forced RTB when carrying).
# =============================================================================

OBS_DIM             = 19
ACT_DIM             = 2
N_STEPS             = 16384
N_EPOCHS            = 10
BATCH_SIZE          = 1024
GAMMA               = 0.99
GAE_LAMBDA          = 0.95
CLIP_RANGE          = 0.2
ENT_COEF            = 0.01
ENT_COEF_FINAL      = 0.001
VF_COEF             = 0.5
LR                  = 3e-4
MAX_GRAD_NORM       = 0.5
SAVE_FREQ           = 200_000
TOTAL_STEPS         = 5_000_000
SEARCH_DURATION_MAX = 4000
STEPS_PER_EPISODE   = 16384
GOSSIP_ALPHA        = 0.2
EMA_ALPHA           = 0.2   # EMA blend factor for performance tracking
GOSSIP_THRESHOLD    = 0.95  # merge only if neighbor_ema >= own_ema * threshold
LOG_STEP_EVERY      = 500
TARGET_LINGER_STEPS = 3     # steps to wait at cluster before declaring depletion


class EntropyScheduleCallback(BaseCallback):
    """Linearly decay ent_coef from initial to final over total_timesteps."""
    def __init__(self, initial, final, total_timesteps):
        super().__init__()
        self.initial         = initial
        self.final           = final
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        progress = max(0.0, 1.0 - self.model.num_timesteps / self.total_timesteps)
        self.model.ent_coef = self.final + progress * (self.initial - self.final)
        return True


class _DictWriter:
    def __init__(self):  self.data = {}
    def write(self, key_values, key_excluded, step=0): self.data = dict(key_values)
    def close(self): pass


class _DummyEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
    action_space      = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)
    def reset(self):   return np.zeros(OBS_DIM, dtype=np.float32)
    def step(self, _): return np.zeros(OBS_DIM), 0.0, False, {}


class EpuckDecentralizedTrainV8(EpuckDecentralizedV4):

    def __init__(self):
        super().__init__()

        self._project_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
        )
        self._robot_name = self.getName()

        cfg = os.path.join(self._project_root, 'current_run_name.txt')
        self._run_name = open(cfg).read().strip() if os.path.exists(cfg) else 'decentralized_indep_v8'

        # ── SB3 PPO ──────────────────────────────────────────────────
        dummy_env   = DummyVecEnv([lambda: _DummyEnv()])
        resume_path = os.path.join(self._project_root,
                                   f"{self._robot_name}_{self._run_name}.zip")
        if os.path.exists(resume_path):
            print(f"[{self._robot_name}] Resuming from {resume_path}")
            self._ppo = PPO.load(resume_path, env=dummy_env, device='cpu')
        else:
            print(f"[{self._robot_name}] Starting fresh | run: {self._run_name}")
            self._ppo = PPO(
                "MlpPolicy", dummy_env,
                n_steps       = N_STEPS,
                batch_size    = BATCH_SIZE,
                n_epochs      = N_EPOCHS,
                gamma         = GAMMA,
                gae_lambda    = GAE_LAMBDA,
                clip_range    = CLIP_RANGE,
                ent_coef      = ENT_COEF,
                vf_coef       = VF_COEF,
                learning_rate = LR,
                max_grad_norm = MAX_GRAD_NORM,
                policy_kwargs = dict(net_arch=[256, 256],
                                     activation_fn=torch.nn.Tanh),
                device  = 'cpu',
                verbose = 0,
            )
        self._stats_writer = _DictWriter()
        self._ppo.set_logger(SB3Logger(folder=None, output_formats=[self._stats_writer]))

        self._entropy_cb       = EntropyScheduleCallback(ENT_COEF, ENT_COEF_FINAL, TOTAL_STEPS)
        self._entropy_cb.model = self._ppo

        # ── Log file ─────────────────────────────────────────────────
        _log_dir = os.path.join(self._project_root, 'logs',
                                f"{self._robot_name}_{self._run_name}")
        os.makedirs(_log_dir, exist_ok=True)
        self._log_path = os.path.join(_log_dir, 'training_log.txt')

        # ── Model gossip (channel 11, 1m range) ──────────────────────
        self._model_emitter  = self.getDevice('model_emitter')
        self._model_receiver = self.getDevice('model_receiver')
        self._model_receiver.enable(self.time_step)

        # ── Rollout buffer tracking ───────────────────────────────────
        self._last_obs         = None
        self._last_raw_action  = np.zeros(ACT_DIM, dtype=np.float32)
        self._last_value       = None
        self._last_log_prob    = None
        self._last_is_ep_start = True

        # ── Step/episode counters ─────────────────────────────────────
        self._step_in_episode = 0
        self._total_steps     = 0
        self._n_updates       = 0

        # ── Reward shaping state ──────────────────────────────────────
        self._prev_base_dist       = None
        self._prev_site_dist       = None
        self._prev_phero_dist      = None
        self._steps_without_pickup = 0
        self._arrival_steps        = 0   # steps spent at cluster before declaring depletion

        # ── Per-episode stats ─────────────────────────────────────────
        self._ep_pickups           = 0
        self._ep_deposits          = 0
        self._ep_total_reward      = 0.0
        self._last_ep_total_reward = 0.0
        self._ema_reward           = 0.0   # exponential moving avg for gossip gating
        self._training_done        = False

    def _log(self, msg):
        print(msg)
        with open(self._log_path, 'a') as f:
            f.write(msg + '\n')

    def _log_file(self, msg):
        with open(self._log_path, 'a') as f:
            f.write(msg + '\n')

    def _get_mode(self, obs_comp, prox):
        wall_dist    = 2.5 - max(abs(obs_comp["pos_x"]), abs(obs_comp["pos_y"]))
        dist_to_base = obs_comp["base_dist_norm"] * 3.5
        if wall_dist < 0.35 or max(prox) > 0.55:
            return "WALL_ESC"
        if not self.carrying and dist_to_base < 0.25:
            return "BASE_ESC"
        if self.carrying:
            return "RTB"
        return "PPO"

    def _log_step(self, obs, obs_comp, raw_action, reward):
        prox      = [s.getValue() / 4096.0 for s in self.ps]
        mode      = self._get_mode(obs_comp, prox)
        wall_dist = 2.5 - max(abs(obs_comp["pos_x"]), abs(obs_comp["pos_y"]))
        ep_num    = self._n_updates + 1

        header = (
            f"\n[EP {ep_num} | Step {self._step_in_episode:5d}] "
            f"picks={self._ep_pickups} deps={self._ep_deposits} | "
            f"rew_step={reward:+.3f} rew_total={self._ep_total_reward:.1f} | "
            f"phero_entries={len(self.pheromone_list)} "
            f"max_w={max((e['weight'] for e in self.pheromone_list), default=0.0):.3f}"
        )
        detail = (
            f"  {self._robot_name}[{mode:8s}]: "
            f"L={raw_action[0]:+.2f} R={raw_action[1]:+.2f} | "
            f"carry={obs[8]:.0f} | "
            f"base={obs[9]:.2f} ang={obs[10]:+.2f} | "
            f"site={obs[11]:.0f} sd={obs[12]:.2f} sa={obs[13]:+.2f} | "
            f"phero={obs[14]:.0f} pd={obs[15]:.2f} pa={obs[16]:+.2f} pden={obs[17]:.2f} | "
            f"search={obs[18]:.2f} | wall={wall_dist:.2f}"
        )
        self._log_file(header + '\n' + detail)

    # =========================================================================
    # deepbots interface
    # =========================================================================

    def create_message(self):
        return super().create_message()

    def use_message_data(self, message):
        if self._training_done:
            return
        if not message or len(message) < 4:
            return

        try:
            pickup_signal = float(message[3])
        except (ValueError, IndexError):
            return

        # ── Episode boundary ──────────────────────────────────────────
        is_ep_start = (self._step_in_episode == 0)
        if is_ep_start and self._total_steps > 0:
            self._on_episode_reset()

        # Receive at step 1 — messages broadcast at step 0 (respawn positions)
        # only arrive in the queue one Webots step later
        if self._step_in_episode == 1:
            self._receive_models()

        # ── Pickup ───────────────────────────────────────────────────
        if pickup_signal > 0.0 and not self.carrying:
            self.carrying = True
            gps = self.gps.getValues()
            px, py = gps[0], gps[1]
            density = max(0, round(pickup_signal))
            self._site_fidelity_pos  = (px, py)
            self._resource_density   = density
            lay_prob = self._poisson_cdf(density, RATE_OF_LAYING_PHEROMONE)
            laid = False
            if random.random() < lay_prob:
                self._add_pheromone(px, py, 1.0, density)  # density included
                laid = True
            self._current_target     = None
            self._ep_pickups        += 1
            self._prev_site_dist     = None
            self._prev_phero_dist    = None
            self._steps_without_pickup = 0
            self._prev_base_dist     = math.sqrt(px ** 2 + py ** 2)
            self._gave_up            = False
            self._give_up_timer      = 0
            self._arrival_steps      = 0
            self._log(f"[PICKUP] {self._robot_name} at ({px:.2f},{py:.2f}) | "
                      f"density={density} lay_prob={lay_prob:.2f} laid={laid} | "
                      f"Ep picks: {self._ep_pickups}")

        # ── Deposit ───────────────────────────────────────────────────
        elif pickup_signal < -0.5 and self.carrying:
            self.carrying = False
            self._ep_deposits += 1
            self._steps_without_pickup = 0
            self._assign_target()
            if self._current_target is not None:
                gps = self.gps.getValues()
                tx, ty = self._current_target[1], self._current_target[2]
                d = math.sqrt((tx - gps[0])**2 + (ty - gps[1])**2)
                if self._current_target[0] == 'site':
                    self._prev_site_dist  = d
                    self._prev_phero_dist = None
                else:
                    self._prev_phero_dist = d
                    self._prev_site_dist  = None
            else:
                self._prev_site_dist  = None
                self._prev_phero_dist = None
            t      = self._current_target
            t_str  = f"{t[0].upper()} ({t[1]:.2f},{t[2]:.2f})" if t else "EXPLORE"
            sf     = self._site_fidelity_pos
            sf_str = f"({sf[0]:.2f},{sf[1]:.2f}) density={self._resource_density}" if sf else "None"
            self._log(f"[DEPOSIT] {self._robot_name} | Ep deps: {self._ep_deposits}")
            self._log(f"  [SITE_FID] {sf_str}")
            self._log(f"  [TARGET]   → {t_str}")

        # ── Target depletion ──────────────────────────────────────────
        # Linger TARGET_LINGER_STEPS steps before declaring cluster empty so the
        # supervisor has time to detect adjacent tags and send a pickup signal.
        if not self.carrying and self._current_target is not None:
            gps = self.gps.getValues()
            tx, ty = self._current_target[1], self._current_target[2]
            if math.sqrt((gps[0]-tx)**2 + (gps[1]-ty)**2) < TARGET_ARRIVAL_DIST:
                self._arrival_steps += 1
                if self._arrival_steps >= TARGET_LINGER_STEPS:
                    self._arrival_steps = 0
                    if self._current_target[0] == 'phero':
                        for entry in self.pheromone_list:
                            if math.sqrt((entry['x']-tx)**2 + (entry['y']-ty)**2) < MERGE_RADIUS:
                                entry['density'] = 0
                                entry['weight']  = PHEROMONE_MIN
                                break
                    elif self._current_target[0] == 'site':
                        self._site_fidelity_pos = None
                        self._resource_density  = 0
                    self._gave_up         = True
                    self._current_target  = None
                    self._prev_site_dist  = None
                    self._prev_phero_dist = None
                    self._assign_target()
                    self._gave_up = False
            else:
                self._arrival_steps = 0

        # ── Search counter ────────────────────────────────────────────
        if not self.carrying:
            self._steps_without_pickup += 1

        # ── Assemble 19D obs ──────────────────────────────────────────
        prox     = [s.getValue() / 4096.0 for s in self.ps]
        obs_comp = self._obs_components()
        search_norm = (0.0 if self.carrying else
                       min(self._steps_without_pickup / SEARCH_DURATION_MAX, 1.0))
        obs = np.array(
            prox + [
                1.0 if self.carrying else 0.0,       # [8]
                obs_comp["base_dist_norm"],           # [9]
                obs_comp["base_angle_norm"],          # [10]
                obs_comp["site_known"],               # [11]
                obs_comp["site_dist_norm"],           # [12]
                obs_comp["site_angle_norm"],          # [13]
                obs_comp["phero_known"],              # [14]
                obs_comp["phero_dist_norm"],          # [15]
                obs_comp["phero_angle_norm"],         # [16]
                obs_comp["phero_density_norm"],       # [17] NEW
                search_norm,                          # [18]
            ],
            dtype=np.float32,
        )

        # ── Reward ────────────────────────────────────────────────────
        reward = self._compute_reward(obs_comp, pickup_signal)
        self._ep_total_reward += reward

        # ── Store previous transition ─────────────────────────────────
        if self._last_obs is not None:
            self._ppo.rollout_buffer.add(
                self._last_obs.reshape(1, -1),
                self._last_raw_action.reshape(1, -1),
                np.array([reward]),
                np.array([self._last_is_ep_start]),
                self._last_value,
                self._last_log_prob,
            )

        # ── PPO inference ─────────────────────────────────────────────
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self._ppo.device)
            actions_t, values_t, log_probs_t = self._ppo.policy.forward(obs_t)

        raw_action = actions_t.cpu().numpy().flatten()

        # ── P1 + BASE_ESC + P2 overrides ─────────────────────────────
        left_cmd, right_cmd = self._apply_overrides(
            raw_action[0], raw_action[1],
            obs_comp["pos_x"], obs_comp["pos_y"],
        )

        # ── Drive motors ──────────────────────────────────────────────
        max_speed    = 6.28
        scale_factor = 6.0
        self.left_motor.setVelocity(
            max(min(left_cmd  * scale_factor, max_speed), -max_speed))
        self.right_motor.setVelocity(
            max(min(right_cmd * scale_factor, max_speed), -max_speed))

        if self._step_in_episode % LOG_STEP_EVERY == 0:
            self._log_step(obs, obs_comp, raw_action, reward)

        self._last_obs         = obs
        self._last_raw_action  = raw_action
        self._last_value       = values_t
        self._last_log_prob    = log_probs_t
        self._last_is_ep_start = is_ep_start

        self._total_steps    += 1
        self._step_in_episode = (self._step_in_episode + 1) % STEPS_PER_EPISODE

        if self._ppo.rollout_buffer.full:
            self._train_ppo(obs)

        if (self._total_steps > 0
                and self._total_steps % SAVE_FREQ == 0
                and self._total_steps < TOTAL_STEPS):
            self._save_checkpoint()

        if self._total_steps >= TOTAL_STEPS and not self._training_done:
            self._save_checkpoint()
            self._log(f"[{self._robot_name}] TRAINING COMPLETE | "
                      f"{self._total_steps} steps | Final model saved.")
            self._training_done = True
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)

    # =========================================================================
    # Episode reset
    # =========================================================================

    def _on_episode_reset(self):
        # Broadcast BEFORE log_episode_summary() so ep_deposits is still set from last episode.
        # Robots are at respawn positions (all within ~0.7m of center, inside 1m emitter range).
        self._broadcast_model()
        self._log_episode_summary()

    def _log_episode_summary(self):
        self._last_ep_total_reward = self._ep_total_reward
        # Update EMA for performance-aware gossip gating
        if self._ema_reward == 0.0:
            self._ema_reward = self._last_ep_total_reward
        else:
            self._ema_reward = ((1 - EMA_ALPHA) * self._ema_reward
                                + EMA_ALPHA * self._last_ep_total_reward)

        ep_num      = self._n_updates + 1
        sim_minutes = (STEPS_PER_EPISODE * self.time_step) / 60000.0
        rate        = self._ep_deposits / sim_minutes if sim_minutes > 0 else 0.0
        phero_count = len(self.pheromone_list)
        phero_max_w = max((e['weight'] for e in self.pheromone_list), default=0.0)

        if ep_num < 30:
            phase = "NEAR    (11 clusters, max_dist=1.2m)"
        elif ep_num < 75:
            phase = "MEDIUM  (11 clusters, max_dist=1.6m)"
        elif ep_num < 150:
            phase = "FAR     (11 clusters, max_dist=2.0m)"
        else:
            phase = "FULL    (11 clusters, max_dist=2.3m)"

        gps       = self.gps.getValues()
        prox      = [s.getValue() / 4096.0 for s in self.ps]
        wall_dist = 2.5 - max(abs(gps[0]), abs(gps[1]))
        d2base    = math.sqrt(gps[0]**2 + gps[1]**2)
        mode      = self._get_mode({"pos_x": gps[0], "pos_y": gps[1],
                                    "base_dist_norm": min(d2base / 3.5, 1.0)}, prox)
        t         = self._current_target
        t_str     = f"{t[0]}({t[1]:.2f},{t[2]:.2f})" if t else "explore"

        summary = (
            f"\n{'='*65}\n"
            f"[EP {ep_num}] {self._robot_name} | "
            f"Picks: {self._ep_pickups} | Deps: {self._ep_deposits} | "
            f"Rate: {rate:.2f} tags/min (sim) | "
            f"TotalSteps: {self._total_steps} | Updates: {self._n_updates}\n"
            f"  Curriculum:  {phase}\n"
            f"  Pheromone:   entries={phero_count} | max_weight={phero_max_w:.3f}\n"
            f"  Reward:      total={self._ep_total_reward:.1f} | "
            f"ema={self._ema_reward:.1f} | ent_coef={self._ppo.ent_coef:.5f}\n"
            f"  [{mode:8s}]: carry={int(self.carrying)} | "
            f"base={d2base:.2f} | wall={wall_dist:.2f} | "
            f"target={t_str} | density={self._resource_density}\n"
            f"{'='*65}"
        )
        self._log(summary)

        self.carrying              = False
        self.pheromone_list        = []
        self._site_fidelity_pos    = None
        self._resource_density     = 0
        self._current_target       = None
        self._last_raw_action      = np.zeros(ACT_DIM, dtype=np.float32)
        self._prev_base_dist       = None
        self._prev_site_dist       = None
        self._prev_phero_dist      = None
        self._steps_without_pickup = 0
        self._arrival_steps        = 0
        self._steps_since_pickup   = 0
        self._carrying_prev        = False
        self._gave_up              = False
        self._give_up_timer        = 0
        self._ep_pickups           = 0
        self._ep_deposits          = 0
        self._ep_total_reward      = 0.0

    # =========================================================================
    # Reward (identical to v7 / centralized per-robot reward)
    # =========================================================================

    def _compute_reward(self, obs_comp, pickup_signal):
        reward    = 0.0
        prox      = [s.getValue() / 4096.0 for s in self.ps]
        gps_x     = obs_comp["pos_x"]
        gps_y     = obs_comp["pos_y"]
        wall_dist = 2.5 - max(abs(gps_x), abs(gps_y))

        max_prox = max(prox)
        if max_prox > 0.1:
            reward -= max_prox * 0.5
        if wall_dist < 0.35:
            reward -= (0.35 - wall_dist) * 0.5

        if pickup_signal > 0.0:
            reward += 5.0
        elif pickup_signal < -0.5:
            reward += 20.0

        dist_from_base = obs_comp["base_dist_norm"] * 3.5

        if not self.carrying:
            if wall_dist >= 0.35:
                avg_speed = (self._last_raw_action[0] + self._last_raw_action[1]) / 2.0
                if avg_speed > 0:
                    reward += avg_speed * 0.15

            if obs_comp["site_known"] > 0.5:
                curr_sd = obs_comp["site_dist_norm"] * 3.5
                if self._prev_site_dist is not None:
                    reward += (self._prev_site_dist - curr_sd) * 30.0
                self._prev_site_dist  = curr_sd
                self._prev_phero_dist = None
                if curr_sd > 0.001:
                    reward += math.cos(obs_comp["site_angle_norm"] * math.pi) * 0.5

            elif obs_comp["phero_known"] > 0.5:
                curr_pd = obs_comp["phero_dist_norm"] * 3.5
                if self._prev_phero_dist is not None:
                    reward += (self._prev_phero_dist - curr_pd) * 30.0
                self._prev_phero_dist = curr_pd
                self._prev_site_dist  = None
                if curr_pd > 0.001:
                    reward += math.cos(obs_comp["phero_angle_norm"] * math.pi) * 0.5

            else:
                self._prev_site_dist  = None
                self._prev_phero_dist = None

            self._prev_base_dist = dist_from_base

        else:
            if self._prev_base_dist is not None:
                reward += (self._prev_base_dist - dist_from_base) * 8.0
            self._prev_base_dist  = dist_from_base
            reward += math.cos(obs_comp["base_angle_norm"] * math.pi) * 0.5
            self._prev_site_dist  = None
            self._prev_phero_dist = None

        reward -= 0.005
        return reward

    # =========================================================================
    # Performance-aware model gossip
    # =========================================================================

    def _broadcast_model(self):
        """Serialize weights + EMA reward and broadcast within 1m.
        Gated on ep_deposits > 0 to avoid spreading stagnant policies."""
        if self._ep_deposits == 0:
            return
        try:
            state = {k: v.cpu().numpy() for k, v in self._ppo.policy.state_dict().items()}
            msg = {
                'weights':    state,
                'avg_reward': self._ema_reward,
                'episode':    self._n_updates,
            }
            payload = zlib.compress(pickle.dumps(msg, protocol=4), level=1)
            encoded = base64.b64encode(payload).decode('ascii')
            self._model_emitter.send(encoded)
            self._log(f"[GOSSIP] {self._robot_name} → broadcast {len(payload)} bytes "
                      f"| ema={self._ema_reward:.1f}")
        except Exception as e:
            self._log(f"[GOSSIP] broadcast error: {e}")

    def _receive_models(self):
        """Merge neighbor models only if neighbor EMA is within 5% below own EMA.
        Threshold = own_ema - |own_ema| * (1 - GOSSIP_THRESHOLD), which correctly
        handles negative rewards (early training) without inverting the comparison."""
        merged = skipped = 0
        while self._model_receiver.getQueueLength() > 0:
            try:
                encoded = self._model_receiver.getString()
                payload = base64.b64decode(encoded.encode('ascii'))
                msg     = pickle.loads(zlib.decompress(payload))
                if not isinstance(msg, dict) or 'weights' not in msg:
                    self._model_receiver.nextPacket()
                    continue
                neighbor_avg = msg.get('avg_reward', float('-inf'))
                tol          = abs(self._ema_reward) * (1.0 - GOSSIP_THRESHOLD)
                threshold    = self._ema_reward - tol
                if neighbor_avg >= threshold:
                    own_sd  = self._ppo.policy.state_dict()
                    nbr_sd  = msg['weights']
                    new_sd  = {
                        k: (1 - GOSSIP_ALPHA) * v
                           + GOSSIP_ALPHA * torch.FloatTensor(nbr_sd[k]).to(self._ppo.device)
                        if k in nbr_sd else v
                        for k, v in own_sd.items()
                    }
                    self._ppo.policy.load_state_dict(new_sd)
                    merged += 1
                else:
                    skipped += 1
                    self._log(f"[GOSSIP] skip: neighbor_ema={neighbor_avg:.1f} "
                              f"< threshold={threshold:.1f}")
            except Exception as e:
                self._log(f"[GOSSIP] receive error: {e}")
            finally:
                self._model_receiver.nextPacket()
        if merged > 0:
            self._log(f"[GOSSIP] {self._robot_name} ← merged {merged}, "
                      f"skipped {skipped} (α={GOSSIP_ALPHA})")

    # =========================================================================
    # PPO training
    # =========================================================================

    def _train_ppo(self, last_obs):
        with torch.no_grad():
            obs_t      = torch.FloatTensor(last_obs).unsqueeze(0).to(self._ppo.device)
            last_value = self._ppo.policy.predict_values(obs_t)

        self._ppo.rollout_buffer.compute_returns_and_advantage(
            last_values = last_value,
            dones       = np.array([True]),
        )

        self._ppo.num_timesteps = self._total_steps
        self._entropy_cb._on_step()

        self._ppo.train()
        self._ppo.rollout_buffer.reset()
        self._n_updates += 1

        s = self._stats_writer.data
        self._log(
            f"{'─'*43}\n"
            f"| {'rollout/':<20} {'':>18} |\n"
            f"|    {'ep_rew_total':<16} {self._last_ep_total_reward:>18.1f} |\n"
            f"|    {'ema_reward':<16} {self._ema_reward:>18.1f} |\n"
            f"| {'time/':<20} {'':>18} |\n"
            f"|    {'iterations':<16} {self._n_updates:>18} |\n"
            f"|    {'total_timesteps':<16} {self._total_steps:>18} |\n"
            f"|    {'ent_coef':<16} {self._ppo.ent_coef:>18.5f} |\n"
            f"| {'train/':<20} {'':>18} |\n"
            f"|    {'approx_kl':<16} {s.get('train/approx_kl', 0):>18.8f} |\n"
            f"|    {'clip_fraction':<16} {s.get('train/clip_fraction', 0):>18.4f} |\n"
            f"|    {'entropy_loss':<16} {s.get('train/entropy_loss', 0):>18.4f} |\n"
            f"|    {'explained_var':<16} {s.get('train/explained_variance', 0):>18.3f} |\n"
            f"|    {'loss':<16} {s.get('train/loss', 0):>18.4f} |\n"
            f"|    {'policy_grad':<16} {s.get('train/policy_gradient_loss', 0):>18.4f} |\n"
            f"|    {'value_loss':<16} {s.get('train/value_loss', 0):>18.3g} |\n"
            f"{'─'*43}"
        )

    def _save_checkpoint(self):
        log_dir = os.path.join(self._project_root, 'logs',
                               f"{self._robot_name}_{self._run_name}")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir,
                            f"{self._robot_name}_{self._run_name}_{self._total_steps}_steps")
        self._ppo.save(path)
        latest = os.path.join(self._project_root,
                              f"{self._robot_name}_{self._run_name}")
        self._ppo.save(latest)
        self._log(f"[{self._robot_name}] Saved checkpoint: {path}.zip")

    # =========================================================================
    # Overrides — P1 + BASE_ESC + P2
    # =========================================================================

    def _apply_overrides(self, left, right, pos_x, pos_y):
        gps       = self.gps.getValues()
        yaw       = self.imu.getRollPitchYaw()[2]
        fwd       = [math.cos(yaw), math.sin(yaw)]
        robot_pos = [gps[0], gps[1]]
        wall_dist = 2.5 - max(abs(pos_x), abs(pos_y))
        prox      = [s.getValue() / 4096.0 for s in self.ps]

        if wall_dist < 0.35 or max(prox) > 0.55:
            return self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=4.0)

        dist_to_base = math.sqrt(pos_x ** 2 + pos_y ** 2)

        if not self.carrying and dist_to_base < 0.25:
            if dist_to_base > 0.001:
                esc_x = pos_x + (pos_x / dist_to_base) * 0.5
                esc_y = pos_y + (pos_y / dist_to_base) * 0.5
            else:
                esc_x, esc_y = 0.5, 0.0
            return self._steer_to(robot_pos, fwd, [esc_x, esc_y], gain=4.0)

        if self.carrying:
            return self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=2.5)

        return [left, right]

    def _steer_to(self, robot_pos, fwd, target, gain=2.5):
        dx   = target[0] - robot_pos[0]
        dy   = target[1] - robot_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 0.01:
            return [0.0, 0.0]
        t_norm = [dx / dist, dy / dist]
        dot    = fwd[0] * t_norm[0] + fwd[1] * t_norm[1]
        cross  = fwd[0] * t_norm[1] - fwd[1] * t_norm[0]
        angle  = math.atan2(cross, dot)
        turn   = max(-1.0, min(1.0, gain * angle / math.pi))
        l      = max(-1.0, min(1.0, 1.0 - turn))
        r      = max(-1.0, min(1.0, 1.0 + turn))
        m      = max(abs(l), abs(r))
        if m > 1.0:
            l /= m
            r /= m
        return [l, r]


if __name__ == '__main__':
    robot = EpuckDecentralizedTrainV8()
    robot.run()
