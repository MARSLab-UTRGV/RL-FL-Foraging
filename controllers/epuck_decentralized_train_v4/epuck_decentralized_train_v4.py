import sys
import os
import math
import pickle
import base64
import numpy as np
import torch
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import Logger as SB3Logger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'epuck_decentralized'))
from epuck_decentralized_v4 import EpuckDecentralizedV4, TARGET_ARRIVAL_DIST, PHEROMONE_MIN

# =============================================================================
# FULLY DECENTRALIZED TRAINING ROBOT — v4
#
# Changes from v3:
#   1. Pheromone: single hotspot → CPFA list-based with roulette selection
#   2. Site fidelity: robot remembers own last pickup, priority over roulette
#   3. Obs: 18D → 20D  (add site [14-16], shift phero to [17-19], drop phero_strength)
#   4. Reward: site approach ×15, phero approach ×15 (to assigned target, not broadcast)
#   5. Target depletion: if arrival < 0.18m with no pickup → roulette for next target
#   6. Gossip FL: unchanged (model sharing via channel 11, FedAvg α=0.2)
#
# Obs layout (20D):
#   [0:8]  prox sensors
#   [8]    tag_visible
#   [9]    tag_dist_norm
#   [10]   tag_angle_norm
#   [11]   carrying
#   [12]   base_dist_norm
#   [13]   base_angle_norm
#   [14]   site_known         ← own last pickup target
#   [15]   site_dist_norm
#   [16]   site_angle_norm
#   [17]   phero_known        ← roulette-selected target
#   [18]   phero_dist_norm
#   [19]   phero_angle_norm
# =============================================================================

OBS_DIM       = 20
ACT_DIM       = 2
N_STEPS       = 16384
N_EPOCHS      = 10
BATCH_SIZE    = 256
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
CLIP_RANGE    = 0.2
ENT_COEF      = 0.15
VF_COEF       = 0.5
LR            = 3e-4
MAX_GRAD_NORM = 0.5
SAVE_FREQ     = 50_000
TOTAL_STEPS   = 7_000_000

GOSSIP_ALPHA      = 0.2
STEPS_PER_EPISODE = 16384
TAG_SEEK_RANGE    = 1.0


class _DictWriter:
    """Captures SB3 logger key-values so we can format them ourselves."""
    def __init__(self):  self.data = {}
    def write(self, key_values, key_excluded, step=0): self.data = dict(key_values)
    def close(self): pass


class _DummyEnv(gym.Env):
    observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
    action_space      = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

    def reset(self):
        return np.zeros(OBS_DIM, dtype=np.float32)

    def step(self, _):
        return np.zeros(OBS_DIM), 0.0, False, {}


class EpuckDecentralizedTrainV4(EpuckDecentralizedV4):

    def __init__(self):
        super().__init__()

        self._project_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
        )
        self._robot_name = self.getName()

        cfg = os.path.join(self._project_root, 'current_run_name.txt')
        self._run_name = open(cfg).read().strip() if os.path.exists(cfg) else 'decentralized_indep_v4'

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

        # ── Log file ─────────────────────────────────────────────────
        _log_dir = os.path.join(self._project_root, 'logs',
                                f"{self._robot_name}_{self._run_name}")
        os.makedirs(_log_dir, exist_ok=True)
        self._log_path = os.path.join(_log_dir, 'training_log.txt')

        # ── Gossip model sharing (channel 11, 2m) ─────────────────────
        try:
            self._model_emitter  = self.getDevice('model_emitter')
            self._model_receiver = self.getDevice('model_receiver')
            self._model_receiver.enable(self.time_step)
            self._gossip_enabled = True
            print(f"[{self._robot_name}] Gossip enabled (channel 11, 2m)")
        except Exception:
            self._gossip_enabled = False
            print(f"[{self._robot_name}] Gossip disabled")
        self._pending_neighbor_weights = []

        # ── Tag obs from supervisor ────────────────────────────────────
        self._tag_visible    = 0.0
        self._tag_dist_norm  = 0.0
        self._tag_angle_norm = 0.0

        # ── Rollout buffer tracking ────────────────────────────────────
        self._last_obs         = None
        self._last_raw_action  = np.zeros(ACT_DIM, dtype=np.float32)
        self._last_value       = None
        self._last_log_prob    = None
        self._last_is_ep_start = True

        # ── Step/episode counters ──────────────────────────────────────
        self._step_in_episode = 0
        self._total_steps     = 0
        self._n_updates       = 0

        # ── Reward shaping state ───────────────────────────────────────
        self._prev_tag_dist   = None
        self._prev_base_dist  = None
        self._prev_site_dist  = None   # approach shaping toward site target
        self._prev_phero_dist = None   # approach shaping toward phero target

        # ── Per-episode stats ──────────────────────────────────────────
        self._ep_pickups       = 0
        self._ep_deposits      = 0
        self._ep_gossip_merges = 0
        self._ep_total_reward  = 0.0
        self._training_done    = False

    def _log(self, msg):
        print(msg)
        with open(self._log_path, 'a') as f:
            f.write(msg + '\n')

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
            self._tag_visible    = float(message[0])
            self._tag_dist_norm  = float(message[1])
            self._tag_angle_norm = float(message[2])
            pickup_signal        = float(message[3])
        except (ValueError, IndexError):
            return

        # ── Drain gossip queue every step ─────────────────────────────
        if self._gossip_enabled:
            self._collect_neighbor_models()

        # ── Episode boundary ──────────────────────────────────────────
        is_ep_start = (self._step_in_episode == 0)
        if is_ep_start and self._total_steps > 0:
            self._on_episode_reset()

        # ── Pickup ────────────────────────────────────────────────────
        if pickup_signal > 0.0 and not self.carrying:
            self.carrying = True
            gps = self.gps.getValues()
            px, py = gps[0], gps[1]
            self._add_pheromone(px, py, pickup_signal)
            self._site_fidelity_pos  = (px, py)
            self._last_pickup_weight = pickup_signal
            self._current_target     = None
            self._ep_pickups        += 1
            self._prev_tag_dist      = None
            self._prev_site_dist     = None
            self._prev_phero_dist    = None
            approx_density = round((pickup_signal - 0.2) / 0.8 * 5)
            self._log(f"[PICKUP] {self._robot_name} at ({px:.2f},{py:.2f}) | "
                      f"strength={pickup_signal:.2f} approx_density~{approx_density} | "
                      f"Ep picks: {self._ep_pickups}")
            self._log(f"  [PHERO] Added ({px:.2f},{py:.2f}) weight={pickup_signal:.2f} | "
                      f"list_size={len(self.pheromone_list)}")

        # ── Deposit ───────────────────────────────────────────────────
        elif pickup_signal < -0.5 and self.carrying:
            self.carrying  = False
            self._ep_deposits += 1
            self._assign_target()
            # Pre-seed distance tracking so approach reward fires from step 1
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
            t = self._current_target
            t_str = f"{t[0].upper()} ({t[1]:.2f},{t[2]:.2f})" if t else "EXPLORE"
            sf = self._site_fidelity_pos
            sf_str = f"({sf[0]:.2f},{sf[1]:.2f}) weight={self._last_pickup_weight:.2f}" if sf else "None"
            self._log(f"[DEPOSIT] {self._robot_name} | Ep deps: {self._ep_deposits}")
            self._log(f"  [SITE_FID] last_pickup={sf_str}")
            self._log(f"  [TARGET]   → {t_str}")

        # ── Target depletion check (not carrying, has target) ─────────
        if not self.carrying and self._current_target is not None:
            gps = self.gps.getValues()
            tx, ty = self._current_target[1], self._current_target[2]
            if math.sqrt((gps[0]-tx)**2 + (gps[1]-ty)**2) < TARGET_ARRIVAL_DIST:
                # Arrived at cluster but no food found — confirm depletion progressively
                if self._current_target[0] == 'site':
                    # Accelerate decay of return probability — mirrors CPFA density gate
                    # falling as cluster depletes. PPO + Poisson gate handle the rest.
                    self._last_pickup_weight = max(self._last_pickup_weight * 0.5, 0.05)
                elif self._current_target[0] == 'phero':
                    # Accelerate decay of this cluster entry — roulette naturally shifts
                    # to fresher/stronger clusters as depleted ones lose weight.
                    for entry in self.pheromone_list:
                        if math.sqrt((entry['x'] - tx) ** 2 +
                                     (entry['y'] - ty) ** 2) < 0.3:
                            entry['weight'] = max(entry['weight'] * 0.5, PHEROMONE_MIN)
                            break
                self._assign_target()
                self._prev_site_dist  = None
                self._prev_phero_dist = None

        # ── Assemble 20D obs ──────────────────────────────────────────
        prox     = [s.getValue() / 4096.0 for s in self.ps]
        obs_comp = self._obs_components()
        obs = np.array(
            prox + [
                self._tag_visible,                    # [8]
                self._tag_dist_norm,                  # [9]
                self._tag_angle_norm,                 # [10]
                1.0 if self.carrying else 0.0,        # [11]
                obs_comp["base_dist_norm"],            # [12]
                obs_comp["base_angle_norm"],           # [13]
                obs_comp["site_known"],                # [14]
                obs_comp["site_dist_norm"],            # [15]
                obs_comp["site_angle_norm"],           # [16]
                obs_comp["phero_known"],               # [17]
                obs_comp["phero_dist_norm"],           # [18]
                obs_comp["phero_angle_norm"],          # [19]
            ],
            dtype=np.float32,
        )

        # ── Compute reward ────────────────────────────────────────────
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

        # ── P1-P3 overrides ───────────────────────────────────────────
        left_cmd, right_cmd = self._apply_overrides(
            raw_action[0], raw_action[1],
            obs_comp["pos_x"], obs_comp["pos_y"],
        )

        # ── Drive motors ──────────────────────────────────────────────
        max_speed    = 6.28
        scale_factor = 6.0
        self.left_motor.setVelocity(max(min(left_cmd  * scale_factor, max_speed), -max_speed))
        self.right_motor.setVelocity(max(min(right_cmd * scale_factor, max_speed), -max_speed))

        # ── Update tracking ───────────────────────────────────────────
        self._last_obs         = obs
        self._last_raw_action  = raw_action
        self._last_value       = values_t
        self._last_log_prob    = log_probs_t
        self._last_is_ep_start = is_ep_start

        self._total_steps     += 1
        self._step_in_episode  = (self._step_in_episode + 1) % STEPS_PER_EPISODE

        # ── Train when buffer full, then gossip ───────────────────────
        if self._ppo.rollout_buffer.full:
            self._train_ppo(obs)
            if self._gossip_enabled and self._n_updates >= 1:
                self._merge_neighbor_models()
                self._broadcast_model()

        # ── Periodic checkpoint ───────────────────────────────────────
        if (self._total_steps > 0
                and self._total_steps % SAVE_FREQ == 0
                and self._total_steps < TOTAL_STEPS):
            self._save_checkpoint()

        # ── Training complete ─────────────────────────────────────────
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
        phero_count = len(self.pheromone_list)
        phero_max_w = max((e['weight'] for e in self.pheromone_list), default=0.0)
        target_str  = self._current_target[0].upper() if self._current_target else 'EXPLORE'

        self._log(
            f"[{self._robot_name}] EP END | "
            f"Picks: {self._ep_pickups} | Deps: {self._ep_deposits} | "
            f"Steps: {self._total_steps} | Updates: {self._n_updates} | "
            f"GossipMerges: {self._ep_gossip_merges} | "
            f"phero_entries={phero_count} max_w={phero_max_w:.3f} | "
            f"target={target_str}"
        )

        self.carrying            = False
        self.pheromone_list      = []
        self._site_fidelity_pos  = None
        self._last_pickup_weight = 0.0
        self._current_target     = None
        self._prev_tag_dist      = None
        self._prev_base_dist     = None
        self._prev_site_dist     = None
        self._prev_phero_dist    = None
        self._ep_pickups         = 0
        self._ep_deposits        = 0
        self._ep_gossip_merges   = 0
        self._ep_total_reward    = 0.0

    # =========================================================================
    # Reward computation (fully onboard)
    # =========================================================================

    def _compute_reward(self, obs_comp, pickup_signal):
        reward    = 0.0
        prox      = [s.getValue() / 4096.0 for s in self.ps]
        gps_x     = obs_comp["pos_x"]
        gps_y     = obs_comp["pos_y"]
        wall_dist = 2.5 - max(abs(gps_x), abs(gps_y))

        # Proximity penalty
        max_prox = max(prox)
        if max_prox > 0.1:
            reward -= max_prox * 0.5

        # Wall penalty
        if wall_dist < 0.35:
            reward -= (0.35 - wall_dist) * 0.5

        # Pickup / deposit bonuses
        if pickup_signal > 0.0:
            reward += 5.0
        elif pickup_signal < -0.5:
            reward += 20.0

        if not self.carrying:
            # ── Tag approach shaping ───────────────────────────────────
            curr_tag_dist = (self._tag_dist_norm * TAG_SEEK_RANGE
                             if self._tag_visible > 0.5 else None)
            if self._prev_tag_dist is not None and curr_tag_dist is not None:
                reward += (self._prev_tag_dist - curr_tag_dist) * 8.0
            self._prev_tag_dist = curr_tag_dist

            if self._tag_visible > 0.5:
                reward += 0.2 + max(math.cos(self._tag_angle_norm * math.pi), 0.0) * 0.3

            dist_from_base = math.sqrt(gps_x ** 2 + gps_y ** 2)

            # Zone reward — stay in active foraging zone
            if 0.8 < dist_from_base < 2.4:
                reward += 0.10

            # Near-base penalty
            if dist_from_base < 0.8:
                reward -= (0.8 - dist_from_base) * 4.0

            # Forward motion bias
            if wall_dist >= 0.35:
                avg_speed = (self._last_raw_action[0] + self._last_raw_action[1]) / 2.0
                if avg_speed > 0:
                    reward += avg_speed * 0.01

            # ── Site fidelity approach (Priority 1 target) ─────────────
            if obs_comp["site_known"] > 0.5:
                curr_sd = obs_comp["site_dist_norm"] * 3.5
                if self._prev_site_dist is not None:
                    reward += (self._prev_site_dist - curr_sd) * 15.0
                self._prev_site_dist  = curr_sd
                self._prev_phero_dist = None
                if curr_sd > 0.001:
                    reward += math.cos(obs_comp["site_angle_norm"] * math.pi) * 0.5

            # ── Pheromone target approach (Priority 2 target) ──────────
            elif obs_comp["phero_known"] > 0.5:
                curr_pd = obs_comp["phero_dist_norm"] * 3.5
                if self._prev_phero_dist is not None:
                    reward += (self._prev_phero_dist - curr_pd) * 15.0
                self._prev_phero_dist = curr_pd
                self._prev_site_dist  = None
                if curr_pd > 0.001:
                    reward += math.cos(obs_comp["phero_angle_norm"] * math.pi) * 0.5

            # ── Free exploration (no target assigned) ──────────────────
            else:
                if dist_from_base > 0.5:
                    reward += min(dist_from_base / 2.3, 1.0) * 0.15
                self._prev_site_dist  = None
                self._prev_phero_dist = None

        else:
            # ── Carrying: approach base ────────────────────────────────
            dist_to_base = obs_comp["base_dist_norm"] * 3.5
            if self._prev_base_dist is not None:
                reward += (self._prev_base_dist - dist_to_base) * 8.0
            self._prev_base_dist  = dist_to_base
            reward += math.cos(obs_comp["base_angle_norm"] * math.pi) * 0.5
            self._prev_site_dist  = None
            self._prev_phero_dist = None

        reward -= 0.005
        return reward

    # =========================================================================
    # PPO training
    # =========================================================================

    def _train_ppo(self, last_obs):
        with torch.no_grad():
            obs_t      = torch.FloatTensor(last_obs).unsqueeze(0).to(self._ppo.device)
            last_value = self._ppo.policy.predict_values(obs_t)

        self._ppo.rollout_buffer.compute_returns_and_advantage(
            last_values = last_value,
            dones       = np.array([False]),
        )
        self._ppo.train()
        self._ppo.rollout_buffer.reset()
        self._n_updates += 1

        s = self._stats_writer.data
        self._log(
            f"{'─'*43}\n"
            f"| {'rollout/':<20} {'':>18} |\n"
            f"|    {'ep_rew_total':<16} {self._ep_total_reward:>18.1f} |\n"
            f"| {'time/':<20} {'':>18} |\n"
            f"|    {'iterations':<16} {self._n_updates:>18} |\n"
            f"|    {'total_timesteps':<16} {self._total_steps:>18} |\n"
            f"| {'train/':<20} {'':>18} |\n"
            f"|    {'approx_kl':<16} {s.get('train/approx_kl', 0):>18.8f} |\n"
            f"|    {'clip_fraction':<16} {s.get('train/clip_fraction', 0):>18.4f} |\n"
            f"|    {'entropy_loss':<16} {s.get('train/entropy_loss', 0):>18.1f} |\n"
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
    # Gossip model sharing (channel 11, 2m — unchanged from v3)
    # =========================================================================

    def _collect_neighbor_models(self):
        while self._model_receiver.getQueueLength() > 0:
            data = base64.b64decode(self._model_receiver.getString())
            self._model_receiver.nextPacket()
            self._pending_neighbor_weights.append(data)

    def _merge_neighbor_models(self):
        if not self._pending_neighbor_weights:
            return
        n_merged = 0
        for data in self._pending_neighbor_weights:
            try:
                neighbor_weights = pickle.loads(data)
                local_state      = self._ppo.policy.state_dict()
                merged_state     = {}
                for k, local_v in local_state.items():
                    if k in neighbor_weights:
                        n_v = torch.tensor(neighbor_weights[k],
                                           dtype=local_v.dtype,
                                           device=local_v.device)
                        merged_state[k] = (1.0 - GOSSIP_ALPHA) * local_v + GOSSIP_ALPHA * n_v
                    else:
                        merged_state[k] = local_v
                self._ppo.policy.load_state_dict(merged_state)
                n_merged += 1
            except Exception as e:
                self._log(f"[{self._robot_name}] Gossip merge error: {e}")

        self._pending_neighbor_weights.clear()
        if n_merged > 0:
            self._ep_gossip_merges += n_merged
            self._log(f"[{self._robot_name}] Gossip merged {n_merged} model(s) "
                      f"| Update {self._n_updates}")

    def _broadcast_model(self):
        try:
            weights = {k: v.detach().cpu().numpy()
                       for k, v in self._ppo.policy.state_dict().items()}
            data = base64.b64encode(pickle.dumps(weights)).decode('ascii')
            self._model_emitter.send(data)
            self._log(f"[{self._robot_name}] Gossip broadcast | "
                      f"{len(data) // 1024} KB | Update {self._n_updates}")
        except Exception as e:
            self._log(f"[{self._robot_name}] Gossip broadcast error: {e}")

    # =========================================================================
    # P1-P2 overrides (unchanged from v3)
    # =========================================================================

    def _apply_overrides(self, left, right, pos_x, pos_y):
        gps       = self.gps.getValues()
        yaw       = self.imu.getRollPitchYaw()[2]
        fwd       = [math.cos(yaw), math.sin(yaw)]
        robot_pos = [gps[0], gps[1]]
        wall_dist = 2.5 - max(abs(pos_x), abs(pos_y))
        prox      = [s.getValue() / 4096.0 for s in self.ps]

        # P1: Wall / obstacle escape
        if wall_dist < 0.35 or max(prox) > 0.55:
            return self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=4.0)

        dist_to_base = math.sqrt(pos_x ** 2 + pos_y ** 2)

        # BASE_ESC: nudge away from nest after deposit (prevents getting stuck on base cylinder)
        if not self.carrying and dist_to_base < 0.25:
            if dist_to_base > 0.001:
                esc_x = pos_x + (pos_x / dist_to_base) * 0.5
                esc_y = pos_y + (pos_y / dist_to_base) * 0.5
            else:
                esc_x, esc_y = 0.5, 0.0
            return self._steer_to(robot_pos, fwd, [esc_x, esc_y], gain=4.0)

        # P2: Return to base when carrying
        if self.carrying:
            return self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=2.5)

        return [left, right]

    def _steer_to(self, robot_pos, fwd, target, gain=4.0):
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
            l /= m; r /= m
        return [l, r]


if __name__ == '__main__':
    robot = EpuckDecentralizedTrainV4()
    robot.run()
