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
from stable_baselines3.common.logger import configure as sb3_configure

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'epuck_decentralized'))
from epuck_decentralized import EpuckDecentralized, INITIAL_TTL

# =============================================================================
# FULLY DECENTRALIZED TRAINING ROBOT CONTROLLER — v3
#
# Changes from v2:
#   1. P4 (tag-seek override) removed — PPO learns tag approach fully
#   2. P3 threshold raised 0.3m → 0.8m (matches near-base penalty boundary)
#   3. Gossip model sharing: each robot broadcasts its PPO policy weights via
#      a dedicated emitter (channel 11, 2m range) after every training update.
#      Received neighbor weights are merged using FedAvg (80% local / 20% neighbor)
#      after the local training update, before the next episode begins.
#
# Per step:
#   1. Receive from supervisor: [tag_visible, tag_dist_norm, tag_angle_norm, pickup_signal]
#   2. Drain model receiver queue (collect neighbor weights, don't merge yet)
#   3. Assemble 18D obs from onboard sensors + supervisor tag obs
#   4. Compute own reward from local state + pickup_signal
#   5. Add transition to own rollout buffer
#   6. Infer action from own PPO policy
#   7. Apply P1-P3 overrides onboard (GPS + IMU)
#   8. Drive motors
#   9. Train own PPO when buffer fills (every N_STEPS steps)
#  10. Merge collected neighbor weights (FedAvg), broadcast own model
#  11. Save own model checkpoint periodically
#
# Supervisor → Robot: [tag_visible, tag_dist_norm, tag_angle_norm, pickup_signal]
# Robot → Supervisor: 17 floats (prox×8, carrying, base_dist, base_angle,
#                                phero×4, gps_x, gps_y)
# =============================================================================

# ── PPO hyperparameters ───────────────────────────────────────────────────────
OBS_DIM       = 18
ACT_DIM       = 2
N_STEPS       = 8192
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
TOTAL_STEPS   = 3_000_000

# ── Gossip model sharing ──────────────────────────────────────────────────────
GOSSIP_ALPHA  = 0.2   # neighbor weight in FedAvg: merged = (1-α)×local + α×neighbor

# ── Episode / arena constants ─────────────────────────────────────────────────
STEPS_PER_EPISODE = 8192
TAG_SEEK_RANGE    = 1.0


class _DummyEnv(gym.Env):
    """Minimal Gym env — used only to initialise SB3's PPO object, never stepped."""
    observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
    action_space      = gym.spaces.Box(-1.0, 1.0, (ACT_DIM,), np.float32)

    def reset(self):
        return np.zeros(OBS_DIM, dtype=np.float32)

    def step(self, _):
        return np.zeros(OBS_DIM), 0.0, False, {}


class EpuckDecentralizedTrainV3(EpuckDecentralized):
    """
    Fully decentralized training robot — v3.
    Inherits all sensor / pheromone infrastructure from EpuckDecentralized.
    Adds: own PPO policy, rollout buffer, reward computation, training loop,
          and gossip model sharing with neighbors within 2m (channel 11).
    """

    def __init__(self):
        super().__init__()

        self._project_root = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
        )
        self._robot_name = self.getName()   # "robot1" … "robot4"

        # Read run name written by supervisor before simulation starts
        cfg = os.path.join(self._project_root, 'current_run_name.txt')
        self._run_name = open(cfg).read().strip() if os.path.exists(cfg) else 'decentralized_indep_v3'

        # ── SB3 PPO (own model per robot) ────────────────────────────────────
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
        self._ppo.set_logger(sb3_configure(None, []))

        # ── Per-robot log file ────────────────────────────────────────────────
        _log_dir = os.path.join(self._project_root, 'logs',
                                f"{self._robot_name}_{self._run_name}")
        os.makedirs(_log_dir, exist_ok=True)
        self._log_path = os.path.join(_log_dir, 'training_log.txt')

        # ── Gossip model sharing (channel 11, 2m range) ───────────────────────
        try:
            self._model_emitter  = self.getDevice('model_emitter')
            self._model_receiver = self.getDevice('model_receiver')
            self._model_receiver.enable(self.time_step)
            self._gossip_enabled = True
            print(f"[{self._robot_name}] Gossip model sharing enabled (channel 11, 2m)")
        except Exception:
            self._gossip_enabled = False
            print(f"[{self._robot_name}] Gossip model sharing disabled (devices not found)")
        self._pending_neighbor_weights = []   # collected each step, merged after training

        # ── Tag obs (received from supervisor each step) ──────────────────────
        self._tag_visible    = 0.0
        self._tag_dist_norm  = 0.0
        self._tag_angle_norm = 0.0

        # ── Rollout buffer tracking ───────────────────────────────────────────
        self._last_obs         = None
        self._last_raw_action  = np.zeros(ACT_DIM, dtype=np.float32)
        self._last_value       = None
        self._last_log_prob    = None
        self._last_is_ep_start = True

        # ── Episode / step counters ───────────────────────────────────────────
        self._step_in_episode = 0
        self._total_steps     = 0
        self._n_updates       = 0

        # ── Reward shaping state ──────────────────────────────────────────────
        self._prev_tag_dist     = None
        self._prev_base_dist    = None
        self._prev_hotspot_dist = None

        # ── Per-episode stats ─────────────────────────────────────────────────
        self._ep_pickups     = 0
        self._ep_deposits    = 0
        self._ep_gossip_merges = 0
        self._training_done  = False

    def _log(self, msg):
        print(msg)
        with open(self._log_path, 'a') as f:
            f.write(msg + '\n')

    # =========================================================================
    # deepbots interface
    # =========================================================================

    def create_message(self):
        """17-float state message to supervisor; also runs pheromone P2P."""
        return super().create_message()

    def use_message_data(self, message):
        """
        Called every simulation step with supervisor's 4-float message.
        Runs the full training loop: obs → reward → buffer → PPO infer → motors.
        """
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

        # ── Drain gossip receiver every step (avoid queue overflow) ───────────
        if self._gossip_enabled:
            self._collect_neighbor_models()

        # ── Episode boundary ──────────────────────────────────────────────────
        is_ep_start = (self._step_in_episode == 0)
        if is_ep_start and self._total_steps > 0:
            self._on_episode_reset()

        # ── Pickup / deposit state changes ────────────────────────────────────
        if pickup_signal > 0.0 and not self.carrying:
            self.carrying = True
            gps = self.gps.getValues()
            self.phero_memory = {
                "hotspot":  (gps[0], gps[1]),
                "strength": pickup_signal,
                "ttl":      INITIAL_TTL,
            }
            self._ep_pickups    += 1
            self._prev_tag_dist  = None
            self._prev_hotspot_dist = None

        elif pickup_signal < -0.5 and self.carrying:
            self.carrying = False
            self._ep_deposits += 1
            obs_comp = self._obs_components()
            if obs_comp["phero_known"] > 0.5:
                self._prev_hotspot_dist = obs_comp["phero_dist_norm"] * 3.5

        # ── Assemble 18D obs ──────────────────────────────────────────────────
        prox     = [s.getValue() / 4096.0 for s in self.ps]
        obs_comp = self._obs_components()
        obs = np.array(
            prox + [
                self._tag_visible,
                self._tag_dist_norm,
                self._tag_angle_norm,
                1.0 if self.carrying else 0.0,
                obs_comp["base_dist_norm"],
                obs_comp["base_angle_norm"],
                obs_comp["phero_known"],
                obs_comp["phero_dist_norm"],
                obs_comp["phero_angle_norm"],
                obs_comp["phero_strength"],
            ],
            dtype=np.float32,
        )

        # ── Compute reward for the transition arriving at this obs ────────────
        reward = self._compute_reward(obs_comp, pickup_signal)

        # ── Store previous transition in rollout buffer ───────────────────────
        if self._last_obs is not None:
            self._ppo.rollout_buffer.add(
                self._last_obs.reshape(1, -1),
                self._last_raw_action.reshape(1, -1),
                np.array([reward]),
                np.array([self._last_is_ep_start]),
                self._last_value,
                self._last_log_prob,
            )

        # ── PPO inference ─────────────────────────────────────────────────────
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self._ppo.device)
            actions_t, values_t, log_probs_t = self._ppo.policy.forward(obs_t)

        raw_action = actions_t.cpu().numpy().flatten()

        # ── P1-P3 overrides (onboard, GPS + IMU) — P4 removed for v3 ─────────
        left_cmd, right_cmd = self._apply_overrides(
            raw_action[0], raw_action[1],
            obs_comp["pos_x"], obs_comp["pos_y"],
        )

        # ── Drive motors ──────────────────────────────────────────────────────
        max_speed    = 6.28
        scale_factor = 6.0
        self.left_motor.setVelocity(
            max(min(left_cmd  * scale_factor, max_speed), -max_speed))
        self.right_motor.setVelocity(
            max(min(right_cmd * scale_factor, max_speed), -max_speed))

        # ── Update tracking for next step ─────────────────────────────────────
        self._last_obs         = obs
        self._last_raw_action  = raw_action
        self._last_value       = values_t
        self._last_log_prob    = log_probs_t
        self._last_is_ep_start = is_ep_start

        # ── Advance step counters ─────────────────────────────────────────────
        self._total_steps     += 1
        self._step_in_episode  = (self._step_in_episode + 1) % STEPS_PER_EPISODE

        # ── Train when rollout buffer is full, then gossip ────────────────────
        if self._ppo.rollout_buffer.full:
            self._train_ppo(obs)
            # Merge neighbor weights collected this episode, then broadcast own model.
            # Merging happens AFTER local training so rollout data stays on-policy,
            # and the NEXT episode starts with the gossip-updated policy.
            if self._gossip_enabled and self._n_updates >= 1:
                self._merge_neighbor_models()
                self._broadcast_model()

        # ── Periodic checkpoint save (skip at TOTAL_STEPS — handled below) ──────
        if self._total_steps > 0 and self._total_steps % SAVE_FREQ == 0 \
                and self._total_steps < TOTAL_STEPS:
            self._save_checkpoint()

        # ── Stop training at TOTAL_STEPS ──────────────────────────────────────
        if self._total_steps >= TOTAL_STEPS and not self._training_done:
            self._save_checkpoint()
            self._log(f"[{self._robot_name}] TRAINING COMPLETE | "
                      f"{self._total_steps} steps | Final model saved.")
            self._training_done = True
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)

    # =========================================================================
    # Episode management
    # =========================================================================

    def _on_episode_reset(self):
        self.carrying        = False
        self.phero_memory    = {"hotspot": None, "strength": 0.0, "ttl": 0}
        self._prev_tag_dist     = None
        self._prev_base_dist    = None
        self._prev_hotspot_dist = None

        self._log(f"[{self._robot_name}] EP END | "
                  f"Picks: {self._ep_pickups} | Deps: {self._ep_deposits} | "
                  f"Steps: {self._total_steps} | Updates: {self._n_updates} | "
                  f"GossipMerges: {self._ep_gossip_merges}")
        self._ep_pickups       = 0
        self._ep_deposits      = 0
        self._ep_gossip_merges = 0

    # =========================================================================
    # Reward computation (fully onboard)
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

        if wall_dist < 0.6:
            reward -= (0.6 - wall_dist) * 0.5

        if pickup_signal > 0.0:
            reward += 5.0
        elif pickup_signal < -0.5:
            reward += 20.0

        if not self.carrying:
            curr_tag_dist = (self._tag_dist_norm * TAG_SEEK_RANGE
                             if self._tag_visible > 0.5 else None)
            if self._prev_tag_dist is not None and curr_tag_dist is not None:
                reward += (self._prev_tag_dist - curr_tag_dist) * 8.0
            self._prev_tag_dist = curr_tag_dist

            if self._tag_visible > 0.5:
                reward += 0.2

            dist_from_base = math.sqrt(gps_x ** 2 + gps_y ** 2)

            if 0.8 < dist_from_base < 2.4:
                reward += 0.10

            if dist_from_base < 0.8:
                reward -= (0.8 - dist_from_base) * 4.0

            if wall_dist >= 0.6:
                avg_speed = (self._last_raw_action[0] + self._last_raw_action[1]) / 2.0
                if avg_speed > 0:
                    reward += avg_speed * 0.01

            if obs_comp["phero_known"] > 0.5:
                curr_hd = obs_comp["phero_dist_norm"] * 3.5
                if self._prev_hotspot_dist is not None:
                    reward += (self._prev_hotspot_dist - curr_hd) * 15.0
                self._prev_hotspot_dist = curr_hd
                reward += math.cos(obs_comp["phero_angle_norm"] * math.pi) * 0.3
            else:
                if dist_from_base > 0.8:
                    reward += min(dist_from_base / 2.3, 1.0) * 0.15
                self._prev_hotspot_dist = None

        else:
            dist_to_base = obs_comp["base_dist_norm"] * 3.5
            if self._prev_base_dist is not None:
                reward += (self._prev_base_dist - dist_to_base) * 8.0
            self._prev_base_dist = dist_to_base
            reward += math.cos(obs_comp["base_angle_norm"] * math.pi) * 0.5
            if self.carrying:
                self._prev_hotspot_dist = None

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

        self._log(f"[{self._robot_name}] Update {self._n_updates} | "
                  f"Step {self._total_steps}")

    def _save_checkpoint(self):
        log_dir = os.path.join(self._project_root, 'logs',
                               f"{self._robot_name}_{self._run_name}")
        os.makedirs(log_dir, exist_ok=True)
        path = os.path.join(log_dir,
                            f"{self._robot_name}_{self._run_name}"
                            f"_{self._total_steps}_steps")
        self._ppo.save(path)
        latest = os.path.join(self._project_root,
                              f"{self._robot_name}_{self._run_name}")
        self._ppo.save(latest)
        self._log(f"[{self._robot_name}] Saved checkpoint: {path}.zip")

    # =========================================================================
    # Gossip model sharing (channel 11, 2m range)
    # =========================================================================

    def _collect_neighbor_models(self):
        """Drain model receiver queue each step to avoid overflow."""
        while self._model_receiver.getQueueLength() > 0:
            data = base64.b64decode(self._model_receiver.getString())
            self._model_receiver.nextPacket()
            self._pending_neighbor_weights.append(data)

    def _merge_neighbor_models(self):
        """
        FedAvg merge with all neighbor weights collected this episode.
        merged = (1 - GOSSIP_ALPHA) × local + GOSSIP_ALPHA × neighbor
        Applied once per neighbor; multiple neighbors are merged sequentially.
        """
        if not self._pending_neighbor_weights:
            return

        n_merged = 0
        for data in self._pending_neighbor_weights:
            try:
                neighbor_weights = pickle.loads(data)
                local_state = self._ppo.policy.state_dict()
                merged_state = {}
                for k, local_v in local_state.items():
                    if k in neighbor_weights:
                        n_v = torch.tensor(
                            neighbor_weights[k],
                            dtype=local_v.dtype,
                            device=local_v.device,
                        )
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
            self._log(f"[{self._robot_name}] Gossip merged {n_merged} neighbor model(s) "
                      f"| Update {self._n_updates}")

    def _broadcast_model(self):
        """Serialize and broadcast own policy weights to neighbors within 2m."""
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
    # P1-P3 overrides (P4 tag-seek removed — PPO learns tag approach in v3)
    # =========================================================================

    def _apply_overrides(self, left, right, pos_x, pos_y):
        gps       = self.gps.getValues()
        yaw       = self.imu.getRollPitchYaw()[2]
        fwd       = [math.cos(yaw), math.sin(yaw)]
        robot_pos = [gps[0], gps[1]]
        wall_dist = 2.5 - max(abs(pos_x), abs(pos_y))

        # P1: wall escape
        if wall_dist < 0.6:
            return self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=4.0)

        # P2: return to base when carrying
        if self.carrying:
            return self._steer_to(robot_pos, fwd, [0.0, 0.0], gain=2.5)

        # P3: base avoidance — 0.8m threshold matches near-base penalty boundary
        dist_to_base = math.sqrt(pos_x ** 2 + pos_y ** 2)
        if dist_to_base < 0.8:
            target = [pos_x * 5.0, pos_y * 5.0]
            return self._steer_to(robot_pos, fwd, target, gain=3.0)

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
            l /= m
            r /= m
        return [l, r]


if __name__ == '__main__':
    robot = EpuckDecentralizedTrainV3()
    robot.run()
