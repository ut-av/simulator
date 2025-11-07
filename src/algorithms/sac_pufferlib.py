#!/usr/bin/env python3
"""
CleanRL-style SAC implementation compatible with PufferLib
Based on https://docs.cleanrl.dev/rl-algorithms/sac/
Optimized for parallel environment execution with visual observations
"""

import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from pufferlib_wrapper import make_vectorized_env
from rl_utils import (
    CNNFeatureExtractor,
    EpisodeMetricsLogger,
    clip_action_to_space,
    extract_episode_metrics,
    layer_init,
    prepare_observation,
    set_random_seeds,
    setup_logging_dirs,
)


@dataclass
class SACConfig:
    """Configuration for SAC training"""
    # Environment
    env_name: str = "donkey-circuit-launch-track-v0"
    num_envs: int = 1  # SAC typically uses fewer parallel envs than PPO
    start_port: int = 9091
    backend: str = "serial"
    
    # Training
    total_timesteps: int = 1000000
    buffer_size: int = 100000
    learning_starts: int = 5000
    batch_size: int = 256
    tau: float = 0.005  # Target network update rate
    gamma: float = 0.99
    
    # Learning rates
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    
    # SAC specific
    alpha: float = 0.2  # Entropy coefficient (or "auto" for automatic tuning)
    autotune: bool = True  # Automatic entropy tuning
    target_entropy_scale: float = -1.0  # Scale for target entropy (-action_dim)
    
    # Networks
    policy_frequency: int = 1  # Update policy every N steps
    target_network_frequency: int = 1  # Update target every N steps
    
    # Logging
    log_dir: str = "./output/tensorboard"
    model_dir: str = "./output/models"
    save_model_freq: int = 50000  # Save every N steps
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Misc
    seed: int = 1
    torch_deterministic: bool = True


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    """
    SAC Actor network with Gaussian policy and tanh squashing
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        self.feature_extractor = CNNFeatureExtractor(observation_space)
        feature_dim = self.feature_extractor.feature_dim
        
        action_dim = np.prod(action_space.shape)
        
        # Policy head
        self.fc1 = layer_init(nn.Linear(feature_dim, 256))
        self.fc_mean = layer_init(nn.Linear(256, action_dim), std=0.01)
        self.fc_logstd = layer_init(nn.Linear(256, action_dim), std=0.01)
        
        # Action scale and bias for rescaling tanh output
        self.register_buffer(
            "action_scale",
            torch.FloatTensor((action_space.high - action_space.low) / 2.0)
        )
        self.register_buffer(
            "action_bias",
            torch.FloatTensor((action_space.high + action_space.low) / 2.0)
        )
    
    def forward(self, x):
        """Forward pass to get action distribution"""
        features = self.feature_extractor(x)
        x = F.relu(self.fc1(features))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std
    
    def get_action(self, x, deterministic=False):
        """Sample action from policy"""
        mean, log_std = self(x)
        std = log_std.exp()
        normal = Normal(mean, std)
        
        if deterministic:
            x_t = mean
        else:
            x_t = normal.rsample()  # Reparameterization trick
        
        # Tanh squashing
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        
        # Compute log probability with tanh correction
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        
        return action, log_prob, mean


class QNetwork(nn.Module):
    """
    SAC Q-network (critic) for state-action value estimation
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        self.feature_extractor = CNNFeatureExtractor(observation_space)
        feature_dim = self.feature_extractor.feature_dim
        
        action_dim = np.prod(action_space.shape)
        
        # Q-value head
        self.fc1 = layer_init(nn.Linear(feature_dim + action_dim, 256))
        self.fc2 = layer_init(nn.Linear(256, 256))
        self.fc3 = layer_init(nn.Linear(256, 1), std=1.0)
    
    def forward(self, x, a):
        """Forward pass to get Q-value"""
        features = self.feature_extractor(x)
        x = torch.cat([features, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q


class ReplayBuffer:
    """Experience replay buffer for off-policy learning"""
    def __init__(self, buffer_size, obs_shape, action_shape, device):
        self.buffer_size = buffer_size
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Allocate storage
        self.observations = torch.zeros((buffer_size,) + obs_shape, dtype=torch.uint8)
        self.next_observations = torch.zeros((buffer_size,) + obs_shape, dtype=torch.uint8)
        self.actions = torch.zeros((buffer_size,) + action_shape)
        self.rewards = torch.zeros((buffer_size,))
        self.dones = torch.zeros((buffer_size,))
    
    def add(self, obs, next_obs, action, reward, done):
        """Add a transition to the buffer"""
        # Store as uint8 to save memory (convert to float on sampling)
        self.observations[self.ptr] = torch.from_numpy(obs).byte()
        self.next_observations[self.ptr] = torch.from_numpy(next_obs).byte()
        self.actions[self.ptr] = torch.from_numpy(action)
        # Convert numpy scalars to Python float/bool
        self.rewards[self.ptr] = float(reward)
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        return dict(
            observations=self.observations[idxs].to(self.device),
            next_observations=self.next_observations[idxs].to(self.device),
            actions=self.actions[idxs].to(self.device),
            rewards=self.rewards[idxs].to(self.device),
            dones=self.dones[idxs].to(self.device),
        )


class SACTrainer:
    """SAC Trainer using PufferLib vectorization"""
    
    def __init__(self, config: SACConfig):
        self.config = config
        
        # Set random seeds
        set_random_seeds(config.seed, config.torch_deterministic)
        
        # Setup device
        self.device = torch.device(config.device)
        
        # Create vectorized environment
        print(f"Creating {config.num_envs} parallel environments...")
        self.envs = make_vectorized_env(
            env_name=config.env_name,
            num_envs=config.num_envs,
            start_port=config.start_port,
            backend=config.backend,
            policy_name="sac",
        )
        
        # Get observation and action spaces
        obs_space = self.envs.single_observation_space
        action_space = self.envs.single_action_space
        
        print(f"Observation space: {obs_space}")
        print(f"Action space: {action_space}")
        
        # Create actor
        self.actor = Actor(obs_space, action_space).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.policy_lr)
        
        # Create Q-networks (double Q-learning)
        self.qf1 = QNetwork(obs_space, action_space).to(self.device)
        self.qf2 = QNetwork(obs_space, action_space).to(self.device)
        self.qf1_target = QNetwork(obs_space, action_space).to(self.device)
        self.qf2_target = QNetwork(obs_space, action_space).to(self.device)
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())
        
        self.q_optimizer = optim.Adam(
            list(self.qf1.parameters()) + list(self.qf2.parameters()),
            lr=config.q_lr
        )
        
        # Automatic entropy tuning
        if config.autotune:
            self.target_entropy = config.target_entropy_scale * np.prod(action_space.shape)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.q_lr)
        else:
            self.alpha = config.alpha
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            buffer_size=config.buffer_size,
            obs_shape=obs_space.shape,
            action_shape=action_space.shape,
            device=self.device,
        )
        
        # Setup logging
        self.log_dir, self.model_dir = setup_logging_dirs(config.log_dir, config.model_dir)
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.global_step = 0
        self.episode_metrics = EpisodeMetricsLogger()
    
    def collect_experience(self, obs, deterministic=False):
        """
        Collect one step of experience from all environments
        Returns: next_obs, metrics_list
        """
        # Get action from policy
        with torch.no_grad():
            obs_tensor = prepare_observation(obs, self.device)
            actions, _, _ = self.actor.get_action(obs_tensor, deterministic=deterministic)
        
        # Convert to numpy and clip
        action_np = clip_action_to_space(actions, self.envs.single_action_space)
        
        # Step environment
        next_obs, reward, terminated, truncated, info = self.envs.step(action_np)
        done = np.logical_or(terminated, truncated)
        
        # Ensure all arrays are numpy arrays
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        if not isinstance(next_obs, np.ndarray):
            next_obs = np.array(next_obs)
        if not isinstance(reward, np.ndarray):
            reward = np.array(reward)
        if not isinstance(done, np.ndarray):
            done = np.array(done)
        
        # Store transitions in replay buffer
        for i in range(self.config.num_envs):
            self.replay_buffer.add(
                obs[i] if self.config.num_envs > 1 else obs,
                next_obs[i] if self.config.num_envs > 1 else next_obs,
                action_np[i] if self.config.num_envs > 1 else action_np,
                reward[i] if self.config.num_envs > 1 else reward,
                done[i] if self.config.num_envs > 1 else done,
            )
        
        # Extract episode metrics
        metrics_list = []
        for i in range(self.config.num_envs):
            d = done[i] if self.config.num_envs > 1 else done
            if isinstance(d, np.ndarray):
                d = d.item()
            metrics = extract_episode_metrics(info, i, d)
            if metrics:
                metrics_list.append(metrics)
        
        return next_obs, metrics_list
    
    def update_networks(self):
        """Update actor and critic networks"""
        # Sample from replay buffer
        data = self.replay_buffer.sample(self.config.batch_size)
        
        # Update Q-functions
        with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = self.actor.get_action(data["next_observations"])
            qf1_next_target = self.qf1_target(data["next_observations"], next_state_actions)
            qf2_next_target = self.qf2_target(data["next_observations"], next_state_actions)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = data["rewards"].flatten() + (1 - data["dones"].flatten()) * self.config.gamma * min_qf_next_target.view(-1)
        
        qf1_a_values = self.qf1(data["observations"], data["actions"]).view(-1)
        qf2_a_values = self.qf2(data["observations"], data["actions"]).view(-1)
        qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
        qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        
        self.q_optimizer.zero_grad()
        qf_loss.backward()
        self.q_optimizer.step()
        
        # Update policy
        if self.global_step % self.config.policy_frequency == 0:
            pi, log_pi, _ = self.actor.get_action(data["observations"])
            qf1_pi = self.qf1(data["observations"], pi)
            qf2_pi = self.qf2(data["observations"], pi)
            min_qf_pi = torch.min(qf1_pi, qf2_pi)
            actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update alpha (temperature parameter)
            if self.config.autotune:
                with torch.no_grad():
                    _, log_pi, _ = self.actor.get_action(data["observations"])
                alpha_loss = (-self.log_alpha.exp() * (log_pi + self.target_entropy)).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
        else:
            actor_loss = torch.tensor(0.0)
            alpha_loss = torch.tensor(0.0) if self.config.autotune else None
        
        # Update target networks
        if self.global_step % self.config.target_network_frequency == 0:
            for param, target_param in zip(self.qf1.parameters(), self.qf1_target.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
            for param, target_param in zip(self.qf2.parameters(), self.qf2_target.parameters()):
                target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        return {
            "qf1_loss": qf1_loss.item(),
            "qf2_loss": qf2_loss.item(),
            "qf_loss": qf_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss.item() if alpha_loss is not None else 0.0,
        }
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"Starting SAC training with PufferLib")
        print(f"{'='*60}")
        print(f"Total timesteps: {self.config.total_timesteps}")
        print(f"Num environments: {self.config.num_envs}")
        print(f"Buffer size: {self.config.buffer_size}")
        print(f"Learning starts: {self.config.learning_starts}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Device: {self.device}")
        print(f"Tensorboard logs: {self.log_dir}")
        print(f"Model checkpoints: {self.model_dir}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Reset environment
        reset_result = self.envs.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        # Training loop
        for global_step in range(self.config.total_timesteps):
            self.global_step = global_step
            
            # Collect experience
            obs, metrics_list = self.collect_experience(obs, deterministic=False)
            
            # Log episode metrics
            for metrics in metrics_list:
                self.episode_metrics.add_metrics(metrics)
            
            # Update networks (after learning_starts)
            if global_step > self.config.learning_starts:
                train_stats = self.update_networks()
                
                # Log training stats
                if global_step % 100 == 0:
                    self.writer.add_scalar("losses/qf1_loss", train_stats["qf1_loss"], global_step)
                    self.writer.add_scalar("losses/qf2_loss", train_stats["qf2_loss"], global_step)
                    self.writer.add_scalar("losses/qf_loss", train_stats["qf_loss"], global_step)
                    self.writer.add_scalar("losses/actor_loss", train_stats["actor_loss"], global_step)
                    self.writer.add_scalar("losses/alpha", train_stats["alpha"], global_step)
                    if self.config.autotune:
                        self.writer.add_scalar("losses/alpha_loss", train_stats["alpha_loss"], global_step)
            
            # Periodic logging
            if global_step % 1000 == 0:
                sps = int(global_step / (time.time() - start_time))
                print(f"Step {global_step}/{self.config.total_timesteps} | SPS: {sps}")
                
                # Log and reset episode metrics
                if len(self.episode_metrics.episode_rewards) > 0:
                    self.episode_metrics.log_to_tensorboard(self.writer, global_step)
                    summary = self.episode_metrics.print_summary()
                    if summary:
                        print(summary)
                    self.episode_metrics.reset()
            
            # Save model
            if global_step > 0 and global_step % self.config.save_model_freq == 0:
                model_path = f"{self.model_dir}/sac_donkey_step{global_step}.pt"
                torch.save({
                    "global_step": global_step,
                    "actor_state_dict": self.actor.state_dict(),
                    "qf1_state_dict": self.qf1.state_dict(),
                    "qf2_state_dict": self.qf2.state_dict(),
                    "qf1_target_state_dict": self.qf1_target.state_dict(),
                    "qf2_target_state_dict": self.qf2_target.state_dict(),
                    "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
                    "q_optimizer_state_dict": self.q_optimizer.state_dict(),
                    "alpha": self.alpha,
                }, model_path)
                print(f"  Saved model to {model_path}")
        
        # Save final model
        final_model_path = f"{self.model_dir}/sac_donkey_final.pt"
        torch.save({
            "global_step": self.global_step,
            "actor_state_dict": self.actor.state_dict(),
            "qf1_state_dict": self.qf1.state_dict(),
            "qf2_state_dict": self.qf2.state_dict(),
            "qf1_target_state_dict": self.qf1_target.state_dict(),
            "qf2_target_state_dict": self.qf2_target.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "alpha": self.alpha,
        }, final_model_path)
        print(f"\nSaved final model to {final_model_path}")
        
        self.envs.close()
        self.writer.close()
        
        total_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {total_time:.2f}s")
        print(f"Average SPS: {int(self.config.total_timesteps / total_time)}")


def main():
    parser = argparse.ArgumentParser(description="SAC training with PufferLib")
    parser.add_argument("--env-name", type=str, default="donkey-circuit-launch-track-v0")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--start-port", type=int, default=9091)
    parser.add_argument("--backend", type=str, default="serial", choices=["serial", "multiprocessing", "ray"])
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--q-lr", type=float, default=3e-4)
    parser.add_argument("--autotune", action="store_true", default=True)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    
    # Create config
    config = SACConfig(
        env_name=args.env_name,
        num_envs=args.num_envs,
        start_port=args.start_port,
        backend=args.backend,
        total_timesteps=args.total_timesteps,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        policy_lr=args.policy_lr,
        q_lr=args.q_lr,
        autotune=args.autotune,
        alpha=args.alpha,
        device=args.device,
        seed=args.seed,
    )
    
    # Create trainer and train
    trainer = SACTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

