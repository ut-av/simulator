#!/usr/bin/env python3
"""
CleanRL-style PPO implementation compatible with PufferLib
Based on https://docs.cleanrl.dev/rl-algorithms/ppo/
Optimized for parallel environment execution
"""

import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from pufferlib_wrapper import make_vectorized_env


@dataclass
class PPOConfig:
    """Configuration for PPO training"""
    # Environment
    env_name: str = "donkey-circuit-launch-track-v0"
    num_envs: int = 4
    start_port: int = 9091
    backend: str = "serial"  # "serial", "multiprocessing", or "ray"
    
    # Training
    total_timesteps: int = 1000000
    learning_rate: float = 3e-4
    num_steps: int = 2048  # Steps per environment per rollout
    batch_size: int = 64
    num_minibatches: int = 32
    update_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    clip_vloss: bool = True
    
    # Logging
    log_dir: str = "./output/pufferlib_runs"
    save_model_freq: int = 10  # Save every N updates
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Misc
    seed: int = 1
    torch_deterministic: bool = True


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize network layer with orthogonal initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CNNActorCritic(nn.Module):
    """
    CNN-based Actor-Critic network for visual observations
    Suitable for DonkeyEnv's camera input
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        # Determine input shape
        # DonkeyEnv provides HxWxC images
        obs_shape = observation_space.shape
        if len(obs_shape) == 3:
            # Assume HxWxC format, convert to CxHxW for PyTorch
            self.input_channels = obs_shape[2]
        else:
            raise ValueError(f"Unexpected observation shape: {obs_shape}")
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            # Create dummy input in CxHxW format
            dummy_input = torch.zeros(1, self.input_channels, obs_shape[0], obs_shape[1])
            cnn_output_size = self.cnn(dummy_input).shape[1]
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(cnn_output_size, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_space.shape[0]), std=0.01),
        )
        
        # Actor log std (learned parameter)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_space.shape[0]))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(cnn_output_size, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
    
    def get_features(self, x):
        """Extract features from observations"""
        # Convert HxWxC to CxHxW and normalize to [0, 1]
        if len(x.shape) == 4:  # Batch of images
            x = x.permute(0, 3, 1, 2)  # BxHxWxC -> BxCxHxW
        else:
            x = x.permute(2, 0, 1)  # HxWxC -> CxHxW
        x = x.float() / 255.0
        return self.cnn(x)
    
    def get_value(self, x):
        """Get value estimate for state"""
        features = self.get_features(x)
        return self.critic(features)
    
    def get_action_and_value(self, x, action=None):
        """Get action distribution and value estimate"""
        features = self.get_features(x)
        
        # Actor: get mean and std for continuous actions
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        
        if action is None:
            action = probs.sample()
        
        # Compute log probability and entropy
        log_prob = probs.log_prob(action).sum(1)
        entropy = probs.entropy().sum(1)
        
        # Critic: get value estimate
        value = self.critic(features)
        
        return action, log_prob, entropy, value


class RolloutBuffer:
    """Buffer for storing rollout data"""
    def __init__(self, num_steps, num_envs, obs_shape, action_shape, device):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device
        
        # Allocate storage
        self.obs = torch.zeros((num_steps, num_envs) + obs_shape).to(device)
        self.actions = torch.zeros((num_steps, num_envs) + action_shape).to(device)
        self.logprobs = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)
        
        self.step = 0
    
    def add(self, obs, action, logprob, reward, done, value):
        """Add a step to the buffer"""
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.logprobs[self.step] = logprob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.step += 1
    
    def reset(self):
        """Reset the buffer"""
        self.step = 0
    
    def compute_returns_and_advantages(self, next_value, next_done, gamma, gae_lambda):
        """Compute returns and advantages using GAE"""
        advantages = torch.zeros_like(self.rewards).to(self.device)
        lastgaelam = 0
        
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
        returns = advantages + self.values
        return returns, advantages


class PPOTrainer:
    """PPO Trainer using PufferLib vectorization"""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        
        # Set random seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.backends.cudnn.deterministic = config.torch_deterministic
        
        # Setup device
        self.device = torch.device(config.device)
        
        # Create vectorized environment
        print(f"Creating {config.num_envs} parallel environments...")
        self.envs = make_vectorized_env(
            env_name=config.env_name,
            num_envs=config.num_envs,
            start_port=config.start_port,
            backend=config.backend,
        )
        
        # Get observation and action spaces
        obs_space = self.envs.single_observation_space
        action_space = self.envs.single_action_space
        
        print(f"Observation space: {obs_space}")
        print(f"Action space: {action_space}")
        
        # Create agent
        self.agent = CNNActorCritic(obs_space, action_space).to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config.learning_rate, eps=1e-5)
        
        # Create rollout buffer
        self.buffer = RolloutBuffer(
            num_steps=config.num_steps,
            num_envs=config.num_envs,
            obs_shape=obs_space.shape,
            action_shape=action_space.shape,
            device=self.device,
        )
        
        # Compute batch sizes
        self.batch_size = int(config.num_envs * config.num_steps)
        self.minibatch_size = int(self.batch_size // config.num_minibatches)
        
        # Setup logging with readable timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = f"{config.log_dir}/{timestamp}"
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.global_step = 0
        self.update = 0
        
    def collect_rollouts(self):
        """Collect rollouts from the environment"""
        self.buffer.reset()
        
        # Reset environments
        reset_result = self.envs.reset()
        # Handle tuple return (obs, info) or just obs
        if isinstance(reset_result, tuple):
            next_obs = reset_result[0]
        else:
            next_obs = reset_result
        # Convert to numpy array first, then to tensor (more efficient)
        if not isinstance(next_obs, np.ndarray):
            next_obs = np.array(next_obs)
        next_obs = torch.from_numpy(next_obs).float().to(self.device)
        next_done = torch.zeros(self.config.num_envs).to(self.device)
        
        episode_rewards = []
        episode_lengths = []
        episode_cte = []
        episode_speed = []
        episode_forward_vel = []
        episode_hits = []
        episode_lap_times = []
        episode_lap_counts = []
        
        # Collect num_steps per environment
        for step in range(self.config.num_steps):
            self.global_step += self.config.num_envs
            
            # Get action from policy
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                value = value.flatten()
            
            # Convert actions to numpy and ensure they match action space
            # Actions should be shape (num_envs, action_dim) with dtype matching action space
            action_np = action.cpu().numpy()
            # Ensure dtype matches action space (usually float32)
            if action_np.dtype != self.envs.single_action_space.dtype:
                action_np = action_np.astype(self.envs.single_action_space.dtype)
            # Clip actions to action space bounds
            action_np = np.clip(
                action_np,
                self.envs.single_action_space.low,
                self.envs.single_action_space.high
            )
            
            # Step environment
            # Pufferlib vectorized environments return (obs, reward, terminated, truncated, info)
            next_obs_np, reward, terminated, truncated, info = self.envs.step(action_np)
            
            # Combine terminated and truncated into done for buffer storage
            # (PPO typically uses 'done' which is True if either terminated or truncated)
            done = np.logical_or(terminated, truncated)
            
            # Store in buffer
            self.buffer.add(
                next_obs,
                action,
                logprob,
                torch.tensor(reward, dtype=torch.float32).to(self.device),
                next_done,
                value,
            )
            
            # Update for next iteration
            # Convert to numpy array first, then to tensor (more efficient)
            if not isinstance(next_obs_np, np.ndarray):
                next_obs_np = np.array(next_obs_np)
            next_obs = torch.from_numpy(next_obs_np).float().to(self.device)
            next_done = torch.tensor(done, dtype=torch.float32).to(self.device)
            
            # Log episode info
            for idx, d in enumerate(done):
                if d:
                    if "episode" in info:
                        ep_info = info["episode"][idx]
                        episode_rewards.append(ep_info["r"])
                        episode_lengths.append(ep_info["l"])
                
                # Log environment-specific metrics from info dict
                # These come from donkey_sim.py's observe() method
                if isinstance(info, dict):
                    # Cross track error (distance from center of track)
                    if "cte" in info:
                        cte_val = info["cte"][idx] if hasattr(info["cte"], "__getitem__") else info["cte"]
                        episode_cte.append(abs(cte_val))
                    
                    # Speed and velocity metrics
                    if "speed" in info:
                        speed_val = info["speed"][idx] if hasattr(info["speed"], "__getitem__") else info["speed"]
                        episode_speed.append(speed_val)
                    
                    if "forward_vel" in info:
                        fwd_vel = info["forward_vel"][idx] if hasattr(info["forward_vel"], "__getitem__") else info["forward_vel"]
                        episode_forward_vel.append(fwd_vel)
                    
                    # Collision detection
                    if "hit" in info:
                        hit_val = info["hit"][idx] if hasattr(info["hit"], "__getitem__") else info["hit"]
                        episode_hits.append(1.0 if hit_val != "none" else 0.0)
                    
                    # Lap timing metrics
                    if "last_lap_time" in info:
                        lap_time = info["last_lap_time"][idx] if hasattr(info["last_lap_time"], "__getitem__") else info["last_lap_time"]
                        if lap_time > 0.0:  # Only log when a lap is completed
                            episode_lap_times.append(lap_time)
                    
                    if "lap_count" in info:
                        lap_count = info["lap_count"][idx] if hasattr(info["lap_count"], "__getitem__") else info["lap_count"]
                        episode_lap_counts.append(lap_count)
        
        # Compute returns and advantages
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            returns, advantages = self.buffer.compute_returns_and_advantages(
                next_value,
                next_done,
                self.config.gamma,
                self.config.gae_lambda,
            )
        
        # Package all metrics for logging
        metrics = {
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths,
            "episode_cte": episode_cte,
            "episode_speed": episode_speed,
            "episode_forward_vel": episode_forward_vel,
            "episode_hits": episode_hits,
            "episode_lap_times": episode_lap_times,
            "episode_lap_counts": episode_lap_counts,
        }
        
        return returns, advantages, metrics
    
    def update_policy(self, returns, advantages):
        """Update policy using PPO"""
        # Flatten batch
        b_obs = self.buffer.obs.reshape((-1,) + self.envs.single_observation_space.shape)
        b_logprobs = self.buffer.logprobs.reshape(-1)
        b_actions = self.buffer.actions.reshape((-1,) + self.envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.buffer.values.reshape(-1)
        
        # Optimize policy for K epochs
        clipfracs = []
        for epoch in range(self.config.update_epochs):
            # Shuffle indices
            b_inds = np.arange(self.batch_size)
            np.random.shuffle(b_inds)
            
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                
                # Get new action probabilities and values
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                
                # Policy loss
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                # Track clipping fraction
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > self.config.clip_coef).float().mean().item()]
                
                # Normalize advantages
                mb_advantages = b_advantages[mb_inds]
                if self.config.normalize_advantages:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                
                # PPO policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config.clip_coef, 1 + self.config.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if self.config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.config.clip_coef,
                        self.config.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - self.config.ent_coef * entropy_loss + v_loss * self.config.vf_coef
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
        
        # Compute explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        return {
            "pg_loss": pg_loss.item(),
            "v_loss": v_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "old_approx_kl": old_approx_kl.item(),
            "approx_kl": approx_kl.item(),
            "clipfrac": np.mean(clipfracs) if len(clipfracs) > 0 else 0.0,
            "explained_variance": explained_var,
        }
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"Starting PPO training with PufferLib")
        print(f"{'='*60}")
        print(f"Total timesteps: {self.config.total_timesteps}")
        print(f"Num environments: {self.config.num_envs}")
        print(f"Steps per rollout: {self.config.num_steps}")
        print(f"Batch size: {self.batch_size}")
        print(f"Minibatch size: {self.minibatch_size}")
        print(f"Device: {self.device}")
        print(f"Experiment name: {self.log_dir}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        num_updates = self.config.total_timesteps // self.batch_size
        
        for update in range(1, num_updates + 1):
            self.update = update
            
            # Collect rollouts
            returns, advantages, metrics = self.collect_rollouts()
            
            # Update policy
            train_stats = self.update_policy(returns, advantages)
            
            # Extract metrics
            episode_rewards = metrics["episode_rewards"]
            episode_lengths = metrics["episode_lengths"]
            episode_cte = metrics["episode_cte"]
            episode_speed = metrics["episode_speed"]
            episode_forward_vel = metrics["episode_forward_vel"]
            episode_hits = metrics["episode_hits"]
            episode_lap_times = metrics["episode_lap_times"]
            episode_lap_counts = metrics["episode_lap_counts"]
            
            # Logging - Episode Performance Metrics
            if len(episode_rewards) > 0:
                # Convert to numpy array if needed and ensure non-empty
                episode_rewards_arr = np.array(episode_rewards) if not isinstance(episode_rewards, np.ndarray) else episode_rewards
                episode_lengths_arr = np.array(episode_lengths) if not isinstance(episode_lengths, np.ndarray) else episode_lengths
                if len(episode_rewards_arr) > 0:
                    self.writer.add_scalar("charts/episodic_return", np.mean(episode_rewards_arr), self.global_step)
                    self.writer.add_scalar("charts/episodic_length", np.mean(episode_lengths_arr), self.global_step)
            
            # Logging - Driving Performance Metrics
            if len(episode_cte) > 0:
                cte_arr = np.array(episode_cte)
                self.writer.add_scalar("driving/cross_track_error", np.mean(cte_arr), self.global_step)
                self.writer.add_scalar("driving/cte_std", np.std(cte_arr), self.global_step)
            
            if len(episode_speed) > 0:
                speed_arr = np.array(episode_speed)
                self.writer.add_scalar("driving/speed", np.mean(speed_arr), self.global_step)
            
            if len(episode_forward_vel) > 0:
                fwd_vel_arr = np.array(episode_forward_vel)
                self.writer.add_scalar("driving/forward_velocity", np.mean(fwd_vel_arr), self.global_step)
            
            if len(episode_hits) > 0:
                hits_arr = np.array(episode_hits)
                self.writer.add_scalar("driving/collision_rate", np.mean(hits_arr), self.global_step)
            
            # Logging - Lap Performance Metrics
            if len(episode_lap_times) > 0:
                lap_times_arr = np.array(episode_lap_times)
                self.writer.add_scalar("laps/lap_time_mean", np.mean(lap_times_arr), self.global_step)
                self.writer.add_scalar("laps/lap_time_min", np.min(lap_times_arr), self.global_step)
                self.writer.add_scalar("laps/lap_time_std", np.std(lap_times_arr), self.global_step)
            
            if len(episode_lap_counts) > 0:
                lap_counts_arr = np.array(episode_lap_counts)
                self.writer.add_scalar("laps/completed_laps", np.sum(lap_counts_arr), self.global_step)
            
            self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], self.global_step)
            self.writer.add_scalar("losses/value_loss", train_stats["v_loss"], self.global_step)
            self.writer.add_scalar("losses/policy_loss", train_stats["pg_loss"], self.global_step)
            self.writer.add_scalar("losses/entropy", train_stats["entropy_loss"], self.global_step)
            self.writer.add_scalar("losses/old_approx_kl", train_stats["old_approx_kl"], self.global_step)
            self.writer.add_scalar("losses/approx_kl", train_stats["approx_kl"], self.global_step)
            self.writer.add_scalar("losses/clipfrac", train_stats["clipfrac"], self.global_step)
            self.writer.add_scalar("losses/explained_variance", train_stats["explained_variance"], self.global_step)
            
            # Print progress
            sps = int(self.global_step / (time.time() - start_time))
            print(f"Update {update}/{num_updates} | Step {self.global_step} | SPS: {sps}")
            if len(episode_rewards) > 0:
                # Convert to numpy array if needed and ensure non-empty
                episode_rewards_arr = np.array(episode_rewards) if not isinstance(episode_rewards, np.ndarray) else episode_rewards
                episode_lengths_arr = np.array(episode_lengths) if not isinstance(episode_lengths, np.ndarray) else episode_lengths
                if len(episode_rewards_arr) > 0 and len(episode_lengths_arr) > 0:
                    print(f"  Reward: {np.mean(episode_rewards_arr):.2f} ± {np.std(episode_rewards_arr):.2f}")
                    print(f"  Length: {np.mean(episode_lengths_arr):.1f}")
            
            # Print driving performance
            if len(episode_cte) > 0:
                print(f"  CTE: {np.mean(episode_cte):.3f} | Speed: {np.mean(episode_speed):.2f} | Collisions: {np.mean(episode_hits):.2%}")
            if len(episode_lap_times) > 0:
                print(f"  Best Lap: {np.min(episode_lap_times):.2f}s | Avg Lap: {np.mean(episode_lap_times):.2f}s")
            
            print(f"  PG Loss: {train_stats['pg_loss']:.4f} | V Loss: {train_stats['v_loss']:.4f}")
            print(f"  Entropy: {train_stats['entropy_loss']:.4f} | KL: {train_stats['approx_kl']:.4f}")
            
            # Save model
            if update % self.config.save_model_freq == 0:
                model_path = f"{self.log_dir}/ppo_donkey_update{update}.pt"
                torch.save({
                    "update": update,
                    "global_step": self.global_step,
                    "model_state_dict": self.agent.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }, model_path)
                print(f"  Saved model to {model_path}")
        
        # Save final model
        final_model_path = f"{self.log_dir}/ppo_donkey_final.pt"
        torch.save({
            "update": self.update,
            "global_step": self.global_step,
            "model_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, final_model_path)
        print(f"\nSaved final model to {final_model_path}")
        
        self.envs.close()
        self.writer.close()
        
        total_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {total_time:.2f}s")
        print(f"Average SPS: {int(self.config.total_timesteps / total_time)}")


def main():
    parser = argparse.ArgumentParser(description="PPO training with PufferLib")
    parser.add_argument("--env-name", type=str, default="donkey-circuit-launch-track-v0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--start-port", type=int, default=9091)
    parser.add_argument("--backend", type=str, default="serial", choices=["serial", "multiprocessing", "ray"])
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-steps", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    
    # Create config
    config = PPOConfig(
        env_name=args.env_name,
        num_envs=args.num_envs,
        start_port=args.start_port,
        backend=args.backend,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        device=args.device,
        seed=args.seed,
    )
    
    # Create trainer and train
    trainer = PPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()

