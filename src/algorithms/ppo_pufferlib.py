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
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
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
    log_dir: str = "./output/tensorboard"
    model_dir: str = "./output/models"
    save_model_freq: int = 10  # Save every N updates
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Misc
    seed: int = 1
    torch_deterministic: bool = True


class CNNActorCritic(nn.Module):
    """
    CNN-based Actor-Critic network for visual observations
    Suitable for DonkeyEnv's camera input
    Uses shared CNNFeatureExtractor from rl_utils
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        
        # Shared feature extractor
        self.feature_extractor = CNNFeatureExtractor(observation_space)
        feature_dim = self.feature_extractor.feature_dim
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_space.shape[0]), std=0.01),
        )
        
        # Actor log std (learned parameter)
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_space.shape[0]))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(feature_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
    
    def get_value(self, x):
        """Get value estimate for state"""
        features = self.feature_extractor(x)
        return self.critic(features)
    
    def get_action_and_value(self, x, action=None):
        """Get action distribution and value estimate"""
        features = self.feature_extractor(x)
        
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
            policy_name="ppo",
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
        
        # Setup logging
        self.log_dir, self.model_dir = setup_logging_dirs(config.log_dir, config.model_dir)
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.global_step = 0
        self.update = 0
        self.episode_metrics = EpisodeMetricsLogger()
        
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
        
        next_obs = prepare_observation(next_obs, self.device)
        next_done = torch.zeros(self.config.num_envs).to(self.device)
        
        # Collect num_steps per environment
        for step in range(self.config.num_steps):
            self.global_step += self.config.num_envs
            
            # Get action from policy
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                value = value.flatten()
            
            # Convert and clip actions to action space
            action_np = clip_action_to_space(action, self.envs.single_action_space)
            
            # Step environment
            next_obs_np, reward, terminated, truncated, info = self.envs.step(action_np)
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
            next_obs = prepare_observation(next_obs_np, self.device)
            next_done = torch.tensor(done, dtype=torch.float32).to(self.device)
            
            # Log episode metrics
            for idx, d in enumerate(done):
                if d:
                    metrics = extract_episode_metrics(info, idx, d)
                    if metrics:
                        self.episode_metrics.add_metrics(metrics)
        
        # Compute returns and advantages
        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
            returns, advantages = self.buffer.compute_returns_and_advantages(
                next_value,
                next_done,
                self.config.gamma,
                self.config.gae_lambda,
            )
        
        return returns, advantages
    
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
        print(f"Tensorboard logs: {self.log_dir}")
        print(f"Model checkpoints: {self.model_dir}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        num_updates = self.config.total_timesteps // self.batch_size
        
        for update in range(1, num_updates + 1):
            self.update = update
            
            # Collect rollouts
            returns, advantages = self.collect_rollouts()
            
            # Update policy
            train_stats = self.update_policy(returns, advantages)
            
            # Log episode metrics to TensorBoard
            self.episode_metrics.log_to_tensorboard(self.writer, self.global_step)
            
            # Log training stats
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
            
            # Print episode metrics summary
            summary = self.episode_metrics.print_summary()
            if summary:
                print(summary)
            
            # Print training stats
            print(f"  PG Loss: {train_stats['pg_loss']:.4f} | V Loss: {train_stats['v_loss']:.4f}")
            print(f"  Entropy: {train_stats['entropy_loss']:.4f} | KL: {train_stats['approx_kl']:.4f}")
            
            # Reset episode metrics for next update
            self.episode_metrics.reset()
            
            # Save model
            if update % self.config.save_model_freq == 0:
                model_path = f"{self.model_dir}/ppo_donkey_update{update}.pt"
                torch.save({
                    "update": update,
                    "global_step": self.global_step,
                    "model_state_dict": self.agent.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }, model_path)
                print(f"  Saved model to {model_path}")
        
        # Save final model
        final_model_path = f"{self.model_dir}/ppo_donkey_final.pt"
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

