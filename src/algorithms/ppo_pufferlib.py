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
    VisualizationWindow,
    clip_action_to_space,
    extract_episode_metrics,
    extract_lap_time_metrics,
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
    visualize: bool = False
    
    # Playback mode
    playback: bool = False
    model_path: Optional[str] = None
    num_episodes: int = 10  # Number of episodes to run in playback mode
    deterministic: bool = True  # Use mean action (no sampling) in playback


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
        print(f"TensorBoard log directory: {self.log_dir}")
        self.writer = SummaryWriter(self.log_dir)
        
        # Training state
        self.global_step = 0
        self.update = 0
        self.episode_metrics = EpisodeMetricsLogger()
        
        # Track last seen lap counts per environment to detect new laps
        self.last_seen_lap_counts = [0] * config.num_envs
        
        # Setup visualization
        self.visualize = config.visualize
        self.visualizer = VisualizationWindow(
            algorithm_name="PPO",
            port=config.start_port
        ) if config.visualize else None
    
    def load_model(self, model_path: str):
        """Load a saved model checkpoint"""
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["model_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.update = checkpoint.get("update", 0)
        print(f"Model loaded successfully (update: {self.update}, global_step: {self.global_step})")
        
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
            
            # Visualization (show first environment)
            if self.visualize and self.visualizer is not None:
                # Get observation from first env (convert from tensor if needed)
                vis_obs = next_obs[0] if self.config.num_envs > 1 else next_obs
                if hasattr(vis_obs, 'cpu'):
                    vis_obs = vis_obs.cpu().numpy()
                vis_action = action[0] if self.config.num_envs > 1 else action
                vis_clipped_action = action_np[0] if self.config.num_envs > 1 else action_np
                vis_reward = reward[0] if isinstance(reward, (np.ndarray, list)) else reward
                if not self.visualizer.update(vis_obs, vis_action, vis_clipped_action, vis_reward):
                    # User closed window
                    self.visualize = False
            
            # Store in buffer
            self.buffer.add(
                next_obs,
                action,
                logprob,
                torch.tensor(reward, dtype=torch.float32).to(self.device),
                next_done,
                value,
            )
            
            # Update for next iteration - vectorized envs auto-reset, so next_obs_np contains reset obs for done envs
            next_obs = prepare_observation(next_obs_np, self.device)
            next_done = torch.tensor(done, dtype=torch.float32).to(self.device)
            
            # Check for new lap times and log immediately to TensorBoard
            for idx in range(self.config.num_envs):
                lap_metrics = extract_lap_time_metrics(info, idx)
                if lap_metrics:
                    # Check if lap_count increased (new lap completed)
                    current_lap_count = lap_metrics.get("lap_count", 0)
                    if current_lap_count > self.last_seen_lap_counts[idx]:
                        # New lap completed, log the lap time
                        if "lap_time" in lap_metrics and lap_metrics["lap_time"] > 0.0:
                            self.writer.add_scalar("laps/lap_time", lap_metrics["lap_time"], self.global_step)
                            self.writer.flush()
                        self.last_seen_lap_counts[idx] = current_lap_count
            
            # Log episode metrics and reset lap count tracking on episode end
            for idx, d in enumerate(done):
                if d:
                    metrics = extract_episode_metrics(info, idx, d)
                    if metrics:
                        self.episode_metrics.add_metrics(metrics)
                    # Reset lap count tracking when episode ends
                    self.last_seen_lap_counts[idx] = 0
        
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
            
            # Flush to ensure data is written to disk
            self.writer.flush()
            
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
        if self.visualizer is not None:
            self.visualizer.close()
        self.writer.flush()
        self.writer.close()
        
        total_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {total_time:.2f}s")
        print(f"Average SPS: {int(self.config.total_timesteps / total_time)}")
    
    def playback(self):
        """Run playback of a trained model"""
        print(f"\n{'='*60}")
        print(f"Starting PPO playback mode")
        print(f"{'='*60}")
        print(f"Model: {self.config.model_path}")
        print(f"Num episodes: {self.config.num_episodes}")
        print(f"Deterministic: {self.config.deterministic}")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Load the model
        if self.config.model_path:
            self.load_model(self.config.model_path)
        else:
            raise ValueError("model_path must be specified for playback mode")
        
        # Set agent to evaluation mode
        self.agent.eval()
        
        # Episode tracking
        episode_count = 0
        episode_metrics = EpisodeMetricsLogger()
        
        # Reset environment
        reset_result = self.envs.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result
        obs = prepare_observation(obs, self.device)
        
        step_count = 0
        start_time = time.time()
        
        # Run episodes
        while episode_count < self.config.num_episodes:
            step_count += self.config.num_envs
            
            # Get action from policy
            with torch.no_grad():
                if self.config.deterministic:
                    # Use mean action (no sampling)
                    features = self.agent.feature_extractor(obs)
                    action = self.agent.actor_mean(features)
                else:
                    # Sample from distribution
                    action, _, _, _ = self.agent.get_action_and_value(obs)
            
            # Convert and clip actions
            action_np = clip_action_to_space(action, self.envs.single_action_space)
            
            # Step environment
            next_obs_np, reward, terminated, truncated, info = self.envs.step(action_np)
            done = np.logical_or(terminated, truncated)
            
            # Visualization (show first environment)
            if self.visualize and self.visualizer is not None:
                vis_obs = obs[0] if self.config.num_envs > 1 else obs
                if hasattr(vis_obs, 'cpu'):
                    vis_obs = vis_obs.cpu().numpy()
                vis_action = action[0] if self.config.num_envs > 1 else action
                vis_clipped_action = action_np[0] if self.config.num_envs > 1 else action_np
                vis_reward = reward[0] if isinstance(reward, (np.ndarray, list)) else reward
                if not self.visualizer.update(vis_obs, vis_action, vis_clipped_action, vis_reward):
                    # User closed window
                    print("\nVisualization window closed. Stopping playback.")
                    break
            
            # Update observation - vectorized envs auto-reset, so next_obs_np contains reset obs for done envs
            obs = prepare_observation(next_obs_np, self.device)
            
            # Log episode metrics after updating observation
            for idx, d in enumerate(done):
                if d:
                    metrics = extract_episode_metrics(info, idx, d)
                    if metrics:
                        episode_metrics.add_metrics(metrics)
                        episode_count += 1
                        
                        # Print detailed episode summary
                        print(f"\n{'='*60}")
                        print(f"Episode {episode_count}/{self.config.num_episodes} Complete")
                        print(f"{'='*60}")
                        
                        # Core metrics
                        if 'reward' in metrics:
                            print(f"  Return: {metrics['reward']:.2f}")
                        if 'length' in metrics:
                            print(f"  Length: {metrics['length']} steps")
                        
                        # Lap metrics
                        if 'lap_count' in metrics:
                            print(f"  Laps Completed: {metrics['lap_count']}")
                        if 'lap_time' in metrics:
                            print(f"  Last Lap Time: {metrics['lap_time']:.2f}s")
                        
                        # Driving performance
                        if 'cte' in metrics:
                            print(f"  Avg Cross-Track Error: {metrics['cte']:.3f}")
                        if 'speed' in metrics:
                            print(f"  Avg Speed: {metrics['speed']:.2f}")
                        if 'forward_vel' in metrics:
                            print(f"  Avg Forward Velocity: {metrics['forward_vel']:.2f}")
                        if 'hit' in metrics:
                            collision_status = "Yes" if metrics['hit'] > 0 else "No"
                            print(f"  Collision: {collision_status}")
                        
                        print(f"{'='*60}\n")
                        
                        # Check if we've completed all episodes
                        if episode_count >= self.config.num_episodes:
                            break
            
            # Break if all requested episodes are done
            if episode_count >= self.config.num_episodes:
                break
        
        # Print final summary
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"PLAYBACK COMPLETE - AGGREGATE STATISTICS")
        print(f"{'='*60}")
        print(f"Total Episodes: {episode_count}")
        print(f"Total Steps: {step_count}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Average SPS: {int(step_count / total_time) if total_time > 0 else 0}")
        print(f"{'='*60}")
        
        # Print detailed aggregate statistics
        metrics_dict = episode_metrics.get_metrics_dict()
        
        # Episode Performance
        if len(metrics_dict['episode_rewards']) > 0:
            rewards = np.array(metrics_dict['episode_rewards'])
            lengths = np.array(metrics_dict['episode_lengths'])
            print(f"\nEpisode Performance:")
            print(f"  Return:  Mean={np.mean(rewards):.2f}, Std={np.std(rewards):.2f}, "
                  f"Min={np.min(rewards):.2f}, Max={np.max(rewards):.2f}")
            print(f"  Length:  Mean={np.mean(lengths):.1f}, Std={np.std(lengths):.1f}, "
                  f"Min={int(np.min(lengths))}, Max={int(np.max(lengths))}")
        
        # Lap Performance
        if len(metrics_dict['episode_lap_counts']) > 0:
            lap_counts = np.array(metrics_dict['episode_lap_counts'])
            total_laps = int(np.sum(lap_counts))
            print(f"\nLap Performance:")
            print(f"  Total Laps Completed: {total_laps}")
            print(f"  Avg Laps per Episode: {np.mean(lap_counts):.2f}")
            print(f"  Episodes with Laps: {np.count_nonzero(lap_counts)}/{len(lap_counts)}")
            
            if len(metrics_dict['episode_lap_times']) > 0:
                lap_times = np.array(metrics_dict['episode_lap_times'])
                print(f"  Best Lap Time: {np.min(lap_times):.2f}s")
                print(f"  Mean Lap Time: {np.mean(lap_times):.2f}s ± {np.std(lap_times):.2f}s")
                print(f"  Worst Lap Time: {np.max(lap_times):.2f}s")
        
        # Driving Performance
        if len(metrics_dict['episode_cte']) > 0:
            cte = np.array(metrics_dict['episode_cte'])
            speeds = np.array(metrics_dict['episode_speed'])
            print(f"\nDriving Performance:")
            print(f"  Cross-Track Error: Mean={np.mean(cte):.3f}, Std={np.std(cte):.3f}, "
                  f"Min={np.min(cte):.3f}, Max={np.max(cte):.3f}")
            print(f"  Speed: Mean={np.mean(speeds):.2f}, Std={np.std(speeds):.2f}, "
                  f"Min={np.min(speeds):.2f}, Max={np.max(speeds):.2f}")
            
            if len(metrics_dict['episode_forward_vel']) > 0:
                fwd_vel = np.array(metrics_dict['episode_forward_vel'])
                print(f"  Forward Velocity: Mean={np.mean(fwd_vel):.2f}, Std={np.std(fwd_vel):.2f}")
        
        # Collision Statistics
        if len(metrics_dict['episode_hits']) > 0:
            hits = np.array(metrics_dict['episode_hits'])
            collision_rate = np.mean(hits) * 100
            num_collisions = int(np.sum(hits))
            print(f"\nCollision Statistics:")
            print(f"  Collision Rate: {collision_rate:.1f}%")
            print(f"  Episodes with Collisions: {num_collisions}/{len(hits)}")
        
        print(f"{'='*60}\n")
        
        # Cleanup
        self.envs.close()
        if self.visualizer is not None:
            self.visualizer.close()


def main():
    parser = argparse.ArgumentParser(description="PPO training/playback with PufferLib")
    
    # Mode selection
    parser.add_argument("--playback", action="store_true", help="run playback mode instead of training")
    parser.add_argument("--model-path", type=str, help="path to saved model checkpoint (required for playback)")
    parser.add_argument("--num-episodes", type=int, default=10, help="number of episodes to run in playback mode")
    parser.add_argument("--deterministic", action="store_true", default=True, help="use deterministic policy (mean action) in playback")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false", help="use stochastic policy (sample actions) in playback")
    
    # Environment
    parser.add_argument("--env-name", type=str, default="donkey-circuit-launch-track-v0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--start-port", type=int, default=9091)
    parser.add_argument("--backend", type=str, default="serial", choices=["serial", "multiprocessing", "ray"])
    
    # Training
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-steps", type=int, default=2048)
    
    # Misc
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--visualize", action="store_true", help="enable pygame visualization window")
    args = parser.parse_args()
    
    # In playback mode, default to 1 environment if not explicitly specified
    num_envs = args.num_envs
    if args.playback and args.num_envs == 4:  # 4 is the default
        # Check if user explicitly set num_envs
        import sys
        if '--num-envs' not in sys.argv:
            num_envs = 1
            print("Playback mode: defaulting to 1 environment (use --num-envs to override)")
    
    # Create config
    config = PPOConfig(
        env_name=args.env_name,
        num_envs=num_envs,
        start_port=args.start_port,
        backend=args.backend,
        total_timesteps=args.total_timesteps,
        learning_rate=args.learning_rate,
        num_steps=args.num_steps,
        device=args.device,
        seed=args.seed,
        visualize=args.visualize,
        playback=args.playback,
        model_path=args.model_path,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
    )
    
    # Create trainer
    trainer = PPOTrainer(config)
    
    # Run playback or training
    if config.playback:
        trainer.playback()
    else:
        trainer.train()


if __name__ == "__main__":
    main()

