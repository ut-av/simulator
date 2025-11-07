#!/usr/bin/env python3
"""
Shared utilities for RL algorithms
Contains common components for PPO, SAC, and other algorithms
"""

import os
import random
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """Initialize network layer with orthogonal initialization"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def set_random_seeds(seed: int, torch_deterministic: bool = True):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic


def setup_logging_dirs(log_dir: str, model_dir: str) -> Tuple[str, str]:
    """Create logging directories with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"{log_dir}/{timestamp}"
    model_path = f"{model_dir}/{timestamp}"
    os.makedirs(model_path, exist_ok=True)
    return log_path, model_path


class CNNFeatureExtractor(nn.Module):
    """
    Shared CNN feature extractor for visual observations
    Used by both PPO and SAC for DonkeyEnv's camera input
    """
    def __init__(self, observation_space):
        super().__init__()
        
        # Determine input shape
        # DonkeyEnv provides HxWxC images
        obs_shape = observation_space.shape
        if len(obs_shape) == 3:
            # Assume HxWxC format, convert to CxHxW for PyTorch
            self.input_channels = obs_shape[2]
            self.height = obs_shape[0]
            self.width = obs_shape[1]
        else:
            raise ValueError(f"Unexpected observation shape: {obs_shape}")
        
        # CNN layers
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
            dummy_input = torch.zeros(1, self.input_channels, self.height, self.width)
            self.feature_dim = self.cnn(dummy_input).shape[1]
    
    def forward(self, x):
        """
        Extract features from observations
        Args:
            x: Observations in HxWxC or BxHxWxC format
        Returns:
            Features: Flattened feature vector(s)
        """
        # Convert HxWxC to CxHxW and normalize to [0, 1]
        if len(x.shape) == 4:  # Batch of images
            x = x.permute(0, 3, 1, 2)  # BxHxWxC -> BxCxHxW
        else:
            x = x.permute(2, 0, 1)  # HxWxC -> CxHxW
        x = x.float() / 255.0
        return self.cnn(x)


def prepare_observation(obs, device):
    """Convert observation to tensor on device"""
    if not isinstance(obs, np.ndarray):
        obs = np.array(obs)
    return torch.from_numpy(obs).float().to(device)


def clip_action_to_space(action, action_space):
    """Clip actions to action space bounds"""
    action_np = action.cpu().numpy() if torch.is_tensor(action) else action
    
    # Ensure dtype matches action space
    if action_np.dtype != action_space.dtype:
        action_np = action_np.astype(action_space.dtype)
    
    # Clip to bounds
    action_np = np.clip(
        action_np,
        action_space.low,
        action_space.high
    )
    
    return action_np


def extract_episode_metrics(info, idx, done):
    """
    Extract episode metrics from info dict
    Returns dict with episode metrics (or None if episode not done)
    """
    metrics = {}
    
    if not done:
        return None
    
    if "episode" in info:
        ep_info = info["episode"][idx]
        metrics["reward"] = ep_info["r"]
        metrics["length"] = ep_info["l"]
    
    if isinstance(info, dict):
        # Cross track error
        if "cte" in info:
            cte_val = info["cte"][idx] if hasattr(info["cte"], "__getitem__") else info["cte"]
            metrics["cte"] = abs(cte_val)
        
        # Speed metrics
        if "speed" in info:
            speed_val = info["speed"][idx] if hasattr(info["speed"], "__getitem__") else info["speed"]
            metrics["speed"] = speed_val
        
        if "forward_vel" in info:
            fwd_vel = info["forward_vel"][idx] if hasattr(info["forward_vel"], "__getitem__") else info["forward_vel"]
            metrics["forward_vel"] = fwd_vel
        
        # Collision detection
        if "hit" in info:
            hit_val = info["hit"][idx] if hasattr(info["hit"], "__getitem__") else info["hit"]
            metrics["hit"] = 1.0 if hit_val != "none" else 0.0
        
        # Lap metrics
        if "last_lap_time" in info:
            lap_time = info["last_lap_time"][idx] if hasattr(info["last_lap_time"], "__getitem__") else info["last_lap_time"]
            if lap_time > 0.0:
                metrics["lap_time"] = lap_time
        
        if "lap_count" in info:
            lap_count = info["lap_count"][idx] if hasattr(info["lap_count"], "__getitem__") else info["lap_count"]
            metrics["lap_count"] = lap_count
    
    return metrics if metrics else None


class EpisodeMetricsLogger:
    """Helper class to accumulate and log episode metrics"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_cte = []
        self.episode_speed = []
        self.episode_forward_vel = []
        self.episode_hits = []
        self.episode_lap_times = []
        self.episode_lap_counts = []
    
    def add_metrics(self, metrics):
        """Add metrics from a single episode"""
        if metrics is None:
            return
        
        if "reward" in metrics:
            self.episode_rewards.append(metrics["reward"])
        if "length" in metrics:
            self.episode_lengths.append(metrics["length"])
        if "cte" in metrics:
            self.episode_cte.append(metrics["cte"])
        if "speed" in metrics:
            self.episode_speed.append(metrics["speed"])
        if "forward_vel" in metrics:
            self.episode_forward_vel.append(metrics["forward_vel"])
        if "hit" in metrics:
            self.episode_hits.append(metrics["hit"])
        if "lap_time" in metrics:
            self.episode_lap_times.append(metrics["lap_time"])
        if "lap_count" in metrics:
            self.episode_lap_counts.append(metrics["lap_count"])
    
    def get_metrics_dict(self):
        """Get all metrics as dictionary"""
        return {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_cte": self.episode_cte,
            "episode_speed": self.episode_speed,
            "episode_forward_vel": self.episode_forward_vel,
            "episode_hits": self.episode_hits,
            "episode_lap_times": self.episode_lap_times,
            "episode_lap_counts": self.episode_lap_counts,
        }
    
    def log_to_tensorboard(self, writer, global_step):
        """Log all accumulated metrics to TensorBoard"""
        # Episode Performance
        if len(self.episode_rewards) > 0:
            rewards_arr = np.array(self.episode_rewards)
            lengths_arr = np.array(self.episode_lengths)
            writer.add_scalar("charts/episodic_return", np.mean(rewards_arr), global_step)
            writer.add_scalar("charts/episodic_length", np.mean(lengths_arr), global_step)
        
        # Driving Performance
        if len(self.episode_cte) > 0:
            cte_arr = np.array(self.episode_cte)
            writer.add_scalar("driving/cross_track_error", np.mean(cte_arr), global_step)
            writer.add_scalar("driving/cte_std", np.std(cte_arr), global_step)
        
        if len(self.episode_speed) > 0:
            speed_arr = np.array(self.episode_speed)
            writer.add_scalar("driving/speed", np.mean(speed_arr), global_step)
        
        if len(self.episode_forward_vel) > 0:
            fwd_vel_arr = np.array(self.episode_forward_vel)
            writer.add_scalar("driving/forward_velocity", np.mean(fwd_vel_arr), global_step)
        
        if len(self.episode_hits) > 0:
            hits_arr = np.array(self.episode_hits)
            writer.add_scalar("driving/collision_rate", np.mean(hits_arr), global_step)
        
        # Lap Performance
        if len(self.episode_lap_times) > 0:
            lap_times_arr = np.array(self.episode_lap_times)
            writer.add_scalar("laps/lap_time_mean", np.mean(lap_times_arr), global_step)
            writer.add_scalar("laps/lap_time_min", np.min(lap_times_arr), global_step)
            writer.add_scalar("laps/lap_time_std", np.std(lap_times_arr), global_step)
        
        if len(self.episode_lap_counts) > 0:
            lap_counts_arr = np.array(self.episode_lap_counts)
            writer.add_scalar("laps/completed_laps", np.sum(lap_counts_arr), global_step)
    
    def print_summary(self):
        """Print summary of metrics"""
        lines = []
        
        if len(self.episode_rewards) > 0:
            rewards_arr = np.array(self.episode_rewards)
            lengths_arr = np.array(self.episode_lengths)
            lines.append(f"  Reward: {np.mean(rewards_arr):.2f} ± {np.std(rewards_arr):.2f}")
            lines.append(f"  Length: {np.mean(lengths_arr):.1f}")
        
        if len(self.episode_cte) > 0:
            cte_arr = np.array(self.episode_cte)
            speed_arr = np.array(self.episode_speed)
            hits_arr = np.array(self.episode_hits)
            lines.append(f"  CTE: {np.mean(cte_arr):.3f} | Speed: {np.mean(speed_arr):.2f} | Collisions: {np.mean(hits_arr):.2%}")
        
        if len(self.episode_lap_times) > 0:
            lap_times_arr = np.array(self.episode_lap_times)
            lines.append(f"  Best Lap: {np.min(lap_times_arr):.2f}s | Avg Lap: {np.mean(lap_times_arr):.2f}s")
        
        return "\n".join(lines)

