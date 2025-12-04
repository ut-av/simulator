#!/usr/bin/env python3
"""
Shared utilities for RL algorithms
Contains common components for PPO, SAC, and other algorithms
"""

import os
import random
from datetime import datetime
from typing import Tuple, Optional

import numpy as np
import pygame
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
    os.makedirs(log_path, exist_ok=True)
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


def extract_lap_time_metrics(info, idx):
    """
    Extract lap time metrics from info dict (can be called even when episode not done)
    Returns dict with lap time metrics (or None if no new lap time)
    """
    metrics = {}
    
    # Handle list of dicts (PufferLib)
    if isinstance(info, list):
        if idx < len(info):
            env_info = info[idx]
            if "last_lap_time" in env_info and env_info["last_lap_time"] > 0.0:
                metrics["lap_time"] = env_info["last_lap_time"]
            if "lap_count" in env_info:
                metrics["lap_count"] = env_info["lap_count"]
        return metrics if metrics else None
    
    if isinstance(info, dict):
        # Lap metrics - check for new lap time
        if "last_lap_time" in info:
            lap_time = info["last_lap_time"][idx] if hasattr(info["last_lap_time"], "__getitem__") else info["last_lap_time"]
            if lap_time > 0.0:
                metrics["lap_time"] = lap_time
        
        if "lap_count" in info:
            lap_count = info["lap_count"][idx] if hasattr(info["lap_count"], "__getitem__") else info["lap_count"]
            metrics["lap_count"] = lap_count
    
    return metrics if metrics else None


def extract_episode_metrics(info, idx, done):
    """
    Extract episode metrics from info dict
    Returns dict with episode metrics (or None if episode not done)
    """
    metrics = {}
    
    if not done:
        return None
    
    # Handle list of dicts (PufferLib)
    if isinstance(info, list):
        if idx < len(info):
            env_info = info[idx]
            
            if "episode" in env_info:
                metrics["reward"] = env_info["episode"]["r"]
                metrics["length"] = env_info["episode"]["l"]
            
            # Cross track error
            if "cte" in env_info:
                metrics["cte"] = abs(env_info["cte"])
            
            # Speed metrics
            if "speed" in env_info:
                metrics["speed"] = env_info["speed"]
            
            if "forward_vel" in env_info:
                metrics["forward_vel"] = env_info["forward_vel"]
            
            # Collision detection
            if "hit" in env_info:
                metrics["hit"] = 1.0 if env_info["hit"] != "none" else 0.0
            
            # Termination info
            if "termination_reason" in env_info:
                metrics["termination_reason"] = env_info["termination_reason"]
            if "off_track" in env_info:
                metrics["off_track"] = env_info["off_track"]
            if "collision" in env_info:
                metrics["collision"] = env_info["collision"]
            if "car_fully_crossed" in env_info:
                metrics["car_fully_crossed"] = env_info["car_fully_crossed"]
            
            # Reward components
            if "reward_components" in env_info:
                metrics["reward_components"] = env_info["reward_components"]
            
            # Lap metrics
            if "last_lap_time" in env_info and env_info["last_lap_time"] > 0.0:
                metrics["lap_time"] = env_info["last_lap_time"]
            
            if "lap_count" in env_info:
                metrics["lap_count"] = env_info["lap_count"]
            
            # Distance traveled
            if "total_distance" in env_info:
                metrics["total_distance"] = env_info["total_distance"]
                
        return metrics if metrics else None
    
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
        
        # Termination info
        if "termination_reason" in info:
            term_val = info["termination_reason"][idx] if hasattr(info["termination_reason"], "__getitem__") else info["termination_reason"]
            metrics["termination_reason"] = term_val
        if "off_track" in info:
            off_val = info["off_track"][idx] if hasattr(info["off_track"], "__getitem__") else info["off_track"]
            metrics["off_track"] = off_val
        if "collision" in info:
            coll_val = info["collision"][idx] if hasattr(info["collision"], "__getitem__") else info["collision"]
            metrics["collision"] = coll_val
        if "car_fully_crossed" in info:
            cross_val = info["car_fully_crossed"][idx] if hasattr(info["car_fully_crossed"], "__getitem__") else info["car_fully_crossed"]
            metrics["car_fully_crossed"] = cross_val
        
        # Reward components
        if "reward_components" in info:
            reward_comp = info["reward_components"][idx] if hasattr(info["reward_components"], "__getitem__") else info["reward_components"]
            metrics["reward_components"] = reward_comp
        
        # Lap metrics
        if "last_lap_time" in info:
            lap_time = info["last_lap_time"][idx] if hasattr(info["last_lap_time"], "__getitem__") else info["last_lap_time"]
            if lap_time > 0.0:
                metrics["lap_time"] = lap_time
        
        if "lap_count" in info:
            lap_count = info["lap_count"][idx] if hasattr(info["lap_count"], "__getitem__") else info["lap_count"]
            metrics["lap_count"] = lap_count
        
        # Distance traveled
        if "total_distance" in info:
            dist_val = info["total_distance"][idx] if hasattr(info["total_distance"], "__getitem__") else info["total_distance"]
            metrics["total_distance"] = dist_val
    
    return metrics if metrics else None


def extract_termination_and_reward_info(info, idx):
    """
    Extract termination reason and reward components from info dict.
    
    Returns dict with:
    - termination_reason: string explaining why episode ended
    - off_track: boolean
    - collision: boolean
    - car_fully_crossed: boolean
    - reward_components: dict with breakdown of reward calculation
    """
    term_info = {
        "termination_reason": "none",
        "off_track": False,
        "collision": False,
        "car_fully_crossed": False,
        "reward_components": {},
    }
    
    # Handle list of dicts (PufferLib vectorized env)
    if isinstance(info, list):
        if idx < len(info):
            env_info = info[idx]
            
            if "termination_reason" in env_info:
                term_info["termination_reason"] = env_info["termination_reason"]
            if "off_track" in env_info:
                term_info["off_track"] = env_info["off_track"]
            if "collision" in env_info:
                term_info["collision"] = env_info["collision"]
            if "car_fully_crossed" in env_info:
                term_info["car_fully_crossed"] = env_info["car_fully_crossed"]
            if "reward_components" in env_info:
                term_info["reward_components"] = env_info["reward_components"]
    
    # Handle dict (single env)
    elif isinstance(info, dict):
        if "termination_reason" in info:
            term_info["termination_reason"] = info["termination_reason"]
        if "off_track" in info:
            term_info["off_track"] = info["off_track"]
        if "collision" in info:
            term_info["collision"] = info["collision"]
        if "car_fully_crossed" in info:
            term_info["car_fully_crossed"] = info["car_fully_crossed"]
        if "reward_components" in info:
            term_info["reward_components"] = info["reward_components"]
    
    return term_info


def extract_step_data(info, idx):
    """
    Extract current step data (both termination and reward info).
    
    Returns dict with all available diagnostic data for logging.
    """
    step_data = {
        "cte": None,
        "max_cte": None,
        "speed": None,
        "forward_vel": None,
        "hit": "none",
        "termination_reason": "none",
        "off_track": False,
        "collision": False,
        "car_fully_crossed": False,
        "missed_checkpoint": False,
        "dq": False,
        "reward_components": {},
        "raw_action": None,
        "smoothed_action": None,
    }
    
    # Handle list of dicts (PufferLib)
    if isinstance(info, list):
        if idx < len(info):
            env_info = info[idx]
            if "cte" in env_info:
                step_data["cte"] = env_info["cte"]
            if "max_cte" in env_info:
                step_data["max_cte"] = env_info["max_cte"]
            if "speed" in env_info:
                step_data["speed"] = env_info["speed"]
            if "forward_vel" in env_info:
                step_data["forward_vel"] = env_info["forward_vel"]
            if "hit" in env_info:
                step_data["hit"] = env_info["hit"]
            if "termination_reason" in env_info:
                step_data["termination_reason"] = env_info["termination_reason"]
            if "off_track" in env_info:
                step_data["off_track"] = env_info["off_track"]
            if "collision" in env_info:
                step_data["collision"] = env_info["collision"]
            if "car_fully_crossed" in env_info:
                step_data["car_fully_crossed"] = env_info["car_fully_crossed"]
            if "missed_checkpoint" in env_info:
                step_data["missed_checkpoint"] = env_info["missed_checkpoint"]
            if "dq" in env_info:
                step_data["dq"] = env_info["dq"]
            if "reward_components" in env_info:
                step_data["reward_components"] = env_info["reward_components"]
            if "raw_action" in env_info:
                step_data["raw_action"] = env_info["raw_action"]
            if "smoothed_action" in env_info:
                step_data["smoothed_action"] = env_info["smoothed_action"]
    
    # Handle dict (single env)
    elif isinstance(info, dict):
        if "cte" in info:
            step_data["cte"] = info["cte"]
        if "max_cte" in info:
            step_data["max_cte"] = info["max_cte"]
        if "speed" in info:
            step_data["speed"] = info["speed"]
        if "forward_vel" in info:
            step_data["forward_vel"] = info["forward_vel"]
        if "hit" in info:
            step_data["hit"] = info["hit"]
        if "termination_reason" in info:
            step_data["termination_reason"] = info["termination_reason"]
        if "off_track" in info:
            step_data["off_track"] = info["off_track"]
        if "collision" in info:
            step_data["collision"] = info["collision"]
        if "car_fully_crossed" in info:
            step_data["car_fully_crossed"] = info["car_fully_crossed"]
        if "missed_checkpoint" in info:
            step_data["missed_checkpoint"] = info["missed_checkpoint"]
        if "dq" in info:
            step_data["dq"] = info["dq"]
        if "reward_components" in info:
            step_data["reward_components"] = info["reward_components"]
        if "raw_action" in info:
            step_data["raw_action"] = info["raw_action"]
        if "smoothed_action" in info:
            step_data["smoothed_action"] = info["smoothed_action"]
    
    return step_data


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
        self.episode_distances = []
        
        # Termination tracking
        self.episode_termination_reasons = []  # List of termination reason strings
        self.termination_counts = {
            "off_track": 0,
            "collision": 0,
            "car_fully_crossed": 0,
            "missed_checkpoint": 0,
            "disqualified": 0,
            "other": 0,
        }
        
        # Reward component tracking
        self.episode_reward_components = []  # List of component dicts per episode
    
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
        if "termination_reason" in metrics:
            term_reason = metrics["termination_reason"]
            self.episode_termination_reasons.append(term_reason)
            if term_reason in self.termination_counts:
                self.termination_counts[term_reason] += 1
            else:
                self.termination_counts["other"] += 1
        if "reward_components" in metrics:
            self.episode_reward_components.append(metrics["reward_components"])
        if "lap_time" in metrics:
            self.episode_lap_times.append(metrics["lap_time"])
        if "lap_count" in metrics:
            self.episode_lap_counts.append(metrics["lap_count"])
        if "total_distance" in metrics:
            self.episode_distances.append(metrics["total_distance"])
    
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
            "episode_distances": self.episode_distances,
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
        
        if len(self.episode_distances) > 0:
            dist_arr = np.array(self.episode_distances)
            writer.add_scalar("driving/distance", np.mean(dist_arr), global_step)
        
        # Lap Performance
        if len(self.episode_lap_times) > 0:
            lap_times_arr = np.array(self.episode_lap_times)
            writer.add_scalar("laps/lap_time_mean", np.mean(lap_times_arr), global_step)
            writer.add_scalar("laps/lap_time_min", np.min(lap_times_arr), global_step)
            writer.add_scalar("laps/lap_time_std", np.std(lap_times_arr), global_step)
        
        if len(self.episode_lap_counts) > 0:
            lap_counts_arr = np.array(self.episode_lap_counts)
            writer.add_scalar("laps/completed_laps", np.sum(lap_counts_arr), global_step)
        
        # Termination reasons
        total_episodes = sum(self.termination_counts.values())
        if total_episodes > 0:
            for reason, count in self.termination_counts.items():
                if count > 0:
                    writer.add_scalar(f"termination/{reason}", count, global_step)
        
        # Reward components (average across episodes)
        if len(self.episode_reward_components) > 0:
            # Collect all components
            done_penalties = []
            off_track_penalties = []
            collision_penalties = []
            
            # Raw values
            speed_rewards = []
            centering_bonuses = []
            distance_rewards = []
            
            # Weights
            speed_weights = []
            centering_weights = []
            distance_weights = []
            
            for components in self.episode_reward_components:
                if "done_penalty" in components:
                    done_penalties.append(components["done_penalty"])
                if "off_track_penalty" in components:
                    off_track_penalties.append(components["off_track_penalty"])
                if "collision_penalty" in components:
                    collision_penalties.append(components["collision_penalty"])
                if "speed_reward" in components:
                    speed_rewards.append(components["speed_reward"])
                    if "speed_weight" in components:
                        speed_weights.append(components["speed_weight"])
                if "centering_bonus" in components:
                    centering_bonuses.append(components["centering_bonus"])
                    if "centering_weight" in components:
                        centering_weights.append(components["centering_weight"])
                if "distance_reward" in components:
                    distance_rewards.append(components["distance_reward"])
                    if "distance_weight" in components:
                        distance_weights.append(components["distance_weight"])
            
            # Log averages
            if done_penalties:
                writer.add_scalar("reward_components/done_penalty_mean", np.mean(done_penalties), global_step)
            if off_track_penalties:
                writer.add_scalar("reward_components/off_track_penalty_mean", np.mean(off_track_penalties), global_step)
            if collision_penalties:
                writer.add_scalar("reward_components/collision_penalty_mean", np.mean(collision_penalties), global_step)
            if speed_rewards:
                raw_mean = np.mean(speed_rewards)
                writer.add_scalar("reward_components/raw_speed_reward_mean", raw_mean, global_step)
                if speed_weights:
                    weight = np.mean(speed_weights)
                    writer.add_scalar("reward_components/weight_speed_reward", weight, global_step)
                    writer.add_scalar("reward_components/weighted_speed_reward_mean", raw_mean * weight, global_step)
            
            if centering_bonuses:
                raw_mean = np.mean(centering_bonuses)
                writer.add_scalar("reward_components/raw_centering_bonus_mean", raw_mean, global_step)
                if centering_weights:
                    weight = np.mean(centering_weights)
                    writer.add_scalar("reward_components/weight_centering_bonus", weight, global_step)
                    writer.add_scalar("reward_components/weighted_centering_bonus_mean", raw_mean * weight, global_step)
            
            if distance_rewards:
                raw_mean = np.mean(distance_rewards)
                writer.add_scalar("reward_components/raw_distance_reward_mean", raw_mean, global_step)
                if distance_weights:
                    weight = np.mean(distance_weights)
                    writer.add_scalar("reward_components/weight_distance_reward", weight, global_step)
                    writer.add_scalar("reward_components/weighted_distance_reward_mean", raw_mean * weight, global_step)
    
    def print_summary(self):
        """Print summary of metrics"""
        lines = []
        
        if len(self.episode_rewards) > 0:
            rewards_arr = np.array(self.episode_rewards)
            lengths_arr = np.array(self.episode_lengths)
            lines.append(f"  Reward: {np.mean(rewards_arr):.2f} ± {np.std(rewards_arr):.2f}")
            lines.append(f"  Length: {np.mean(lengths_arr):.1f}")
        
            cte_str = f"CTE: {np.mean(self.episode_cte):.3f}" if len(self.episode_cte) > 0 else "CTE: N/A"
            speed_str = f"Speed: {np.mean(self.episode_speed):.2f}" if len(self.episode_speed) > 0 else "Speed: N/A"
            hits_str = f"Collisions: {np.mean(self.episode_hits):.2%}" if len(self.episode_hits) > 0 else "Collisions: N/A"
            
            dist_str = ""
            if len(self.episode_distances) > 0:
                dist_arr = np.array(self.episode_distances)
                dist_str = f" | Dist: {np.mean(dist_arr):.1f}m"
            
            lines.append(f"  {cte_str} | {speed_str} | {hits_str}{dist_str}")
        
        # Termination summary
        total_episodes = sum(self.termination_counts.values())
        if total_episodes > 0:
            reasons_str = " | ".join([f"{reason}: {count}" for reason, count in self.termination_counts.items() if count > 0])
            lines.append(f"  Termination: {reasons_str}")
        
        if len(self.episode_lap_times) > 0:
            lap_times_arr = np.array(self.episode_lap_times)
            lines.append(f"  Best Lap: {np.min(lap_times_arr):.2f}s | Avg Lap: {np.mean(lap_times_arr):.2f}s")
        
        return "\n".join(lines)


def process_observation_image(obs):
    """
    Process observation image for visualization
    Converts various formats to HxWxC uint8 numpy array
    """
    # Process observation image
    try:
        arr = np.array(obs)
    except Exception:
        arr = obs
    
    # Handle batch dimension - take first environment
    if hasattr(arr, 'ndim') and arr.ndim == 4:
        arr = arr[0]
    
    # Convert channels-first (C, H, W) to channels-last (H, W, C)
    if hasattr(arr, 'ndim') and arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
        try:
            arr = arr.transpose(1, 2, 0)
        except Exception:
            pass
    
    # Handle grayscale 2D image
    if hasattr(arr, 'ndim') and arr.ndim == 2:
        try:
            arr = np.stack([arr, arr, arr], axis=-1)
        except Exception:
            pass
    
    # Ensure uint8 dtype
    if hasattr(arr, 'dtype'):
        try:
            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)
        except Exception:
            pass
    
    # Transpose (H, W, C) to (W, H, C) for Pygame
    # Pygame expects the first dimension to be width (x) and second to be height (y)
    try:
        arr = arr.transpose(1, 0, 2)
    except Exception:
        pass
    
    return arr


def extract_action_value(action):
    """Extract action value from tensor/array, handling batch dimensions"""
    if hasattr(action, 'cpu'):
        action = action.cpu().numpy()
    if hasattr(action, 'ndim') and action.ndim > 1:
        action = action[0]
    return action


def extract_reward_value(reward):
    """Extract reward value from tensor/array"""
    if hasattr(reward, 'item'):
        return reward.item()
    elif hasattr(reward, '__len__') and len(reward) > 0:
        return float(reward[0])
    else:
        return float(reward)


class VisualizationWindow:
    """Pygame visualization window for training"""
    def __init__(self, algorithm_name="RL", port=None):
        pygame.init()
        self.font = pygame.font.SysFont(None, 20)
        self.small_font = pygame.font.SysFont(None, 16)
        self.initialized = False
        self.reward_history = []
        self.max_history = 200
        self.algorithm_name = algorithm_name
        self.port = port
        
        # UI dimensions
        self.ui_height = 640
        self.scaled_w = 0
        self.scaled_h = 0
        
        # Track individual reward components
        self.component_history = {
            "speed_reward": [],
            "centering_bonus": [],
            "done_penalty": [],
            "off_track_penalty": [],
            "collision_penalty": [],
            "distance_reward": [],
        }
    
    def update(self, obs, action, clipped_action, reward, diagnostic_data=None):
        """
        Update the visualization window with new data.
        
        Args:
            obs: Observation image
            action: Action from policy
            clipped_action: Clipped action
            reward: Reward value
            diagnostic_data: Optional dict with step diagnostics
                - cte: cross-track error
                - speed: speed
                - forward_vel: forward velocity
                - hit: collision object name
                - termination_reason: why episode might end
                - reward_components: dict of reward breakdown
        """
        # Process observation
        obs_img = process_observation_image(obs)
        
        # Extract action and reward values
        action = extract_action_value(action)
        clipped_action = extract_action_value(clipped_action)
        reward_val = extract_reward_value(reward)
        
        # Setup window dimensions
        w, h = obs_img.shape[:2]
        scale_factor = 1
        self.scaled_w, self.scaled_h = w * scale_factor, h * scale_factor
        
        # UI layout: actions (150px) + diagnostics (150px) + component plots (240px)
        # Use instance variable ui_height
        
        if not self.initialized:
            self.screen = pygame.display.set_mode((self.scaled_w, self.scaled_h + self.ui_height))
            # Set window title with algorithm name and port
            title_parts = [self.algorithm_name]
            if self.port is not None:
                title_parts.append(f"Port {self.port}")
            pygame.display.set_caption(" | ".join(title_parts))
            self.initialized = True
        
        # Display observation
        surface = pygame.surfarray.make_surface(obs_img)
        scaled_surface = pygame.transform.scale(surface, (self.scaled_w, self.scaled_h))
        self.screen.blit(scaled_surface, (0, 0))
        
        # Clear UI area
        pygame.draw.rect(self.screen, (0, 0, 0), (0, self.scaled_h, self.scaled_w, self.ui_height))
        
        # Draw UI elements
        # Draw UI elements
        bar_y = self.scaled_h + 5
        bar_width = 100
        bar_height = 15
        label_x = 5
        bar_x = 90
        value_x = bar_x + bar_width + 5
        
        # Steer action bar
        steer_value_raw = float(action[0])
        steer_label = self.font.render("Steer:", True, (255, 255, 255))
        self.screen.blit(steer_label, (label_x, bar_y))
        pygame.draw.rect(self.screen, (255, 0, 0), (bar_x, bar_y, bar_width, bar_height), 2)
        center_x = bar_x + bar_width // 2
        if steer_value_raw >= 0:
            width = int(steer_value_raw * (bar_width // 2))
            pygame.draw.rect(self.screen, (255, 0, 0), (center_x, bar_y, width, bar_height))
        else:
            width = int(abs(steer_value_raw) * (bar_width // 2))
            pygame.draw.rect(self.screen, (255, 0, 0), (center_x - width, bar_y, width, bar_height))
        steer_text = self.font.render(f"{steer_value_raw:.2f}", True, (255, 255, 255))
        self.screen.blit(steer_text, (value_x, bar_y))
        
        # Throttle action bar
        throttle_value_raw = float(action[1])
        throttle_label = self.font.render("Throttle:", True, (255, 255, 255))
        self.screen.blit(throttle_label, (label_x, bar_y + 20))
        pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y + 20, bar_width, bar_height), 2)
        width = int(throttle_value_raw * bar_width)
        pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, bar_y + 20, width, bar_height))
        throttle_text = self.font.render(f"{throttle_value_raw:.2f}", True, (255, 255, 255))
        self.screen.blit(throttle_text, (value_x, bar_y + 20))
        
        # Clipped actions
        clipped_y = bar_y + 40
        steer_clipped_raw = float(clipped_action[0])
        steer_clipped_label = self.font.render("Steer (c):", True, (255, 255, 255))
        self.screen.blit(steer_clipped_label, (label_x, clipped_y))
        pygame.draw.rect(self.screen, (255, 0, 0), (bar_x, clipped_y, bar_width, bar_height), 2)
        center_x = bar_x + bar_width // 2
        if steer_clipped_raw >= 0:
            width = int(steer_clipped_raw * (bar_width // 2))
            pygame.draw.rect(self.screen, (255, 0, 0), (center_x, clipped_y, width, bar_height))
        else:
            width = int(abs(steer_clipped_raw) * (bar_width // 2))
            pygame.draw.rect(self.screen, (255, 0, 0), (center_x - width, clipped_y, width, bar_height))
        steer_clipped_text = self.font.render(f"{steer_clipped_raw:.2f}", True, (255, 255, 255))
        self.screen.blit(steer_clipped_text, (value_x, clipped_y))
        
        throttle_clipped_raw = float(clipped_action[1])
        throttle_clipped_label = self.font.render("Throttle (c):", True, (255, 255, 255))
        self.screen.blit(throttle_clipped_label, (label_x, clipped_y + 20))
        pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, clipped_y + 20, bar_width, bar_height), 2)
        width = int(throttle_clipped_raw * bar_width)
        pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, clipped_y + 20, width, bar_height))
        throttle_clipped_text = self.font.render(f"{throttle_clipped_raw:.2f}", True, (255, 255, 255))
        self.screen.blit(throttle_clipped_text, (value_x, clipped_y + 20))
        
        # Smoothed/Raw Actions (if available)
        smoothed_y = clipped_y + 40
        raw_action = diagnostic_data.get("raw_action")
        smoothed_action = diagnostic_data.get("smoothed_action")
        
        if raw_action is not None and smoothed_action is not None:
            # Raw Action (Blue)
            raw_steer = float(raw_action[0])
            raw_throttle = float(raw_action[1])
            
            # Smoothed Action (Purple)
            smooth_steer = float(smoothed_action[0])
            smooth_throttle = float(smoothed_action[1])
            
            # Draw Steer Comparison
            steer_smooth_label = self.font.render("Steer (Sm/Raw):", True, (255, 255, 255))
            self.screen.blit(steer_smooth_label, (label_x, smoothed_y))
            pygame.draw.rect(self.screen, (255, 0, 0), (bar_x, smoothed_y, bar_width, bar_height), 2)
            
            center_x = bar_x + bar_width // 2
            
            # Draw Raw Steer (Blue outline/thin bar)
            if raw_steer >= 0:
                width = int(raw_steer * (bar_width // 2))
                pygame.draw.rect(self.screen, (0, 100, 255), (center_x, smoothed_y + 5, width, bar_height - 10))
            else:
                width = int(abs(raw_steer) * (bar_width // 2))
                pygame.draw.rect(self.screen, (0, 100, 255), (center_x - width, smoothed_y + 5, width, bar_height - 10))
                
            # Draw Smoothed Steer (Purple solid)
            if smooth_steer >= 0:
                width = int(smooth_steer * (bar_width // 2))
                # Use alpha blending for overlay? Pygame rects don't support alpha directly without surface
                # Just draw it slightly smaller or on top
                pygame.draw.rect(self.screen, (200, 0, 255), (center_x, smoothed_y, width, 5)) # Top strip
                pygame.draw.rect(self.screen, (200, 0, 255), (center_x, smoothed_y + bar_height - 5, width, 5)) # Bottom strip
            else:
                width = int(abs(smooth_steer) * (bar_width // 2))
                pygame.draw.rect(self.screen, (200, 0, 255), (center_x - width, smoothed_y, width, 5))
                pygame.draw.rect(self.screen, (200, 0, 255), (center_x - width, smoothed_y + bar_height - 5, width, 5))
            
            steer_smooth_text = self.font.render(f"{smooth_steer:.2f}/{raw_steer:.2f}", True, (255, 255, 255))
            self.screen.blit(steer_smooth_text, (value_x, smoothed_y))
            
            # Draw Throttle Comparison
            throttle_smooth_label = self.font.render("Throt (Sm/Raw):", True, (255, 255, 255))
            self.screen.blit(throttle_smooth_label, (label_x, smoothed_y + 20))
            pygame.draw.rect(self.screen, (0, 255, 0), (bar_x, smoothed_y + 20, bar_width, bar_height), 2)
            
            # Draw Raw Throttle (Blue)
            width = int(raw_throttle * bar_width)
            pygame.draw.rect(self.screen, (0, 100, 255), (bar_x, smoothed_y + 25, width, bar_height - 10))
            
            # Draw Smoothed Throttle (Purple)
            width = int(smooth_throttle * bar_width)
            pygame.draw.rect(self.screen, (200, 0, 255), (bar_x, smoothed_y + 20, width, 5))
            pygame.draw.rect(self.screen, (200, 0, 255), (bar_x, smoothed_y + 20 + bar_height - 5, width, 5))
            
            throttle_smooth_text = self.font.render(f"{smooth_throttle:.2f}/{raw_throttle:.2f}", True, (255, 255, 255))
            self.screen.blit(throttle_smooth_text, (value_x, smoothed_y + 20))
            
            reward_y = smoothed_y + 45
        else:
            reward_y = clipped_y + 45
        reward_text = self.font.render(f"Reward: {reward_val:.2f}", True, (255, 255, 255))
        self.screen.blit(reward_text, (label_x, reward_y))
        
        # Show diagnostic data if available
        diag_y = reward_y + 25
        if diagnostic_data:
            # CTE and speed
            cte_val = diagnostic_data.get("cte", 0.0)
            max_cte_val = diagnostic_data.get("max_cte", 2.0)
            speed_val = diagnostic_data.get("speed", 0.0)
            fwd_vel = diagnostic_data.get("forward_vel", 0.0)
            hit_val = diagnostic_data.get("hit", "none")
            
            # Calculate CTE ratio for proximity to termination
            cte_ratio = abs(cte_val) / max_cte_val if max_cte_val > 0 else 0.0
            
            diag_text = f"CTE: {cte_val:.2f} | Spd: {speed_val:.2f}"
            diag_surf = self.font.render(diag_text, True, (200, 200, 200))
            self.screen.blit(diag_surf, (label_x, diag_y))
            
            # Termination reason
            term_reason = diagnostic_data.get("termination_reason", "none")
            if term_reason != "none":
                term_color = (255, 0, 0) if term_reason != "none" else (255, 255, 255)
                term_text = f"Reason: {term_reason}"
                term_surf = self.font.render(term_text, True, term_color)
                self.screen.blit(term_surf, (label_x, diag_y + 25))
            
            # Collision indicator
            if hit_val != "none":
                hit_surf = self.font.render(f"HIT: {hit_val}", True, (255, 0, 0))
                self.screen.blit(hit_surf, (label_x, diag_y + 25))
            
            # Termination condition proximity indicators
            term_y = diag_y + 50
            
            # CTE proximity bar (shows how close to off-track termination)
            cte_label = self.font.render("CTE Proximity:", True, (255, 255, 255))
            self.screen.blit(cte_label, (label_x, term_y))
            
            # Draw CTE bar with thresholds
            cte_bar_x = bar_x
            cte_bar_width = bar_width
            cte_bar_height = bar_height
            
            # Background bar
            pygame.draw.rect(self.screen, (60, 60, 60), (cte_bar_x, term_y, cte_bar_width, cte_bar_height))
            
            # Draw threshold markers
            # 1.0 = max_cte (off-track warning)
            threshold_1_x = int(cte_bar_x + cte_bar_width * 1.0)
            pygame.draw.line(self.screen, (255, 165, 0), (threshold_1_x, term_y), (threshold_1_x, term_y + cte_bar_height), 2)
            
            # 1.5 = car fully crossed
            threshold_2_x = int(cte_bar_x + min(1.5, 2.0) * cte_bar_width / 2.0)
            if threshold_2_x <= cte_bar_x + cte_bar_width:
                pygame.draw.line(self.screen, (255, 0, 0), (threshold_2_x, term_y), (threshold_2_x, term_y + cte_bar_height), 2)
            
            # Current CTE position (capped at 2.0 for display)
            cte_display_ratio = min(cte_ratio, 2.0) / 2.0  # Normalize to 0-1 for bar display
            cte_fill_width = int(cte_display_ratio * cte_bar_width)
            
            # Color based on proximity: green -> yellow -> orange -> red
            if cte_ratio < 0.7:
                cte_color = (0, 255, 0)  # Green - safe
            elif cte_ratio < 0.9:
                cte_color = (255, 255, 0)  # Yellow - warning
            elif cte_ratio < 1.0:
                cte_color = (255, 165, 0)  # Orange - danger
            else:
                cte_color = (255, 0, 0)  # Red - off-track
            
            pygame.draw.rect(self.screen, cte_color, (cte_bar_x, term_y, cte_fill_width, cte_bar_height))
            pygame.draw.rect(self.screen, (255, 255, 255), (cte_bar_x, term_y, cte_bar_width, cte_bar_height), 2)
            
            # CTE ratio text
            cte_text = f"{cte_ratio:.2f} / 1.00"
            if cte_ratio >= 1.5:
                cte_text += " CROSSED!"
            elif cte_ratio >= 1.0:
                cte_text += " OFF-TRACK!"
            cte_surf = self.font.render(cte_text, True, cte_color)
            self.screen.blit(cte_surf, (value_x, term_y))
            
            # Other termination flags
            flag_y = term_y + 25
            flags = []
            if diagnostic_data.get("off_track", False):
                flags.append(("OFF-TRACK", (255, 165, 0)))
            if diagnostic_data.get("collision", False):
                flags.append(("COLLISION", (255, 0, 0)))
            if diagnostic_data.get("car_fully_crossed", False):
                flags.append(("FULLY-CROSSED", (255, 0, 0)))
            if diagnostic_data.get("missed_checkpoint", False):
                flags.append(("MISSED-CP", (255, 100, 100)))
            if diagnostic_data.get("dq", False):
                flags.append(("DISQUALIFIED", (255, 0, 0)))
            
            if flags:
                flag_text = " | ".join([f[0] for f in flags])
                flag_color = flags[0][1]  # Use color of first flag
                flag_surf = self.font.render(flag_text, True, flag_color)
                self.screen.blit(flag_surf, (label_x, flag_y))
            
            # Reward components - display current values
            reward_comp = diagnostic_data.get("reward_components", {})
            if reward_comp:
                comp_y = flag_y + 25
                comp_texts = [
                    f"Speed: {reward_comp.get('speed_reward', 0.0):.2f} (x{reward_comp.get('speed_weight', 1.0):.1f})",
                    f"Center: {reward_comp.get('centering_bonus', 0.0):.2f} (x{reward_comp.get('centering_weight', 1.0):.1f})",
                    f"Done: {reward_comp.get('done_penalty', 0.0):.2f}",
                    f"OffTrack: {reward_comp.get('off_track_penalty', 0.0):.2f}",
                    f"Dist: {reward_comp.get('distance_reward', 0.0):.2f} (x{reward_comp.get('distance_weight', 1.0):.1f})",
                ]
                for i, text in enumerate(comp_texts):
                    comp_surf = self.font.render(text, True, (100, 200, 100))
                    self.screen.blit(comp_surf, (label_x + (i % 2) * 160, comp_y + (i // 2) * 20))
            
            # Update component history
            for key in self.component_history.keys():
                value = reward_comp.get(key, 0.0)
                self.component_history[key].append(value)
                if len(self.component_history[key]) > self.max_history:
                    self.component_history[key].pop(0)
        else:
            # If no diagnostic data, append zeros to maintain history length
            for key in self.component_history.keys():
                self.component_history[key].append(0.0)
                if len(self.component_history[key]) > self.max_history:
                    self.component_history[key].pop(0)
        
        # Update reward history
        self.reward_history.append(reward_val)
        if len(self.reward_history) > self.max_history:
            self.reward_history.pop(0)
        
        # Draw plots
        self._draw_reward_plot(self.scaled_w, self.scaled_h + self.ui_height)
        self._draw_reward_components_plot(self.scaled_w, self.scaled_h + self.ui_height, reward_comp)
        
        pygame.display.flip()
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        
        return True
    
    def _draw_reward_plot(self, screen_width, screen_height):
        """Draw a scrolling reward plot"""
        if len(self.reward_history) < 2:
            return
        
        plot_width = 300
        plot_height = 80
        plot_x = 10
        # Position below text area
        plot_y = screen_height - self.ui_height + 310
        
        # Draw plot background
        pygame.draw.rect(self.screen, (40, 40, 40), (plot_x, plot_y, plot_width, plot_height))
        pygame.draw.rect(self.screen, (100, 100, 100), (plot_x, plot_y, plot_width, plot_height), 2)
        
        # Draw grid lines
        for i in range(5):
            y = plot_y + (plot_height * i) // 4
            pygame.draw.line(self.screen, (60, 60, 60), (plot_x, y), (plot_x + plot_width, y), 1)
        
        # Calculate scaling
        rewards_array = np.array(self.reward_history)
        min_reward = rewards_array.min()
        max_reward = rewards_array.max()
        reward_range = max_reward - min_reward if max_reward > min_reward else 1
        
        # Draw reward line
        for i in range(len(self.reward_history) - 1):
            y1 = plot_y + plot_height - ((self.reward_history[i] - min_reward) / reward_range * plot_height)
            y2 = plot_y + plot_height - ((self.reward_history[i + 1] - min_reward) / reward_range * plot_height)
            x1 = plot_x + (i / max(len(self.reward_history) - 1, 1)) * plot_width
            x2 = plot_x + ((i + 1) / max(len(self.reward_history) - 1, 1)) * plot_width
            pygame.draw.line(self.screen, (255, 255, 0), (x1, y1), (x2, y2), 2)
        
        # Draw labels
        title = self.font.render("Total Reward", True, (255, 255, 255))
        self.screen.blit(title, (plot_x + 5, plot_y - 25))
        min_text = self.small_font.render(f"Min: {min_reward:.2f}", True, (150, 150, 150))
        self.screen.blit(min_text, (plot_x + 5, plot_y + plot_height + 5))
        max_text = self.small_font.render(f"Max: {max_reward:.2f}", True, (150, 150, 150))
        self.screen.blit(max_text, (plot_x + 5, plot_y + plot_height + 25))
    
    def _draw_reward_components_plot(self, screen_width, screen_height, current_components=None):
        """Draw individual reward component plots"""
        # Skip if not enough data
        if len(self.reward_history) < 2:
            return
            
        if current_components is None:
            current_components = {}
        
        # Plot configuration - bottom row dedicated to component plots
        plot_width = 145
        plot_height = 50
        plot_spacing = 10
        # Center the plots horizontally
        total_width_needed = 2 * plot_width + 1 * plot_spacing
        start_x = (screen_width - total_width_needed) // 2
        # Position at very bottom with margin
        start_y = screen_height - plot_height - 5
        
        # Component configurations: (key, label, color)
        components = [
            ("speed_reward", "Speed Reward", (0, 255, 100)),
            ("centering_bonus", "Centering Bonus", (100, 150, 255)),
            ("done_penalty", "Done Penalty", (255, 100, 100)),
            ("off_track_penalty", "Off-Track Penalty", (255, 150, 0)),
            ("collision_penalty", "Collision Penalty", (255, 0, 0)),
            ("distance_reward", "Distance Reward", (0, 255, 255)),
        ]
        
        # Draw each component plot
        for idx, (key, label, color) in enumerate(components):
            history = self.component_history[key]
            if len(history) < 2:
                continue
            
            # Calculate position (2 columns, 3 rows)
            col = idx % 2
            row = idx // 2
            plot_x = start_x + col * (plot_width + plot_spacing)
            plot_x = start_x + col * (plot_width + plot_spacing)
            plot_y = start_y - row * (plot_height + 40)
            
            # Draw plot background
            pygame.draw.rect(self.screen, (30, 30, 30), (plot_x, plot_y, plot_width, plot_height))
            pygame.draw.rect(self.screen, (80, 80, 80), (plot_x, plot_y, plot_width, plot_height), 1)
            
            # Draw grid lines
            for i in range(3):
                y = plot_y + (plot_height * i) // 2
                pygame.draw.line(self.screen, (50, 50, 50), (plot_x, y), (plot_x + plot_width, y), 1)
            
            # Calculate scaling
            history_array = np.array(history)
            min_val = history_array.min()
            max_val = history_array.max()
            val_range = max_val - min_val if max_val > min_val else 1
            
            # Draw zero line if range includes both positive and negative
            if min_val < 0 and max_val > 0:
                zero_y = plot_y + plot_height - ((-min_val) / val_range * plot_height)
                pygame.draw.line(self.screen, (100, 100, 100), (plot_x, zero_y), (plot_x + plot_width, zero_y), 1)
            
            # Draw component line
            for i in range(len(history) - 1):
                y1 = plot_y + plot_height - ((history[i] - min_val) / val_range * plot_height)
                y2 = plot_y + plot_height - ((history[i + 1] - min_val) / val_range * plot_height)
                x1 = plot_x + (i / max(len(history) - 1, 1)) * plot_width
                x2 = plot_x + ((i + 1) / max(len(history) - 1, 1)) * plot_width
                pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), 2)
            
            # Draw labels
            title = self.small_font.render(label, True, color)
            self.screen.blit(title, (plot_x + 3, plot_y - 18))
            
            # Show weight if available
            weight_key = key.replace("_reward", "_weight").replace("_bonus", "_weight")
            if "penalty" not in key and weight_key in current_components:
                weight = current_components[weight_key]
                weight_text = self.small_font.render(f"x{weight:.1f}", True, (150, 150, 150))
                self.screen.blit(weight_text, (plot_x + 3, plot_y - 32))
            
            # Current value
            current_val = history[-1] if history else 0.0
            current_text = self.small_font.render(f"{current_val:.2f}", True, (200, 200, 200))
            self.screen.blit(current_text, (plot_x + plot_width - 45, plot_y + 2))
            
            # Min/max values (smaller font)
            if abs(min_val) > 0.01 or abs(max_val) > 0.01:
                range_text = self.small_font.render(f"[{min_val:.2f}, {max_val:.2f}]", True, (120, 120, 120))
                self.screen.blit(range_text, (plot_x + 3, plot_y + plot_height + 3))
    
    def close(self):
        """Close the visualization window"""
        if self.initialized:
            pygame.quit()

