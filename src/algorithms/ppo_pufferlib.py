#!/usr/bin/env python3
"""
CleanRL-style PPO implementation compatible with PufferLib
Based on https://docs.cleanrl.dev/rl-algorithms/ppo/
Optimized for parallel environment execution
"""

import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import os


import argparse
from datetime import datetime
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
    max_cte: float = 2.0  # Maximum cross-track error before termination (lower = stricter)
    frame_skip: int = 1  # Number of frames to skip between actions
    
    # Curriculum Learning
    # Note: Curriculum uses lap_count_proximity (passing near start point) rather than
    # lap_count (crossing finish line) for more robust lap completion detection
    curriculum_learning: bool = False  # Enable curriculum learning for speed
    curriculum_initial_speed: float = 0.25  # Initial max speed (m/s)
    curriculum_target_speed: float = 3.0  # Target max speed (m/s)
    curriculum_speed_increment: float = 0.1  # Speed increase per success
    curriculum_speed_decrement: float = 0.05  # Speed decrease per failure
    curriculum_success_threshold: int = 3  # Consecutive laps needed to increase speed
    curriculum_failure_threshold: int = 5  # Failed episodes before decreasing speed
    curriculum_min_lap_time: float = 30.0  # Minimum lap time to count as success (seconds)
    
    # Random Spawn
    random_spawn_enabled: bool = False  # Enable random spawning anywhere on track
    random_spawn_max_cte_offset: float = 0.0  # Max lateral offset from centerline (meters)
    random_spawn_max_rotation_offset: float = 0.0  # Max rotation offset from tangent (degrees)
    
    # Action Smoothing & Control
    action_smoothing: bool = False # Enable action smoothing
    action_smoothing_sigma: float = 1.0  # Sigma for Gaussian smoothing
    action_history_len: int = 120  # Length of action history for smoothing
    min_throttle: float = 0.0  # Minimum throttle value (if > 0)
    
    # Reward Weights
    reward_speed_weight: float = 1.0  # Weight for speed reward component
    reward_centering_weight: float = 1.0  # Weight for centering reward component
    reward_distance_weight: float = 0.0  # Weight for distance reward component
    reward_done_penalty: float = -1.0  # Penalty for episode termination (crash/off-track)
    reward_lin_combination: bool = False  # Use linear combination of reward terms
    
    # Centering Reward Spline
    centering_setpoint_x: float = 0.3  # X-position for spline points 1 and 3 (0.0 to 1.0)
    centering_setpoint_y: float = 0.8  # Y-value for spline points 1 and 3 (0.0 to 1.0)
    
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
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Video Recording
    record_videos: bool = False  # Enable video recording of episodes during training

    
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


class CurriculumManager:
    """Manages curriculum learning for speed adjustment"""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.enabled = config.curriculum_learning
        
        if not self.enabled:
            return
        
        # Current speed limit
        self.current_max_speed = config.curriculum_initial_speed
        
        # Performance tracking
        self.consecutive_successes = 0
        self.consecutive_failures = 0
        self.recent_episodes = []  # Track recent episode outcomes
        self.total_laps_completed = 0
        
        print(f"\nCurriculum Learning Enabled:")
        print(f"  Initial max speed: {self.current_max_speed:.2f} m/s")
        print(f"  Target max speed: {config.curriculum_target_speed:.2f} m/s")
        print(f"  Success threshold: {config.curriculum_success_threshold} consecutive laps")
        print(f"  Failure threshold: {config.curriculum_failure_threshold} failed episodes")
    
    def get_max_speed(self) -> float:
        """Get current maximum speed limit"""
        if not self.enabled:
            return float('inf')  # No limit
        return self.current_max_speed
    
    def update(self, lap_count: int, lap_time: float, episode_length: int) -> dict:
        """Update curriculum based on episode outcome
        
        Returns:
            dict: Status information about curriculum changes
        """
        if not self.enabled:
            return {}
        
        status = {
            'speed_changed': False,
            'old_speed': self.current_max_speed,
            'new_speed': self.current_max_speed,
            'reason': ''
        }
        
        # Determine if episode was successful
        # Success = completed at least 1 lap with reasonable lap time
        is_success = (
            lap_count > 0 and 
            lap_time >= self.config.curriculum_min_lap_time and
            lap_time < 300.0  # Reasonable upper bound
        )
        
        self.recent_episodes.append(is_success)
        if len(self.recent_episodes) > 20:  # Keep last 20 episodes
            self.recent_episodes.pop(0)
        
        if is_success:
            self.total_laps_completed += lap_count
            self.consecutive_successes += 1
            self.consecutive_failures = 0
            
            # Check if we should increase speed
            if self.consecutive_successes >= self.config.curriculum_success_threshold:
                if self.current_max_speed < self.config.curriculum_target_speed:
                    old_speed = self.current_max_speed
                    self.current_max_speed = min(
                        self.current_max_speed + self.config.curriculum_speed_increment,
                        self.config.curriculum_target_speed
                    )
                    status['speed_changed'] = True
                    status['old_speed'] = old_speed
                    status['new_speed'] = self.current_max_speed
                    status['reason'] = f'{self.consecutive_successes} consecutive successful laps'
                    self.consecutive_successes = 0  # Reset counter
        else:
            self.consecutive_failures += 1
            self.consecutive_successes = 0
            
            # Check if we should decrease speed
            if self.consecutive_failures >= self.config.curriculum_failure_threshold:
                if self.current_max_speed > self.config.curriculum_initial_speed:
                    old_speed = self.current_max_speed
                    self.current_max_speed = max(
                        self.current_max_speed - self.config.curriculum_speed_decrement,
                        self.config.curriculum_initial_speed
                    )
                    status['speed_changed'] = True
                    status['old_speed'] = old_speed
                    status['new_speed'] = self.current_max_speed
                    status['reason'] = f'{self.consecutive_failures} consecutive failures'
                    self.consecutive_failures = 0  # Reset counter
        
        return status
    
    def get_stats(self) -> dict:
        """Get curriculum statistics"""
        if not self.enabled:
            return {}
        
        success_rate = 0.0
        if len(self.recent_episodes) > 0:
            success_rate = sum(self.recent_episodes) / len(self.recent_episodes)
        
        return {
            'current_max_speed': self.current_max_speed,
            'consecutive_successes': self.consecutive_successes,
            'consecutive_failures': self.consecutive_failures,
            'recent_success_rate': success_rate,
            'total_laps_completed': self.total_laps_completed,
        }


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
        
        # Create curriculum manager
        self.curriculum = CurriculumManager(config)
        
        # Create vectorized environment
        print(f"Creating {config.num_envs} parallel environments...")
        env_config = {
            "max_cte": config.max_cte,
            "frame_skip": config.frame_skip,
        }
        # Add max_speed to env_config if curriculum learning is enabled
        if config.curriculum_learning:
            env_config["max_speed"] = self.curriculum.get_max_speed()
        
        # Add random spawn parameters to env_config
        if config.random_spawn_enabled:
            env_config["random_spawn_enabled"] = True
            env_config["random_spawn_max_cte_offset"] = config.random_spawn_max_cte_offset
            env_config["random_spawn_max_rotation_offset"] = config.random_spawn_max_rotation_offset
            print(f"Random spawn enabled: lateral offset ±{config.random_spawn_max_cte_offset}m, "
                  f"rotation offset ±{config.random_spawn_max_rotation_offset}°")
        
        # Add reward weights to env_config
        env_config["reward_speed_weight"] = config.reward_speed_weight
        env_config["reward_centering_weight"] = config.reward_centering_weight
        env_config["reward_distance_weight"] = config.reward_distance_weight
        env_config["reward_done_penalty"] = config.reward_done_penalty
        env_config["reward_lin_combination"] = config.reward_lin_combination
        print(f"Reward weights: speed={config.reward_speed_weight}, centering={config.reward_centering_weight}, distance={config.reward_distance_weight}")
        print(f"Done penalty: {config.reward_done_penalty}")
        print(f"Reward linear combination: {config.reward_lin_combination}")
        
        # Add centering setpoints to env_config
        env_config["centering_setpoints"] = (config.centering_setpoint_x, config.centering_setpoint_y)
        print(f"Centering reward setpoints: x={config.centering_setpoint_x}, y={config.centering_setpoint_y}")
        
        # Add action smoothing and control parameters to env_config
        env_config["action_smoothing"] = config.action_smoothing
        env_config["action_smoothing_sigma"] = config.action_smoothing_sigma
        env_config["action_history_len"] = config.action_history_len
        env_config["min_throttle"] = config.min_throttle
        print(f"Action smoothing: {config.action_smoothing} (sigma={config.action_smoothing_sigma}, history={config.action_history_len})")
        print(f"Min throttle: {config.min_throttle}")
        
        # Add playback mode to env_config
        env_config["playback"] = config.playback
        
        self.envs = make_vectorized_env(
            env_name=config.env_name,
            num_envs=config.num_envs,
            start_port=config.start_port,
            backend=config.backend,
            policy_name="ppo",
            env_config=env_config,
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
        self.run_id = os.path.basename(self.log_dir)
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
        
        # Video Recording State
        self.recording_state = "IDLE"  # IDLE, WAITING_FOR_EPISODE_START, RECORDING
        self.current_video_filename = ""
        self.video_dir = os.path.expanduser(f"~/roboracer_ws/simulator/output/videos/{self.run_id}")
        self.recording_trigger_update = 0
        self.recording_metadata = ""
        if config.record_videos:
            os.makedirs(self.video_dir, exist_ok=True)
            print(f"Video recording enabled. Videos will be saved to: {self.video_dir}")

    def send_env_message(self, env_idx, message):
        """Send a message to a specific environment"""
        try:
            # Handle Serial backend
            if hasattr(self.envs, 'envs'):
                # Traverse wrappers to find DonkeyEnv
                env = self.envs.envs[env_idx]
                while hasattr(env, 'env'):
                    if hasattr(env, 'viewer') and hasattr(env.viewer, 'handler'):
                        break
                    env = env.env
                
                if hasattr(env, 'viewer') and hasattr(env.viewer, 'handler'):
                    env.viewer.handler.send_msg(message)
                else:
                    # Try one more level if it's a PufferEnv wrapper
                    if hasattr(env, 'env_creator'):
                         # This path is harder to reach dynamically without unwrapping
                         pass
                    # print(f"Warning: Could not find viewer/handler for env {env_idx}")
                    pass
            # Handle Multiprocessing backend (if supported by PufferLib/Gym)
            elif hasattr(self.envs, 'call'):
                # Attempt to call send_msg on the env
                # This is backend-specific and might not work easily
                pass
        except Exception as e:
            print(f"Error sending message to env {env_idx}: {e}")

    def process_and_log_video(self, video_path, step):
        """Read video file and log to TensorBoard"""
        try:
            import torchvision
            
            # Retry loop to wait for file to be written
            max_retries = 10
            retry_delay = 0.5
            
            for i in range(max_retries):
                if os.path.exists(video_path) and os.path.getsize(video_path) > 0:
                    try:

                        time.sleep(1.0)  # Wait for file to be fully released

                        
                        # read_video returns (T, H, W, C) in [0, 255]
                        # add_video expects (N, T, C, H, W)
                        video, _, _ = torchvision.io.read_video(video_path, pts_unit='sec')
                        
                        if len(video) > 0:
                            # Permute to (T, C, H, W)
                            video = video.permute(0, 3, 1, 2)
                            # Add batch dimension (1, T, C, H, W)
                            video = video.unsqueeze(0)
                            
                            self.writer.add_video("recording/episode", video, step, fps=20)
                            self.writer.flush()
                            print(f"Logged video to TensorBoard: {video_path}")
                            

                                
                            return
                        else:
                            print(f"Warning: Empty video file {video_path} (attempt {i+1}/{max_retries})")

                    except Exception as e:
                        print(f"Warning: Error processing/reading video {video_path} (attempt {i+1}/{max_retries}): {e}")
                else:
                    print(f"Waiting for video file {video_path}... (attempt {i+1}/{max_retries})")
                
                time.sleep(retry_delay)
            
            print(f"Error: Failed to process video {video_path} after {max_retries} attempts")
                
        except ImportError as e:
            print(f"Warning: torchvision not installed or import failed: {e}")
            print("To use video logging, please install torchvision and av: uv add torchvision av")
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")

    
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
                
                # Extract diagnostic data from first environment
                from rl_utils import extract_step_data
                vis_diag = extract_step_data(info, 0) if info else None
                
                if not self.visualizer.update(vis_obs, vis_action, vis_clipped_action, vis_reward, vis_diag):
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
                    # Use lap_count_proximity for curriculum learning if available
                    current_lap_count = lap_metrics.get("lap_count_proximity", lap_metrics.get("lap_count", 0))
                    if current_lap_count > self.last_seen_lap_counts[idx]:
                        # New lap completed, log the lap time
                        if "lap_time" in lap_metrics and lap_metrics["lap_time"] > 0.0:
                            self.writer.add_scalar("laps/lap_time", lap_metrics["lap_time"], self.global_step)
                            self.writer.flush()
                        self.last_seen_lap_counts[idx] = current_lap_count
                        self.last_seen_lap_counts[idx] = current_lap_count
            
            # Video Recording Logic (Env 0 only)
            if self.config.record_videos:
                if self.recording_state == "WAITING_FOR_EPISODE_START":
                    if done[0]:
                        # Start recording next episode
                        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"{self.video_dir}/update_{self.recording_trigger_update}.mp4"
                        
                        # Save model checkpoint for this video
                        model_path = f"{self.model_dir}/update_{self.recording_trigger_update}.pt"
                        torch.save({
                            "update": self.recording_trigger_update,
                            "global_step": self.global_step,
                            "model_state_dict": self.agent.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                        }, model_path)
                        print(f"Saved model checkpoint for video to {model_path}")
                        
                        # Save metadata file
                        metadata_path = f"{self.video_dir}/update_{self.recording_trigger_update}.txt"
                        with open(metadata_path, "w") as f:
                            f.write(self.recording_metadata)
                        print(f"Saved metadata to {metadata_path}")

                        # On Windows/Unity, path might need to be absolute or relative to project. 
                        # We pass absolute path.
                        self.current_video_filename = filename
                        self.send_env_message(0, {"msg_type": "record_video", "filename": filename})
                        self.recording_state = "RECORDING"
                        print(f"Started recording video: {filename}")
                
                elif self.recording_state == "RECORDING":
                    if done[0]:
                        # Stop recording
                        self.send_env_message(0, {"msg_type": "stop_recording"})
                        self.recording_state = "PROCESSING"
                        print(f"Stopped recording video. Processing...")
                        
                        # Process in background or main thread? Main thread for simplicity.
                        # Wait a bit for file to be written?
                        # Unity Recorder might take a moment.
                        # We'll process it in the next update or immediately?
                        # Let's try immediately but with a small delay if needed, 
                        # or just log it.
                        self.process_and_log_video(self.current_video_filename, self.global_step)
                        self.recording_state = "IDLE"

            for idx, d in enumerate(done):
                if d:
                    metrics = extract_episode_metrics(info, idx, d)
                    if metrics:
                        self.episode_metrics.add_metrics(metrics)
                        
                        # Update curriculum based on episode outcome
                        if self.config.curriculum_learning:
                            # Use lap_count_proximity for curriculum learning if available
                            lap_count = metrics.get('lap_count_proximity', metrics.get('lap_count', 0))
                            lap_time = metrics.get('lap_time', 0.0)
                            episode_length = metrics.get('length', 0)
                            
                            curriculum_status = self.curriculum.update(lap_count, lap_time, episode_length)
                            
                            # If speed changed, update environment configuration
                            if curriculum_status.get('speed_changed', False):
                                new_speed = curriculum_status['new_speed']
                                print(f"\n[Curriculum] Speed adjusted: {curriculum_status['old_speed']:.2f} -> {new_speed:.2f} m/s")
                                print(f"[Curriculum] Reason: {curriculum_status['reason']}")
                                
                                # Note: We can't directly update running environments' max_speed,
                                # but the change will take effect on next episode reset
                                # For now, we'll log it and environments will get updated on reset
                    
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
        
        if self.config.model_path:
            self.load_model(self.config.model_path)
            
        start_time = time.time()
        num_updates = self.config.total_timesteps // self.batch_size
        
        for update in range(self.update + 1, num_updates + 1):
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
            
            # Log curriculum learning stats
            if self.config.curriculum_learning:
                curriculum_stats = self.curriculum.get_stats()
                self.writer.add_scalar("curriculum/max_speed", curriculum_stats['current_max_speed'], self.global_step)
                self.writer.add_scalar("curriculum/consecutive_successes", curriculum_stats['consecutive_successes'], self.global_step)
                self.writer.add_scalar("curriculum/consecutive_failures", curriculum_stats['consecutive_failures'], self.global_step)
                self.writer.add_scalar("curriculum/recent_success_rate", curriculum_stats['recent_success_rate'], self.global_step)
                self.writer.add_scalar("curriculum/total_laps_completed", curriculum_stats['total_laps_completed'], self.global_step)
            
            # Flush to ensure data is written to disk
            self.writer.flush()
            
            # Print progress
            sps = int(self.global_step / (time.time() - start_time))
            progress_str = f"Update {update}/{num_updates} | Step {self.global_step} | SPS: {sps}"
            print(progress_str)
            
            # Collect metadata for recording
            metadata_lines = [progress_str]
            
            # Print episode metrics summary
            summary = self.episode_metrics.print_summary()
            if summary:
                print(summary)
                metadata_lines.append(summary)
            
            # Print training stats
            train_stats_str = f"  PG Loss: {train_stats['pg_loss']:.4f} | V Loss: {train_stats['v_loss']:.4f}\n" \
                              f"  Entropy: {train_stats['entropy_loss']:.4f} | KL: {train_stats['approx_kl']:.4f}"
            print(train_stats_str)
            metadata_lines.append(train_stats_str)
            
            # Print curriculum stats
            if self.config.curriculum_learning:
                curriculum_stats = self.curriculum.get_stats()
                curr_str = f"  [Curriculum] Max Speed: {curriculum_stats['current_max_speed']:.2f} m/s | " \
                           f"Success Rate: {curriculum_stats['recent_success_rate']:.1%} | " \
                           f"Total Laps: {curriculum_stats['total_laps_completed']}"
                print(curr_str)
                metadata_lines.append(curr_str)
            
            # Trigger video recording for next episode
            if self.config.record_videos and self.recording_state == "IDLE":
                self.recording_state = "WAITING_FOR_EPISODE_START"
                self.recording_trigger_update = update
                self.recording_metadata = "\n".join(metadata_lines)
                print("Video recording requested for next episode.")
            
            # Reset episode metrics for next update
            self.episode_metrics.reset()
            
            # Save model
            # if update % self.config.save_model_freq == 0:
            #     model_path = f"{self.model_dir}/ppo_donkey_update{update}.pt"
            #     torch.save({
            #         "update": update,
            #         "global_step": self.global_step,
            #         "model_state_dict": self.agent.state_dict(),
            #         "optimizer_state_dict": self.optimizer.state_dict(),
            #     }, model_path)
            #     print(f"  Saved model to {model_path}")
        
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
            
            # Check if any environment is done (episode ended/car reset)
            if np.any(done):
                # Log episode metrics and check for completion
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
                                if 'off_track' in metrics:
                                    off_track_status = "Yes" if metrics['off_track'] else "No"
                                    print(f"  Off-Track: {off_track_status}")
                            
                            print(f"{'='*60}\n")
                            
                            # Check if we've completed all episodes
                            if episode_count >= self.config.num_episodes:
                                break
                
                # Break if all requested episodes are done
                if episode_count >= self.config.num_episodes:
                    break

            # Visualization (show first environment)
            if self.visualize and self.visualizer is not None:
                vis_obs = obs[0] if self.config.num_envs > 1 else obs
                if hasattr(vis_obs, 'cpu'):
                    vis_obs = vis_obs.cpu().numpy()
                vis_action = action[0] if self.config.num_envs > 1 else action
                vis_clipped_action = action_np[0] if self.config.num_envs > 1 else action_np
                vis_reward = reward[0] if isinstance(reward, (np.ndarray, list)) else reward
                
                # Extract diagnostic data from first environment
                from rl_utils import extract_step_data
                vis_diag = extract_step_data(info, 0) if info else None
                
                if not self.visualizer.update(vis_obs, vis_action, vis_clipped_action, vis_reward, vis_diag):
                    # User closed window
                    print("\nVisualization window closed. Stopping playback.")
                    break
            
            # Update observation - vectorized envs auto-reset, so next_obs_np contains reset obs for done envs
            obs = prepare_observation(next_obs_np, self.device)
        
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
        
        # Off-Track Statistics
        if 'episode_termination_reasons' in metrics_dict:
            reasons = metrics_dict['episode_termination_reasons']
            # Count off_track occurrences (either as boolean flag or termination reason)
            # Note: metrics_dict doesn't explicitly store 'episode_off_track' list in EpisodeMetricsLogger
            # but we can infer from termination reasons if that's how it's tracked, 
            # OR we should update EpisodeMetricsLogger to store off_track explicitly if needed.
            # Looking at EpisodeMetricsLogger, it stores 'episode_termination_reasons'.
            
            num_off_track = reasons.count("off_track")
            off_track_rate = (num_off_track / len(reasons)) * 100 if len(reasons) > 0 else 0.0
            
            print(f"\nOff-Track Statistics:")
            print(f"  Off-Track Rate: {off_track_rate:.1f}%")
            print(f"  Episodes Off-Track: {num_off_track}/{len(reasons)}")
        
        print(f"{'='*60}\n")
        
        # Cleanup
        self.envs.close()
        if self.visualizer is not None:
            self.visualizer.close()


def main():
    parser = argparse.ArgumentParser(description="PPO training/playback with PufferLib")
    
    # Mode selection
    parser.add_argument("--playback", action="store_true", help="run playback mode instead of training")
    parser.add_argument("--model-path", type=str, help="path to saved model checkpoint (required for playback, optional for resuming training)")
    parser.add_argument("--num-episodes", type=int, default=10, help="number of episodes to run in playback mode")
    parser.add_argument("--deterministic", action="store_true", default=True, help="use deterministic policy (mean action) in playback")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false", help="use stochastic policy (sample actions) in playback")
    
    # Environment
    parser.add_argument("--env-name", type=str, default="donkey-circuit-launch-track-v0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--start-port", type=int, default=9091)
    parser.add_argument("--backend", type=str, default="serial", choices=["serial", "multiprocessing", "ray"])
    parser.add_argument("--max-cte", type=float, default=2.0, help="maximum cross-track error before termination (lower = stricter)")
    parser.add_argument("--frame-skip", type=int, default=1, help="number of frames to skip between actions")
    
    # Curriculum Learning
    parser.add_argument("--curriculum-learning", action="store_true", help="enable curriculum learning for speed")
    parser.add_argument("--curriculum-initial-speed", type=float, default=0.25, help="initial max speed for curriculum (m/s)")
    parser.add_argument("--curriculum-target-speed", type=float, default=3.0, help="target max speed for curriculum (m/s)")
    parser.add_argument("--curriculum-speed-increment", type=float, default=0.1, help="speed increase per success")
    parser.add_argument("--curriculum-speed-decrement", type=float, default=0.05, help="speed decrease per failure")
    parser.add_argument("--curriculum-success-threshold", type=int, default=3, help="consecutive laps needed to increase speed")
    parser.add_argument("--curriculum-failure-threshold", type=int, default=5, help="failed episodes before decreasing speed")
    parser.add_argument("--curriculum-min-lap-time", type=float, default=30.0, help="minimum lap time to count as success (seconds)")
    
    # Random Spawn
    parser.add_argument("--random-spawn", action="store_true", help="enable random spawning anywhere on track")
    parser.add_argument("--random-spawn-max-cte-offset", type=float, default=1.0, help="max lateral offset from centerline (meters)")
    parser.add_argument("--random-spawn-max-rotation-offset", type=float, default=15.0, help="max rotation offset from tangent (degrees)")
    
    # Action Smoothing & Control
    parser.add_argument("--action-smoothing", action="store_true", default=False, help="enable action smoothing (default: False)")
    parser.add_argument("--no-action-smoothing", dest="action_smoothing", action="store_false", help="disable action smoothing")
    parser.add_argument("--action-smoothing-sigma", type=float, default=1.0, help="sigma for Gaussian smoothing")
    parser.add_argument("--action-history-len", type=int, default=120, help="length of action history for smoothing")
    parser.add_argument("--min-throttle", type=float, default=0.0, help="minimum throttle value")
    
    # Reward Weights
    parser.add_argument("--reward-speed-weight", type=float, default=1.0, help="weight for speed reward component")
    parser.add_argument("--reward-centering-weight", type=float, default=1.0, help="weight for centering reward component")
    parser.add_argument("--reward-lin-combination", action="store_true", help="use linear combination of reward terms")
    parser.add_argument("--centering-setpoint-x", type=float, default=0.3, help="x-position for spline points 1 and 3 (0.0 to 1.0)")
    parser.add_argument("--centering-setpoint-y", type=float, default=0.8, help="y-value for spline points 1 and 3 (0.0 to 1.0)")
    
    # Training
    parser.add_argument("--total-timesteps", type=int, default=1000000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--num-steps", type=int, default=2048)
    
    # Logging
    parser.add_argument("--log-dir", type=str, default="./output/tensorboard", help="tensorboard log directory")
    parser.add_argument("--model-dir", type=str, default="./output/models", help="model checkpoint directory")
    
    # Misc
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--visualize", action="store_true", help="enable pygame visualization window")
    parser.add_argument("--record-videos", action="store_true", help="enable video recording of episodes")
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
        max_cte=args.max_cte,
        frame_skip=args.frame_skip,
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
        curriculum_learning=args.curriculum_learning,
        curriculum_initial_speed=args.curriculum_initial_speed,
        curriculum_target_speed=args.curriculum_target_speed,
        curriculum_speed_increment=args.curriculum_speed_increment,
        curriculum_speed_decrement=args.curriculum_speed_decrement,
        curriculum_success_threshold=args.curriculum_success_threshold,
        curriculum_failure_threshold=args.curriculum_failure_threshold,
        curriculum_min_lap_time=args.curriculum_min_lap_time,
        random_spawn_enabled=args.random_spawn,
        random_spawn_max_cte_offset=args.random_spawn_max_cte_offset,
        random_spawn_max_rotation_offset=args.random_spawn_max_rotation_offset,
        reward_speed_weight=args.reward_speed_weight,
        reward_centering_weight=args.reward_centering_weight,
        reward_lin_combination=args.reward_lin_combination,
        centering_setpoint_x=args.centering_setpoint_x,
        centering_setpoint_y=args.centering_setpoint_y,
        record_videos=args.record_videos,
        action_smoothing=args.action_smoothing,
        action_smoothing_sigma=args.action_smoothing_sigma,
        action_history_len=args.action_history_len,
        min_throttle=args.min_throttle,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
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

