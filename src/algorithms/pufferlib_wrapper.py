#!/usr/bin/env python3
"""
PufferLib wrapper for DonkeyEnv
Enables vectorization and parallel simulation
"""

import pathlib
import sys
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import uuid
import gym
import gym_donkeycar
import pufferlib
import pufferlib.emulation
import pufferlib.vector
from typing import Dict, Any, Optional, Tuple
from utils.sim_starter import start_sim


class GymToGymnasiumWrapper(gym.Wrapper):
    """
    Wrapper to convert legacy Gym environment to Gymnasium-compatible interface.
    - Makes reset() accept seed parameter and return (observation, info)
    - Makes step() return (observation, reward, terminated, truncated, info)
    - Handles legacy Gym's reset() that returns only observation
    - Handles legacy Gym's step() that returns (observation, reward, done, info)
    """
    def __init__(self, env):
        super().__init__(env)
        self._last_info = {}
    
    def reset(self, seed: Optional[int] = None, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment and return (observation, info) tuple.
        Compatible with Gymnasium API.
        """
        # Handle seed if provided (legacy Gym uses seed() method)
        if seed is not None:
            if hasattr(self.env, 'seed'):
                self.env.seed(seed)
        
        # Call legacy reset() which returns only observation
        observation = self.env.reset()
        
        # Get info from the last step or create empty dict
        # For DonkeyEnv, we can get info by calling observe() but that would advance the env
        # So we'll return empty info dict and let step() populate it
        info = self._last_info.copy() if self._last_info else {}
        
        return observation, info
    
    def step(self, action):
        """
        Step the environment and return Gymnasium-compatible 5-tuple.
        Returns: (observation, reward, terminated, truncated, info)
        """
        # Legacy Gym returns (observation, reward, done, info)
        observation, reward, done, info = self.env.step(action)
        self._last_info = info
        
        # Convert legacy Gym's 'done' to Gymnasium's 'terminated' and 'truncated'
        # Since legacy Gym doesn't distinguish termination vs truncation,
        # we treat 'done=True' as 'terminated=True' and 'truncated=False'
        terminated = bool(done)
        truncated = False
        
        return observation, reward, terminated, truncated, info


def make_donkey_env(env_name: str = "donkey-circuit-launch-track-v0", port: int = 9091):
    """
    Create a DonkeyEnv wrapped for PufferLib
    Launches simulator instance before creating environment.
    
    Args:
        env_name: Name of the donkey sim environment
        port: Port to use for TCP connection
    
    Returns:
        PufferLib-wrapped environment
    """
    def env_creator():
        conf = {
            "host": "127.0.0.1",
            "port": port,
            "body_style": "donkey",
            "body_rgb": (128, 128, 128),
            "car_name": f"agent_{port}",
            "font_size": 100,
            "racer_name": "PPO_Puffer",
            "country": "USA",
            "bio": "Learning to drive w PufferLib PPO",
            "guid": str(uuid.uuid4()),
            "max_cte": 10,
            "max_speed": 2.0,
        }
        # Use start_sim to launch simulator and create environment
        env = start_sim(env_name=env_name, port=port, conf=conf)
        # Wrap with compatibility layer for Gymnasium
        return GymToGymnasiumWrapper(env)
    
    # Wrap with PufferLib's Gym emulation layer
    # Note: PufferLib expects Gymnasium, but can work with legacy Gym via emulation
    return pufferlib.emulation.GymnasiumPufferEnv(env_creator=env_creator)


def make_vectorized_env(
    env_name: str = "donkey-circuit-launch-track-v0",
    num_envs: int = 4,
    start_port: int = 9091,
    backend: str = "serial",
    policy_name: str = "agent",
    env_config: dict = None,
):
    """
    Create a vectorized environment with multiple DonkeyEnv instances
    
    Args:
        env_name: Name of the donkey sim environment
        num_envs: Number of parallel environments
        start_port: Starting port number (each env gets start_port + i)
        backend: Vectorization backend - "serial", "multiprocessing", or "ray"
        policy_name: Name of the policy (e.g., "ppo", "sac") - used for car naming
        env_config: Dict of environment configuration parameters (max_cte, frame_skip, etc.)
    
    Returns:
        PufferLib vectorized environment
    """
    if env_config is None:
        env_config = {}
    # Create environment creators for each parallel instance
    # Each needs a unique port to connect to separate simulator instances
    # Each will launch its own simulator instance via start_sim
    env_creators = []
    for i in range(num_envs):
        port = start_port + i
        
        def make_env(port=port, policy_name=policy_name, **kwargs):  # Capture port and policy_name in closure, accept kwargs from pufferlib (buf, seed, etc.)
            def env_creator():
                conf = {
                    "host": "127.0.0.1",
                    "port": port,
                    "body_style": "donkey",
                    "body_rgb": (128, 128, 128),
                    "font_size": 100,
                    "country": "USA",
                    "guid": str(uuid.uuid4()),
                    "max_cte": env_config.get("max_cte", 10),
                    "frame_skip": env_config.get("frame_skip", 1),
                    "max_speed": env_config.get("max_speed", float('inf')),  # Default to no limit
                    "policy_name": policy_name,  # Pass policy name to start_sim
                    
                    # Random spawn parameters
                    "random_spawn_enabled": env_config.get("random_spawn_enabled", False),
                    "random_spawn_max_cte_offset": env_config.get("random_spawn_max_cte_offset", 0.0),
                    "random_spawn_max_rotation_offset": env_config.get("random_spawn_max_rotation_offset", 0.0),
                    
                    # Reward weights
                    "reward_speed_weight": env_config.get("reward_speed_weight", 1.0),
                    "reward_centering_weight": env_config.get("reward_centering_weight", 1.0),
                    
                    # Centering setpoints
                    "centering_setpoints": env_config.get("centering_setpoints", (0.3, 0.8)),
                    
                    # Action Smoothing & Control
                    "action_smoothing": env_config.get("action_smoothing", False),
                    "action_smoothing_sigma": env_config.get("action_smoothing_sigma", 1.0),
                    "action_history_len": env_config.get("action_history_len", 120),
                    "min_throttle": env_config.get("min_throttle", 0.0),
                }
                # Use start_sim to launch simulator and create environment
                env = start_sim(env_name=env_name, port=port, conf=conf)
                # Wrap with compatibility layer for Gymnasium
                return GymToGymnasiumWrapper(env)
            
            # Wrap with PufferLib's Gym emulation layer using env_creator
            # This ensures proper handling of legacy Gym reset() that returns only observation
            return pufferlib.emulation.GymnasiumPufferEnv(env_creator=env_creator, **kwargs)
        
        env_creators.append(make_env)
    
    # Map backend strings to PufferLib backend classes
    backend_map = {
        "serial": pufferlib.vector.Serial,
        "multiprocessing": pufferlib.vector.Multiprocessing,
        "ray": pufferlib.vector.Ray,
    }
    
    if backend.lower() not in backend_map:
        raise ValueError(f"Invalid backend: {backend}. Choose from {list(backend_map.keys())}")
    
    # Create vectorized environment
    # When passing a list of creators, env_args and env_kwargs must be lists of length num_envs
    # env_args must be a list of lists/tuples, env_kwargs must be a list of dicts
    # Use empty lists/dicts if no args/kwargs are needed
    env_args = [[] for _ in range(num_envs)]
    env_kwargs = [{} for _ in range(num_envs)]
    
    vecenv = pufferlib.vector.make(
        env_creators,
        env_args=env_args,
        env_kwargs=env_kwargs,
        backend=backend_map[backend.lower()],
        num_envs=num_envs,
    )
    
    return vecenv


if __name__ == "__main__":
    # Test the wrapper
    print("Testing single environment...")
    env = make_donkey_env()
    obs = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    
    # Test a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"Step {i}: reward={reward:.3f}, done={done}")
        if done:
            obs = env.reset()
    
    env.close()
    print("\nSingle environment test complete!")
    
    # Note: Testing vectorized environments requires multiple simulator instances
    # print("\nTesting vectorized environment...")
    # vecenv = make_vectorized_env(num_envs=2, backend="serial")
    # obs = vecenv.reset()
    # print(f"Vectorized observation shape: {obs.shape}")
    # vecenv.close()

