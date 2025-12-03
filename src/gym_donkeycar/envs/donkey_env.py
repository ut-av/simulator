"""
file: donkey_env.py
author: Tawn Kramer
date: 2018-08-31
"""
import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

from gym_donkeycar.envs.donkey_proc import DonkeyUnityProcess
from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimContoller

logger = logging.getLogger(__name__)


def supply_defaults(conf: Dict[str, Any]) -> None:
    """
    Update the config dictonnary
    with defaults when values are missing.

    :param conf: The user defined config dict,
        passed to the environment constructor.
    """
    defaults = [
        ("start_delay", 5.0),
        ("max_cte", 8.0),
        ("frame_skip", 1),
        ("cam_resolution", (120, 160, 3)),
        ("log_level", logging.INFO),
        ("host", "localhost"),
        ("port", 9091),
        ("steer_limit", 1.0),
        ("throttle_min", 0.0),
        ("throttle_max", 1.0),
        ("throttle_max", 1.0),
        ("max_speed", float('inf')),  # Maximum speed in m/s (default: no limit)
        # Sigma represents the smoothing window in frames. To smooth over ~0.5 seconds (30 frames), you need a sigma around 10.0 - 20.0.
        ("action_smoothing", True),
        ("action_smoothing_sigma", 1.0),
        ("action_history_len", 120),
        ("min_throttle", 0.0),
    ]

    for key, val in defaults:
        if key not in conf:
            conf[key] = val
            print(f"Setting default: {key} {val}")


class DonkeyEnv(gym.Env):
    """
    OpenAI Gym Environment for Donkey

    :param level: name of the level to load
    :param conf: configuration dictionary
    """

    metadata = {"render.modes": ["human", "rgb_array"]}

    ACTION_NAMES: List[str] = ["steer", "throttle"]
    VAL_PER_PIXEL: int = 255

    def __init__(self, level: str, conf: Optional[Dict[str, Any]] = None):
        print("starting DonkeyGym env")
        self.viewer = None
        self.proc = None

        if conf is None:
            conf = {}

        conf["level"] = level

        # ensure defaults are supplied if missing.
        supply_defaults(conf)

        # set logging level
        logging.basicConfig(level=conf["log_level"])

        logger.debug("DEBUG ON")
        logger.debug(conf)

        # start Unity simulation subprocess
        # NOTE: Simulator launching is now handled by sim_starter.py
        # This block is disabled to prevent duplicate simulator instances
        self.proc = None
        if "exe_path" in conf:
            logger.warning(
                "DEPRECATION WARNING: 'exe_path' should not be passed to DonkeyEnv. "
                "Simulator launching should be handled by sim_starter.py to avoid duplicate instances. "
                "Ignoring 'exe_path' and assuming simulator is already running on port {}.".format(conf["port"])
            )
            # Do NOT launch simulator - it should already be running
            # self.proc = DonkeyUnityProcess()
            # self.proc.start(conf["exe_path"], host="0.0.0.0", port=conf["port"])
            # time.sleep(conf["start_delay"])

        # start simulation com
        self.viewer = DonkeyUnitySimContoller(conf=conf)

        # Note: for some RL algorithms, it would be better to normalize the action space to [-1, 1]
        # and then rescale to proper limtis
        # steering and throttle
        self.action_space = spaces.Box(
            low=np.array([-float(conf["steer_limit"]), float(conf["throttle_min"])]),
            high=np.array([float(conf["steer_limit"]), float(conf["throttle_max"])]),
            dtype=np.float32,
        )

        # camera sensor data
        self.observation_space = spaces.Box(0, self.VAL_PER_PIXEL, self.viewer.get_sensor_size(), dtype=np.uint8)

        # simulation related variables.
        self.seed()

        # Frame Skipping
        self.frame_skip = conf["frame_skip"]

        # wait until the car is loaded in the scene
        self.viewer.wait_until_loaded()

        # Action Smoothing
        self.action_smoothing = conf["action_smoothing"]
        self.action_smoothing_sigma = conf["action_smoothing_sigma"]
        self.action_history_len = conf["action_history_len"]
        self.min_throttle = conf["min_throttle"]
        self.action_history = deque(maxlen=self.action_history_len)
        
        if self.action_smoothing:
            # Pre-compute Gaussian weights
            # We want the most recent action (index N-1) to have the highest weight
            # Weights are based on distance from the most recent action
            # w_i = exp(- (N-1-i)^2 / (2 * sigma^2))
            indices = np.arange(self.action_history_len)
            # Distance from the end: 0 for last item, 1 for second to last, etc.
            distances = (self.action_history_len - 1) - indices
            self.smoothing_weights = np.exp(-0.5 * (distances / self.action_smoothing_sigma) ** 2)
            # Normalize weights to sum to 1
            self.smoothing_weights /= np.sum(self.smoothing_weights)
            print(f"Action smoothing enabled: sigma={self.action_smoothing_sigma}, history_len={self.action_history_len}")
            print(f"Smoothing weights: {self.smoothing_weights}")

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if hasattr(self, "viewer") and self.viewer is not None:
            self.viewer.quit()
        if hasattr(self, "proc") and self.proc is not None:
            self.proc.quit()

    def set_reward_fn(self, reward_fn: Callable) -> None:
        self.viewer.set_reward_fn(reward_fn)

    def set_episode_over_fn(self, ep_over_fn: Callable) -> None:
        self.viewer.set_episode_over_fn(ep_over_fn)

    def seed(self, seed: Optional[int] = None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        # Apply action smoothing if enabled
        if self.action_smoothing:
            self.action_history.append(action)
            
            # If we don't have enough history yet, just use available history
            # We need to adjust weights dynamically or just pad?
            # Simpler: just use what we have and re-normalize weights for the current length
            current_len = len(self.action_history)
            if current_len < self.action_history_len:
                # Use subset of weights corresponding to the most recent 'current_len' items
                # The weights are computed for positions 0..N-1. 
                # The most recent is always at N-1 (highest weight).
                # So we should take the last 'current_len' weights.
                weights = self.smoothing_weights[-current_len:]
                weights = weights / np.sum(weights)
            else:
                weights = self.smoothing_weights
                
            # Compute weighted average
            # action_history is (T, A), weights is (T,)
            # We want sum(w_i * a_i)
            history_array = np.array(self.action_history)
            smoothed_action = np.average(history_array, axis=0, weights=weights)
            
            # Debug: Print raw vs smoothed action
            #print(f"Raw: {action} -> Smoothed: {smoothed_action}")
            
            # Update info with raw action for debugging if needed
            # But we can't easily pass it out unless we modify the return signature or info dict
            # For now, just use smoothed_action for the environment
            action_to_take = smoothed_action
            
            # Apply minimum throttle rescaling
            # If throttle > 0, map it to [min_throttle, 1.0]
            if action_to_take[1] > 0.01:
                action_to_take[1] = self.min_throttle + (1.0 - self.min_throttle) * action_to_take[1]
        else:
            action_to_take = action

        for _ in range(self.frame_skip):
            self.viewer.take_action(action_to_take)
            observation, reward, done, info = self.viewer.observe()
            
        if self.action_smoothing:
            info["raw_action"] = action
            info["smoothed_action"] = action_to_take
            
        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        # Activate hand brake, so the car does not move
        self.viewer.handler.send_control(0, 0, 1.0)
        time.sleep(0.1)
        self.viewer.reset()
        self.viewer.handler.send_control(0, 0, 1.0)
        time.sleep(0.1)
        observation, reward, done, info = self.viewer.observe()
        
        # Reset action history on reset
        if hasattr(self, "action_history"):
            self.action_history.clear()
            
        return observation

    def render(self, mode: str = "human", close: bool = False) -> Optional[np.ndarray]:
        if close:
            self.viewer.quit()

        return self.viewer.render(mode)

    def is_game_over(self) -> bool:
        return self.viewer.is_game_over()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class GeneratedRoadsEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="generated_road", *args, **kwargs)


class WarehouseEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="warehouse", *args, **kwargs)


class AvcSparkfunEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="sparkfun_avc", *args, **kwargs)


class GeneratedTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="generated_track", *args, **kwargs)


class MountainTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="mountain_track", *args, **kwargs)


class RoboRacingLeagueTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="roboracingleague_1", *args, **kwargs)


class WaveshareEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="waveshare", *args, **kwargs)


class MiniMonacoEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="mini_monaco", *args, **kwargs)


class WarrenTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="warren", *args, **kwargs)


class ThunderhillTrackEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="thunderhill", *args, **kwargs)


class CircuitLaunchEnv(DonkeyEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(level="circuit_launch", *args, **kwargs)