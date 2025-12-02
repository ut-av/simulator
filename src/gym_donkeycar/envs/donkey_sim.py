"""
file: donkey_sim.py
author: Tawn Kramer
date: 2018-08-31
"""
import base64
import logging
import math
import os
import time
import types
from io import BytesIO
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
from PIL import Image
from scipy.interpolate import interp1d

from gym_donkeycar.core.fps import FPSTimer
from gym_donkeycar.core.message import IMesgHandler
from gym_donkeycar.core.sim_client import SimClient

logger = logging.getLogger(__name__)


# Math helpers added by CireNeikual (222464)
def euler_to_quat(e):
    cx = np.cos(e[0] * 0.5)
    sx = np.sin(e[0] * 0.5)
    cy = np.cos(e[1] * 0.5)
    sy = np.sin(e[1] * 0.5)
    cz = np.cos(e[2] * 0.5)
    sz = np.sin(e[2] * 0.5)

    x = sz * cx * cy - cz * sx * sy
    y = cz * sx * cy + sz * cx * sy
    z = cz * cx * sy - sz * sx * cy
    w = cz * cx * cy + sz * sx * sy

    return [x, y, z, w]


def cross(v0, v1):
    return [v0[1] * v1[2] - v0[2] * v1[1], v0[2] * v1[0] - v0[0] * v1[2], v0[0] * v1[1] - v0[1] * v1[0]]


def rotate_vec(q, v):
    uv = cross(q[0:3], v)
    uuv = cross(q[0:3], uv)

    scaleUv = 2.0 * q[3]

    uv[0] *= scaleUv
    uv[1] *= scaleUv
    uv[2] *= scaleUv

    uuv[0] *= 2.0
    uuv[1] *= 2.0
    uuv[2] *= 2.0

    return [v[0] + uv[0] + uuv[0], v[1] + uv[1] + uuv[1], v[2] + uv[2] + uuv[2]]


class DonkeyUnitySimContoller:
    def __init__(self, conf: Dict[str, Any]):
        logger.setLevel(conf["log_level"])

        self.address = (conf["host"], conf["port"])

        self.handler = DonkeyUnitySimHandler(conf=conf)

        self.client = SimClient(self.address, self.handler)

    def set_car_config(
        self,
        body_style: str,
        body_rgb: Tuple[int, int, int],
        car_name: str,
        font_size: int,
    ) -> None:
        self.handler.send_car_config(body_style, body_rgb, car_name, font_size)

    def set_cam_config(self, **kwargs) -> None:
        self.handler.send_cam_config(**kwargs)

    def set_reward_fn(self, reward_fn: Callable) -> None:
        self.handler.set_reward_fn(reward_fn)

    def set_episode_over_fn(self, ep_over_fn: Callable) -> None:
        self.handler.set_episode_over_fn(ep_over_fn)

    def wait_until_loaded(self) -> None:
        time.sleep(0.1)
        while not self.handler.loaded:
            logger.warning("waiting for sim to start..")
            time.sleep(1.0)
        logger.info("sim started!")

    def reset(self) -> None:
        self.handler.reset()

    def get_sensor_size(self) -> Tuple[int, int, int]:
        return self.handler.get_sensor_size()

    def take_action(self, action: np.ndarray):
        self.handler.take_action(action)

    def observe(self) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        return self.handler.observe()

    def quit(self) -> None:
        self.client.stop()

    def exit_scene(self) -> None:
        self.handler.send_exit_scene()

    def render(self, mode: str) -> None:
        pass

    def is_game_over(self) -> bool:
        return self.handler.is_game_over()

    def calc_reward(self, done: bool) -> float:
        return self.handler.calc_reward(done)


class DonkeyUnitySimHandler(IMesgHandler):
    def __init__(self, conf: Dict[str, Any]):
        self.conf = conf
        self.SceneToLoad = conf["level"]
        self.loaded = False
        self.max_cte = conf["max_cte"]
        self.max_speed = conf.get("max_speed", float('inf'))  # Maximum speed limit in m/s
        self.centering_setpoints = conf.get("centering_setpoints", (0.3, 0.8))  # (x_pos, y_value) for spline points 1 and 3
        
        # Reward weights (will be normalized)
        self.reward_speed_weight = conf.get("reward_speed_weight", 1.0)
        self.reward_centering_weight = conf.get("reward_centering_weight", 1.0)
        self.reward_lin_combination = conf.get("reward_lin_combination", False)
        
        # Normalize weights to sum to 1
        total_weight = self.reward_speed_weight + self.reward_centering_weight
        if total_weight > 0:
            self.reward_speed_weight /= total_weight
            self.reward_centering_weight /= total_weight
        else:
            self.reward_speed_weight = 0.5
            self.reward_centering_weight = 0.5
        
        # Random spawn configuration
        self.random_spawn_enabled = conf.get("random_spawn_enabled", False)
        self.random_spawn_max_cte_offset = conf.get("random_spawn_max_cte_offset", 0.0)  # Max distance from centerline
        self.random_spawn_max_rotation_offset = conf.get("random_spawn_max_rotation_offset", 0.0)  # Max rotation offset in degrees
        
        # Distance reward weight
        self.reward_distance_weight = conf.get("reward_distance_weight", 0.0)
        
        # Done penalty
        self.reward_done_penalty = conf.get("reward_done_penalty", -1.0)
        
        # Debug: Print random spawn configuration
        print(f"[DonkeyEnv] Random spawn configuration:")
        print(f"  random_spawn_enabled: {self.random_spawn_enabled}")
        print(f"  random_spawn_max_cte_offset: {self.random_spawn_max_cte_offset}")
        print(f"  random_spawn_max_rotation_offset: {self.random_spawn_max_rotation_offset}")
        
        self.timer = FPSTimer()

        # sensor size - height, width, depth
        self.camera_img_size = conf["cam_resolution"]
        self.image_array = np.zeros(self.camera_img_size)
        self.image_array_b = None
        self.last_obs = self.image_array
        self.time_received = time.time()
        self.last_received = self.time_received
        self.hit = "none"
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.speed = 0.0
        self.forward_vel = 0.0
        self.missed_checkpoint = False
        self.dq = False
        self.over = False
        self.reset_complete = False
        self.client = None
        self.fns = {
            "telemetry": self.on_telemetry,
            "scene_selection_ready": self.on_scene_selection_ready,
            "scene_names": self.on_recv_scene_names,
            "car_loaded": self.on_car_loaded,
            "car_reset_done": self.on_car_reset_done,
            "cross_start": self.on_cross_start,
            "race_start": self.on_race_start,
            "race_stop": self.on_race_stop,
            "DQ": self.on_DQ,
            "ping": self.on_ping,
            "aborted": self.on_abort,
            "missed_checkpoint": self.on_missed_checkpoint,
            "need_car_config": self.on_need_car_config,
            "collision_with_starting_line": self.on_collision_with_starting_line,
        }
        self.gyro_x = 0.0
        self.gyro_y = 0.0
        self.gyro_z = 0.0
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        self.lidar = []

        # car in Unity lefthand coordinate system: roll is Z, pitch is X and yaw is Y
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

        # variables required for lidar points decoding into array format
        self.lidar_deg_per_sweep_inc = 1
        self.lidar_num_sweep_levels = 1
        self.lidar_deg_ang_delta = 1

        self.last_lap_time = 0.0
        self.current_lap_time = 0.0
        self.starting_line_index = -1
        self.lap_count = 0  # Increments when crossing finish line
        self.lap_count_proximity = 0  # Increments when passing near starting point while pointing in correct direction
        self.start_position = None  # Will store (x, y, z) of starting position
        self.start_yaw = 0.0  # Starting yaw angle
        self.last_proximity_check = False  # Track if we were near start last frame
        self.has_left_start = False  # Track if car has moved away from start position

        # Tracking for termination conditions and reward components
        self.termination_reason = "none"
        self.off_track = False
        self.collision = False
        self.car_fully_crossed = False
        
        # Reward components for logging
        self.reward_components = {
            "done_penalty": 0.0,
            "off_track_penalty": 0.0,
            "collision_penalty": 0.0,
            "speed_reward": 0.0,
            "centering_bonus": 0.0,
            "distance_reward": 0.0,
            "total_reward": 0.0,
            "speed_weight": self.reward_speed_weight,
            "centering_weight": self.reward_centering_weight,
            "distance_weight": self.reward_distance_weight,
        }
        
        self.total_distance = 0.0
        self.last_pos = None
        self.dist_since_last_reward = 0.0

    def on_connect(self, client: SimClient) -> None:  # pytype: disable=signature-mismatch
        logger.debug("socket connected")
        self.client = client

    def on_disconnect(self) -> None:
        logger.debug("socket disconnected")
        self.client = None

    def on_abort(self, message: Dict[str, Any]) -> None:
        self.client.stop()

    def on_need_car_config(self, message: Dict[str, Any]) -> None:
        logger.info("on need car config")
        self.loaded = True
        self.send_config(self.conf)

    def on_collision_with_starting_line(self, message: Dict[str, Any]) -> None:
        if self.current_lap_time == 0.0:
            self.current_lap_time = message["timeStamp"]
            self.starting_line_index = message["starting_line_index"]
            # Store starting position for proximity-based lap completion
            if self.start_position is None:
                self.start_position = (self.x, self.y, self.z)
                self.start_yaw = self.yaw
                #logger.info(f"Starting position set: pos=({self.x:.2f}, {self.y:.2f}, {self.z:.2f}), yaw={self.yaw:.1f}°")
        elif self.starting_line_index == message["starting_line_index"]:
            time_at_crossing = message["timeStamp"]
            self.last_lap_time = float(time_at_crossing - self.current_lap_time)
            self.current_lap_time = time_at_crossing
            self.lap_count += 1
            lap_msg = f"New lap time (finish line): {round(self.last_lap_time, 2)} seconds"
            logger.info(lap_msg)

    def check_proximity_lap_completion(self) -> None:
        """
        Check if the car has completed a lap by passing near the starting point
        while pointing in the correct direction.
        
        The car must first move at least 1.5x the proximity distance away from the start
        before we begin checking for lap completion. This prevents false positives
        immediately after spawning.
        """
        if self.start_position is None:
            return
        
        # Calculate distance to starting point (2D distance, ignoring y)
        start_x, start_y, start_z = self.start_position
        distance_to_start = math.sqrt((self.x - start_x)**2 + (self.z - start_z)**2)
        
        # Proximity threshold for lap completion
        proximity_threshold = 2.0 * self.max_cte
        
        # Check if car has left the start area (must be at least 1.5x proximity distance away)
        if not self.has_left_start:
            if distance_to_start >= (proximity_threshold * 1.5):
                self.has_left_start = True
                logger.debug(f"Car has left start area (distance = {distance_to_start:.2f}m)")
            # Don't check for lap completion until car has left start area
            return
        
        # Check if within track width of starting point (using max_cte as proxy for track width)
        within_proximity = distance_to_start <= proximity_threshold
        
        if within_proximity:
            # Check if pointing in roughly the correct direction
            # Calculate angle difference (in degrees)
            yaw_diff = abs(self.yaw - self.start_yaw)
            # Normalize to [-180, 180]
            while yaw_diff > 180:
                yaw_diff -= 360
            yaw_diff = abs(yaw_diff)
            
            # Allow up to 90 degrees difference
            correct_direction = yaw_diff <= 90.0
            
            # Increment lap count if we just entered the proximity zone and are pointing correctly
            if correct_direction and not self.last_proximity_check:
                self.lap_count_proximity += 1
                logger.info(f"Lap completed (proximity): count = {self.lap_count_proximity}, distance = {distance_to_start:.2f}m, yaw_diff = {yaw_diff:.1f}°")
        
        self.last_proximity_check = within_proximity

    @staticmethod
    def extract_keys(dict_: Dict[str, Any], list_: List[str]) -> Dict[str, Any]:
        return_dict = {}
        for key in list_:
            if key in dict_:
                return_dict[key] = dict_[key]
        return return_dict

    def send_config(self, conf: Dict[str, Any]) -> None:
        if "degPerSweepInc" in conf:
            raise ValueError("LIDAR config keys were renamed to use snake_case name instead of CamelCase")

        logger.info("sending car config.")
        # both ways work, car_config shouldn't interfere with other config, so keeping the two alternative
        self.set_car_config(conf)
        if "car_config" in conf.keys():
            self.set_car_config(conf["car_config"])
            logger.info("done sending car config.")

        if "cam_config" in conf.keys():
            cam_config = self.extract_keys(
                conf["cam_config"],
                [
                    "img_w",
                    "img_h",
                    "img_d",
                    "img_enc",
                    "fov",
                    "fish_eye_x",
                    "fish_eye_y",
                    "offset_x",
                    "offset_y",
                    "offset_z",
                    "rot_x",
                    "rot_y",
                    "rot_z",
                ],
            )
            self.send_cam_config(**cam_config)
            logger.info(f"done sending cam config. {cam_config}")

        if "cam_config_b" in conf.keys():
            cam_config_b = self.extract_keys(
                conf["cam_config_b"],
                [
                    "img_w",
                    "img_h",
                    "img_d",
                    "img_enc",
                    "fov",
                    "fish_eye_x",
                    "fish_eye_y",
                    "offset_x",
                    "offset_y",
                    "offset_z",
                    "rot_x",
                    "rot_y",
                    "rot_z",
                ],
            )
            self.send_cam_config(**cam_config_b, msg_type="cam_config_b")
            logger.info(f"done sending cam config B. {cam_config_b}")
            self.image_array_b = np.zeros(self.camera_img_size)

        if "lidar_config" in conf.keys():
            if "degPerSweepInc" in conf:
                raise ValueError("LIDAR config keys were renamed to use snake_case name instead of CamelCase")

            lidar_config = self.extract_keys(
                conf["lidar_config"],
                [
                    "deg_per_sweep_inc",
                    "deg_ang_down",
                    "deg_ang_delta",
                    "num_sweeps_levels",
                    "max_range",
                    "noise",
                    "offset_x",
                    "offset_y",
                    "offset_z",
                    "rot_x",
                ],
            )
            self.send_lidar_config(**lidar_config)
            logger.info(f"done sending lidar config., {lidar_config}")

        # what follows is needed in order not to break older conf

        cam_config = self.extract_keys(
            conf,
            [
                "img_w",
                "img_h",
                "img_d",
                "img_enc",
                "fov",
                "fish_eye_x",
                "fish_eye_y",
                "offset_x",
                "offset_y",
                "offset_z",
                "rot_x",
                "rot_y",
                "rot_z",
            ],
        )
        if cam_config != {}:
            self.send_cam_config(**cam_config)
            logger.info(f"done sending cam config. {cam_config}")
            logger.warning(
                """This way of passing cam_config is deprecated,
                please wrap the parameters in a sub-dictionary with the key 'cam_config'.
                Example: GYM_CONF = {'cam_config':"""
                + str(cam_config)
                + "}"
            )

        lidar_config = self.extract_keys(
            conf,
            [
                "deg_per_sweep_inc",
                "deg_ang_down",
                "deg_ang_delta",
                "num_sweeps_levels",
                "max_range",
                "noise",
                "offset_x",
                "offset_y",
                "offset_z",
                "rot_x",
            ],
        )
        if lidar_config != {}:
            self.send_lidar_config(**lidar_config)
            logger.info(f"done sending lidar config., {lidar_config}")
            logger.warning(
                """This way of passing lidar_config is deprecated,
                please wrap the parameters in a sub-dictionary with the key 'lidar_config'.
                Example: GYM_CONF = {'lidar_config':"""
                + str(lidar_config)
                + "}"
            )

    def set_car_config(self, conf: Dict[str, Any]) -> None:
        if "body_style" in conf:
            self.send_car_config(
                conf["body_style"],
                conf["body_rgb"],
                conf["car_name"],
                conf["font_size"],
                conf.get("max_speed", 30.0),  # Default to 30.0 m/s if not specified
            )

    def set_racer_bio(self, conf: Dict[str, Any]) -> None:
        if "bio" in conf:
            self.send_racer_bio(
                conf["racer_name"],
                conf["car_name"],
                conf["bio"],
                conf["country"],
                conf["guid"],
            )

    def on_recv_message(self, message: Dict[str, Any]) -> None:
        if "msg_type" not in message:
            logger.warn("expected msg_type field")
            return
        msg_type = message["msg_type"]
        logger.debug("got message :" + msg_type)
        if msg_type in self.fns:
            self.fns[msg_type](message)
        else:
            logger.warning(f"unknown message type {msg_type}")

    # ------- Env interface ---------- #

    def reset(self) -> None:
        logger.debug("reseting")
        self.reset_complete = False
        self.send_reset_car()
        
        # Wait for reset confirmation from simulator
        timeout = 5.0  # 5 second timeout
        start_time = time.time()
        while not self.reset_complete and (time.time() - start_time) < timeout:
            time.sleep(0.01)
        
        if not self.reset_complete:
            logger.warning("Reset confirmation not received from simulator within timeout")
        
        self.timer.reset()
        self.image_array = np.zeros(self.camera_img_size)
        self.image_array_b = None
        self.last_obs = self.image_array
        self.time_received = time.time()
        self.last_received = self.time_received
        self.hit = "none"
        self.cte = 0.0
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.speed = 0.0
        self.forward_vel = 0.0
        self.over = False
        self.missed_checkpoint = False
        self.dq = False
        self.gyro_x = 0.0
        self.gyro_y = 0.0
        self.gyro_z = 0.0
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_z = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_z = 0.0
        self.lidar = []
        self.current_lap_time = 0.0
        self.last_lap_time = 0.0
        self.lap_count = 0
        self.lap_count_proximity = 0
        self.start_position = None
        self.start_yaw = 0.0
        self.last_proximity_check = False
        self.has_left_start = False

        # car
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        
        # Reset termination tracking
        self.termination_reason = "none"
        self.off_track = False
        self.collision = False
        self.car_fully_crossed = False
        self.reward_components = {
            "done_penalty": 0.0,
            "off_track_penalty": 0.0,
            "collision_penalty": 0.0,
            "speed_reward": 0.0,
            "centering_bonus": 0.0,
            "distance_reward": 0.0,
            "total_reward": 0.0,
            "speed_weight": self.reward_speed_weight,
            "centering_weight": self.reward_centering_weight,
            "distance_weight": self.reward_distance_weight,
        }
        
        self.total_distance = 0.0
        self.last_pos = None
        self.dist_since_last_reward = 0.0

    def get_sensor_size(self) -> Tuple[int, int, int]:
        return self.camera_img_size

    def take_action(self, action: np.ndarray) -> None:
        steer = action[0]
        throttle = action[1]
        
        # Apply speed limiting if max_speed is set
        if self.max_speed < float('inf') and self.speed > self.max_speed:
            # If current speed exceeds limit, reduce throttle to zero
            # This prevents acceleration beyond the speed limit
            throttle = 0.0
        
        self.send_control(steer, throttle)

    def observe(self) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        while self.last_received == self.time_received:
            time.sleep(0.001)

        self.last_received = self.time_received
        observation = self.image_array
        done = self.is_game_over()
        reward = self.calc_reward(done)

        info = {
            "pos": (self.x, self.y, self.z),
            "cte": self.cte,
            "max_cte": self.max_cte,
            "speed": self.speed,
            "forward_vel": self.forward_vel,
            "hit": self.hit,
            "gyro": (self.gyro_x, self.gyro_y, self.gyro_z),
            "accel": (self.accel_x, self.accel_y, self.accel_z),
            "vel": (self.vel_x, self.vel_y, self.vel_z),
            "lidar": (self.lidar),
            "car": (self.roll, self.pitch, self.yaw),
            "last_lap_time": self.last_lap_time,
            "lap_count": self.lap_count,
            "lap_count_proximity": self.lap_count_proximity,
            "termination_reason": self.termination_reason,
            "off_track": self.off_track,
            "collision": self.collision,
            "car_fully_crossed": self.car_fully_crossed,
            "reward_components": self.reward_components.copy(),
            "missed_checkpoint": self.missed_checkpoint,
            "missed_checkpoint": self.missed_checkpoint,
            "dq": self.dq,
            "total_distance": self.total_distance,
        }

        # Add the second image to the dict
        if self.image_array_b is not None:
            info["image_b"] = self.image_array_b

        # self.timer.on_frame()

        return observation, reward, done, info

    def is_game_over(self) -> bool:
        return self.over

    # ------ RL interface ----------- #

    def set_reward_fn(self, reward_fn: Callable[[], float]):
        """
        allow users to set their own reward function
        """
        self.calc_reward = types.MethodType(reward_fn, self)
        logger.debug("custom reward fn set.")

    def calc_reward(self, done: bool) -> float:
        """
        Calculate reward and track components for logging.
        
        Reward components:
        - Done penalty: reward_done_penalty if episode marked as done
        - Off-track penalty: reward_done_penalty if CTE exceeds max_cte.
          CTE (Cross Track Error) is provided by the simulator and represents the 
          lateral distance from the track center. We penalize high CTE to encourage 
          the agent to stay within the track boundaries and follow the optimal line.
        - Collision penalty: -2.0 if car hit something
        - Speed reward: forward_vel when moving forward
        - Centering bonus: spline-based bonus [0, 1] when moving forward
        
        Total reward (when moving forward):
        If reward_lin_combination is True:
            reward = (speed_weight * speed_reward + centering_weight * centering_bonus)
            If speed < 0.1, reward is further scaled by speed to ensure speed matters.
            Weights are normalized to sum to 1.
        Else (default):
            reward = (1.0 - |CTE|/max_cte) * speed_reward
        """
        # Reset components
        self.reward_components = {
            "done_penalty": 0.0,
            "off_track_penalty": 0.0,
            "collision_penalty": 0.0,
            "speed_reward": 0.0,
            "centering_bonus": 0.0,
            "distance_reward": 0.0,
            "total_reward": 0.0,
            "speed_weight": self.reward_speed_weight,
            "centering_weight": self.reward_centering_weight,
            "distance_weight": self.reward_distance_weight,
        }
        
        # Calculate distance reward
        raw_distance = self.dist_since_last_reward
        weighted_distance = raw_distance * self.reward_distance_weight
        self.dist_since_last_reward = 0.0  # Reset accumulator
        self.reward_components["distance_reward"] = raw_distance
        
        reward = 0.0

        if done:
            self.reward_components["done_penalty"] = self.reward_done_penalty
            reward = self.reward_done_penalty
            self.reward_components["total_reward"] = reward
            return reward

        if self.cte > self.max_cte:
            self.reward_components["off_track_penalty"] = self.reward_done_penalty
            reward = self.reward_done_penalty
            self.reward_components["total_reward"] = reward
            return reward

        # Collision
        if self.hit != "none":
            self.reward_components["collision_penalty"] = -2.0
            reward = -2.0
            self.reward_components["total_reward"] = reward
            return reward

        # Going fast close to the center of lane yields best reward
        if self.forward_vel > 0.0:
            speed_reward = self.forward_vel
            
            # 5-point spline for centering bonus:
            # Point 0: x=-1.0 (left edge), y=0.0
            # Point 1: x=-centering_setpoints[0], y=centering_setpoints[1]
            # Point 2: x=0.0 (center), y=1.0
            # Point 3: x=centering_setpoints[0], y=centering_setpoints[1]
            # Point 4: x=1.0 (right edge), y=0.0
            normalized_cte = self.cte / self.max_cte  # Range: [-1, 1]
            x_pos, y_val = self.centering_setpoints
            spline_x = np.array([-1.0, -x_pos, 0.0, x_pos, 1.0])
            spline_y = np.array([0.0, y_val, 1.0, y_val, 0.0])
            centering_spline = interp1d(spline_x, spline_y, kind='cubic', bounds_error=False, fill_value=0.0)
            centering_bonus = float(centering_spline(normalized_cte))
            
            if self.reward_lin_combination:
                # Calculate weighted sum of components
                reward = (self.reward_speed_weight * speed_reward + 
                         self.reward_centering_weight * centering_bonus)
                
                # Scale reward by speed when moving slowly to ensure speed matters
                if speed_reward < 0.1:
                    reward *= speed_reward
            else:
                # Default product-based reward (using spline centering)
                reward = centering_bonus * speed_reward
            
            self.reward_components["speed_reward"] = speed_reward
            self.reward_components["centering_bonus"] = centering_bonus
            
            # Add distance reward
            reward += weighted_distance
            
            self.reward_components["total_reward"] = reward
            return reward

        # In reverse, reward doesn't have centering term (prevents exploits)
        self.reward_components["speed_reward"] = self.forward_vel
        
        # Add distance reward
        reward = self.forward_vel + weighted_distance
        
        self.reward_components["total_reward"] = reward
        return reward

    # ------ Socket interface ----------- #

    def on_telemetry(self, message: Dict[str, Any]) -> None:
        img_string = message["image"]
        image = Image.open(BytesIO(base64.b64decode(img_string)))

        # always update the image_array as the observation loop will hang if not changing.
        self.image_array = np.asarray(image)
        self.time_received = time.time()

        if "image_b" in message:
            img_string_b = message["image_b"]
            image_b = Image.open(BytesIO(base64.b64decode(img_string_b)))
            self.image_array_b = np.asarray(image_b)

        if "pos_x" in message:
            self.x = message["pos_x"]
            self.y = message["pos_y"]
            self.z = message["pos_z"]
            
            # Update distance traveled
            if self.last_pos is not None:
                dist = math.sqrt(
                    (self.x - self.last_pos[0])**2 + 
                    (self.y - self.last_pos[1])**2 + 
                    (self.z - self.last_pos[2])**2
                )
                self.total_distance += dist
                self.dist_since_last_reward += dist
            
            self.last_pos = (self.x, self.y, self.z)

        if "speed" in message:
            self.speed = message["speed"]

        e = [self.pitch * np.pi / 180.0, self.yaw * np.pi / 180.0, self.roll * np.pi / 180.0]
        q = euler_to_quat(e)

        forward = rotate_vec(q, [0.0, 0.0, 1.0])

        # dot
        self.forward_vel = forward[0] * self.vel_x + forward[1] * self.vel_y + forward[2] * self.vel_z

        if "gyro_x" in message:
            self.gyro_x = message["gyro_x"]
            self.gyro_y = message["gyro_y"]
            self.gyro_z = message["gyro_z"]
        if "accel_x" in message:
            self.accel_x = message["accel_x"]
            self.accel_y = message["accel_y"]
            self.accel_z = message["accel_z"]
        if "vel_x" in message:
            self.vel_x = message["vel_x"]
            self.vel_y = message["vel_y"]
            self.vel_z = message["vel_z"]

        if "roll" in message:
            self.roll = message["roll"]
            self.pitch = message["pitch"]
            self.yaw = message["yaw"]

        # Cross track error not always present.
        # Will be missing if path is not setup in the given scene.
        # It should be setup in the 4 scenes available now.
        if "cte" in message:
            self.cte = message["cte"]

        if "lidar" in message:
            self.lidar = self.process_lidar_packet(message["lidar"])

        # don't update hit once session over
        if self.over:
            return

        if "hit" in message:
            self.hit = message["hit"]
            if self.hit != "none":
                logger.info(f"Hit detected: {self.hit}")

        # Check proximity-based lap completion
        self.check_proximity_lap_completion()

        self.determine_episode_over()

    def on_cross_start(self, message: Dict[str, Any]) -> None:
        logger.info(f"crossed start line: lap_time {message['lap_time']}")

    def on_race_start(self, message: Dict[str, Any]) -> None:
        logger.debug("race started")

    def on_race_stop(self, message: Dict[str, Any]) -> None:
        logger.debug("race stoped")

    def on_missed_checkpoint(self, message: Dict[str, Any]) -> None:
        logger.info("racer missed checkpoint")
        self.missed_checkpoint = True

    def on_DQ(self, message: Dict[str, Any]) -> None:
        logger.info("racer DQ")
        self.dq = True

    def on_ping(self, message: Dict[str, Any]) -> None:
        """
        no reply needed at this point. Server sends these as a keep alive to make sure clients haven't gone away.
        """
        pass

    def set_episode_over_fn(self, ep_over_fn: Callable[[], bool]):
        """
        allow userd to define their own episode over function
        """
        self.determine_episode_over = types.MethodType(ep_over_fn, self)
        logger.debug("custom ep_over fn set.")

    def determine_episode_over(self):
        """
        Determine if episode is over and track the reason.
        
        Termination conditions (in priority order):
        1. Ignore large CTE if near startup (startup noise)
        2. Car crosses centerline (CTE exceeds max_cte)
        3. Entire car has fully crossed the line (CTE > max_cte for sustained period)
        4. Physical collision with object
        5. Missed checkpoint
        6. Disqualified
        """
        # We have a few initial frames on start that are sometimes very large CTE 
        # when it's behind the path just slightly. We ignore those.
        # Reduced from 2x to 1.5x for quicker termination
        if math.fabs(self.cte) > 1.5 * self.max_cte:
            return
        
        # Check if car's center point has crossed the line
        if math.fabs(self.cte) > self.max_cte:
            self.off_track = True
            # For "entire car crossed" logic: use a stricter threshold
            # If CTE is significantly beyond max, the entire car has crossed
            if math.fabs(self.cte) > self.max_cte * 1.2:
                self.car_fully_crossed = True
            
            if not self.over:
                logger.debug(f"game over: cte {self.cte}, max_cte {self.max_cte}")
                if self.car_fully_crossed:
                    logger.debug(f"Car fully crossed the line: CTE={self.cte:.2f}")
                    self.termination_reason = "car_fully_crossed"
                else:
                    self.termination_reason = "off_track"
            self.over = True
            return
        
        # Physical collision
        if self.hit != "none":
            self.collision = True
            if not self.over:
                logger.debug(f"game over: hit {self.hit}")
                self.termination_reason = "collision"
            self.over = True
            return
        
        # Missed checkpoint
        if self.missed_checkpoint:
            if not self.over:
                logger.debug("game over: missed checkpoint")
                self.termination_reason = "missed_checkpoint"
            self.over = True
            return
        
        # Disqualified
        if self.dq:
            if not self.over:
                logger.debug("game over: disqualified")
                self.termination_reason = "disqualified"
            self.over = True
            return

        # Disable reset in RACE mode
        if os.environ.get("RACE") == "True":
            self.over = False

    def on_scene_selection_ready(self, message: Dict[str, Any]) -> None:
        logger.debug("SceneSelectionReady")
        self.send_get_scene_names()

    def on_car_loaded(self, message: Dict[str, Any]) -> None:
        logger.debug("car loaded")
        self.loaded = True
        # Enable hand brake, so the car doesn't move
        self.send_control(0, 0, 1.0)
        self.on_need_car_config({})

    def on_car_reset_done(self, message: Dict[str, Any]) -> None:
        """Handle confirmation that car reset is complete"""
        logger.debug("car reset complete (msg received)")
        self.reset_complete = True

    def on_recv_scene_names(self, message: Dict[str, Any]) -> None:
        if message:
            names = message["scene_names"]
            logger.debug(f"SceneNames: {names}")
            print("loading scene", self.SceneToLoad)
            if self.SceneToLoad in names:
                self.send_load_scene(self.SceneToLoad)
            else:
                raise ValueError(f"Scene name {self.SceneToLoad} not in scene list {names}")

    def send_control(self, steer: float, throttle: float, brake: float = 0.0) -> None:
        """
        Send command to simulator.

        :param steer: desired steering
        :param throttle: desired throttle
        :param brake: whether to activate or not hand brake
            (can be a continuous value)
        """
        if not self.loaded:
            return
        msg = {
            "msg_type": "control",
            "steering": str(steer),
            "throttle": str(throttle),
            "brake": str(brake),
        }
        self.queue_message(msg)

    def send_reset_car(self) -> None:
        msg = {"msg_type": "reset_car"}
        
        # Add random spawn parameters if enabled
        if self.random_spawn_enabled:
            msg["random_spawn_enabled"] = "true"
            msg["random_spawn_max_cte_offset"] = str(self.random_spawn_max_cte_offset)
            msg["random_spawn_max_rotation_offset"] = str(self.random_spawn_max_rotation_offset)
            #print(f"[DonkeyEnv] Sending reset_car with random spawn: cte_offset={self.random_spawn_max_cte_offset}, rotation_offset={self.random_spawn_max_rotation_offset}")
        else:
            pass
            #print(f"[DonkeyEnv] Sending reset_car WITHOUT random spawn (random_spawn_enabled={self.random_spawn_enabled})")
        
        self.queue_message(msg)

    def send_get_scene_names(self) -> None:
        msg = {"msg_type": "get_scene_names"}
        self.queue_message(msg)

    def send_load_scene(self, scene_name: str) -> None:
        msg = {"msg_type": "load_scene", "scene_name": scene_name}
        self.queue_message(msg)

    def send_exit_scene(self) -> None:
        msg = {"msg_type": "exit_scene"}
        self.queue_message(msg)

    def send_car_config(
        self,
        body_style: str = "donkey",
        body_rgb: Tuple[int, int, int] = (255, 255, 255),
        car_name: str = "car",
        font_size: int = 100,
        max_speed: float = 30.0,
    ):
        """
        # body_style = "donkey" | "bare" | "car01" | "f1" | "cybertruck"
        # body_rgb  = (128, 128, 128) tuple of ints
        # car_name = "string less than 64 char"
        # max_speed = maximum speed in m/s (default 30.0)
        """
        assert isinstance(body_style, str)
        assert isinstance(body_rgb, list) or isinstance(body_rgb, tuple)
        assert len(body_rgb) == 3
        assert isinstance(car_name, str)
        assert isinstance(font_size, int) or isinstance(font_size, str)

        msg = {
            "msg_type": "car_config",
            "body_style": body_style,
            "body_r": str(body_rgb[0]),
            "body_g": str(body_rgb[1]),
            "body_b": str(body_rgb[2]),
            "car_name": car_name,
            "font_size": str(font_size),
            "max_speed": str(max_speed),
        }
        self.blocking_send(msg)
        time.sleep(0.1)

    def send_racer_bio(self, racer_name: str, car_name: str, bio: str, country: str, guid: str) -> None:
        # body_style = "donkey" | "bare" | "car01" choice of string
        # body_rgb  = (128, 128, 128) tuple of ints
        # car_name = "string less than 64 char"
        # guid = "some random string"
        msg = {
            "msg_type": "racer_info",
            "racer_name": racer_name,
            "car_name": car_name,
            "bio": bio,
            "country": country,
            "guid": guid,
        }
        self.blocking_send(msg)
        time.sleep(0.1)

    def send_cam_config(
        self,
        msg_type: str = "cam_config",
        img_w: int = 0,
        img_h: int = 0,
        img_d: int = 0,
        img_enc: Union[str, int] = 0,  # 0 is default value
        fov: int = 0,
        fish_eye_x: float = 0.0,
        fish_eye_y: float = 0.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
        offset_z: float = 0.0,
        rot_x: float = 0.0,
        rot_y: float = 0.0,
        rot_z: float = 0.0,
    ) -> None:
        """Camera config
        set any field to Zero to get the default camera setting.
        offset_x moves camera left/right
        offset_y moves camera up/down
        offset_z moves camera forward/back
        rot_x will rotate the camera
        with fish_eye_x/y == 0.0 then you get no distortion
        img_enc can be one of JPG|PNG|TGA
        """
        msg = {
            "msg_type": msg_type,
            "fov": str(fov),
            "fish_eye_x": str(fish_eye_x),
            "fish_eye_y": str(fish_eye_y),
            "img_w": str(img_w),
            "img_h": str(img_h),
            "img_d": str(img_d),
            "img_enc": str(img_enc),
            "offset_x": str(offset_x),
            "offset_y": str(offset_y),
            "offset_z": str(offset_z),
            "rot_x": str(rot_x),
            "rot_y": str(rot_y),
            "rot_z": str(rot_z),
        }
        self.blocking_send(msg)
        time.sleep(0.1)

    def send_lidar_config(
        self,
        deg_per_sweep_inc: float = 2.0,
        deg_ang_down: float = 0.0,
        deg_ang_delta: float = -1.0,
        num_sweeps_levels: int = 1,
        max_range: float = 50.0,
        noise: float = 0.5,
        offset_x: float = 0.0,
        offset_y: float = 0.5,
        offset_z: float = 0.5,
        rot_x: float = 0.0,
    ):
        """Lidar config
        offset_x moves lidar left/right
        the offset_y moves lidar up/down
        the offset_z moves lidar forward/back
        deg_per_sweep_inc : as the ray sweeps around, how many degrees does it advance per sample (int)
        deg_ang_down : what is the starting angle for the initial sweep compared to the forward vector
        deg_ang_delta : what angle change between sweeps
        num_sweeps_levels : how many complete 360 sweeps (int)
        max_range : what it max distance we will register a hit
        noise : what is the scalar on the perlin noise applied to point position

        Here's some sample settings that similate a more sophisticated lidar:
        msg = '{ "msg_type" : "lidar_config",
        "degPerSweepInc" : "2.0", "degAngDown" : "25", "degAngDelta" : "-1.0",
        "numSweepsLevels" : "25", "maxRange" : "50.0", "noise" : "0.2",
        "offset_x" : "0.0", "offset_y" : "1.0", "offset_z" : "1.0", "rot_x" : "0.0" }'
        And here's some sample settings that similate a simple RpLidar A2 one level horizontal scan.
        msg = '{ "msg_type" : "lidar_config", "degPerSweepInc" : "2.0",
        "degAngDown" : "0.0", "degAngDelta" : "-1.0", "numSweepsLevels" : "1",
        "maxRange" : "50.0", "noise" : "0.4",
        "offset_x" : "0.0", "offset_y" : "0.5", "offset_z" : "0.5", "rot_x" : "0.0" }'
        """
        msg = {
            "msg_type": "lidar_config",
            "degPerSweepInc": str(deg_per_sweep_inc),
            "degAngDown": str(deg_ang_down),
            "degAngDelta": str(deg_ang_delta),
            "numSweepsLevels": str(num_sweeps_levels),
            "maxRange": str(max_range),
            "noise": str(noise),
            "offset_x": str(offset_x),
            "offset_y": str(offset_y),
            "offset_z": str(offset_z),
            "rot_x": str(rot_x),
        }
        self.blocking_send(msg)
        time.sleep(0.1)

        self.lidar_deg_per_sweep_inc = float(deg_per_sweep_inc)
        self.lidar_num_sweep_levels = int(num_sweeps_levels)
        self.lidar_deg_ang_delta = float(deg_ang_delta)

    def process_lidar_packet(self, lidar_info: List[Dict[str, float]]) -> np.ndarray:
        point_per_sweep = int(360 / self.lidar_deg_per_sweep_inc)
        points_num = round(abs(self.lidar_num_sweep_levels * point_per_sweep))
        reconstructed_lidar_info = [-1 for _ in range(points_num)]  # we chose -1 to be the "None" value

        if lidar_info is not None:
            for point in lidar_info:
                rx = point["rx"]
                ry = point["ry"]
                d = point["d"]

                x_index = round(abs(rx / self.lidar_deg_per_sweep_inc))
                y_index = round(abs(ry / self.lidar_deg_ang_delta))

                reconstructed_lidar_info[point_per_sweep * y_index + x_index] = d

        return np.array(reconstructed_lidar_info)

    def blocking_send(self, msg: Dict[str, Any]) -> None:
        if self.client is None:
            logger.debug(f"skipping: \n {msg}")
            return

        logger.debug(f"blocking send \n {msg}")
        self.client.send_now(msg)

    def queue_message(self, msg: Dict[str, Any]) -> None:
        if self.client is None:
            logger.debug(f"skipping: \n {msg}")
            return

        logger.debug(f"sending \n {msg}")
        self.client.queue_message(msg)
