"""
CarController - High-level interface for controlling cars in the simulator.

Provides convenient methods for common control operations.
"""

import logging
import math
from typing import Tuple, Optional
from .sim_api_client import SimApiClient

logger = logging.getLogger(__name__)


class CarController:
    """
    High-level controller for simulated cars.

    Wraps SimApiClient to provide convenient methods for car control.
    """

    def __init__(self, car_id: int = 0, client: Optional[SimApiClient] = None):
        """
        Initialize a CarController.

        Args:
            car_id: ID of the car to control
            client: SimApiClient instance. If None, creates a new one.
        """
        self.car_id = car_id
        self.client = client or SimApiClient()

        # Current control state
        self.throttle = 0.0
        self.steering = 0.0
        self.brake = 0.0

    def set_throttle(self, throttle: float) -> None:
        """
        Set throttle level.

        Args:
            throttle: Value from 0.0 (no throttle) to 1.0 (full throttle)
        """
        self.throttle = max(0.0, min(1.0, float(throttle)))

    def set_steering(self, steering: float) -> None:
        """
        Set steering angle.

        Args:
            steering: Value from -1.0 (full left) to 1.0 (full right)
        """
        self.steering = max(-1.0, min(1.0, float(steering)))

    def set_brake(self, brake: float) -> None:
        """
        Set brake level.

        Args:
            brake: Value from 0.0 (no brake) to 1.0 (full brake)
        """
        self.brake = max(0.0, min(1.0, float(brake)))

    def send_control(self) -> bool:
        """
        Send current control state to the simulator.

        Returns:
            True if control was sent successfully
        """
        return self.client.control_car(
            self.car_id,
            self.steering,
            self.throttle,
            self.brake
        )

    def drive(self, throttle: float, steering: float = 0.0) -> bool:
        """
        Drive the car with specified throttle and steering.

        Args:
            throttle: Throttle level (0.0-1.0)
            steering: Steering level (-1.0-1.0)

        Returns:
            True if control was sent successfully
        """
        self.set_throttle(throttle)
        self.set_steering(steering)
        self.set_brake(0.0)
        return self.send_control()

    def stop(self) -> bool:
        """
        Stop the car (apply full brake).

        Returns:
            True if control was sent successfully
        """
        self.set_throttle(0.0)
        self.set_steering(0.0)
        self.set_brake(1.0)
        return self.send_control()

    def turn_left(self, throttle: float = 0.5, angle: float = 0.5) -> bool:
        """
        Turn the car to the left.

        Args:
            throttle: Throttle level (0.0-1.0)
            angle: Steering angle (-1.0-1.0)

        Returns:
            True if control was sent successfully
        """
        return self.drive(throttle, -abs(angle))

    def turn_right(self, throttle: float = 0.5, angle: float = 0.5) -> bool:
        """
        Turn the car to the right.

        Args:
            throttle: Throttle level (0.0-1.0)
            angle: Steering angle (-1.0-1.0)

        Returns:
            True if control was sent successfully
        """
        return self.drive(throttle, abs(angle))

    def set_position(self, x: float, y: float, z: float = 0.0,
                    roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> bool:
        """
        Set the car's position and rotation.

        Args:
            x, y, z: Position coordinates
            roll, pitch, yaw: Rotation angles in radians

        Returns:
            True if position was set successfully
        """
        return self.client.set_car_position(
            self.car_id, x, y, z, roll, pitch, yaw
        )

    def reset(self) -> bool:
        """
        Reset the car to its initial state.

        Returns:
            True if reset was successful
        """
        self.throttle = 0.0
        self.steering = 0.0
        self.brake = 0.0
        return self.client.reset_car(self.car_id)

    def follow_curvature(self, velocity: float, curvature: float) -> bool:
        """
        Follow a curved path using curvature-based control.

        This is similar to Ackermann steering where curvature = 1/turn_radius.

        Args:
            velocity: Desired velocity (maps to throttle)
            curvature: Path curvature (1/meters)
                      Positive = left turn, Negative = right turn

        Returns:
            True if control was sent successfully
        """
        # Map velocity to throttle (assuming linear mapping)
        throttle = max(0.0, min(1.0, abs(velocity)))

        # Map curvature to steering angle (clamped to -1 to 1)
        # Assuming max curvature of 1.0 (1 meter radius)
        steering = max(-1.0, min(1.0, curvature))

        return self.drive(throttle, steering)

    def get_state(self) -> dict:
        """
        Get the current control state.

        Returns:
            Dictionary with current throttle, steering, and brake values
        """
        return {
            "car_id": self.car_id,
            "throttle": self.throttle,
            "steering": self.steering,
            "brake": self.brake
        }

    def __repr__(self) -> str:
        state = self.get_state()
        return (f"CarController(car_id={state['car_id']}, "
                f"throttle={state['throttle']:.2f}, "
                f"steering={state['steering']:.2f}, "
                f"brake={state['brake']:.2f})")
