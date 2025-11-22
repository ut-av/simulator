"""
Core ROS message definitions for simulator control.

Mirrors the message structures defined in ROSSimApi.cs
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import json


@dataclass
class CarControlMsg:
    """Control message for car throttle, steering, and brake"""
    car_id: int
    steering: float
    throttle: float
    brake: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class SetCarPositionMsg:
    """Message to set car position and rotation"""
    car_id: int
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class SpawnCarMsg:
    """Message to spawn a new car in the simulation"""
    car_id: int
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class DespawnCarMsg:
    """Message to despawn a car from the simulation"""
    car_id: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class ResetCarMsg:
    """Message to reset a car to its initial state"""
    car_id: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class SimResetMsg:
    """Message to reset the entire simulation"""
    reset_all_cars: bool = True
    reload_scene: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class SimStateMsg:
    """Current state of the simulation"""
    active_car_count: int
    simulation_time: float
    time_scale: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
