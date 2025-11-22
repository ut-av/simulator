"""
roboracer_ros - ROS interface for controlling RoboRacer simulation.

This package provides a Python interface to control cars in the RoboRacer 
simulation via ROS topics. It supports both interactive and single-command modes.
"""

__version__ = "0.1.0"
__author__ = "RoboRacer Team"

from .sim_api_client import SimApiClient
from .car_controller import CarController

__all__ = ["SimApiClient", "CarController"]
