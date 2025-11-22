#!/usr/bin/env python3
"""
Autonomous control loop example.

This example shows how to run a continuous control loop
that performs repeated driving patterns.
"""

import time
import logging
import math
from roboracer_ros import CarController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def drive_square(controller: CarController, side_length: float = 5.0, velocity: float = 0.5) -> None:
    """
    Drive the car in a square pattern.

    Args:
        controller: CarController instance
        side_length: Length of square sides in meters (approximate)
        velocity: Throttle level (0.0-1.0)
    """
    logger.info(f"Driving in a square with {side_length}m sides at velocity {velocity}")

    time_per_side = side_length / velocity  # Approximate time to drive one side

    for i in range(4):
        logger.info(f"Driving side {i+1}/4")

        # Drive straight
        controller.drive(throttle=velocity, steering=0.0)
        time.sleep(time_per_side)

        # Turn right (approximately 90 degrees)
        logger.info("Turning right")
        controller.turn_right(throttle=0.2, angle=0.8)
        time.sleep(0.8)

    # Stop
    controller.stop()
    logger.info("Square pattern complete")


def drive_circle(controller: CarController, radius: float = 3.0, velocity: float = 0.5,
                 duration: float = 10.0) -> None:
    """
    Drive the car in a circular pattern.

    Args:
        controller: CarController instance
        radius: Turn radius in meters
        velocity: Throttle level (0.0-1.0)
        duration: How long to drive (seconds)
    """
    logger.info(f"Driving in circle with {radius}m radius at velocity {velocity}")

    # Curvature = 1/radius
    curvature = 1.0 / radius

    start_time = time.time()
    while time.time() - start_time < duration:
        # Use curvature-based control to drive in a circle
        controller.follow_curvature(velocity=velocity, curvature=curvature)
        time.sleep(0.1)

    # Stop
    controller.stop()
    logger.info("Circle pattern complete")


def drive_figure8(controller: CarController, radius: float = 2.0, velocity: float = 0.5) -> None:
    """
    Drive the car in a figure-8 pattern.

    Args:
        controller: CarController instance
        radius: Radius of each circle in figure-8
        velocity: Throttle level (0.0-1.0)
    """
    logger.info(f"Driving figure-8 with {radius}m radius loops")

    curvature = 1.0 / radius
    cycle_time = 2 * math.pi * radius / velocity  # Time for one complete circle

    # First circle (left)
    logger.info("Driving left loop")
    start = time.time()
    while time.time() - start < cycle_time / 2:
        controller.follow_curvature(velocity=velocity, curvature=curvature)
        time.sleep(0.1)

    # Second circle (right) - negative curvature
    logger.info("Driving right loop")
    start = time.time()
    while time.time() - start < cycle_time / 2:
        controller.follow_curvature(velocity=velocity, curvature=-curvature)
        time.sleep(0.1)

    # Stop
    controller.stop()
    logger.info("Figure-8 pattern complete")


def main():
    """Run autonomous control examples."""
    logger.info("Starting autonomous control example")

    controller = CarController(car_id=0)

    try:
        # Example 1: Simple forward and stop
        logger.info("\n=== Example 1: Simple Forward Drive ===")
        controller.drive(throttle=0.5, steering=0.0)
        time.sleep(2)
        controller.stop()
        time.sleep(1)

        # Example 2: Drive in a circle
        logger.info("\n=== Example 2: Circular Drive ===")
        drive_circle(controller, radius=3.0, velocity=0.4, duration=8.0)
        time.sleep(1)

        # Example 3: Drive in a square
        logger.info("\n=== Example 3: Square Drive ===")
        drive_square(controller, side_length=4.0, velocity=0.4)
        time.sleep(1)

        logger.info("All examples completed successfully")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        controller.stop()

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        controller.stop()

    finally:
        controller.client.shutdown()


if __name__ == "__main__":
    main()
