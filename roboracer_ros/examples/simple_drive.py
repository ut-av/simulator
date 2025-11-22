#!/usr/bin/env python3
"""
Simple example showing basic car control.

This example demonstrates the most basic use case:
driving a car forward, stopping it, and resetting.
"""

import time
import logging
from roboracer_ros import CarController

# Setup logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Run the simple drive example."""
    logger.info("Starting simple drive example")

    # Create a controller for car 0
    controller = CarController(car_id=0)

    try:
        # Drive forward at 50% throttle for 3 seconds
        logger.info("Driving forward at 50% throttle")
        controller.drive(throttle=0.5)

        for i in range(3):
            time.sleep(1)
            print(f"  {3-i} seconds remaining...")

        # Stop the car
        logger.info("Stopping car")
        controller.stop()
        time.sleep(0.5)

        # Print final state
        print("\nFinal car state:")
        print(controller.get_state())

        logger.info("Example completed successfully")

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        controller.stop()

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)

    finally:
        # Always clean up
        controller.client.shutdown()


if __name__ == "__main__":
    main()
