#!/usr/bin/env python3
"""
Interactive demo showing the Python API in an interactive environment.

This script demonstrates various car control features interactively.
"""

import time
import logging
import sys
from roboracer_ros import CarController, SimApiClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def print_menu():
    """Print the interactive menu."""
    print("\n" + "=" * 50)
    print("RoboRacer Interactive Demo")
    print("=" * 50)
    print("1. Simple forward drive")
    print("2. Turn left")
    print("3. Turn right")
    print("4. Drive in a circle")
    print("5. Drive in a square")
    print("6. Set position")
    print("7. Reset car")
    print("8. Show car state")
    print("9. Control multiple cars")
    print("0. Exit")
    print("=" * 50)


def drive_circle_interactive(controller: CarController):
    """Interactive circle driving demo."""
    print("\nDriving in a circle...")
    print("(Circle will complete in ~10 seconds)")

    curvature = 0.5  # 1/2 meter radius
    for i in range(10):
        controller.follow_curvature(velocity=0.4, curvature=curvature)
        time.sleep(1)
        print(f"  {10-i-1} seconds remaining...")

    controller.stop()
    print("Circle complete!")


def drive_square_interactive(controller: CarController):
    """Interactive square driving demo."""
    print("\nDriving in a square...")

    for side in range(4):
        print(f"Side {side+1}/4: Driving straight")
        controller.drive(throttle=0.4, steering=0.0)
        time.sleep(2)

        print(f"Side {side+1}/4: Turning")
        controller.turn_right(throttle=0.2, angle=0.8)
        time.sleep(0.8)

    controller.stop()
    print("Square complete!")


def main():
    """Run the interactive demo."""
    logger.info("Starting interactive demo")

    controller = CarController(car_id=0)

    try:
        while True:
            print_menu()

            try:
                choice = input("Enter choice (0-9): ").strip()

                if choice == "1":
                    print("\nDriving forward at 50% throttle...")
                    controller.drive(throttle=0.5, steering=0.0)
                    time.sleep(3)
                    controller.stop()
                    print("Stopped!")

                elif choice == "2":
                    print("\nTurning left...")
                    controller.turn_left(throttle=0.4, angle=0.5)
                    time.sleep(2)
                    controller.stop()
                    print("Stopped!")

                elif choice == "3":
                    print("\nTurning right...")
                    controller.turn_right(throttle=0.4, angle=0.5)
                    time.sleep(2)
                    controller.stop()
                    print("Stopped!")

                elif choice == "4":
                    drive_circle_interactive(controller)

                elif choice == "5":
                    drive_square_interactive(controller)

                elif choice == "6":
                    try:
                        x = float(input("Enter X coordinate: "))
                        y = float(input("Enter Y coordinate: "))
                        z = float(input("Enter Z coordinate (default 0): ") or "0")
                        controller.set_position(x, y, z)
                        print(f"Position set to ({x}, {y}, {z})")
                    except ValueError:
                        print("Invalid input!")

                elif choice == "7":
                    print("\nResetting car...")
                    controller.reset()
                    print("Car reset!")

                elif choice == "8":
                    print("\nCurrent car state:")
                    state = controller.get_state()
                    for key, value in state.items():
                        if isinstance(value, float):
                            print(f"  {key}: {value:.2f}")
                        else:
                            print(f"  {key}: {value}")

                elif choice == "9":
                    print("\nMultiple car control demo")
                    print("This would control multiple cars...")
                    print("Creating controllers for cars 0 and 1...")

                    client = SimApiClient()
                    car0 = CarController(car_id=0, client=client)
                    car1 = CarController(car_id=1, client=client)

                    print("Car 0: Driving forward")
                    car0.drive(throttle=0.5, steering=0.0)
                    time.sleep(1)

                    print("Car 1: Turning left")
                    car1.turn_left(throttle=0.5, angle=0.3)
                    time.sleep(1)

                    print("Stopping both cars")
                    car0.stop()
                    car1.stop()

                    client.shutdown()

                elif choice == "0":
                    print("\nExiting...")
                    controller.stop()
                    break

                else:
                    print("Invalid choice. Please try again.")

            except KeyboardInterrupt:
                print("\nInterrupt received. Enter 'quit' to exit or try another command.")
                controller.stop()

            time.sleep(0.5)

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        controller.stop()

    finally:
        controller.client.shutdown()
        logger.info("Demo completed")


if __name__ == "__main__":
    main()
