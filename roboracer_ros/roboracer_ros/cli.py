"""
Command-line interface for controlling cars in the RoboRacer simulator.

Supports both interactive and single-command modes.
"""

import logging
import argparse
import sys
from typing import Optional
import time
from .car_controller import CarController
from .sim_api_client import SimApiClient

logger = logging.getLogger(__name__)


class RoboRacerCLI:
    """Interactive command-line interface for car control."""

    # Available commands
    COMMANDS = {
        "help": "Show this help message",
        "drive": "drive <throttle> [steering] - Drive car",
        "throttle": "throttle <value> - Set throttle (0.0-1.0)",
        "steering": "steering <value> - Set steering (-1.0-1.0)",
        "brake": "brake <value> - Apply brake (0.0-1.0)",
        "stop": "Stop the car",
        "reset": "Reset car to initial state",
        "forward": "forward [throttle] - Drive forward",
        "backward": "backward [throttle] - Drive backward",
        "left": "left [throttle] [angle] - Turn left",
        "right": "right [throttle] [angle] - Turn right",
        "position": "position <x> <y> [z] - Set car position",
        "state": "Show current car state",
        "quit": "Exit interactive mode",
    }

    def __init__(self, car_id: int = 0):
        """
        Initialize the CLI.

        Args:
            car_id: ID of the car to control
        """
        self.car_id = car_id
        self.client = SimApiClient()
        self.controller = CarController(car_id, self.client)
        self.running = True

    def print_welcome(self) -> None:
        """Print welcome message and instructions."""
        print("\n" + "=" * 60)
        print("RoboRacer Car Control Interface")
        print("=" * 60)
        print(f"Controlling car ID: {self.car_id}")
        print("\nType 'help' for available commands or 'quit' to exit.\n")

    def print_help(self) -> None:
        """Print help message."""
        print("\n" + "-" * 60)
        print("Available Commands:")
        print("-" * 60)
        for cmd, desc in self.COMMANDS.items():
            print(f"  {desc}")
        print("-" * 60 + "\n")

    def print_state(self) -> None:
        """Print current car state."""
        state = self.controller.get_state()
        print(f"\nCurrent State:")
        print(f"  Car ID:    {state['car_id']}")
        print(f"  Throttle:  {state['throttle']:.2f}")
        print(f"  Steering:  {state['steering']:.2f}")
        print(f"  Brake:     {state['brake']:.2f}\n")

    def handle_command(self, line: str) -> bool:
        """
        Handle a single command.

        Args:
            line: Command string

        Returns:
            False if should quit, True otherwise
        """
        if not line.strip():
            return True

        parts = line.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]

        try:
            if cmd == "help":
                self.print_help()
            elif cmd == "drive":
                if not args:
                    print("Error: drive requires at least throttle value")
                    return True
                throttle = float(args[0])
                steering = float(args[1]) if len(args) > 1 else 0.0
                self.controller.drive(throttle, steering)
                print(f"Driving: throttle={throttle:.2f}, steering={steering:.2f}")
            elif cmd == "throttle":
                if not args:
                    print("Error: throttle requires a value")
                    return True
                throttle = float(args[0])
                self.controller.set_throttle(throttle)
                self.controller.send_control()
                print(f"Throttle set to {throttle:.2f}")
            elif cmd == "steering":
                if not args:
                    print("Error: steering requires a value")
                    return True
                steering = float(args[0])
                self.controller.set_steering(steering)
                self.controller.send_control()
                print(f"Steering set to {steering:.2f}")
            elif cmd == "brake":
                if not args:
                    print("Error: brake requires a value")
                    return True
                brake = float(args[0])
                self.controller.set_brake(brake)
                self.controller.send_control()
                print(f"Brake set to {brake:.2f}")
            elif cmd == "stop":
                self.controller.stop()
                print("Car stopped")
            elif cmd == "reset":
                self.controller.reset()
                print("Car reset to initial state")
            elif cmd == "forward":
                throttle = float(args[0]) if args else 0.5
                self.controller.drive(throttle, 0.0)
                print(f"Driving forward at throttle={throttle:.2f}")
            elif cmd == "backward":
                throttle = float(args[0]) if args else 0.5
                self.controller.drive(-throttle, 0.0)
                print(f"Driving backward at throttle={throttle:.2f}")
            elif cmd == "left":
                throttle = float(args[0]) if args else 0.5
                angle = float(args[1]) if len(args) > 1 else 0.5
                self.controller.turn_left(throttle, angle)
                print(f"Turning left: throttle={throttle:.2f}, angle={angle:.2f}")
            elif cmd == "right":
                throttle = float(args[0]) if args else 0.5
                angle = float(args[1]) if len(args) > 1 else 0.5
                self.controller.turn_right(throttle, angle)
                print(f"Turning right: throttle={throttle:.2f}, angle={angle:.2f}")
            elif cmd == "position":
                if len(args) < 2:
                    print("Error: position requires x and y coordinates")
                    return True
                x, y = float(args[0]), float(args[1])
                z = float(args[2]) if len(args) > 2 else 0.0
                self.controller.set_position(x, y, z)
                print(f"Set position to ({x:.2f}, {y:.2f}, {z:.2f})")
            elif cmd == "state":
                self.print_state()
            elif cmd == "quit":
                self.running = False
                print("Exiting...")
                return False
            else:
                print(f"Unknown command: {cmd}. Type 'help' for available commands.")
        except ValueError as e:
            print(f"Error: Invalid argument - {e}")
        except Exception as e:
            print(f"Error: {e}")
            logger.exception("Command execution error")

        return True

    def run_interactive(self) -> None:
        """Run interactive command loop."""
        self.print_welcome()

        while self.running:
            try:
                line = input("roboracer> ").strip()
                if not self.handle_command(line):
                    break
            except KeyboardInterrupt:
                print("\nInterrupt received. Type 'quit' to exit.")
            except EOFError:
                print("\nEnd of input")
                break

        self.cleanup()

    def run_single_command(self, cmd_line: str) -> None:
        """
        Execute a single command and exit.

        Args:
            cmd_line: Command string to execute
        """
        success = self.handle_command(cmd_line)
        self.cleanup()

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.client:
            self.client.shutdown()


def create_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="roboracer-cli",
        description="Control RoboRacer simulation cars via ROS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  roboracer-cli

  # Single command - drive the car
  roboracer-cli drive 0.5 0.2

  # Single command - stop the car
  roboracer-cli stop

  # Control specific car
  roboracer-cli --car 1 drive 0.7

  # Set verbose logging
  roboracer-cli -v drive 0.5
        """
    )

    parser.add_argument(
        "command",
        nargs="*",
        help="Command to execute (optional). If not provided, enters interactive mode."
    )

    parser.add_argument(
        "-c", "--car",
        type=int,
        default=0,
        help="Car ID to control (default: 0)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    return parser


def setup_logging(level: str) -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else args.log_level
    setup_logging(log_level)

    try:
        cli = RoboRacerCLI(car_id=args.car)

        if args.command:
            # Single command mode
            command_str = " ".join(args.command)
            logger.info(f"Executing command: {command_str}")
            cli.run_single_command(command_str)
        else:
            # Interactive mode
            cli.run_interactive()

        return 0

    except KeyboardInterrupt:
        print("\nInterrupt received. Exiting.")
        return 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
