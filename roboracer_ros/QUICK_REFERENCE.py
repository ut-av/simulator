#!/usr/bin/env python3
"""
Quick reference - RoboRacer ROS Package Commands

This is a handy reference for the most common commands.
For more details, see README.md or run: roboracer-cli help
"""

QUICK_REFERENCE = """
╔════════════════════════════════════════════════════════════════════╗
║                  RoboRacer ROS Control Quick Reference            ║
╚════════════════════════════════════════════════════════════════════╝

🚀 QUICK START
  pip install -e roboracer_ros
  roboracer-cli

📋 COMMON COMMANDS

  Drive Commands:
    roboracer-cli drive 0.5              # Drive forward at 50% throttle
    roboracer-cli drive 0.5 0.2          # Drive with left steering
    roboracer-cli forward 0.5            # Drive forward
    roboracer-cli backward 0.5           # Drive backward
    roboracer-cli left 0.5               # Turn left
    roboracer-cli right 0.5              # Turn right
    roboracer-cli stop                   # Stop car

  Control Parameters:
    roboracer-cli throttle 0.7           # Set throttle to 70%
    roboracer-cli steering 0.3           # Steer right 30%
    roboracer-cli brake 0.5              # Apply 50% brake

  Position:
    roboracer-cli position 10 5 0        # Teleport to (10, 5, 0)

  Car Management:
    roboracer-cli reset                  # Reset car to start
    roboracer-cli state                  # Show car state
    roboracer-cli --car 1 drive 0.5      # Control car ID 1

  Interactive Mode:
    roboracer-cli                        # Start interactive shell
    roboracer> help
    roboracer> drive 0.5
    roboracer> stop
    roboracer> quit

🐍 PYTHON API QUICK START

  from roboracer_ros import CarController

  controller = CarController(car_id=0)
  
  # Drive
  controller.drive(throttle=0.5, steering=0.2)
  controller.stop()
  
  # Position
  controller.set_position(x=10, y=5, z=0)
  
  # Advanced
  controller.follow_curvature(velocity=0.5, curvature=0.5)
  controller.reset()
  
  # Cleanup
  controller.client.shutdown()

📖 DOCUMENTATION

  README.md                 - Full documentation
  QUICKSTART.md             - 5-minute getting started
  DEVELOPMENT.md            - Developer guide
  IMPLEMENTATION_SUMMARY.md - What was built
  examples/README.md        - Examples explanation

💻 EXAMPLES

  python examples/simple_drive.py        # Basic drive example
  python examples/autonomous_loop.py     # Patterns: circle, square, figure-8
  python examples/interactive_demo.py    # Interactive menu demo

⚙️ PACKAGE STRUCTURE

  roboracer_ros/
    cli.py                - Command-line interface
    car_controller.py     - High-level car control
    sim_api_client.py     - Low-level ROS interface
    messages.py           - Message definitions
  
  examples/
    simple_drive.py       - Simplest example
    autonomous_loop.py    - Advanced patterns
    interactive_demo.py   - Interactive menu
  
  tests/
    test_car_controller.py
    test_sim_api_client.py

🔧 INSTALLATION & SETUP

  # Install package
  pip install -e roboracer_ros
  
  # Run tests
  pytest
  
  # Enable debug logging
  roboracer-cli -v drive 0.5
  
  # Install development tools
  pip install -e "roboracer_ros[dev]"

📝 PARAMETER RANGES

  throttle: 0.0 (stop) to 1.0 (full speed)
  steering: -1.0 (full left) to 1.0 (full right)
  brake:    0.0 (no brake) to 1.0 (full brake)
  
  Position: x, y, z coordinates
  Rotation: roll, pitch, yaw in radians (0.0 = default)

🚗 CONTROLLING MULTIPLE CARS

  from roboracer_ros import SimApiClient, CarController
  
  client = SimApiClient()
  car0 = CarController(car_id=0, client=client)
  car1 = CarController(car_id=1, client=client)
  
  car0.drive(throttle=0.5)
  car1.turn_left(throttle=0.4)

🔍 TROUBLESHOOTING

  Problem: "rclpy not found"
  Solution: source /opt/ros/humble/setup.bash

  Problem: "Connection refused"
  Solution: Make sure simulator is running with --ros flag

  Problem: "Car doesn't move"
  Solution: Check car ID, ensure UtavDriver component attached

❓ GET HELP

  roboracer-cli help                 # Show all commands
  roboracer-cli -h                   # Show CLI help
  roboracer-cli -v <command>         # Enable debug logging
  
  Interactive:
    roboracer> help                  # List commands
    roboracer> state                 # Show car state

╔════════════════════════════════════════════════════════════════════╗
║  For more information, see README.md or QUICKSTART.md              ║
╚════════════════════════════════════════════════════════════════════╝
"""

if __name__ == "__main__":
    print(QUICK_REFERENCE)
