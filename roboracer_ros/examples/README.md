# Examples for roboracer_ros

This directory contains example scripts demonstrating how to use the roboracer_ros package.

## Quick Start

### Using the CLI

The easiest way to get started is with the command-line interface:

```bash
# Interactive mode
roboracer-cli

# Single commands
roboracer-cli drive 0.5 0.2
roboracer-cli stop
roboracer-cli position 0 0 0
```

## Example Scripts

### 1. simple_drive.py

The simplest example - just drives forward and stops.

```bash
python examples/simple_drive.py
```

What it does:
1. Creates a CarController for car 0
2. Drives forward at 50% throttle for 3 seconds
3. Stops the car
4. Prints the final state

### 2. autonomous_loop.py

Demonstrates autonomous control patterns including circles, squares, and figure-8s.

```bash
python examples/autonomous_loop.py
```

Features:
- Simple forward drive
- Circular driving patterns
- Square patterns with turns
- Figure-8 patterns
- Curvature-based control

### 3. interactive_demo.py

An interactive menu-driven demo that lets you try different control modes.

```bash
python examples/interactive_demo.py
```

Features:
- Menu-driven interface
- Single drive commands
- Circle driving
- Square driving
- Position control
- Multiple car control
- Car state display

## Running from Python

### Basic Usage

```python
from roboracer_ros import CarController

# Create controller
controller = CarController(car_id=0)

# Drive
controller.drive(throttle=0.5, steering=0.2)

# Stop
controller.stop()
```

### Advanced Usage

```python
from roboracer_ros import CarController, SimApiClient
import time

# Create a shared client for multiple cars
client = SimApiClient()

# Control multiple cars
car0 = CarController(car_id=0, client=client)
car1 = CarController(car_id=1, client=client)

# Drive them
car0.drive(throttle=0.5)
car1.turn_left(throttle=0.4)

# Let them run for a bit
time.sleep(2)

# Stop
car0.stop()
car1.stop()

# Clean up
client.shutdown()
```

## Prerequisites

1. **Simulator running**:
   ```bash
   # Make sure ROS is enabled
   ./sim.x86_64 --ros --port 9091
   ```

2. **ROS 2 installed**:
   ```bash
   source /opt/ros/humble/setup.bash
   ```

3. **roboracer_ros installed**:
   ```bash
   pip install -e ..
   ```

## Troubleshooting

### "rclpy not found"

Make sure ROS 2 is installed and sourced:
```bash
source /opt/ros/humble/setup.bash
```

### Car doesn't respond

1. Check that simulator is running with `--ros` flag
2. Verify car ID is correct (usually 0)
3. Check that UtavDriver component is attached to car in Unity
4. Look at Unity console for error messages

### Examples won't run

1. Make sure you're in the right directory
2. Make sure roboracer_ros is installed: `pip install -e ..`
3. Check that Python version is 3.8+: `python --version`

## Creating Your Own Example

Here's a template:

```python
#!/usr/bin/env python3
"""Description of what this example does."""

import logging
from roboracer_ros import CarController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting example")
    
    controller = CarController(car_id=0)
    
    try:
        # Your code here
        controller.drive(throttle=0.5)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        controller.stop()
    
    finally:
        controller.client.shutdown()

if __name__ == "__main__":
    main()
```

## Next Steps

- Read the [README.md](../README.md) for full documentation
- Try the [CLI mode](../README.md#interactive-mode)
- Integrate with your own ROS nodes
