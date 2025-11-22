"""
SimApiClient - Low-level ROS interface to ROSSimApi topics.

Handles publishing and subscribing to ROS topics for simulator control.
"""

import logging
from typing import Optional, Callable, Dict, Any
import time
import json

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Int32, Bool, String
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    rclpy = None
    Node = object
    Int32 = None
    Bool = None
    String = None


logger = logging.getLogger(__name__)


class SimApiClient:
    """
    Client for communicating with ROSSimApi in the simulator.
    
    Manages ROS node lifecycle and provides methods to publish control messages
    to various simulator topics.
    """

    # ROS Topic names (must match ROSSimApi.cs)
    SPAWN_CAR_TOPIC = "/sim/spawn_car"
    DESPAWN_CAR_TOPIC = "/sim/despawn_car"
    RESET_CAR_TOPIC = "/sim/reset_car"
    CAR_CONTROL_TOPIC = "/sim/car_control"
    SET_CAR_POSITION_TOPIC = "/sim/set_car_position"
    SIM_STATE_TOPIC = "/sim/state"
    SIM_RESET_TOPIC = "/sim/reset"

    def __init__(self, node_name: str = "roboracer_controller", auto_init: bool = True):
        """
        Initialize the SimApiClient.

        Args:
            node_name: Name for the ROS node
            auto_init: If True, automatically initialize ROS and create node
        """
        if not ROS_AVAILABLE:
            raise RuntimeError(
                "ROS 2 (rclpy) is not installed. Install with: pip install rclpy"
            )

        self.node_name = node_name
        self.node: Optional[Node] = None
        self.is_initialized = False
        self._state_callback: Optional[Callable] = None
        self._subscribers: Dict[str, Any] = {}

        if auto_init:
            self.initialize()

    def initialize(self) -> None:
        """Initialize ROS node and publishers."""
        if self.is_initialized:
            logger.warning("SimApiClient already initialized")
            return

        try:
            if not rclpy.ok():
                rclpy.init()

            self.node = Node(self.node_name)
            self.is_initialized = True
            logger.info(f"SimApiClient initialized with node: {self.node_name}")

        except Exception as e:
            logger.error(f"Failed to initialize ROS: {e}")
            raise

    def shutdown(self) -> None:
        """Shutdown ROS node and cleanup resources."""
        if self.node is not None:
            self.node.destroy_node()
            self.is_initialized = False
            logger.info("SimApiClient shutdown complete")

    def spawn_car(self, car_id: int, x: float, y: float, z: float,
                  roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> bool:
        """
        Spawn a car at the specified position and rotation.

        Args:
            car_id: Unique identifier for the car
            x, y, z: Position coordinates
            roll, pitch, yaw: Rotation angles in radians

        Returns:
            True if message was published, False otherwise
        """
        if not self.is_initialized:
            logger.error("SimApiClient not initialized")
            return False

        try:
            msg = Int32()
            msg.data = car_id
            pub = self.node.create_publisher(Int32, self.SPAWN_CAR_TOPIC, 10)
            pub.publish(msg)
            logger.info(f"Published spawn request for car {car_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to spawn car: {e}")
            return False

    def despawn_car(self, car_id: int) -> bool:
        """
        Despawn a car from the simulation.

        Args:
            car_id: ID of the car to despawn

        Returns:
            True if message was published, False otherwise
        """
        if not self.is_initialized:
            logger.error("SimApiClient not initialized")
            return False

        try:
            msg = Int32()
            msg.data = car_id
            pub = self.node.create_publisher(Int32, self.DESPAWN_CAR_TOPIC, 10)
            pub.publish(msg)
            logger.info(f"Published despawn request for car {car_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to despawn car: {e}")
            return False

    def reset_car(self, car_id: int) -> bool:
        """
        Reset a car to its initial state.

        Args:
            car_id: ID of the car to reset

        Returns:
            True if message was published, False otherwise
        """
        if not self.is_initialized:
            logger.error("SimApiClient not initialized")
            return False

        try:
            msg = Int32()
            msg.data = car_id
            pub = self.node.create_publisher(Int32, self.RESET_CAR_TOPIC, 10)
            pub.publish(msg)
            logger.info(f"Published reset request for car {car_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset car: {e}")
            return False

    def control_car(self, car_id: int, steering: float, throttle: float, brake: float) -> bool:
        """
        Send control inputs to a car.

        Args:
            car_id: ID of the car to control
            steering: Steering input (-1.0 to 1.0)
            throttle: Throttle input (0.0 to 1.0)
            brake: Brake input (0.0 to 1.0)

        Returns:
            True if message was published, False otherwise
        """
        if not self.is_initialized:
            logger.error("SimApiClient not initialized")
            return False

        try:
            # Create a JSON message with the control data
            msg = String()
            msg.data = json.dumps({
                "car_id": car_id,
                "steering": float(steering),
                "throttle": float(throttle),
                "brake": float(brake)
            })

            pub = self.node.create_publisher(String, self.CAR_CONTROL_TOPIC, 10)
            pub.publish(msg)
            logger.debug(f"Sent control for car {car_id}: steering={steering}, throttle={throttle}, brake={brake}")
            return True
        except Exception as e:
            logger.error(f"Failed to control car: {e}")
            return False

    def set_car_position(self, car_id: int, x: float, y: float, z: float,
                        roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> bool:
        """
        Set a car's position and rotation.

        Args:
            car_id: ID of the car
            x, y, z: Position coordinates
            roll, pitch, yaw: Rotation angles in radians

        Returns:
            True if message was published, False otherwise
        """
        if not self.is_initialized:
            logger.error("SimApiClient not initialized")
            return False

        try:
            msg = String()
            msg.data = json.dumps({
                "car_id": car_id,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "roll": float(roll),
                "pitch": float(pitch),
                "yaw": float(yaw)
            })

            pub = self.node.create_publisher(String, self.SET_CAR_POSITION_TOPIC, 10)
            pub.publish(msg)
            logger.info(f"Set car {car_id} position to ({x}, {y}, {z})")
            return True
        except Exception as e:
            logger.error(f"Failed to set car position: {e}")
            return False

    def reset_simulation(self, reset_all_cars: bool = True, reload_scene: bool = False) -> bool:
        """
        Reset the simulation.

        Args:
            reset_all_cars: If True, reset all spawned cars
            reload_scene: If True, reload the entire scene

        Returns:
            True if message was published, False otherwise
        """
        if not self.is_initialized:
            logger.error("SimApiClient not initialized")
            return False

        try:
            msg = Bool()
            msg.data = reset_all_cars
            pub = self.node.create_publisher(Bool, self.SIM_RESET_TOPIC, 10)
            pub.publish(msg)
            logger.info(f"Published simulation reset request (reset_all={reset_all_cars})")
            return True
        except Exception as e:
            logger.error(f"Failed to reset simulation: {e}")
            return False

    def subscribe_to_state(self, callback: Callable) -> bool:
        """
        Subscribe to simulation state updates.

        Args:
            callback: Function to call when state is received

        Returns:
            True if subscription was successful, False otherwise
        """
        if not self.is_initialized:
            logger.error("SimApiClient not initialized")
            return False

        try:
            self._state_callback = callback

            def state_callback(msg):
                try:
                    state_data = json.loads(msg.data) if isinstance(msg, type(msg)) else msg.data
                    callback(state_data)
                except Exception as e:
                    logger.error(f"Error in state callback: {e}")

            sub = self.node.create_subscription(
                String,
                self.SIM_STATE_TOPIC,
                state_callback,
                10
            )
            self._subscribers["state"] = sub
            logger.info("Subscribed to simulation state topic")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to state: {e}")
            return False

    def spin_once(self, timeout_sec: float = 0.1) -> None:
        """
        Process one iteration of ROS callbacks.

        Args:
            timeout_sec: Maximum time to wait for callbacks
        """
        if self.node is not None:
            try:
                rclpy.spin_once(self.node, timeout_sec=timeout_sec)
            except Exception as e:
                logger.error(f"Error during spin_once: {e}")
