"""
Unit tests for SimApiClient.

Tests ROS interface without requiring an actual ROS setup.
"""

import unittest
import json
from unittest.mock import Mock, patch, MagicMock
from roboracer_ros.sim_api_client import SimApiClient


class TestSimApiClientNoROS(unittest.TestCase):
    """Test SimApiClient error handling when ROS is not available."""

    @patch("roboracer_ros.sim_api_client.ROS_AVAILABLE", False)
    def test_initialization_without_ros(self):
        """Test that proper error is raised when ROS is not available."""
        with self.assertRaises(RuntimeError) as context:
            SimApiClient()

        self.assertIn("rclpy", str(context.exception))


class TestSimApiClientMocked(unittest.TestCase):
    """Test SimApiClient with mocked ROS dependencies."""

    def setUp(self):
        """Set up test fixtures with mocked ROS."""
        # We'll patch the ROS modules
        self.ros_patcher = patch("roboracer_ros.sim_api_client.rclpy")
        self.node_patcher = patch("roboracer_ros.sim_api_client.Node")
        self.int32_patcher = patch("roboracer_ros.sim_api_client.Int32")
        self.bool_patcher = patch("roboracer_ros.sim_api_client.Bool")
        self.string_patcher = patch("roboracer_ros.sim_api_client.String")

        self.mock_rclpy = self.ros_patcher.start()
        self.mock_node_class = self.node_patcher.start()
        self.mock_int32 = self.int32_patcher.start()
        self.mock_bool = self.bool_patcher.start()
        self.mock_string = self.string_patcher.start()

        # Configure mocks
        self.mock_rclpy.ok.return_value = True
        self.mock_node = MagicMock()
        self.mock_node_class.return_value = self.mock_node

        # Patch ROS_AVAILABLE to True
        self.ros_available_patcher = patch("roboracer_ros.sim_api_client.ROS_AVAILABLE", True)
        self.ros_available_patcher.start()

    def tearDown(self):
        """Clean up patches."""
        self.ros_patcher.stop()
        self.node_patcher.stop()
        self.int32_patcher.stop()
        self.bool_patcher.stop()
        self.string_patcher.stop()
        self.ros_available_patcher.stop()

    def test_initialization(self):
        """Test SimApiClient initialization."""
        client = SimApiClient(auto_init=False)

        self.assertEqual(client.node_name, "roboracer_controller")
        self.assertFalse(client.is_initialized)

    def test_initialize(self):
        """Test ROS initialization."""
        client = SimApiClient(auto_init=False)
        client.initialize()

        self.assertTrue(client.is_initialized)
        self.assertIsNotNone(client.node)
        self.mock_rclpy.init.assert_called_once()

    def test_topic_names(self):
        """Test that topic names match ROSSimApi."""
        client = SimApiClient(auto_init=False)

        self.assertEqual(client.SPAWN_CAR_TOPIC, "/sim/spawn_car")
        self.assertEqual(client.DESPAWN_CAR_TOPIC, "/sim/despawn_car")
        self.assertEqual(client.RESET_CAR_TOPIC, "/sim/reset_car")
        self.assertEqual(client.CAR_CONTROL_TOPIC, "/sim/car_control")
        self.assertEqual(client.SET_CAR_POSITION_TOPIC, "/sim/set_car_position")
        self.assertEqual(client.SIM_STATE_TOPIC, "/sim/state")
        self.assertEqual(client.SIM_RESET_TOPIC, "/sim/reset")

    def test_spawn_car_not_initialized(self):
        """Test spawn_car when not initialized."""
        client = SimApiClient(auto_init=False)
        result = client.spawn_car(0, 0.0, 0.0, 0.0)

        self.assertFalse(result)

    def test_spawn_car(self):
        """Test spawn_car when initialized."""
        client = SimApiClient(auto_init=True)
        self.mock_node.create_publisher.return_value = MagicMock()

        result = client.spawn_car(0, 10.0, 5.0, 0.0)

        self.assertTrue(result)
        self.mock_node.create_publisher.assert_called()

    def test_despawn_car(self):
        """Test despawn_car."""
        client = SimApiClient(auto_init=True)
        self.mock_node.create_publisher.return_value = MagicMock()

        result = client.despawn_car(0)

        self.assertTrue(result)

    def test_reset_car(self):
        """Test reset_car."""
        client = SimApiClient(auto_init=True)
        self.mock_node.create_publisher.return_value = MagicMock()

        result = client.reset_car(0)

        self.assertTrue(result)

    def test_control_car(self):
        """Test control_car."""
        client = SimApiClient(auto_init=True)
        self.mock_node.create_publisher.return_value = MagicMock()

        result = client.control_car(0, steering=0.5, throttle=0.8, brake=0.0)

        self.assertTrue(result)

    def test_set_car_position(self):
        """Test set_car_position."""
        client = SimApiClient(auto_init=True)
        self.mock_node.create_publisher.return_value = MagicMock()

        result = client.set_car_position(
            0, x=10.0, y=5.0, z=0.0,
            roll=0.0, pitch=0.0, yaw=1.57
        )

        self.assertTrue(result)

    def test_reset_simulation(self):
        """Test reset_simulation."""
        client = SimApiClient(auto_init=True)
        self.mock_node.create_publisher.return_value = MagicMock()

        result = client.reset_simulation(reset_all_cars=True, reload_scene=False)

        self.assertTrue(result)

    def test_shutdown(self):
        """Test shutdown."""
        client = SimApiClient(auto_init=True)
        client.shutdown()

        self.assertFalse(client.is_initialized)
        self.mock_node.destroy_node.assert_called_once()


class TestMessageSerialization(unittest.TestCase):
    """Test message serialization."""

    def test_control_car_message_serialization(self):
        """Test that control messages are properly serialized."""
        control_data = {
            "car_id": 0,
            "steering": 0.5,
            "throttle": 0.8,
            "brake": 0.0
        }

        # Simulate what happens in control_car
        json_str = json.dumps(control_data)
        parsed = json.loads(json_str)

        self.assertEqual(parsed["car_id"], 0)
        self.assertEqual(parsed["steering"], 0.5)
        self.assertEqual(parsed["throttle"], 0.8)
        self.assertEqual(parsed["brake"], 0.0)

    def test_position_message_serialization(self):
        """Test that position messages are properly serialized."""
        position_data = {
            "car_id": 0,
            "x": 10.0,
            "y": 5.0,
            "z": 0.0,
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": 1.57
        }

        json_str = json.dumps(position_data)
        parsed = json.loads(json_str)

        self.assertEqual(parsed["car_id"], 0)
        self.assertEqual(parsed["x"], 10.0)
        self.assertEqual(parsed["y"], 5.0)


if __name__ == "__main__":
    unittest.main()
