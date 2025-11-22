"""
Unit tests for CarController.

Tests basic car control functionality without requiring ROS.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from roboracer_ros.car_controller import CarController


class TestCarController(unittest.TestCase):
    """Test cases for CarController."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the client to avoid ROS dependency in tests
        self.mock_client = Mock()
        self.controller = CarController(car_id=0, client=self.mock_client)

    def test_initialization(self):
        """Test CarController initialization."""
        self.assertEqual(self.controller.car_id, 0)
        self.assertEqual(self.controller.throttle, 0.0)
        self.assertEqual(self.controller.steering, 0.0)
        self.assertEqual(self.controller.brake, 0.0)

    def test_set_throttle_clamping(self):
        """Test that throttle is clamped to valid range."""
        # Test lower bound
        self.controller.set_throttle(-0.5)
        self.assertEqual(self.controller.throttle, 0.0)

        # Test upper bound
        self.controller.set_throttle(1.5)
        self.assertEqual(self.controller.throttle, 1.0)

        # Test valid value
        self.controller.set_throttle(0.5)
        self.assertEqual(self.controller.throttle, 0.5)

    def test_set_steering_clamping(self):
        """Test that steering is clamped to valid range."""
        # Test lower bound
        self.controller.set_steering(-1.5)
        self.assertEqual(self.controller.steering, -1.0)

        # Test upper bound
        self.controller.set_steering(1.5)
        self.assertEqual(self.controller.steering, 1.0)

        # Test valid value
        self.controller.set_steering(0.3)
        self.assertEqual(self.controller.steering, 0.3)

    def test_set_brake_clamping(self):
        """Test that brake is clamped to valid range."""
        # Test lower bound
        self.controller.set_brake(-0.5)
        self.assertEqual(self.controller.brake, 0.0)

        # Test upper bound
        self.controller.set_brake(1.5)
        self.assertEqual(self.controller.brake, 1.0)

        # Test valid value
        self.controller.set_brake(0.7)
        self.assertEqual(self.controller.brake, 0.7)

    def test_send_control_calls_client(self):
        """Test that send_control calls the client."""
        self.mock_client.control_car.return_value = True

        self.controller.set_throttle(0.5)
        self.controller.set_steering(0.2)
        self.controller.set_brake(0.0)

        result = self.controller.send_control()

        self.assertTrue(result)
        self.mock_client.control_car.assert_called_once_with(
            0, 0.2, 0.5, 0.0
        )

    def test_drive(self):
        """Test drive command."""
        self.mock_client.control_car.return_value = True

        result = self.controller.drive(throttle=0.5, steering=0.2)

        self.assertTrue(result)
        self.assertEqual(self.controller.throttle, 0.5)
        self.assertEqual(self.controller.steering, 0.2)
        self.assertEqual(self.controller.brake, 0.0)
        self.mock_client.control_car.assert_called_once()

    def test_stop(self):
        """Test stop command."""
        self.mock_client.control_car.return_value = True

        # First, set some values
        self.controller.drive(throttle=0.5, steering=0.3)
        self.mock_client.reset_mock()

        # Now stop
        result = self.controller.stop()

        self.assertTrue(result)
        self.assertEqual(self.controller.throttle, 0.0)
        self.assertEqual(self.controller.steering, 0.0)
        self.assertEqual(self.controller.brake, 1.0)

    def test_turn_left(self):
        """Test left turn command."""
        self.mock_client.control_car.return_value = True

        result = self.controller.turn_left(throttle=0.5, angle=0.5)

        self.assertTrue(result)
        self.assertEqual(self.controller.throttle, 0.5)
        self.assertEqual(self.controller.steering, -0.5)  # Negative = left

    def test_turn_right(self):
        """Test right turn command."""
        self.mock_client.control_car.return_value = True

        result = self.controller.turn_right(throttle=0.5, angle=0.5)

        self.assertTrue(result)
        self.assertEqual(self.controller.throttle, 0.5)
        self.assertEqual(self.controller.steering, 0.5)  # Positive = right

    def test_set_position(self):
        """Test position setting."""
        self.mock_client.set_car_position.return_value = True

        result = self.controller.set_position(x=10.0, y=5.0, z=1.0)

        self.assertTrue(result)
        self.mock_client.set_car_position.assert_called_once_with(
            0, 10.0, 5.0, 1.0, 0.0, 0.0, 0.0
        )

    def test_reset(self):
        """Test car reset."""
        self.mock_client.reset_car.return_value = True

        # Set some values first
        self.controller.drive(throttle=0.5, steering=0.3)

        result = self.controller.reset()

        self.assertTrue(result)
        # State should be reset
        self.assertEqual(self.controller.throttle, 0.0)
        self.assertEqual(self.controller.steering, 0.0)
        self.assertEqual(self.controller.brake, 0.0)
        self.mock_client.reset_car.assert_called_once_with(0)

    def test_follow_curvature(self):
        """Test curvature-based control."""
        self.mock_client.control_car.return_value = True

        result = self.controller.follow_curvature(velocity=0.5, curvature=0.5)

        self.assertTrue(result)
        # Curvature maps to steering
        self.assertEqual(self.controller.steering, 0.5)

    def test_get_state(self):
        """Test state getter."""
        self.controller.set_throttle(0.5)
        self.controller.set_steering(0.2)
        self.controller.set_brake(0.1)

        state = self.controller.get_state()

        self.assertEqual(state["car_id"], 0)
        self.assertEqual(state["throttle"], 0.5)
        self.assertEqual(state["steering"], 0.2)
        self.assertEqual(state["brake"], 0.1)

    def test_repr(self):
        """Test string representation."""
        self.controller.set_throttle(0.5)
        self.controller.set_steering(0.2)

        repr_str = repr(self.controller)

        self.assertIn("CarController", repr_str)
        self.assertIn("car_id=0", repr_str)
        self.assertIn("throttle=0.50", repr_str)
        self.assertIn("steering=0.20", repr_str)


class TestMultipleControllers(unittest.TestCase):
    """Test cases for controlling multiple cars."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock()

    def test_multiple_cars_same_client(self):
        """Test controlling multiple cars with same client."""
        car0 = CarController(car_id=0, client=self.mock_client)
        car1 = CarController(car_id=1, client=self.mock_client)

        self.mock_client.control_car.return_value = True

        car0.drive(throttle=0.5)
        car1.drive(throttle=0.3)

        # Both should have called control_car with correct car_id
        calls = self.mock_client.control_car.call_args_list
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0][0], 0)  # First call, car_id arg
        self.assertEqual(calls[1][0][0], 1)  # Second call, car_id arg


if __name__ == "__main__":
    unittest.main()
