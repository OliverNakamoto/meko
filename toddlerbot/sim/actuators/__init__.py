"""Actuator models for robot simulation.

This module provides an abstraction layer for different actuator types,
allowing seamless switching between Dynamixel and Feetech motor models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional
from jax import Array
import jax.numpy as jnp

from .planner_state import PlannerState


class ActuatorInterface(ABC):
    """Abstract interface for actuator models.

    All actuator implementations must conform to this interface to be
    compatible with the simulation environment.
    """

    @abstractmethod
    def __init__(self, params_dict: Dict[str, Any], robot_config: Any):
        """Initialize actuator with parameters.

        Args:
            params_dict: Dictionary containing per-joint actuator parameters
            robot_config: Robot configuration object with motor specifications
        """
        pass

    @abstractmethod
    def init_planner_state(self, n_joints: int, initial_positions: Optional[Array] = None) -> Any:
        """Initialize planner state for all joints.

        Args:
            n_joints: Number of joints
            initial_positions: Optional initial joint positions (rad). If None, zeros are used.

        Returns:
            Initial planner state (implementation-specific)
        """
        pass

    @abstractmethod
    def step(
        self,
        q: Array,
        q_dot: Array,
        action: Array,
        dt: float,
        planner_state: Any,
        noise_dict: Optional[Dict[str, Array]] = None
    ) -> Tuple[Array, Any]:
        """Compute actuator torques for one timestep.

        Args:
            q: Current joint positions (n_joints,) rad
            q_dot: Current joint velocities (n_joints,) rad/s
            action: Desired joint positions (n_joints,) rad
            dt: Timestep in seconds
            planner_state: Current planner state from previous step
            noise_dict: Optional dictionary of noise/randomization parameters

        Returns:
            Tuple of:
                - tau: Computed torques (n_joints,) N·m
                - new_planner_state: Updated planner state
        """
        pass

    @abstractmethod
    def get_diagnostics(self) -> Dict[str, Array]:
        """Return diagnostic information for debugging/logging.

        Returns:
            Dictionary with diagnostic data (duty cycles, voltages, etc.)
        """
        pass


class ActuatorFactory:
    """Factory for creating actuator instances based on configuration."""

    @staticmethod
    def create(
        actuator_family: str,
        params_dict: Dict[str, Any],
        robot_config: Any
    ) -> ActuatorInterface:
        """Create an actuator instance.

        Args:
            actuator_family: Type of actuator ("dynamixel" or "feetech")
            params_dict: Dictionary containing actuator parameters
            robot_config: Robot configuration object

        Returns:
            ActuatorInterface implementation instance

        Raises:
            ValueError: If actuator_family is not recognized
        """
        if actuator_family == "dynamixel":
            from .dynamixel import DynamixelActuators
            return DynamixelActuators(params_dict, robot_config)
        elif actuator_family == "feetech":
            from .feetech import FeetechActuators
            return FeetechActuators(params_dict, robot_config)
        else:
            raise ValueError(
                f"Unknown actuator_family: {actuator_family}. "
                f"Must be 'dynamixel' or 'feetech'"
            )


__all__ = [
    "ActuatorInterface",
    "ActuatorFactory",
    "PlannerState",
]
