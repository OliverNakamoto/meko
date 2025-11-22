"""Planner state dataclass for Feetech actuators."""

from dataclasses import dataclass
from jax import Array


@dataclass
class PlannerState:
    """State maintained by the trapezoidal velocity planner.

    The planner maintains its own internal position and velocity
    separate from the physics simulation to enable smooth trajectory
    planning with acceleration/deceleration profiles.
    """
    position: Array  # (n_joints,) rad - planner's target position
    velocity: Array  # (n_joints,) rad/s - planner's current velocity
    last_torque: Array  # (n_joints,) N·m - last computed torque
