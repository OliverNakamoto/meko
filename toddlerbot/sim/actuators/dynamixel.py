"""Dynamixel actuator implementation using asymmetric saturation model."""

from typing import Dict, Any, Tuple, Optional
from jax import Array
import jax.numpy as jnp

from toddlerbot.sim.actuators import ActuatorInterface
from toddlerbot.utils.array_utils import ArrayType
from toddlerbot.utils.array_utils import array_lib as np


class DynamixelActuators(ActuatorInterface):
    """Dynamixel motor controller with asymmetric saturation.

    Implements the existing Dynamixel PD controller with asymmetric
    saturation based on velocity-dependent torque limits and passive-
    active damping ratio.
    """

    def __init__(self, params_dict: Dict[str, Any], robot_config: Any):
        """Initialize Dynamixel actuators.

        Args:
            params_dict: Dictionary with Dynamixel-specific parameters
            robot_config: Robot configuration object with motor specifications
        """
        self.kp = np.array(robot_config.motor_kp_sim, dtype=np.float32)
        self.kd = np.array(robot_config.motor_kd_sim, dtype=np.float32)
        self.tau_max = np.array(robot_config.motor_tau_max, dtype=np.float32)
        self.q_dot_max = np.array(robot_config.motor_q_dot_max, dtype=np.float32)
        self.tau_q_dot_max = np.array(robot_config.motor_tau_q_dot_max, dtype=np.float32)
        self.q_dot_tau_max = np.array(robot_config.motor_q_dot_tau_max, dtype=np.float32)
        self.tau_brake_max = np.array(robot_config.motor_tau_brake_max, dtype=np.float32)
        self.kd_min = np.array(robot_config.motor_kd_min, dtype=np.float32)
        self.passive_active_ratio = robot_config.passive_active_ratio

        # Store diagnostics
        self.last_torque = None

    def init_planner_state(self, n_joints: int, initial_positions: Optional[Array] = None) -> None:
        """Dynamixel doesn't use planner state.

        Args:
            n_joints: Number of joints
            initial_positions: Unused for Dynamixel

        Returns:
            None (Dynamixel has no planner state)
        """
        return None

    def step(
        self,
        q: Array,
        q_dot: Array,
        action: Array,
        dt: float,
        planner_state: Any,
        noise_dict: Optional[Dict[str, Array]] = None
    ) -> Tuple[Array, Any]:
        """Compute torques using Dynamixel asymmetric saturation model.

        Args:
            q: Current joint positions (n_joints,) rad
            q_dot: Current joint velocities (n_joints,) rad/s
            action: Desired joint positions (n_joints,) rad
            dt: Timestep (unused for Dynamixel)
            planner_state: Unused for Dynamixel
            noise_dict: Optional domain randomization parameters

        Returns:
            Tuple of (torques, None)
        """
        if noise_dict is None:
            noise_dict = {}

        # Compute q_dot_dot for passive-active ratio
        # Approximate acceleration from velocity (simple backward diff)
        # In practice, this is provided by the environment
        q_dot_dot = jnp.zeros_like(q_dot)  # Placeholder

        # Apply noise/randomization
        kp = self.kp * noise_dict.get("kp", 1.0)
        kd = self.kd * noise_dict.get("kd", 1.0)
        tau_max = self.tau_max * noise_dict.get("tau_max", 1.0)
        q_dot_tau_max = self.q_dot_tau_max * noise_dict.get("q_dot_tau_max", 1.0)
        q_dot_max = self.q_dot_max * noise_dict.get("q_dot_max", 1.0)
        kd_min = self.kd_min * noise_dict.get("kd_min", 1.0)
        tau_brake_max = self.tau_brake_max * noise_dict.get("tau_brake_max", 1.0)
        tau_q_dot_max = self.tau_q_dot_max * noise_dict.get("tau_q_dot_max", 1.0)
        passive_active_ratio = self.passive_active_ratio * noise_dict.get(
            "passive_active_ratio", 1.0
        )

        # PD control with passive-active ratio
        error = action - q
        real_kp = jnp.where(q_dot_dot * error < 0, kp * passive_active_ratio, kp)
        tau_m = real_kp * error - (kd_min + kd) * q_dot
        abs_q_dot = jnp.abs(q_dot)

        # Linear taper between (q_dot_tau_max, tau_max) and (q_dot_max, tau_q_dot_max)
        slope = (tau_q_dot_max - tau_max) / (q_dot_max - q_dot_tau_max)
        taper_limit = tau_max + slope * (abs_q_dot - q_dot_tau_max)

        tau_acc_limit = jnp.where(abs_q_dot <= q_dot_tau_max, tau_max, taper_limit)

        # Asymmetric saturation
        tau_m_clamped = jnp.where(
            jnp.logical_and(abs_q_dot > q_dot_max, q_dot * action > 0),
            # Dynamixel self-protection: reverse torque
            jnp.where(
                q_dot > 0,
                jnp.ones_like(tau_m) * -tau_brake_max,
                jnp.ones_like(tau_m) * tau_brake_max,
            ),
            jnp.where(
                q_dot > 0,
                jnp.clip(tau_m, -tau_brake_max, tau_acc_limit),
                jnp.clip(tau_m, -tau_acc_limit, tau_brake_max),
            ),
        )

        self.last_torque = tau_m_clamped
        return tau_m_clamped, None

    def get_diagnostics(self) -> Dict[str, Array]:
        """Return diagnostic information.

        Returns:
            Dictionary with diagnostic data
        """
        return {
            "actuator/torque": self.last_torque if self.last_torque is not None else jnp.array([]),
        }
