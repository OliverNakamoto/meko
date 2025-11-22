"""Feetech actuator implementation with electrical model and trapezoidal planner."""

from typing import Dict, Any, Tuple, Optional
from jax import Array
import jax.numpy as jnp
import numpy as np

from toddlerbot.sim.actuators import ActuatorInterface
from toddlerbot.sim.actuators.planner_state import PlannerState


class FeetechActuators(ActuatorInterface):
    """Feetech servo motor controller with electrical model.

    Implements a trapezoidal velocity planner followed by a PD controller
    with error gain, PWM clipping, and a voltage-to-torque electrical model.

    Based on the kbot-sim implementation for STS series servos.
    """

    def __init__(self, params_dict: Dict[str, Any], robot_config: Any):
        """Initialize Feetech actuators.

        Args:
            params_dict: Dictionary with Feetech-specific parameters
            robot_config: Robot configuration object with motor specifications
        """
        # Load per-joint control gains (real, not scaled)
        self.kp = jnp.array(robot_config.motor_kp_real, dtype=jnp.float32)
        self.kd = jnp.array(robot_config.motor_kd_real, dtype=jnp.float32)

        # Load Feetech electrical parameters
        self.kt = jnp.array(robot_config.motor_kt, dtype=jnp.float32)
        self.R = jnp.array(robot_config.motor_R, dtype=jnp.float32)
        self.vin = jnp.array(robot_config.motor_vin, dtype=jnp.float32)
        self.max_pwm = jnp.array(robot_config.motor_max_pwm, dtype=jnp.float32)
        self.error_gain = jnp.array(robot_config.motor_error_gain, dtype=jnp.float32)

        # Load planner parameters
        self.vmax = jnp.array(robot_config.motor_vmax, dtype=jnp.float32)
        self.amax = jnp.array(robot_config.motor_amax, dtype=jnp.float32)

        # Encoder resolution and deadband
        encoder_resolution = jnp.array(
            robot_config.motor_encoder_resolution_deg, dtype=jnp.float32
        ) * jnp.pi / 180.0  # Convert to radians
        self.deadband = 2.0 * encoder_resolution
        self.decay_factor = 0.8

        # Store number of joints
        self.n_joints = len(self.kp)

        # Diagnostics
        self.last_duty = None
        self.last_voltage = None
        self.last_torque = None

    def init_planner_state(
        self, n_joints: int, initial_positions: Optional[Array] = None
    ) -> PlannerState:
        """Initialize planner state for all joints.

        Args:
            n_joints: Number of joints
            initial_positions: Optional initial joint positions (rad). If None, zeros are used.

        Returns:
            Initial PlannerState with position, velocity, and torque
        """
        if initial_positions is None:
            initial_positions = jnp.zeros(n_joints, dtype=jnp.float32)

        return PlannerState(
            position=initial_positions,
            velocity=jnp.zeros(n_joints, dtype=jnp.float32),
            last_torque=jnp.zeros(n_joints, dtype=jnp.float32),
        )

    def trapezoidal_step(
        self,
        target_position: Array,
        current_position: Array,
        current_velocity: Array,
        vmax: Array,
        amax: Array,
        deadband: Array,
        dt: float,
    ) -> Tuple[Array, Array]:
        """Execute one step of the trapezoidal velocity planner.

        Args:
            target_position: Desired joint positions (rad)
            current_position: Planner's current position (rad)
            current_velocity: Planner's current velocity (rad/s)
            vmax: Maximum velocity per joint (rad/s)
            amax: Maximum acceleration per joint (rad/s²)
            deadband: Position deadband per joint (rad)
            dt: Timestep (s)

        Returns:
            Tuple of (new_position, new_velocity)
        """
        # Compute position error
        position_error = target_position - current_position
        abs_error = jnp.abs(position_error)
        error_sign = jnp.sign(position_error)

        # Deadband logic: decay velocity if within deadband
        in_deadband = abs_error <= deadband
        velocity_decayed = current_velocity * self.decay_factor

        # Update position with decaying velocity when in deadband
        position_in_deadband = current_position + velocity_decayed * dt

        # Trapezoidal planning when outside deadband
        # Calculate stopping distance: d = v² / (2a)
        stopping_distance = (current_velocity ** 2) / (2.0 * amax + 1e-8)

        # Determine if we should accelerate or decelerate
        should_decelerate = jnp.abs(stopping_distance) >= abs_error

        # Acceleration direction
        # If decelerating: opposite to velocity
        # If accelerating: toward target
        accel_direction = jnp.where(
            should_decelerate,
            -jnp.sign(current_velocity),
            error_sign
        )

        # Apply acceleration
        new_velocity_unclamped = current_velocity + accel_direction * amax * dt

        # Clamp velocity to vmax
        new_velocity = jnp.clip(new_velocity_unclamped, -vmax, vmax)

        # Update position
        new_position = current_position + new_velocity * dt

        # Choose deadband or trapezoidal result
        final_position = jnp.where(in_deadband, position_in_deadband, new_position)
        final_velocity = jnp.where(in_deadband, velocity_decayed, new_velocity)

        return final_position, final_velocity

    def step(
        self,
        q: Array,
        q_dot: Array,
        action: Array,
        dt: float,
        planner_state: PlannerState,
        noise_dict: Optional[Dict[str, Array]] = None
    ) -> Tuple[Array, PlannerState]:
        """Compute actuator torques using Feetech electrical model.

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
        if noise_dict is None:
            noise_dict = {}

        # Apply domain randomization
        kp = self.kp * noise_dict.get("kp", 1.0)
        kd = self.kd * noise_dict.get("kd", 1.0)
        kt = self.kt * noise_dict.get("kt", 1.0)
        R = self.R * noise_dict.get("R", 1.0)
        vin = self.vin * noise_dict.get("vin", 1.0)
        max_pwm = self.max_pwm * noise_dict.get("max_pwm", 1.0)
        error_gain = self.error_gain * noise_dict.get("error_gain", 1.0)
        vmax = self.vmax * noise_dict.get("vmax", 1.0)
        amax = self.amax * noise_dict.get("amax", 1.0)
        deadband = self.deadband * noise_dict.get("encoder_zero_offset", 1.0)

        # Trapezoidal planner step
        new_planner_pos, new_planner_vel = self.trapezoidal_step(
            target_position=action,
            current_position=planner_state.position,
            current_velocity=planner_state.velocity,
            vmax=vmax,
            amax=amax,
            deadband=deadband,
            dt=dt,
        )

        # PD control with error gain (applied to Kp only, NOT Kd)
        pos_error = new_planner_pos - q
        vel_error = new_planner_vel - q_dot

        raw_duty = kp * error_gain * pos_error + kd * vel_error

        # Clip to max PWM
        duty = jnp.clip(raw_duty, -max_pwm, max_pwm)

        # Electrical model: voltage to torque
        # Simplified model WITHOUT back-EMF (matches kbot-sim)
        # τ = (PWM * Vin * Kt) / R
        voltage = duty * vin
        torque = voltage * kt / (R + 1e-8)  # Add small epsilon to prevent division by zero

        # Ensure torque is finite
        torque = jnp.where(jnp.isfinite(torque), torque, 0.0)

        # Update diagnostics
        self.last_duty = duty
        self.last_voltage = voltage
        self.last_torque = torque

        # Create new planner state
        new_planner_state = PlannerState(
            position=new_planner_pos,
            velocity=new_planner_vel,
            last_torque=torque,
        )

        return torque, new_planner_state

    def get_diagnostics(self) -> Dict[str, Array]:
        """Return diagnostic information for debugging/logging.

        Returns:
            Dictionary with diagnostic data
        """
        diagnostics = {}

        if self.last_duty is not None:
            diagnostics["actuator/duty"] = self.last_duty
        if self.last_voltage is not None:
            diagnostics["actuator/voltage"] = self.last_voltage
        if self.last_torque is not None:
            diagnostics["actuator/torque"] = self.last_torque

        return diagnostics
