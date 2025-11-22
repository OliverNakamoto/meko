#!/usr/bin/env python3
"""
Simple test script to verify Feetech actuator integration.

This script tests:
1. Loading robot config with Feetech motors
2. Initializing ActuatorFactory
3. Basic actuator step function
4. Planner state management
"""

import jax
import jax.numpy as jnp
import numpy as np

from toddlerbot.sim.robot import Robot
from toddlerbot.sim.actuators import ActuatorFactory, PlannerState


def test_actuator_factory():
    """Test ActuatorFactory creates correct actuator types."""
    print("Testing ActuatorFactory...")

    # Create a dummy robot config
    class DummyRobot:
        actuator_family = "feetech"
        motor_kp_real = [22.0] * 3
        motor_kd_real = [12.0] * 3
        motor_kt = [1.0] * 3
        motor_R = [1.4] * 3
        motor_vin = [12.1] * 3
        motor_max_pwm = [0.97] * 3
        motor_error_gain = [0.163] * 3
        motor_vmax = [2.0] * 3
        motor_amax = [17.45] * 3
        motor_encoder_resolution_deg = [0.087] * 3

    robot = DummyRobot()

    # Test Feetech creation
    actuator = ActuatorFactory.create("feetech", {}, robot)
    print(f"✓ Created Feetech actuator: {type(actuator).__name__}")

    # Test Dynamixel creation
    robot.actuator_family = "dynamixel"
    robot.motor_kp_sim = [1.0] * 3
    robot.motor_kd_sim = [0.5] * 3
    robot.motor_tau_max = [1.0] * 3
    robot.motor_q_dot_max = [5.0] * 3
    robot.motor_tau_q_dot_max = [0.5] * 3
    robot.motor_q_dot_tau_max = [1.0] * 3
    robot.motor_tau_brake_max = [1.5] * 3
    robot.motor_kd_min = [0.1] * 3
    robot.passive_active_ratio = 3.0

    actuator = ActuatorFactory.create("dynamixel", {}, robot)
    print(f"✓ Created Dynamixel actuator: {type(actuator).__name__}")


def test_feetech_step():
    """Test Feetech actuator step function."""
    print("\nTesting Feetech actuator step...")

    # Create robot config
    class DummyRobot:
        actuator_family = "feetech"
        motor_kp_real = [22.0] * 3
        motor_kd_real = [12.0] * 3
        motor_kt = [1.0] * 3
        motor_R = [1.4] * 3
        motor_vin = [12.1] * 3
        motor_max_pwm = [0.97] * 3
        motor_error_gain = [0.163] * 3
        motor_vmax = [2.0] * 3
        motor_amax = [17.45] * 3
        motor_encoder_resolution_deg = [0.087] * 3

    robot = DummyRobot()
    actuator = ActuatorFactory.create("feetech", {}, robot)

    # Initialize planner state
    n_joints = 3
    initial_pos = jnp.array([0.0, 0.5, -0.5])
    planner_state = actuator.init_planner_state(n_joints, initial_pos)

    print(f"✓ Initialized planner state")
    print(f"  Position shape: {planner_state.position.shape}")
    print(f"  Initial positions: {planner_state.position}")

    # Test step
    q = jnp.array([0.0, 0.5, -0.5])
    q_dot = jnp.array([0.0, 0.0, 0.0])
    action = jnp.array([0.1, 0.6, -0.4])  # Small movements
    dt = 0.02

    tau, new_planner_state = actuator.step(q, q_dot, action, dt, planner_state)

    print(f"✓ Step completed")
    print(f"  Computed torques: {tau}")
    print(f"  New planner position: {new_planner_state.position}")
    print(f"  New planner velocity: {new_planner_state.velocity}")

    # Verify torques are finite
    assert jnp.all(jnp.isfinite(tau)), "Torques contain NaN or Inf!"
    print(f"✓ Torques are finite")

    # Test diagnostics
    diagnostics = actuator.get_diagnostics()
    print(f"✓ Diagnostics: {list(diagnostics.keys())}")


def test_trapezoidal_planner():
    """Test trapezoidal planner convergence."""
    print("\nTesting trapezoidal planner convergence...")

    class DummyRobot:
        actuator_family = "feetech"
        motor_kp_real = [22.0]
        motor_kd_real = [12.0]
        motor_kt = [1.0]
        motor_R = [1.4]
        motor_vin = [12.1]
        motor_max_pwm = [0.97]
        motor_error_gain = [0.163]
        motor_vmax = [2.0]
        motor_amax = [17.45]
        motor_encoder_resolution_deg = [0.087]

    robot = DummyRobot()
    actuator = ActuatorFactory.create("feetech", {}, robot)

    # Start at 0, target 0.5 rad
    planner_state = actuator.init_planner_state(1, jnp.array([0.0]))
    target = jnp.array([0.5])
    dt = 0.001

    positions = [0.0]
    velocities = [0.0]

    # Simulate for 0.5 seconds
    for step in range(500):
        q = planner_state.position
        q_dot = planner_state.velocity

        tau, planner_state = actuator.step(q, q_dot, target, dt, planner_state)
        positions.append(float(planner_state.position[0]))
        velocities.append(float(planner_state.velocity[0]))

    final_position = positions[-1]
    final_velocity = velocities[-1]

    print(f"  Final position: {final_position:.4f} (target: 0.5)")
    print(f"  Final velocity: {final_velocity:.4f}")
    print(f"  Position error: {abs(final_position - 0.5):.4f}")

    # Check convergence (should be close to target)
    assert abs(final_position - 0.5) < 0.01, "Planner did not converge!"
    print(f"✓ Planner converged to target")


def test_pwm_clipping():
    """Test that PWM is properly clipped."""
    print("\nTesting PWM clipping...")

    class DummyRobot:
        actuator_family = "feetech"
        motor_kp_real = [1000.0]  # Very high gain to force saturation
        motor_kd_real = [100.0]
        motor_kt = [1.0]
        motor_R = [1.4]
        motor_vin = [12.1]
        motor_max_pwm = [0.97]
        motor_error_gain = [1.0]
        motor_vmax = [2.0]
        motor_amax = [17.45]
        motor_encoder_resolution_deg = [0.087]

    robot = DummyRobot()
    actuator = ActuatorFactory.create("feetech", {}, robot)

    # Large error to force saturation
    q = jnp.array([0.0])
    q_dot = jnp.array([0.0])
    action = jnp.array([2.0])  # Large target

    planner_state = actuator.init_planner_state(1, q)
    tau, _ = actuator.step(q, q_dot, action, 0.02, planner_state)

    diagnostics = actuator.get_diagnostics()
    duty = diagnostics["actuator/duty"]

    print(f"  Duty cycle: {duty[0]:.4f}")
    print(f"  Max PWM: {robot.motor_max_pwm[0]:.4f}")

    # Verify clipping
    assert abs(duty[0]) <= robot.motor_max_pwm[0] + 1e-6, "PWM not clipped!"
    print(f"✓ PWM properly clipped")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Feetech Actuator Integration Test")
    print("=" * 60)

    try:
        test_actuator_factory()
        test_feetech_step()
        test_trapezoidal_planner()
        test_pwm_clipping()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
