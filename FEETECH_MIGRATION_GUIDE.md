# Feetech Motor Migration Guide

This guide explains how to use Feetech motors in ToddlerBot instead of Dynamixel motors.

## Overview

The ToddlerBot codebase now supports **two actuator families**:
1. **Dynamixel** - Original asymmetric saturation model
2. **Feetech** - Electrical model with trapezoidal planner (from kbot-sim/Zeroth)

You can switch between them via configuration, maintaining full backwards compatibility.

---

## Quick Start

### 1. Update Your Robot Configuration

Edit your robot's `robot.yml` file (or `default.yml` for global config):

```yaml
actuators:
  actuator_family: feetech  # Change from "dynamixel" to "feetech"
```

### 2. Specify Feetech Motor Models

Update motor definitions in your robot config:

```yaml
motors:
  left_ankle_pitch:
    motor: feetech_sts3250  # Changed from XC330
    group: leg
    zero_pos: 0
    home_pos: 0
    kp: 22        # Real hardware gains (NO scaling needed)
    kd: 12
```

### 3. Train or Run

No code changes needed! The environment will automatically use Feetech actuators:

```bash
python -m toddlerbot.locomotion.ppo_train --robot_name=your_robot
```

---

## What Changed Under the Hood

### Motor Model

**Dynamixel Model (Original):**
- PD control with asymmetric saturation
- Velocity-dependent torque limits
- Passive-active damping ratio
- Formula: `τ = kp * (a - q) - kd * q_dot` → asymmetric clipping

**Feetech Model (New):**
- Trapezoidal velocity planner (smooth acceleration/deceleration)
- PD control with error gain
- Electrical model (kt, R, vin)
- Formula: `PWM = kp * error_gain * pos_error + kd * vel_error` → `τ = (PWM * Vin * Kt) / R`

### Key Differences

| Feature | Dynamixel | Feetech |
|---------|-----------|---------|
| **Control gains** | kp_sim = kp_real / 150 | kp_sim = kp_real (no scaling) |
| **Torque limits** | Asymmetric (tau_max, tau_brake_max) | Electrical (kt, R, vin, max_pwm) |
| **Trajectory planning** | None (direct PD) | Trapezoidal planner |
| **Planner state** | None | position, velocity, last_torque |
| **Back-EMF** | Implicit in saturation | Not modeled (simplified) |

---

## Available Feetech Motor Types

### STS3250

```yaml
feetech_sts3250:
  # Electrical parameters (from system ID)
  kt: 1.0005874626213263      # N·m/A
  R: 1.3890462492623645       # Ω
  vin: 12.1                   # V
  max_pwm: 0.97               # 0-1
  error_gain: 0.16293639      # dimensionless

  # MuJoCo physics
  armature: 0.04              # kg·m²
  damping: 1.3464038511725651  # N·m·s/rad
  frictionloss: 0.2           # N·m

  # Performance limits
  max_torque: 8.716           # N·m
  max_velocity: 8.938         # rad/s

  # Planner parameters
  vmax: 2.0                   # rad/s
  amax: 17.45                 # rad/s²
  encoder_resolution_deg: 0.087
```

### STS3215_12V

```yaml
feetech_sts3215_12v:
  # Electrical parameters
  kt: 1.0                     # N·m/A
  R: 2.2136477795617733       # Ω
  vin: 12.1                   # V
  max_pwm: 0.9964             # 0-1
  error_gain: 0.16292703      # dimensionless

  # MuJoCo physics
  armature: 0.04              # kg·m²
  damping: 1.2305092028680242  # N·m·s/rad
  frictionloss: 0.162         # N·m

  # Performance limits
  max_torque: 5.466           # N·m
  max_velocity: 4.856         # rad/s

  # Planner parameters
  vmax: 2.0                   # rad/s
  amax: 17.45                 # rad/s²
  encoder_resolution_deg: 0.087
```

---

## Domain Randomization

Feetech motors have additional randomization parameters:

```python
# In MJXConfig.DomainRandConfig:
kt_range: [0.9, 1.1]                    # ±10% torque constant
R_range: [0.9, 1.1]                     # ±10% resistance
vin_range: [0.95, 1.05]                 # ±5% supply voltage
max_pwm_range: [0.95, 1.05]             # ±5% PWM limit
error_gain_range: [0.9, 1.1]            # ±10% error gain
vmax_range: [0.8, 1.2]                  # ±20% max velocity
amax_range: [0.8, 1.2]                  # ±20% max acceleration
encoder_zero_offset_range: [-0.035, 0.035]  # ±2° encoder offset
```

These are applied automatically during `env.reset()` when `add_domain_rand=True`.

---

## Tuning Recommendations

### If Training Diverges

1. **Check torque scaling**: Feetech max torques differ from Dynamixel
   - STS3250: ~8.7 N·m vs XC330: ~0.68 N·m
   - May need to adjust reward torque penalty coefficients

2. **Adjust vmax/amax**: Default values are conservative
   - Increase `vmax` for faster movements
   - Increase `amax` for more aggressive acceleration

3. **Tune PD gains**: Feetech uses real gains (no kp_ratio/kd_ratio)
   - Start with manufacturer recommendations (kp=22, kd=12 for legs)
   - Tune based on tracking error and oscillation

### If Sim-to-Real Gap is Large

1. **Run system ID** for your specific motors
   - Measure kt, R, vin under load
   - Update `default.yml` with measured values

2. **Calibrate encoder deadband**
   - Measure actual encoder resolution
   - Adjust `encoder_resolution_deg` if needed

3. **Tune error_gain**
   - Controls PD output scaling
   - Adjust if torques are too aggressive/weak

---

## Implementation Details

### Trapezoidal Planner

The planner maintains internal state separate from physics:

```python
# Deadband behavior
if |position_error| <= deadband:
    velocity *= 0.8  # Decay factor

# Trapezoidal planning
stopping_distance = velocity² / (2 * amax)
if error > stopping_distance:
    accelerate toward target
else:
    decelerate

# Velocity clipping
velocity = clip(velocity, -vmax, vmax)
```

**Key parameters:**
- `deadband = 2 * encoder_resolution` = 0.174° = 0.003 rad
- `decay_factor = 0.8`

### PD Controller

```python
pos_error = planner_position - current_position
vel_error = planner_velocity - current_velocity

raw_duty = kp * error_gain * pos_error + kd * vel_error
duty = clip(raw_duty, -max_pwm, max_pwm)
```

**Note:** Error gain is only applied to Kp, NOT Kd!

### Electrical Model

```python
voltage = duty * vin
torque = voltage * kt / R
```

**Simplified model** - does NOT include back-EMF term (`kt * ω`). This matches the kbot-sim implementation.

### Planner State

Stored in `state.info["planner_state"]`:
```python
@dataclass
class PlannerState:
    position: Array  # (n_joints,) rad
    velocity: Array  # (n_joints,) rad/s
    last_torque: Array  # (n_joints,) N·m
```

---

## Testing

### Basic Functionality Test

```bash
python test_feetech_integration.py
```

This tests:
- ActuatorFactory creation
- Step function
- Trapezoidal planner convergence
- PWM clipping

### Single-Joint Hardware Test

Before full deployment, test one joint:

```python
from toddlerbot.sim.real_world import RealWorld

# Test single joint in isolation
robot = Robot("your_robot")
real = RealWorld(robot)

# Send sinusoidal command (1Hz, small amplitude)
import numpy as np
for t in np.linspace(0, 2*np.pi, 100):
    target = 0.1 * np.sin(t)
    real.set_motor_position(0, target)  # Joint 0
    time.sleep(0.02)

    # Monitor position error
    pos = real.get_motor_position(0)
    error = target - pos
    print(f"Error: {error:.3f} rad")
```

**Safety:** Have emergency stop ready!

---

## Comparison: Dynamixel vs Feetech

### When to Use Feetech

✅ You have Feetech STS motors
✅ You have system ID data (kt, R, vin)
✅ You want smooth trajectory planning
✅ You need model-based torque estimation

### When to Stick with Dynamixel

✅ You have Dynamixel motors
✅ You already have trained policies
✅ You need asymmetric braking behavior
✅ You don't have electrical parameters

### Can I Mix Both?

⚠️ **Current implementation does NOT support mixing actuator types per-joint.**

All joints must use the same `actuator_family`. If you need mixed actuators, you'll need to:
1. Extend `ActuatorFactory` to accept per-joint family
2. Update `mjx_env.py` to handle mixed planner states
3. Implement per-joint parameter loading in `robot.py`

---

## Troubleshooting

### Error: "Unknown actuator_family"

Check `actuator_family` in your config is exactly `"dynamixel"` or `"feetech"` (case-sensitive).

### Error: KeyError for motor parameters

Ensure your motor model (e.g., `feetech_sts3250`) is defined in `default.yml` under `actuators:`.

### Torques are NaN/Inf

- Check that `R > 0` (resistance must be positive)
- Verify `kt`, `vin` are reasonable values
- Add safety checks: `torque = jnp.where(jnp.isfinite(torque), torque, 0.0)`

### Planner doesn't converge

- Increase `vmax` and `amax` for faster convergence
- Check `dt` matches your control rate (default: 0.02s = 50Hz)
- Verify deadband isn't too large

### Physics explodes in simulation

- Reduce `amax` (too aggressive acceleration)
- Check joint `armature`, `damping`, `frictionloss` values
- Ensure `max_pwm` isn't too high (typical: 0.95-0.97)

---

## Architecture Reference

### File Structure

```
toddlerbot/
├── sim/
│   ├── actuators/
│   │   ├── __init__.py           # ActuatorInterface, ActuatorFactory
│   │   ├── dynamixel.py          # DynamixelActuators
│   │   ├── feetech.py            # FeetechActuators
│   │   └── planner_state.py      # PlannerState dataclass
│   ├── robot.py                  # Robot config loader (updated)
│   └── motor_control.py          # Legacy MotorController (kept for compatibility)
├── locomotion/
│   ├── mjx_env.py                # MJX environment (updated)
│   └── mjx_config.py             # DomainRandConfig (updated)
├── descriptions/
│   └── default.yml               # Motor definitions (updated)
└── test_feetech_integration.py   # Test script
```

### Key Changes by File

| File | Change | Purpose |
|------|--------|---------|
| `default.yml` | Added `actuator_family`, Feetech motor types | Configuration |
| `robot.py` | Load Feetech params (kt, R, vin, etc.) | Parameter loading |
| `actuators/__init__.py` | ActuatorInterface, ActuatorFactory | Abstraction layer |
| `actuators/feetech.py` | Trapezoidal planner + electrical model | Feetech implementation |
| `actuators/dynamixel.py` | Wrapped existing MotorController | Backwards compatibility |
| `mjx_env.py` | Planner state init/update, physics override | Environment integration |
| `mjx_config.py` | Feetech domain rand ranges | Randomization |

---

## Next Steps

1. ✅ **Basic integration complete** - All core components implemented
2. 🔄 **Test with existing robot** - Verify on your robot configuration
3. 🔄 **Hardware validation** - Single-joint bench test
4. 🔄 **Full training run** - 10M steps, compare to Dynamixel baseline
5. 🔄 **Real robot deployment** - Staged testing (arms → legs → walking)
6. ⏳ **Advanced features** (future):
   - Per-joint actuator types (mixed Dynamixel/Feetech)
   - Back-EMF modeling
   - Thermal limits
   - Current sensor integration for hardware

---

## References

- **Feetech implementation source**: `/home/oliverz/humanoid_labs/zeroth/kbot-sim/zbot-policy-walking/train.py`
- **Migration plan**: This implementation follows the comprehensive plan in the initial discussion
- **Original Dynamixel model**: `toddlerbot/sim/motor_control.py:MotorController`

---

## Contact

For questions or issues with Feetech integration, please refer to:
- GitHub Issues: `https://github.com/hshi74/toddlerbot/issues`
- Migration plan document (see conversation history)

**Last updated**: 2025-11-22
