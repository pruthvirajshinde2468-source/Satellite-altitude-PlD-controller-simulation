# Satellite Attitude PID Controller Simulation

A comprehensive simulation of satellite attitude (orientation) control using a PID (Proportional-Integral-Derivative) controller. This project demonstrates how satellites maintain their desired orientation despite external disturbances.

## Project Overview

### Objective
Maintain the satellite's orientation (attitude) at a desired angle despite external disturbances like solar pressure, gravity gradients, and magnetic torques.

### Challenge
External disturbances constantly try to change the satellite's orientation:
- **Solar Pressure Torque**: Force from solar radiation pressure
- **Gravity Gradient Torque**: Differential gravitational force across satellite dimensions
- **Magnetic Torques**: Interaction with Earth's magnetic field

### Solution
Use thrusters or reaction wheels to apply corrective torques through a PID controller that continuously adjusts control outputs based on current attitude error.

---

## PID Controller Theory

### What is a PID Controller?
A PID controller computes a control output based on three terms:

```
u(t) = Kp × e(t) + Ki × ∫e(t)dt + Kd × de(t)/dt
```

Where:
- **u(t)**: Control output (torque in our case)
- **e(t)**: Error (setpoint - current angle)
- **Kp, Ki, Kd**: Tuning parameters

### The Three Terms

#### 1. **Proportional Term (P)**
```
P_term = Kp × error
```
- **What it does**: Provides immediate response proportional to current error
- **Effect**: 
  - Larger Kp → Faster response, but can cause oscillations
  - Smaller Kp → Slower response, less oscillation
- **Problem**: Always leaves some steady-state error

#### 2. **Integral Term (I)**
```
I_term = Ki × ∫error dt
```
- **What it does**: Accumulates past errors over time
- **Effect**: Eliminates steady-state error
- **Challenge**: "Integral windup" - integral keeps growing when actuator is saturated
- **Solution**: Anti-windup saturation limit

#### 3. **Derivative Term (D)**
```
D_term = Kd × de(t)/dt
```
- **What it does**: Predicts future error based on rate of change
- **Effect**: 
  - Damps oscillations
  - Improves stability
  - Reduces overshoot
- **Challenge**: Sensitive to measurement noise

---

## Satellite Dynamics

The satellite dynamics are governed by the rotational equation of motion:

```
I × α = τ_control - τ_disturbance - damping
```

Where:
- **I**: Moment of inertia (kg·m²) - resistance to rotation
- **α**: Angular acceleration (rad/s²) - d²θ/dt²
- **τ_control**: Control torque from PID controller (N·m)
- **τ_disturbance**: External disturbance torques (N·m)
- **damping**: Natural damping in the system (N·m·s/rad)

### Rearranged Form
```
d²θ/dt² = (τ_control - b × dθ/dt) / I
```

Or in simple terms:
```
turn_speed_change = (our_force - natural_slowdown × current_speed) / satellite_weight
```

---

## Control Loop Process

```
┌─────────────────┐
│  Disturbances   │
└────────┬────────┘
         │
         ▼
    ┌─────────┐
    │Satellite│──┐
    └─────────┘  │
                 ▼
         ┌───────────────┐
         │Current Angle  │
         └───────┬───────┘
                 │
        ┌────────▼────────┐
        │Calculate Error  │
        │setpoint - angle │
        └────────┬────────┘
                 │
        ┌────────▼────────┐
        │  PID Controller │
        │  Compute u(t)   │
        └────────┬────────┘
                 │
        ┌────────▼──────────┐
        │Apply Control     │
        │Torque to         │
        │Satellite         │
        └─────────┬────────┘
                  │
                  └─────────┐
                            │
                     (repeat)
```

### Step-by-Step Process
1. **Measure** current satellite angle θ
2. **Calculate error**: e = desired_angle - current_angle
3. **Compute PID terms**:
   - P: Immediate correction based on current error
   - I: Correction based on accumulated past errors
   - D: Predictive correction based on error trend
4. **Apply** the calculated control torque
5. **Repeat** continuously at high frequency

---

## Project Structure

```
satellite-altitude-pid-controller-simulation/
├── satellite_dynamics.py      # Satellite model and disturbance model
├── pid_controller.py          # PID controller implementation
├── simulation.py              # Main simulation loop
├── visualization.py           # Plotting and analysis tools
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

### Module Descriptions

#### `satellite_dynamics.py`
- **SatelliteDynamics**: Models satellite attitude dynamics
- **SatelliteState**: Stores current angle, velocity, and acceleration
- **DisturbanceModel**: Generates realistic disturbance torques (solar, gravity, magnetic)

#### `pid_controller.py`
- **PIDController**: Implements the PID control algorithm
- **PIDGains**: Stores Kp, Ki, Kd tuning parameters
- **PIDState**: Tracks integral accumulation and previous error

#### `simulation.py`
- **SatelliteSimulation**: Main simulation class
  - Integrates satellite dynamics, PID controller, and disturbances
  - Implements the control loop
  - Logs data for analysis
  - Computes performance statistics

#### `visualization.py`
- Functions for plotting and analyzing results:
  - `plot_satellite_attitude()`: Angle and error over time
  - `plot_pid_terms()`: Individual PID components
  - `plot_torques()`: Control vs disturbance torques
  - `plot_velocity()`: Angular velocity profile
  - `plot_comprehensive_analysis()`: All-in-one summary plot

---

## Getting Started

### Installation

1. Clone or download the project
2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Running the Simulation

```bash
python simulation.py
```

This will:
1. Run a 150-second simulation
2. Print progress every 10 seconds
3. Display performance statistics
4. Save results to `satellite_simulation.json`

### Visualizing Results

```bash
python visualization.py
```

This will:
1. Load the simulation results
2. Generate multiple analysis plots
3. Save PNG files with visualizations
4. Display detailed statistics

---

## Performance Metrics

The simulation computes several key metrics:

| Metric | Description |
|--------|-------------|
| **Final Angle** | Satellite attitude at end of simulation |
| **Final Error** | Remaining angle error at end |
| **Max Error** | Maximum angle error during simulation |
| **Mean Error** | Average absolute error |
| **Steady-State Error** | Error in final 10% of simulation |
| **Overshoot** | How much angle exceeds setpoint |
| **Settling Time** | Time for angle to settle within tolerance |

---

## Tuning the PID Controller

The current gains are:
- **Kp = 0.8**: Proportional gain
- **Ki = 0.05**: Integral gain
- **Kd = 1.2**: Derivative gain

### Tuning Guidelines

**Increase Kp if:**
- Response is too slow
- Steady-state error is large

**Decrease Kp if:**
- Response oscillates too much
- System becomes unstable

**Increase Ki if:**
- Steady-state error remains after P tuning
- System settles slowly

**Decrease Ki if:**
- System overshoot increases
- Integral windup occurs

**Increase Kd if:**
- Oscillations occur
- Overshoot is excessive

**Decrease Kd if:**
- Response is too sluggish
- Noise amplification is problematic

### Tuning Script Example

```python
from simulation import SatelliteSimulation

sim = SatelliteSimulation()

# Try different gains
for kp in [0.5, 0.8, 1.2]:
    for kd in [0.8, 1.2, 1.5]:
        sim.set_pid_gains(kp=kp, kd=kd)
        stats = sim.run(duration=150.0)
        print(f"Kp={kp}, Kd={kd}: Settling time = {stats['settling_time']}s")
```

---

## Key Formulas Reference

### Satellite Dynamics
```
θ̈ = (τ_control - b × θ̇) / I

Where:
θ = satellite angle (radians)
θ̇ = angular velocity (rad/s)
θ̈ = angular acceleration (rad/s²)
I = moment of inertia
b = damping coefficient
τ_control = control torque (N·m)
```

### PID Control Law
```
u(t) = Kp × e(t) + Ki × ∫e(τ) dτ + Kd × de(t)/dt

Where:
u(t) = control output (torque)
e(t) = error = setpoint - measured
Kp = proportional gain
Ki = integral gain
Kd = derivative gain
```

### Disturbance Model
```
τ_disturbance = τ_solar + τ_gravity + τ_magnetic

τ_solar(t) = A_s × sin(ω_s × t)
τ_gravity(t) = A_g × sin(ω_g × t)
τ_magnetic(t) = A_m × sin(ω_m × t)
```

---

## Expected Results

A well-tuned PID controller should:

1. **Settle quickly**: Reach desired angle in 10-20 seconds
2. **Minimize overshoot**: Not exceed setpoint by more than 5-10%
3. **Reject disturbances**: Maintain attitude despite continuous disturbances
4. **Achieve low steady-state error**: Error < 0.1° in steady state
5. **Handle saturation**: Gracefully degrade if control torque is limited

---

## Physical Interpretation

### Real-World Application
This simulation models actual satellite attitude control systems used in:
- **Earth observation satellites**: Must point at targets
- **Communication satellites**: Require precise antenna pointing
- **Scientific spacecraft**: Need stable measurement platforms
- **Spacecraft during orbital maneuvers**: Maintain attitude control

### Real Satellite Example
A typical small satellite might have:
- Inertia: 50-200 kg·m²
- Max disturbance torques: 0.001-0.01 N·m
- Available control torque: 0.01-0.1 N·m
- Attitude accuracy requirement: ±1 degree or better

---

## Troubleshooting

### If simulation diverges (angle grows unbounded):
- Reduce Kp and Kd
- Check integral windup limit
- Verify actuator saturation limit

### If settlement is too slow:
- Increase Kp
- Increase Kd slightly
- Check if control torque is saturated

### If oscillations occur:
- Reduce Kp
- Increase Kd (derivative damping)
- Check for measurement noise (in real systems)

---

## Future Enhancements

Possible extensions to this project:

1. **Multi-axis control**: 3-axis attitude control (roll, pitch, yaw)
2. **Quaternion representation**: More accurate for large rotations
3. **Adaptive control**: Automatically tune gains based on performance
4. **Noise simulation**: Add measurement and actuator noise
5. **Reaction wheel model**: Detailed reaction wheel dynamics
6. **Thruster plume interaction**: Model thruster plume effects
7. **Magnetic torquer model**: Add magnetic field interaction
8. **State estimation**: Implement attitude observer

---

## References

- Wie, B. "Space Vehicle Dynamics and Control" (2nd ed.)
- Sidi, M. J. "Spacecraft Dynamics and Control"
- Bennett, F. V. "Spacecraft Attitude Dynamics and Control"
- Astrom, K. J., & Murray, R. M. "Feedback Systems: An Introduction for Scientists and Engineers"

---

## License

This project is provided as educational material for understanding satellite attitude control and PID controller fundamentals.

---

## Author Notes

This simulation provides a clear, educational implementation of satellite attitude control. While simplified compared to real-world systems, it captures the essential physics and control principles used in actual spacecraft attitude control systems.

The modular design allows easy modifications for different scenarios, tuning experiments, or extensions to more complex dynamics.
