"""
Satellite PID Controller - Quick Start Guide

This file provides quick examples to get started with the simulator.
"""

# =============================================================================
# EXAMPLE 1: Basic Simulation Run
# =============================================================================
"""
from simulation import SatelliteSimulation

# Create simulator with default parameters
sim = SatelliteSimulation()

# Run for 100 seconds
stats = sim.run(duration=100.0)

# Print summary
sim.print_summary(stats)

# Save results
sim.save_data("my_simulation.json")
"""


# =============================================================================
# EXAMPLE 2: Modify Satellite Parameters
# =============================================================================
"""
from simulation import SatelliteSimulation

# Create simulator with custom parameters
sim = SatelliteSimulation(
    inertia=150.0,      # kg·m² (heavier satellite)
    damping=0.8,        # N·m·s/rad (more damping)
    desired_angle=0.0   # radians (nadir pointing)
)

# Run simulation
stats = sim.run(duration=150.0)
sim.print_summary(stats)
"""


# =============================================================================
# EXAMPLE 3: Tune PID Gains
# =============================================================================
"""
from simulation import SatelliteSimulation
import numpy as np

sim = SatelliteSimulation()

# Set custom PID gains
sim.set_pid_gains(
    kp=1.0,    # Proportional gain
    ki=0.1,    # Integral gain
    kd=1.5     # Derivative gain
)

# Run simulation
stats = sim.run(duration=150.0)
print(f"Settling time: {stats['settling_time']:.2f} seconds")
print(f"Max error: {np.degrees(stats['max_error']):.2f} degrees")
print(f"Steady-state error: {np.degrees(stats['steady_state_error']):.4f} degrees")
"""


# =============================================================================
# EXAMPLE 4: Visualize Results
# =============================================================================
"""
from visualization import *

# Load previous simulation data
data = load_simulation_data("my_simulation.json")

# Print detailed statistics
print_detailed_statistics(data)

# Create individual plots
plot_satellite_attitude(data)
plot_pid_terms(data)
plot_torques(data)
plot_velocity(data)

# Or create comprehensive plot
plot_comprehensive_analysis(data)
"""


# =============================================================================
# EXAMPLE 5: Gain Tuning Loop
# =============================================================================
"""
from simulation import SatelliteSimulation
import numpy as np

print("Tuning Kp (Kd and Ki held constant)")
print("=" * 50)

results = []
for kp in np.linspace(0.5, 1.5, 6):
    sim = SatelliteSimulation()
    sim.set_pid_gains(kp=kp, ki=0.05, kd=1.2)
    stats = sim.run(duration=100.0)
    results.append({
        'kp': kp,
        'settling_time': stats['settling_time'],
        'max_error': np.degrees(stats['max_error']),
        'overshoot': np.degrees(stats['overshoot'])
    })
    print(f"Kp={kp:.2f}: Settling={stats['settling_time']:.1f}s, "
          f"MaxErr={np.degrees(stats['max_error']):.2f}°, "
          f"Overshoot={np.degrees(stats['overshoot']):.2f}°")

# Find best Kp
best = min(results, key=lambda x: x['settling_time'])
print(f"\\nBest Kp: {best['kp']:.2f} (settling time: {best['settling_time']:.1f}s)")
"""


# =============================================================================
# EXAMPLE 6: Compare Multiple Scenarios
# =============================================================================
"""
from simulation import SatelliteSimulation
import numpy as np

# Scenario A: Nominal satellite
print("Scenario A: Nominal Satellite")
sim_a = SatelliteSimulation(inertia=100, damping=0.5)
stats_a = sim_a.run(duration=150)
print(f"Settling time: {stats_a['settling_time']:.2f}s")

# Scenario B: Heavy satellite (requires different tuning)
print("\\nScenario B: Heavy Satellite")
sim_b = SatelliteSimulation(inertia=300, damping=1.0)
sim_b.set_pid_gains(kp=2.0, ki=0.1, kd=1.5)  # Scaled gains
stats_b = sim_b.run(duration=200)
print(f"Settling time: {stats_b['settling_time']:.2f}s")

# Scenario C: Low-damping space environment
print("\\nScenario C: Low-Damping Satellite")
sim_c = SatelliteSimulation(inertia=100, damping=0.05)
sim_c.set_pid_gains(kp=0.8, ki=0.05, kd=2.0)  # More derivative damping
stats_c = sim_c.run(duration=150)
print(f"Settling time: {stats_c['settling_time']:.2f}s")
"""


# =============================================================================
# EXAMPLE 7: Analyze Disturbances
# =============================================================================
"""
from satellite_dynamics import DisturbanceModel
import numpy as np

# Create disturbance model
disturbance = DisturbanceModel(
    max_solar_torque=0.001,
    max_gravity_torque=0.0015,
    max_magnetic_torque=0.001
)

# Get individual components over time
print("Disturbance Analysis")
print("=" * 50)
print("Time(s)  | Solar(Nm) | Gravity(Nm) | Magnetic(Nm) | Total(Nm)")
print("-" * 55)

for t in np.linspace(0, 100, 11):
    components = disturbance.get_disturbance_components(t)
    print(f"{t:6.1f} | {components['solar']:9.6f} | "
          f"{components['gravity']:11.6f} | "
          f"{components['magnetic']:12.6f} | {components['total']:9.6f}")
"""


# =============================================================================
# EXAMPLE 8: Step-by-Step Manual Simulation
# =============================================================================
"""
from satellite_dynamics import SatelliteDynamics, DisturbanceModel
from pid_controller import PIDController
import numpy as np

# Create components
satellite = SatelliteDynamics(inertia=100.0, damping_coeff=0.5)
controller = PIDController(kp=0.8, ki=0.05, kd=1.2)
disturbance_model = DisturbanceModel()

# Simulation parameters
dt = 0.01  # 10 ms time step
t = 0.0
setpoint = 0.0

print("Manual Step-by-Step Simulation")
print("=" * 60)
print("Time(s) | Angle(°) | Error(°) | Control(Nm) | Disturbance(Nm)")
print("-" * 60)

for step in range(15000):  # 150 seconds
    # Get current angle
    current_angle = satellite.state.angle
    
    # Compute PID control
    control_torque = controller.compute(setpoint, current_angle, dt)
    
    # Get disturbance
    dist_torque = disturbance_model.get_total_disturbance(t)
    
    # Update satellite dynamics
    satellite.update(control_torque, dist_torque, dt)
    
    # Print every 1000 steps (10 seconds)
    if step % 1000 == 0:
        error = setpoint - current_angle
        print(f"{t:7.2f} | {np.degrees(current_angle):8.2f} | "
              f"{np.degrees(error):8.2f} | {control_torque:11.6f} | "
              f"{dist_torque:15.6f}")
    
    t += dt

print("-" * 60)
print(f"Final angle: {np.degrees(satellite.state.angle):.2f}°")
print(f"Final error: {np.degrees(setpoint - satellite.state.angle):.4f}°")
"""


if __name__ == "__main__":
    print("""
    Satellite PID Controller - Quick Start Guide
    =============================================
    
    This file contains 8 example use cases:
    
    1. Basic Simulation Run
       - Create simulator with defaults and run
    
    2. Modify Satellite Parameters
       - Change inertia, damping, desired angle
    
    3. Tune PID Gains
       - Set custom Kp, Ki, Kd values
    
    4. Visualize Results
       - Create plots from simulation data
    
    5. Gain Tuning Loop
       - Automatically test different Kp values
    
    6. Compare Multiple Scenarios
       - Compare nominal, heavy, and low-damping satellites
    
    7. Analyze Disturbances
       - Examine solar, gravity, and magnetic torque components
    
    8. Step-by-Step Manual Simulation
       - Manual control loop implementation
    
    GETTING STARTED:
    
    1. Install dependencies:
       pip install -r requirements.txt
    
    2. Run simulation:
       python simulation.py
    
    3. Visualize results:
       python visualization.py
    
    4. Run all scenarios:
       python scenarios.py
    
    5. Copy-paste examples from this file to get started!
    """)
