"""
Example Scenarios and Tuning Guide

This script demonstrates different ways to use the satellite PID controller
simulation with various scenarios and tuning parameters.
"""

import numpy as np
from simulation import SatelliteSimulation
from visualization import plot_comprehensive_analysis, load_simulation_data, print_detailed_statistics


def scenario_1_baseline():
    """
    Scenario 1: Baseline controller performance
    
    Default PID gains optimized for general performance
    """
    print("\n" + "=" * 70)
    print("SCENARIO 1: BASELINE CONTROLLER")
    print("=" * 70)
    
    sim = SatelliteSimulation(
        inertia=100.0,
        damping=0.5,
        desired_angle=0.0
    )
    
    print("Configuration:")
    print("  Inertia: 100 kg·m²")
    print("  Damping: 0.5 N·m·s/rad")
    print("  Desired Angle: 0.0° (nadir pointing)")
    
    stats = sim.run(duration=150.0)
    sim.print_summary(stats)
    sim.save_data("scenario1_baseline.json")
    
    return stats


def scenario_2_aggressive_control():
    """
    Scenario 2: Aggressive control (high gains)
    
    Higher proportional gain for faster response
    Useful for agile satellite maneuvers
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: AGGRESSIVE CONTROL (High Gains)")
    print("=" * 70)
    
    sim = SatelliteSimulation(
        inertia=100.0,
        damping=0.5,
        desired_angle=0.0
    )
    
    # Set more aggressive gains
    sim.set_pid_gains(kp=1.5, ki=0.1, kd=1.8)
    
    print("Configuration:")
    print("  Inertia: 100 kg·m²")
    print("  Damping: 0.5 N·m·s/rad")
    print("  Desired Angle: 0.0°")
    print("  PID Gains: Kp=1.5, Ki=0.1, Kd=1.8 (Aggressive)")
    
    stats = sim.run(duration=150.0)
    sim.print_summary(stats)
    sim.save_data("scenario2_aggressive.json")
    
    return stats


def scenario_3_conservative_control():
    """
    Scenario 3: Conservative control (low gains)
    
    Lower gains for stable, smooth response
    Useful for sensitive payloads
    """
    print("\n" + "=" * 70)
    print("SCENARIO 3: CONSERVATIVE CONTROL (Low Gains)")
    print("=" * 70)
    
    sim = SatelliteSimulation(
        inertia=100.0,
        damping=0.5,
        desired_angle=0.0
    )
    
    # Set conservative gains
    sim.set_pid_gains(kp=0.4, ki=0.02, kd=0.6)
    
    print("Configuration:")
    print("  Inertia: 100 kg·m²")
    print("  Damping: 0.5 N·m·s/rad")
    print("  Desired Angle: 0.0°")
    print("  PID Gains: Kp=0.4, Ki=0.02, Kd=0.6 (Conservative)")
    
    stats = sim.run(duration=150.0)
    sim.print_summary(stats)
    sim.save_data("scenario3_conservative.json")
    
    return stats


def scenario_4_low_damping_satellite():
    """
    Scenario 4: Low-damping satellite
    
    Satellite with minimal natural damping (e.g., in space)
    Requires more aggressive derivative control
    """
    print("\n" + "=" * 70)
    print("SCENARIO 4: LOW-DAMPING SATELLITE (Space Environment)")
    print("=" * 70)
    
    sim = SatelliteSimulation(
        inertia=100.0,
        damping=0.05,  # Very low damping
        desired_angle=0.0
    )
    
    # Increase derivative gain for stability
    sim.set_pid_gains(kp=0.8, ki=0.05, kd=2.0)
    
    print("Configuration:")
    print("  Inertia: 100 kg·m²")
    print("  Damping: 0.05 N·m·s/rad (Low - Space Environment)")
    print("  Desired Angle: 0.0°")
    print("  PID Gains: Kp=0.8, Ki=0.05, Kd=2.0 (High derivative)")
    
    stats = sim.run(duration=150.0)
    sim.print_summary(stats)
    sim.save_data("scenario4_low_damping.json")
    
    return stats


def scenario_5_heavy_satellite():
    """
    Scenario 5: Large, heavy satellite
    
    Higher inertia requires different tuning
    More difficult to control due to momentum
    """
    print("\n" + "=" * 70)
    print("SCENARIO 5: HEAVY SATELLITE (High Inertia)")
    print("=" * 70)
    
    sim = SatelliteSimulation(
        inertia=500.0,  # Much higher inertia
        damping=1.0,
        desired_angle=0.0
    )
    
    # Scale up proportional gain for heavier satellite
    sim.set_pid_gains(kp=2.0, ki=0.1, kd=1.5)
    
    print("Configuration:")
    print("  Inertia: 500 kg·m² (Heavy Satellite)")
    print("  Damping: 1.0 N·m·s/rad")
    print("  Desired Angle: 0.0°")
    print("  PID Gains: Kp=2.0, Ki=0.1, Kd=1.5 (Scaled for inertia)")
    
    stats = sim.run(duration=200.0)  # Longer duration for heavier satellite
    sim.print_summary(stats)
    sim.save_data("scenario5_heavy_satellite.json")
    
    return stats


def scenario_6_angle_tracking():
    """
    Scenario 6: Angle tracking (not nadir-pointing)
    
    Satellite maintaining non-zero desired angle
    Simulates attitude maneuvers
    """
    print("\n" + "=" * 70)
    print("SCENARIO 6: ANGLE TRACKING (Non-zero Setpoint)")
    print("=" * 70)
    
    desired_angle_deg = 30.0
    desired_angle_rad = np.radians(desired_angle_deg)
    
    sim = SatelliteSimulation(
        inertia=100.0,
        damping=0.5,
        desired_angle=desired_angle_rad  # 30 degrees
    )
    
    print("Configuration:")
    print("  Inertia: 100 kg·m²")
    print("  Damping: 0.5 N·m·s/rad")
    print(f"  Desired Angle: {desired_angle_deg}° (Non-zero setpoint)")
    print("  PID Gains: Kp=0.8, Ki=0.05, Kd=1.2 (Default)")
    
    stats = sim.run(duration=150.0)
    sim.print_summary(stats)
    sim.save_data("scenario6_angle_tracking.json")
    
    return stats


def compare_scenarios():
    """
    Compare all scenarios side-by-side
    """
    print("\n\n" + "=" * 70)
    print("COMPARISON OF ALL SCENARIOS")
    print("=" * 70)
    
    results = {
        'Baseline': scenario_1_baseline(),
        'Aggressive': scenario_2_aggressive_control(),
        'Conservative': scenario_3_conservative_control(),
        'Low Damping': scenario_4_low_damping_satellite(),
        'Heavy': scenario_5_heavy_satellite(),
        'Angle Track': scenario_6_angle_tracking(),
    }
    
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"{'Scenario':<15} {'Settling Time (s)':<20} {'Max Error (°)':<20} {'Steady-State (°)':<20}")
    print("-" * 75)
    
    for scenario_name, stats in results.items():
        print(f"{scenario_name:<15} {stats['settling_time']:<20.2f} {np.degrees(stats['max_error']):<20.2f} {np.degrees(stats['steady_state_error']):<20.4f}")
    
    print("=" * 70)


def manual_tuning_guide():
    """
    Interactive tuning guide to find optimal gains
    """
    print("\n" + "=" * 70)
    print("MANUAL TUNING GUIDE")
    print("=" * 70)
    print("""
This guide shows how to systematically tune PID gains:

1. START WITH PROPORTIONAL GAIN (Kp)
   - Set Ki=0, Kd=0
   - Increase Kp until oscillations appear
   - Set Kp to about half the oscillation point
   
2. ADD DERIVATIVE GAIN (Kd)
   - Increase Kd to dampen oscillations
   - Too much Kd = sluggish response
   
3. ADD INTEGRAL GAIN (Ki)
   - Increase Ki to eliminate steady-state error
   - Too much Ki = overshoot and instability

Example script:
""")
    
    print("""
from simulation import SatelliteSimulation
import numpy as np

# Test different Kp values
print("Testing Kp values (Ki=0, Kd=0):")
for kp in np.linspace(0.2, 2.0, 10):
    sim = SatelliteSimulation()
    sim.set_pid_gains(kp=kp, ki=0, kd=0)
    stats = sim.run(duration=100)
    settling = stats['settling_time']
    error = np.degrees(stats['steady_state_error'])
    print(f"Kp={kp:.2f}: Settling={settling:.1f}s, Error={error:.4f}°")

# Once Kp is chosen, test Kd
print("\\nTesting Kd values (Kp=0.8, Ki=0):")
for kd in np.linspace(0.0, 3.0, 10):
    sim = SatelliteSimulation()
    sim.set_pid_gains(kp=0.8, ki=0, kd=kd)
    stats = sim.run(duration=100)
    print(f"Kd={kd:.2f}: Settling={stats['settling_time']:.1f}s")

# Finally test Ki for steady-state elimination
print("\\nTesting Ki values (Kp=0.8, Kd=1.2):")
for ki in np.linspace(0.0, 0.2, 10):
    sim = SatelliteSimulation()
    sim.set_pid_gains(kp=0.8, ki=ki, kd=1.2)
    stats = sim.run(duration=100)
    sse = np.degrees(stats['steady_state_error'])
    print(f"Ki={ki:.3f}: SSE={sse:.4f}°")
    """)


if __name__ == "__main__":
    # Run all scenarios
    compare_scenarios()
    
    # Print tuning guide
    manual_tuning_guide()
    
    print("\n" + "=" * 70)
    print("SCENARIO EXECUTION COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - scenario1_baseline.json")
    print("  - scenario2_aggressive.json")
    print("  - scenario3_conservative.json")
    print("  - scenario4_low_damping.json")
    print("  - scenario5_heavy_satellite.json")
    print("  - scenario6_angle_tracking.json")
    print("\nTo visualize results, use: python visualization.py")
