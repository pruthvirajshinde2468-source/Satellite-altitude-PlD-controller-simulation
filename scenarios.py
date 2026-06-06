"""
Scenarios, Comparison Studies, and Robustness Analysis

New in this version:
  - Multi-step attitude manoeuvre sequence
  - Monte Carlo robustness study (parameter uncertainty)
  - Degraded-actuator scenario (reduced wheel torque)
  - Frequency-domain comparison via Bode plots
  - Integrator accuracy comparison (Euler vs. RK4)
"""

import numpy as np
from simulation import SatelliteSimulation
from analysis import compare_pid_bode, print_stability_report
from visualization import (plot_comprehensive_analysis, load_simulation_data,
                            print_detailed_statistics)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def _run(label: str, sim: SatelliteSimulation, duration: float = 150.0,
         save: str = None) -> dict:
    stats = sim.run(duration=duration)
    sim.print_summary(stats)
    if save:
        sim.save_data(save)
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# Existing scenarios (updated to use new API)
# ──────────────────────────────────────────────────────────────────────────────

def scenario_1_baseline() -> dict:
    """Default PID gains — general-purpose reference."""
    _header("SCENARIO 1: BASELINE CONTROLLER")
    sim = SatelliteSimulation(inertia=100.0, damping=0.5, desired_angle=0.0)
    return _run("Baseline", sim, save="scenario1_baseline.json")


def scenario_2_aggressive() -> dict:
    """High gains — fast response, more overshoot."""
    _header("SCENARIO 2: AGGRESSIVE CONTROL  (Kp=1.5, Ki=0.10, Kd=1.8)")
    sim = SatelliteSimulation(inertia=100.0, damping=0.5, desired_angle=0.0)
    sim.set_pid_gains(kp=1.5, ki=0.1, kd=1.8)
    return _run("Aggressive", sim, save="scenario2_aggressive.json")


def scenario_3_conservative() -> dict:
    """Low gains — slow, smooth response."""
    _header("SCENARIO 3: CONSERVATIVE CONTROL  (Kp=0.4, Ki=0.02, Kd=0.6)")
    sim = SatelliteSimulation(inertia=100.0, damping=0.5, desired_angle=0.0)
    sim.set_pid_gains(kp=0.4, ki=0.02, kd=0.6)
    return _run("Conservative", sim, save="scenario3_conservative.json")


def scenario_4_low_damping() -> dict:
    """Very low structural damping (space environment)."""
    _header("SCENARIO 4: LOW-DAMPING SATELLITE  (b=0.05)")
    sim = SatelliteSimulation(inertia=100.0, damping=0.05, desired_angle=0.0)
    sim.set_pid_gains(kp=0.8, ki=0.05, kd=2.0)
    return _run("LowDamping", sim, save="scenario4_low_damping.json")


def scenario_5_heavy() -> dict:
    """High-inertia satellite — requires higher gains."""
    _header("SCENARIO 5: HEAVY SATELLITE  (I=500 kg·m²)")
    sim = SatelliteSimulation(inertia=500.0, damping=1.0, desired_angle=0.0)
    sim.set_pid_gains(kp=2.0, ki=0.1, kd=1.5)
    return _run("Heavy", sim, duration=200.0, save="scenario5_heavy.json")


def scenario_6_angle_tracking() -> dict:
    """Non-zero setpoint (attitude manoeuvre to 30°)."""
    _header("SCENARIO 6: ANGLE TRACKING  (setpoint = 30°)")
    sim = SatelliteSimulation(inertia=100.0, damping=0.5,
                              desired_angle=np.radians(30.0))
    return _run("Tracking", sim, save="scenario6_angle_tracking.json")


# ──────────────────────────────────────────────────────────────────────────────
# New scenarios
# ──────────────────────────────────────────────────────────────────────────────

def scenario_7_manoeuvre_sequence() -> dict:
    """
    Scenario 7: Multi-step attitude manoeuvre sequence.

    The satellite targets three successive setpoints:
        0 s → 0°  (initial, nadir-pointing)
       40 s → 20° (observation mode)
       90 s → -15° (another target)
      130 s → 0°  (return to nadir)
    """
    _header("SCENARIO 7: MULTI-STEP MANOEUVRE SEQUENCE")
    sim = SatelliteSimulation(inertia=100.0, damping=0.5, desired_angle=0.0)
    sim.schedule_manoeuvre(40.0,  np.radians(20.0))
    sim.schedule_manoeuvre(90.0,  np.radians(-15.0))
    sim.schedule_manoeuvre(130.0, np.radians(0.0))
    stats = _run("ManoeuvreSeq", sim, duration=170.0,
                 save="scenario7_manoeuvre_sequence.json")
    return stats


def scenario_8_sensor_noise() -> dict:
    """
    Scenario 8: Effect of sensor measurement noise.

    Compares clean vs. noisy control to illustrate the importance of the
    derivative filter.
    """
    _header("SCENARIO 8: SENSOR NOISE COMPARISON")

    sim_clean = SatelliteSimulation(inertia=100.0, damping=0.5,
                                    use_noise=False)
    stats_clean = sim_clean.run(duration=100.0)

    sim_noisy = SatelliteSimulation(inertia=100.0, damping=0.5,
                                    use_noise=True)
    stats_noisy = sim_noisy.run(duration=100.0)

    print("\nNoise impact:")
    print(f"  Clean — settling={stats_clean['settling_time']:.1f}s  "
          f"SSE={np.degrees(stats_clean['steady_state_error']):.4f}°  "
          f"IAE={stats_clean['IAE']:.4f}")
    print(f"  Noisy — settling={stats_noisy['settling_time']:.1f}s  "
          f"SSE={np.degrees(stats_noisy['steady_state_error']):.4f}°  "
          f"IAE={stats_noisy['IAE']:.4f}")
    return stats_noisy


def scenario_9_degraded_actuator() -> dict:
    """
    Scenario 9: Degraded reaction wheel (50 % torque loss).

    Simulates partial actuator failure — the wheel can only deliver
    half the commanded torque.
    """
    _header("SCENARIO 9: DEGRADED ACTUATOR  (max_torque = 0.05 N·m)")
    sim = SatelliteSimulation(inertia=100.0, damping=0.5, desired_angle=0.0)
    sim.wheel.max_torque = 0.05  # halved
    return _run("Degraded", sim, duration=200.0,
                save="scenario9_degraded_actuator.json")


def scenario_10_integrator_comparison() -> dict:
    """
    Scenario 10: Euler vs. RK4 accuracy comparison.

    Uses a large time step (dt=0.1 s) to make the difference visible.
    At dt=0.01 s the results are nearly identical; errors emerge at dt=0.1 s.
    """
    _header("SCENARIO 10: EULER vs. RK4  (dt = 0.1 s)")

    results = {}
    for method in ('euler', 'rk4'):
        sim = SatelliteSimulation(inertia=100.0, damping=0.5,
                                  desired_angle=0.0, integrator=method)
        sim.dt = 0.1  # coarse time step to expose integration error
        stats = sim.run(duration=150.0)
        results[method] = stats
        print(f"\n  {method.upper():5s}  settling={stats['settling_time']:.1f}s  "
              f"SSE={np.degrees(stats['steady_state_error']):.4f}°  "
              f"IAE={stats['IAE']:.4f}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
def monte_carlo_robustness(n_runs: int = 50,
                            inertia_variation: float = 0.20,
                            disturbance_variation: float = 0.50,
                            seed: int = 42) -> list:
    """
    Monte Carlo robustness study.

    Randomly perturbs:
      - Satellite inertia          ± inertia_variation  (fraction)
      - Each disturbance magnitude ± disturbance_variation (fraction)

    Reports statistics on settling time, steady-state error, and IAE.

    Args:
        n_runs:               Number of Monte Carlo samples
        inertia_variation:    ± fraction of nominal inertia  (default ±20 %)
        disturbance_variation: ± fraction of nominal disturbance (default ±50 %)
        seed:                 RNG seed for reproducibility

    Returns:
        List of per-run result dicts.
    """
    _header(f"MONTE CARLO ROBUSTNESS  ({n_runs} runs)")
    print(f"  Inertia variation    : ±{inertia_variation*100:.0f}%")
    print(f"  Disturbance variation: ±{disturbance_variation*100:.0f}%")
    print("-" * 70)

    rng     = np.random.default_rng(seed)
    results = []

    base_inertia = 100.0

    for i in range(n_runs):
        inertia  = base_inertia * (1.0 + rng.uniform(-inertia_variation,
                                                       inertia_variation))
        sim = SatelliteSimulation(inertia=inertia, damping=0.5, desired_angle=0.0)

        # Perturb disturbance magnitudes
        sim.disturbance.max_solar   *= (1.0 + rng.uniform(-disturbance_variation,
                                                            disturbance_variation))
        sim.disturbance.max_gravity *= (1.0 + rng.uniform(-disturbance_variation,
                                                            disturbance_variation))
        sim.disturbance.max_magnetic *= (1.0 + rng.uniform(-disturbance_variation,
                                                             disturbance_variation))

        stats = sim.run(duration=150.0)
        results.append({
            'run':           i + 1,
            'inertia':       inertia,
            'settling_time': stats['settling_time'],
            'sse_deg':       np.degrees(stats['steady_state_error']),
            'IAE':           stats['IAE'],
        })
        print(f"  Run {i+1:3d}/{n_runs}  I={inertia:6.1f} kg·m²  "
              f"settling={stats['settling_time']:6.1f}s  "
              f"SSE={np.degrees(stats['steady_state_error']):.4f}°")

    # Summary statistics
    st  = [r['settling_time'] for r in results]
    sse = [r['sse_deg']       for r in results]
    iae = [r['IAE']           for r in results]

    print("\n" + "=" * 70)
    print("MONTE CARLO SUMMARY")
    print("=" * 70)
    print(f"  Settling Time  :  mean={np.mean(st):6.1f}s  "
          f"std={np.std(st):5.1f}s  max={np.max(st):6.1f}s")
    print(f"  Steady-State   :  mean={np.mean(sse):.4f}°  "
          f"std={np.std(sse):.4f}°  max={np.max(sse):.4f}°")
    print(f"  IAE            :  mean={np.mean(iae):.4f}  "
          f"std={np.std(iae):.4f}  max={np.max(iae):.4f}")
    print("=" * 70)

    return results


# ──────────────────────────────────────────────────────────────────────────────
def compare_scenarios():
    """Run all baseline scenarios and print a comparison table."""
    _header("FULL SCENARIO COMPARISON")

    results = {
        'Baseline':     scenario_1_baseline(),
        'Aggressive':   scenario_2_aggressive(),
        'Conservative': scenario_3_conservative(),
        'Low Damping':  scenario_4_low_damping(),
        'Heavy':        scenario_5_heavy(),
        'Angle Track':  scenario_6_angle_tracking(),
    }

    print("\n" + "=" * 80)
    print("SCENARIO COMPARISON TABLE")
    print("=" * 80)
    hdr = f"{'Scenario':<15}  {'Settling(s)':<14}  {'MaxErr(°)':<12}  {'SSE(°)':<12}  {'IAE':<12}"
    print(hdr)
    print("-" * 80)
    for name, s in results.items():
        print(f"{name:<15}  {s['settling_time']:<14.2f}  "
              f"{np.degrees(s['max_error']):<12.3f}  "
              f"{np.degrees(s['steady_state_error']):<12.4f}  "
              f"{s['IAE']:<12.4f}")
    print("=" * 80)


def frequency_analysis():
    """Bode plots and stability margins for baseline + aggressive + conservative."""
    _header("FREQUENCY-DOMAIN ANALYSIS")

    INERTIA, DAMPING = 100.0, 0.5
    print_stability_report(0.8,  0.05, 1.2, INERTIA, DAMPING)
    print_stability_report(1.5,  0.10, 1.8, INERTIA, DAMPING)
    print_stability_report(0.4,  0.02, 0.6, INERTIA, DAMPING)

    compare_pid_bode([
        {'kp': 0.8,  'ki': 0.05, 'kd': 1.2, 'label': 'Baseline'},
        {'kp': 1.5,  'ki': 0.10, 'kd': 1.8, 'label': 'Aggressive'},
        {'kp': 0.4,  'ki': 0.02, 'kd': 0.6, 'label': 'Conservative'},
    ], inertia=INERTIA, damping=DAMPING,
       save_filename="plot_bode_comparison.png")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Core scenarios
    compare_scenarios()

    # New advanced scenarios
    scenario_7_manoeuvre_sequence()
    scenario_8_sensor_noise()
    scenario_9_degraded_actuator()
    scenario_10_integrator_comparison()

    # Robustness study (reduce n_runs for a quick test)
    monte_carlo_robustness(n_runs=20)

    # Frequency analysis
    frequency_analysis()

    print("\n" + "=" * 70)
    print("ALL SCENARIOS COMPLETE")
    print("=" * 70)
