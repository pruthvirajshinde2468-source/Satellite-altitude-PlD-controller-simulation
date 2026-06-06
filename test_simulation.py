"""
Unit and Integration Tests for Satellite PID Controller Simulation

Covers:
  - SatelliteDynamics (Euler + RK4, angle wrapping, reset)
  - DisturbanceModel
  - SensorModel
  - PIDController (bug fixes: correct integral units, anti-windup, D filter)
  - ReactionWheel (torque/momentum limits)
  - SatelliteSimulation (end-to-end convergence, manoeuvres, noise)
  - analysis module (performance indices)
"""

import unittest
import numpy as np

from satellite_dynamics import SatelliteDynamics, DisturbanceModel, SensorModel, ReactionWheel
from pid_controller import PIDController
from simulation import SatelliteSimulation
from analysis import compute_performance_indices, compute_stability_margins


# ══════════════════════════════════════════════════════════════════════════════
class TestSatelliteDynamics(unittest.TestCase):

    def setUp(self):
        self.sat = SatelliteDynamics(inertia=100.0, damping_coeff=0.5)

    def test_initial_state(self):
        self.assertEqual(self.sat.state.angle, 0.0)
        self.assertEqual(self.sat.state.angular_velocity, 0.0)

    def test_zero_torque_stationary(self):
        for _ in range(200):
            self.sat.update(0.0, 0.0, 0.01)
        self.assertAlmostEqual(self.sat.state.angle, 0.0, places=10)
        self.assertAlmostEqual(self.sat.state.angular_velocity, 0.0, places=10)

    def test_positive_torque_increases_angle(self):
        for _ in range(100):
            self.sat.update(0.01, 0.0, 0.01)
        self.assertGreater(self.sat.state.angle, 0.0)

    def test_damping_reduces_velocity(self):
        self.sat.state.angular_velocity = 0.5
        for _ in range(1000):
            self.sat.update(0.0, 0.0, 0.01)
        self.assertLess(abs(self.sat.state.angular_velocity), 0.5)

    def test_reset(self):
        self.sat.update(0.01, 0.0, 0.01)
        self.sat.reset(initial_angle=0.1, initial_velocity=0.2)
        self.assertAlmostEqual(self.sat.state.angle, 0.1)
        self.assertAlmostEqual(self.sat.state.angular_velocity, 0.2)
        self.assertAlmostEqual(self.sat.state.angular_acceleration, 0.0)

    def test_rk4_euler_agree_at_small_dt(self):
        """RK4 and Euler should produce nearly identical results for very small dt."""
        sat_rk4   = SatelliteDynamics(inertia=100.0, damping_coeff=0.5)
        sat_euler = SatelliteDynamics(inertia=100.0, damping_coeff=0.5)
        tau = 0.05
        dt  = 0.001  # very small — integration errors negligible
        for _ in range(500):
            sat_rk4.update(tau, 0.0, dt, method='rk4')
            sat_euler.update(tau, 0.0, dt, method='euler')
        self.assertAlmostEqual(sat_rk4.state.angle, sat_euler.state.angle, places=4)

    def test_rk4_more_accurate_at_large_dt(self):
        """At large dt, RK4 stays closer to the fine-dt reference than Euler does."""
        dt_coarse = 0.10
        dt_fine   = 0.001
        tau       = 0.05
        duration  = 5.0

        # Reference: Euler with tiny dt
        ref = SatelliteDynamics(inertia=100.0, damping_coeff=0.5)
        for _ in range(int(duration / dt_fine)):
            ref.update(tau, 0.0, dt_fine, method='euler')
        ref_angle = ref.state.angle

        sat_rk4   = SatelliteDynamics(inertia=100.0, damping_coeff=0.5)
        sat_euler = SatelliteDynamics(inertia=100.0, damping_coeff=0.5)
        for _ in range(int(duration / dt_coarse)):
            sat_rk4.update(tau, 0.0, dt_coarse, method='rk4')
            sat_euler.update(tau, 0.0, dt_coarse, method='euler')

        err_rk4   = abs(sat_rk4.state.angle   - ref_angle)
        err_euler = abs(sat_euler.state.angle - ref_angle)
        self.assertLess(err_rk4, err_euler,
                        "RK4 should be more accurate than Euler at large dt")

    def test_angle_wrapping(self):
        """Angle must stay in [-π, π]."""
        self.sat.state.angle = 3.0  # near π
        for _ in range(500):
            self.sat.update(0.005, 0.0, 0.01)  # slowly increase angle
        self.assertGreaterEqual(self.sat.state.angle, -np.pi)
        self.assertLessEqual(self.sat.state.angle,  np.pi)


# ══════════════════════════════════════════════════════════════════════════════
class TestDisturbanceModel(unittest.TestCase):

    def setUp(self):
        self.dist = DisturbanceModel(max_solar_torque=0.001,
                                     max_gravity_torque=0.0015,
                                     max_magnetic_torque=0.001)

    def test_bounded(self):
        max_possible = 0.001 + 0.0015 + 0.001
        for t in np.linspace(0, 1000, 500):
            self.assertLessEqual(abs(self.dist.get_total_disturbance(t)), max_possible + 1e-12)

    def test_components_sum_to_total(self):
        for t in np.linspace(0, 100, 20):
            c = self.dist.get_disturbance_components(t)
            self.assertAlmostEqual(c['total'], c['solar'] + c['gravity'] + c['magnetic'],
                                   places=12)

    def test_zero_at_origin(self):
        self.assertAlmostEqual(self.dist.get_total_disturbance(0.0), 0.0, places=12)


# ══════════════════════════════════════════════════════════════════════════════
class TestSensorModel(unittest.TestCase):

    def test_disabled_returns_true_values(self):
        sensor = SensorModel(enabled=False)
        a, v = sensor.measure(1.23, 0.45)
        self.assertEqual(a, 1.23)
        self.assertEqual(v, 0.45)

    def test_enabled_adds_noise(self):
        sensor = SensorModel(angle_std=0.01, velocity_std=0.005, enabled=True, seed=0)
        angles = [sensor.measure(0.0, 0.0)[0] for _ in range(1000)]
        # Mean should be near 0, std near 0.01
        self.assertAlmostEqual(np.mean(angles), 0.0, delta=0.003)
        self.assertAlmostEqual(np.std(angles), 0.01, delta=0.003)

    def test_noise_std_scales_with_parameter(self):
        sensor_lo = SensorModel(angle_std=0.001, enabled=True, seed=42)
        sensor_hi = SensorModel(angle_std=0.100, enabled=True, seed=42)
        n   = 5000
        lo  = np.std([sensor_lo.measure(0.0, 0.0)[0] for _ in range(n)])
        hi  = np.std([sensor_hi.measure(0.0, 0.0)[0] for _ in range(n)])
        self.assertLess(lo, hi)


# ══════════════════════════════════════════════════════════════════════════════
class TestPIDController(unittest.TestCase):

    def setUp(self):
        self.pid = PIDController(kp=1.0, ki=0.1, kd=0.5,
                                  integral_limit=1.0, output_limit=0.1,
                                  derivative_filter_tau=0.0)  # no filter for deterministic D tests

    def test_gains(self):
        g = self.pid.get_gains()
        self.assertEqual(g.kp, 1.0)
        self.assertEqual(g.ki, 0.1)
        self.assertEqual(g.kd, 0.5)

    def test_zero_error_zero_output(self):
        out = self.pid.compute(0.0, 0.0, 0.01)
        self.assertAlmostEqual(out, 0.0)
        self.assertAlmostEqual(self.pid.get_terms()['proportional'], 0.0)

    def test_proportional_term(self):
        out = self.pid.compute(1.0, 0.0, 0.01)
        self.assertAlmostEqual(self.pid.p_term, 1.0)  # Kp*error = 1*1

    def test_integral_accumulates_correctly(self):
        """Raw integral must accumulate in rad*s when output is not saturated."""
        # Use large output_limit and tiny gains so output never saturates,
        # allowing the integral to grow freely.
        pid = PIDController(kp=0.0, ki=0.1, kd=0.0,
                            integral_limit=10.0, output_limit=100.0,
                            derivative_filter_tau=0.0)
        for _ in range(100):
            pid.compute(1.0, 0.0, 0.01)
        # Raw integral = error * steps * dt = 1.0 * 100 * 0.01 = 1.0 rad*s
        acc = pid.get_terms()['integral_accumulation']
        self.assertAlmostEqual(acc, 1.0, places=5)
        # Scaled i_term = ki * raw = 0.1 * 1.0 = 0.1
        self.assertAlmostEqual(pid.i_term, 0.1, places=5)

    def test_derivative_sign(self):
        """Derivative must be negative when error is decreasing."""
        self.pid.reset()
        self.pid.compute(1.0, 0.0, 0.01)   # error = 1.0
        self.pid.compute(1.0, 0.5, 0.01)   # error = 0.5 — decreasing
        self.assertLess(self.pid.d_term, 0.0)

    def test_output_saturation(self):
        for _ in range(200):
            out = self.pid.compute(100.0, 0.0, 0.01)
        self.assertLessEqual(abs(out), 0.1 + 1e-9)

    def test_anti_windup_bounds_integral(self):
        """With large constant error, scaled i_term must not exceed output_limit."""
        for _ in range(5000):
            self.pid.compute(1.0, 0.0, 0.01)
        # i_term = ki * clamp(raw, ±1.0) ≤ 0.1 * 1.0 = 0.1
        self.assertLessEqual(abs(self.pid.i_term), 0.1 + 1e-9)

    def test_derivative_filter_smooths(self):
        """Filtered D response must change more slowly than unfiltered."""
        pid_filtered   = PIDController(kp=1.0, ki=0.0, kd=1.0,
                                       output_limit=1e6,   # no saturation
                                       derivative_filter_tau=0.5)
        pid_unfiltered = PIDController(kp=1.0, ki=0.0, kd=1.0,
                                       output_limit=1e6,
                                       derivative_filter_tau=0.0)
        # Large step in error
        pid_filtered.compute(0.0, 0.0, 0.01)
        pid_unfiltered.compute(0.0, 0.0, 0.01)
        pid_filtered.compute(1.0, 0.0, 0.01)
        pid_unfiltered.compute(1.0, 0.0, 0.01)
        self.assertLess(abs(pid_filtered.d_term), abs(pid_unfiltered.d_term))

    def test_gain_update(self):
        self.pid.set_gains(kp=2.0, ki=0.2, kd=1.0)
        g = self.pid.get_gains()
        self.assertEqual(g.kp, 2.0)
        self.assertEqual(g.ki, 0.2)
        self.assertEqual(g.kd, 1.0)

    def test_reset_clears_state(self):
        for _ in range(100):
            self.pid.compute(1.0, 0.0, 0.01)
        self.pid.reset()
        self.assertAlmostEqual(self.pid._integral, 0.0)
        self.assertAlmostEqual(self.pid._prev_error, 0.0)
        self.assertAlmostEqual(self.pid._filtered_deriv, 0.0)
        self.assertAlmostEqual(self.pid.output, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
class TestReactionWheel(unittest.TestCase):

    def setUp(self):
        self.wheel = ReactionWheel(wheel_inertia=0.01, max_torque=0.1, max_momentum=1.0)

    def test_torque_within_motor_limit(self):
        actual = self.wheel.apply_torque(200.0, 0.01)
        self.assertLessEqual(abs(actual), 0.1 + 1e-9)

    def test_momentum_does_not_exceed_limit(self):
        for _ in range(10000):
            self.wheel.apply_torque(0.1, 0.01)
        self.assertLessEqual(abs(self.wheel.momentum), 1.0 + 1e-9)

    def test_saturation_fraction_range(self):
        self.assertAlmostEqual(self.wheel.saturation_fraction, 0.0)
        for _ in range(100):
            self.wheel.apply_torque(0.1, 0.01)
        sf = self.wheel.saturation_fraction
        self.assertGreaterEqual(sf, 0.0)
        self.assertLessEqual(sf, 1.0 + 1e-9)

    def test_reset(self):
        self.wheel.apply_torque(0.1, 1.0)
        self.wheel.reset()
        self.assertAlmostEqual(self.wheel.momentum, 0.0)
        self.assertAlmostEqual(self.wheel.speed, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
class TestSatelliteSimulation(unittest.TestCase):

    def setUp(self):
        self.sim = SatelliteSimulation(inertia=100.0, damping=0.5, desired_angle=0.0)

    def test_initial_time_zero(self):
        self.assertEqual(self.sim.time, 0.0)
        self.assertEqual(len(self.sim.data['time']), 0)

    def test_step_returns_required_keys(self):
        sd = self.sim.step()
        for key in ('time', 'angle', 'error', 'control_torque',
                    'disturbance_torque', 'setpoint', 'wheel_momentum'):
            self.assertIn(key, sd)

    def test_data_logged_per_step(self):
        for _ in range(50):
            self.sim.step()
        self.assertEqual(len(self.sim.data['time']), 50)

    def test_convergence(self):
        stats = self.sim.run(duration=100.0)
        self.assertLess(abs(stats['final_error']), 0.02)

    def test_settling_within_duration(self):
        stats = self.sim.run(duration=100.0)
        self.assertLess(stats['settling_time'], 100.0)

    def test_pid_gains_update(self):
        self.sim.set_pid_gains(kp=2.0, ki=0.2, kd=1.0)
        g = self.sim.controller.get_gains()
        self.assertEqual(g.kp, 2.0)

    def test_statistics_keys_present(self):
        stats = self.sim.run(duration=50.0)
        for key in ('final_angle', 'final_error', 'max_error', 'settling_time',
                    'IAE', 'ISE', 'ITAE', 'ITSE', 'max_wheel_momentum'):
            self.assertIn(key, stats)

    def test_manoeuvre_changes_setpoint(self):
        self.sim.schedule_manoeuvre(0.5, np.radians(30.0))
        # Run past the trigger time
        for _ in range(100):
            self.sim.step()
        self.assertAlmostEqual(self.sim.setpoint, np.radians(30.0), places=6)

    def test_noise_does_not_crash(self):
        sim = SatelliteSimulation(inertia=100.0, damping=0.5, use_noise=True)
        stats = sim.run(duration=20.0)
        self.assertIsInstance(stats['final_error'], float)


# ══════════════════════════════════════════════════════════════════════════════
class TestIntegration(unittest.TestCase):

    def test_system_stability(self):
        sim = SatelliteSimulation()
        stats = sim.run(duration=100.0)
        self.assertLess(stats['settling_time'], 100.0)
        self.assertLess(abs(stats['final_error']), 0.05)

    def test_multiple_setpoints_converge(self):
        for sp_deg in [0, 15, 30, 45]:
            sim = SatelliteSimulation(desired_angle=np.radians(sp_deg))
            stats = sim.run(duration=150.0)
            # System must settle within the run time
            self.assertLess(stats['settling_time'], 150.0,
                            f"System did not settle at setpoint {sp_deg} deg")
            # SSE < 3 deg is physically correct given persistent sinusoidal disturbances
            sse_deg = np.degrees(stats['steady_state_error'])
            self.assertLess(sse_deg, 3.0,
                            f"Excessive SSE at setpoint {sp_deg} deg")

    def test_disturbance_rejection(self):
        sim = SatelliteSimulation()
        stats = sim.run(duration=150.0)
        # Mean error includes transient; 1 deg is realistic with persistent disturbances
        self.assertLess(np.degrees(stats['mean_error']), 1.0)

    def test_performance_indices_positive(self):
        sim = SatelliteSimulation()
        stats = sim.run(duration=50.0)
        for key in ('IAE', 'ISE', 'ITAE', 'ITSE'):
            self.assertGreater(stats[key], 0.0)


# ══════════════════════════════════════════════════════════════════════════════
class TestAnalysis(unittest.TestCase):

    def test_performance_indices_zero_error(self):
        t = np.linspace(0, 10, 1000)
        e = np.zeros(1000)
        indices = compute_performance_indices(t, e, 0.01)
        for key in ('IAE', 'ISE', 'ITAE', 'ITSE'):
            self.assertAlmostEqual(indices[key], 0.0, places=10)

    def test_performance_indices_positive_for_nonzero_error(self):
        t = np.linspace(0, 10, 1000)
        e = np.ones(1000)
        indices = compute_performance_indices(t, e, 0.01)
        for key in ('IAE', 'ISE', 'ITAE', 'ITSE'):
            self.assertGreater(indices[key], 0.0)

    def test_itae_penalises_late_errors_more(self):
        t = np.linspace(0, 10, 1000)
        e_early = np.zeros(1000)
        e_early[:100] = 1.0          # error only at the beginning
        e_late  = np.zeros(1000)
        e_late[-100:] = 1.0          # error only at the end

        idx_early = compute_performance_indices(t, e_early, 0.01)
        idx_late  = compute_performance_indices(t, e_late,  0.01)
        # ITAE should penalise late errors more
        self.assertGreater(idx_late['ITAE'], idx_early['ITAE'])

    def test_stability_margins_computed(self):
        m = compute_stability_margins(kp=0.8, ki=0.05, kd=1.2,
                                       inertia=100.0, damping=0.5)
        self.assertIn('phase_margin_deg',  m)
        self.assertIn('gain_margin_db',    m)
        self.assertIsNotNone(m['phase_margin_deg'])
        # Baseline tuning should have positive phase margin (stable)
        self.assertGreater(m['phase_margin_deg'], 0.0)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    unittest.main(verbosity=2)
