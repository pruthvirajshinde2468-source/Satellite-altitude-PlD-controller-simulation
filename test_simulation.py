"""
Unit Tests for Satellite PID Controller Simulation

Validates that all components work correctly
"""

import unittest
import numpy as np
from satellite_dynamics import SatelliteDynamics, DisturbanceModel, SatelliteState
from pid_controller import PIDController
from simulation import SatelliteSimulation


class TestSatelliteDynamics(unittest.TestCase):
    """Test satellite dynamics model"""
    
    def setUp(self):
        """Create test fixture"""
        self.satellite = SatelliteDynamics(inertia=100.0, damping_coeff=0.5)
    
    def test_initialization(self):
        """Test satellite initializes with correct parameters"""
        self.assertEqual(self.satellite.inertia, 100.0)
        self.assertEqual(self.satellite.damping, 0.5)
        self.assertEqual(self.satellite.state.angle, 0.0)
        self.assertEqual(self.satellite.state.angular_velocity, 0.0)
    
    def test_zero_torque_remains_stationary(self):
        """With no torque, stationary satellite should remain stationary"""
        for _ in range(100):
            self.satellite.update(control_torque=0.0, disturbance_torque=0.0, dt=0.01)
        
        self.assertAlmostEqual(self.satellite.state.angle, 0.0, places=6)
        self.assertAlmostEqual(self.satellite.state.angular_velocity, 0.0, places=6)
    
    def test_positive_torque_increases_angle(self):
        """Positive control torque should increase angle"""
        initial_angle = self.satellite.state.angle
        
        for _ in range(100):
            self.satellite.update(control_torque=0.01, disturbance_torque=0.0, dt=0.01)
        
        final_angle = self.satellite.state.angle
        self.assertGreater(final_angle, initial_angle)
    
    def test_damping_reduces_velocity(self):
        """Damping should prevent velocity from growing indefinitely"""
        # Give satellite initial angular velocity
        self.satellite.state.angular_velocity = 0.5
        
        velocities = [self.satellite.state.angular_velocity]
        
        for _ in range(1000):
            self.satellite.update(control_torque=0.0, disturbance_torque=0.0, dt=0.01)
            velocities.append(self.satellite.state.angular_velocity)
        
        # Final velocity should be less than initial
        self.assertLess(abs(velocities[-1]), abs(velocities[0]))
    
    def test_reset_functionality(self):
        """Test reset returns satellite to initial conditions"""
        # Change state
        self.satellite.update(control_torque=0.01, disturbance_torque=0.0, dt=0.01)
        
        # Reset
        self.satellite.reset(initial_angle=0.0, initial_velocity=0.0)
        
        # Check state
        self.assertEqual(self.satellite.state.angle, 0.0)
        self.assertEqual(self.satellite.state.angular_velocity, 0.0)
        self.assertEqual(self.satellite.state.angular_acceleration, 0.0)
    
    def test_state_update_order(self):
        """Test that angular acceleration, velocity, angle are updated in correct order"""
        dt = 0.01
        tau_control = 0.1
        
        self.satellite.update(control_torque=tau_control, disturbance_torque=0.0, dt=dt)
        
        # With control torque and zero velocity:
        # a = tau / I = 0.1 / 100 = 0.001 rad/s²
        expected_acceleration = 0.1 / 100.0
        self.assertAlmostEqual(self.satellite.state.angular_acceleration, expected_acceleration, places=6)
        
        # v should increase by a * dt
        expected_velocity = expected_acceleration * dt
        self.assertAlmostEqual(self.satellite.state.angular_velocity, expected_velocity, places=6)


class TestDisturbanceModel(unittest.TestCase):
    """Test disturbance model"""
    
    def setUp(self):
        """Create test fixture"""
        self.disturbance = DisturbanceModel(
            max_solar_torque=0.001,
            max_gravity_torque=0.0015,
            max_magnetic_torque=0.001
        )
    
    def test_disturbance_within_bounds(self):
        """Total disturbance should be within physical limits"""
        for t in np.linspace(0, 1000, 100):
            torque = self.disturbance.get_total_disturbance(t)
            
            # Maximum total disturbance should not exceed sum of maxima
            max_possible = 0.001 + 0.0015 + 0.001  # 0.0035 N·m
            self.assertLessEqual(abs(torque), max_possible)
    
    def test_components_sum_to_total(self):
        """Individual components should sum to total"""
        for t in np.linspace(0, 100, 10):
            components = self.disturbance.get_disturbance_components(t)
            total = components['total']
            sum_components = (components['solar'] + components['gravity'] + 
                            components['magnetic'])
            
            self.assertAlmostEqual(total, sum_components, places=6)
    
    def test_disturbance_periodicity(self):
        """Disturbances should be periodic"""
        torque_t0 = self.disturbance.get_total_disturbance(0.0)
        torque_t100 = self.disturbance.get_total_disturbance(100.0)  # One period
        
        # Should be close but not exact due to different frequencies
        self.assertGreater(abs(torque_t0 - torque_t100), 0.0)


class TestPIDController(unittest.TestCase):
    """Test PID controller"""
    
    def setUp(self):
        """Create test fixture"""
        self.controller = PIDController(kp=1.0, ki=0.1, kd=0.5)
    
    def test_initialization(self):
        """Test controller initializes with correct gains"""
        gains = self.controller.get_gains()
        self.assertEqual(gains.kp, 1.0)
        self.assertEqual(gains.ki, 0.1)
        self.assertEqual(gains.kd, 0.5)
    
    def test_proportional_term(self):
        """Test proportional term responds to current error"""
        # Zero error should give zero P term
        output = self.controller.compute(setpoint=0.0, measured_value=0.0, dt=0.01)
        terms = self.controller.get_terms()
        self.assertAlmostEqual(terms['proportional'], 0.0, places=6)
        
        # Reset and test with error
        self.controller.reset()
        output = self.controller.compute(setpoint=1.0, measured_value=0.0, dt=0.01)
        terms = self.controller.get_terms()
        
        # P = Kp * error = 1.0 * 1.0 = 1.0
        self.assertAlmostEqual(terms['proportional'], 1.0, places=6)
    
    def test_integral_accumulation(self):
        """Test integral term accumulates over time"""
        self.controller.reset()
        
        # Constant error over multiple steps
        for _ in range(100):
            output = self.controller.compute(setpoint=1.0, measured_value=0.0, dt=0.01)
        
        terms = self.controller.get_terms()
        
        # Integral should have accumulated
        self.assertGreater(abs(terms['integral']), 0.0)
    
    def test_derivative_term(self):
        """Test derivative term responds to error rate"""
        self.controller.reset()
        
        # First step: error = 1.0
        self.controller.compute(setpoint=1.0, measured_value=0.0, dt=0.01)
        
        # Second step: error = 0.5 (error decreasing)
        # de/dt = (0.5 - 1.0) / 0.01 = -50 rad/s²
        output = self.controller.compute(setpoint=1.0, measured_value=0.5, dt=0.01)
        terms = self.controller.get_terms()
        
        # D = Kd * de/dt = 0.5 * (-50) = -25
        # (Derivative should be negative as error is decreasing)
        self.assertLess(terms['derivative'], 0.0)
    
    def test_output_saturation(self):
        """Test controller output is saturated"""
        self.controller.reset()
        
        # Large error that would produce output > saturation limit
        for _ in range(100):
            output = self.controller.compute(setpoint=100.0, measured_value=0.0, dt=0.01)
        
        # Output should be limited to output_limit (0.1 by default)
        self.assertLessEqual(abs(output), 0.1)
    
    def test_anti_windup(self):
        """Test anti-windup prevents excessive integral accumulation"""
        self.controller.reset()
        
        # Apply large constant error
        for _ in range(1000):
            self.controller.compute(setpoint=1.0, measured_value=0.0, dt=0.01)
        
        terms = self.controller.get_terms()
        
        # Integral should be bounded
        self.assertLessEqual(abs(terms['integral']), 1.0)
    
    def test_gain_update(self):
        """Test gains can be updated"""
        self.controller.set_gains(kp=2.0, ki=0.2, kd=1.0)
        gains = self.controller.get_gains()
        
        self.assertEqual(gains.kp, 2.0)
        self.assertEqual(gains.ki, 0.2)
        self.assertEqual(gains.kd, 1.0)


class TestSatelliteSimulation(unittest.TestCase):
    """Test complete satellite simulation"""
    
    def setUp(self):
        """Create test fixture"""
        self.sim = SatelliteSimulation(inertia=100.0, damping=0.5, desired_angle=0.0)
    
    def test_simulation_initialization(self):
        """Test simulation initializes correctly"""
        self.assertEqual(self.sim.setpoint, 0.0)
        self.assertEqual(self.sim.time, 0.0)
        self.assertEqual(len(self.sim.data['time']), 0)
    
    def test_single_step_execution(self):
        """Test single simulation step executes"""
        step_data = self.sim.step()
        
        # Check that all expected data is returned
        self.assertIn('time', step_data)
        self.assertIn('angle', step_data)
        self.assertIn('error', step_data)
        self.assertIn('control_torque', step_data)
        self.assertIn('disturbance_torque', step_data)
    
    def test_data_logging(self):
        """Test data is logged correctly"""
        initial_count = len(self.sim.data['time'])
        
        for _ in range(100):
            self.sim.step()
        
        final_count = len(self.sim.data['time'])
        self.assertEqual(final_count - initial_count, 100)
    
    def test_convergence_to_setpoint(self):
        """Test controller converges to setpoint"""
        stats = self.sim.run(duration=100.0)
        
        # Final error should be small
        final_error = abs(stats['final_error'])
        self.assertLess(final_error, 0.01)  # Less than 0.01 radians
    
    def test_pid_gains_update(self):
        """Test PID gains can be updated"""
        initial_gains = self.sim.controller.get_gains()
        
        self.sim.set_pid_gains(kp=2.0, ki=0.2, kd=1.0)
        
        updated_gains = self.sim.controller.get_gains()
        self.assertEqual(updated_gains.kp, 2.0)
        self.assertNotEqual(updated_gains.kp, initial_gains.kp)
    
    def test_statistics_computation(self):
        """Test statistics are computed correctly"""
        stats = self.sim.run(duration=50.0)
        
        # Check that all statistics are present
        self.assertIn('final_angle', stats)
        self.assertIn('final_error', stats)
        self.assertIn('max_error', stats)
        self.assertIn('mean_error', stats)
        self.assertIn('steady_state_error', stats)
        self.assertIn('settling_time', stats)
        
        # Check sanity
        self.assertGreater(stats['max_error'], 0.0)  # Should have had some error
        self.assertLess(stats['settling_time'], 50.0)  # Should settle within duration


class TestIntegration(unittest.TestCase):
    """Integration tests for complete system"""
    
    def test_system_stability(self):
        """Test that controlled satellite is stable"""
        sim = SatelliteSimulation()
        
        # Run simulation
        stats = sim.run(duration=100.0)
        
        # System should settle
        self.assertLess(stats['settling_time'], 100.0)
        
        # Final error should be small
        self.assertLess(abs(stats['final_error']), 0.05)
    
    def test_different_setpoints(self):
        """Test controller works with different setpoints"""
        for setpoint_deg in [0, 15, 30, 45]:
            setpoint_rad = np.radians(setpoint_deg)
            sim = SatelliteSimulation(desired_angle=setpoint_rad)
            stats = sim.run(duration=100.0)
            
            # Should converge near setpoint
            error_deg = np.degrees(abs(stats['final_error']))
            self.assertLess(error_deg, 1.0)
    
    def test_disturbance_rejection(self):
        """Test controller rejects disturbances"""
        sim = SatelliteSimulation()
        stats = sim.run(duration=150.0)
        
        # Mean error should be small despite continuous disturbances
        mean_error_deg = np.degrees(stats['mean_error'])
        self.assertLess(mean_error_deg, 0.5)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
