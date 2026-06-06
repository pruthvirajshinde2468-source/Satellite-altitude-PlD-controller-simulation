"""
Satellite Attitude Control Simulation

Main simulation loop that integrates:
- Satellite dynamics model
- PID controller
- Disturbance model
- Data logging and visualization

Simulates satellite attitude control over time with external disturbances.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
from satellite_dynamics import SatelliteDynamics, DisturbanceModel, SatelliteState
from pid_controller import PIDController


class SatelliteSimulation:
    """Complete satellite attitude control simulation"""
    
    def __init__(self, inertia: float = 100.0, damping: float = 0.5,
                 desired_angle: float = 0.0):
        """
        Initialize simulation
        
        Args:
            inertia: Satellite moment of inertia (kg·m²)
            damping: Damping coefficient (N·m·s/rad)
            desired_angle: Desired satellite attitude (radians)
        """
        # Satellite model
        self.satellite = SatelliteDynamics(inertia=inertia, damping_coeff=damping)
        
        # PID controller (gains tuned for this system)
        self.controller = PIDController(kp=0.8, ki=0.05, kd=1.2,
                                       integral_limit=1.0, output_limit=0.1)
        
        # Disturbance model
        self.disturbance = DisturbanceModel(
            max_solar_torque=0.001,
            max_gravity_torque=0.0015,
            max_magnetic_torque=0.001
        )
        
        # Simulation parameters
        self.setpoint = desired_angle  # Desired angle (radians)
        self.time = 0.0
        self.dt = 0.01  # Time step (10 ms)
        
        # Data logging
        self.data = {
            'time': [],
            'angle': [],
            'velocity': [],
            'error': [],
            'control_torque': [],
            'disturbance_torque': [],
            'p_term': [],
            'i_term': [],
            'd_term': [],
        }
        
    def step(self) -> dict:
        """
        Execute one simulation step
        
        Step-by-step process:
        1. Measure current satellite angle
        2. Calculate error: e(t) = setpoint - current_angle
        3. Compute PID output: u(t) = f(error)
        4. Get disturbance torque
        5. Update satellite dynamics
        6. Log data
        
        Returns:
            Dictionary with current step data
        """
        # Step 1: Measure current state
        current_angle = self.satellite.state.angle
        current_velocity = self.satellite.state.angular_velocity
        
        # Step 2: Calculate error
        error = self.setpoint - current_angle
        
        # Step 3: Compute PID control torque
        control_torque = self.controller.compute(self.setpoint, current_angle, self.dt)
        
        # Step 4: Get disturbance torque from environment
        disturbance_torque = self.disturbance.get_total_disturbance(self.time)
        
        # Step 5: Update satellite dynamics
        self.satellite.update(control_torque, disturbance_torque, self.dt)
        
        # Step 6: Log data
        pid_terms = self.controller.get_terms()
        step_data = {
            'time': self.time,
            'angle': current_angle,
            'velocity': current_velocity,
            'error': error,
            'control_torque': control_torque,
            'disturbance_torque': disturbance_torque,
            'p_term': pid_terms['proportional'],
            'i_term': pid_terms['integral'],
            'd_term': pid_terms['derivative'],
        }
        
        self._log_data(step_data)
        self.time += self.dt
        
        return step_data
    
    def _log_data(self, step_data: dict):
        """Log simulation data for analysis"""
        for key, value in step_data.items():
            self.data[key].append(value)
    
    def run(self, duration: float = 100.0) -> dict:
        """
        Run complete simulation
        
        Args:
            duration: Simulation duration (seconds)
            
        Returns:
            Dictionary with simulation results and statistics
        """
        print(f"Starting satellite attitude control simulation...")
        print(f"Setpoint: {np.degrees(self.setpoint):.1f}°")
        print(f"Duration: {duration:.1f} seconds")
        print(f"Time step: {self.dt} seconds")
        print("-" * 60)
        
        steps = int(duration / self.dt)
        last_print_time = 0.0
        
        for step_num in range(steps):
            # Execute simulation step
            step_data = self.step()
            
            # Print progress every 10 seconds
            if self.time - last_print_time >= 10.0:
                angle_deg = np.degrees(step_data['angle'])
                error_deg = np.degrees(step_data['error'])
                print(f"t = {self.time:6.1f}s | Angle = {angle_deg:7.2f}° | Error = {error_deg:7.2f}° | "
                      f"Torque = {step_data['control_torque']:7.4f} N·m")
                last_print_time = self.time
        
        print("-" * 60)
        print("Simulation complete!")
        
        return self._compute_statistics()
    
    def _compute_statistics(self) -> dict:
        """Compute simulation statistics"""
        angles = np.array(self.data['angle'])
        errors = np.array(self.data['error'])
        
        stats = {
            'final_angle': angles[-1],
            'final_error': errors[-1],
            'max_error': np.max(np.abs(errors)),
            'mean_error': np.mean(np.abs(errors)),
            'steady_state_error': np.mean(np.abs(errors[-int(len(errors)*0.1):])),
            'overshoot': np.max(angles) - self.setpoint if np.max(angles) > self.setpoint else 0.0,
            'settling_time': self._calculate_settling_time(),
            'total_steps': len(angles),
            'simulation_time': self.time,
        }
        
        return stats
    
    def _calculate_settling_time(self, tolerance: float = 0.05) -> float:
        """
        Calculate time for angle to settle within tolerance
        
        Args:
            tolerance: Settling tolerance (radians)
            
        Returns:
            Settling time in seconds
        """
        errors = np.array(self.data['error'])
        times = np.array(self.data['time'])
        
        # Find first time after which error stays within tolerance
        settled_indices = np.where(np.abs(errors) <= tolerance)[0]
        
        if len(settled_indices) > 0:
            # Check if it stays settled (within 100 consecutive points)
            for idx in settled_indices:
                if idx + 100 < len(errors):
                    if np.all(np.abs(errors[idx:idx+100]) <= tolerance):
                        return times[idx]
        
        return times[-1]  # Not settled within duration
    
    def print_summary(self, stats: dict):
        """Print simulation summary"""
        print("\n" + "=" * 60)
        print("SATELLITE ATTITUDE CONTROL SUMMARY")
        print("=" * 60)
        print(f"Final Angle:          {np.degrees(stats['final_angle']):8.2f}°")
        print(f"Final Error:          {np.degrees(stats['final_error']):8.2f}°")
        print(f"Max Error:            {np.degrees(stats['max_error']):8.2f}°")
        print(f"Mean Error:           {np.degrees(stats['mean_error']):8.2f}°")
        print(f"Steady-State Error:   {np.degrees(stats['steady_state_error']):8.2f}°")
        print(f"Overshoot:            {np.degrees(stats['overshoot']):8.2f}°")
        print(f"Settling Time:        {stats['settling_time']:8.2f}s")
        print("=" * 60)
    
    def save_data(self, filename: str = "simulation_results.json"):
        """
        Save simulation data to file
        
        Args:
            filename: Output filename
        """
        output = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'simulation_duration': self.time,
                'time_step': self.dt,
                'setpoint_degrees': np.degrees(self.setpoint),
            },
            'data': self.data,
            'statistics': self._compute_statistics(),
        }
        
        # Convert numpy types to Python types for JSON serialization
        output_serializable = self._make_serializable(output)
        
        with open(filename, 'w') as f:
            json.dump(output_serializable, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
    
    def _make_serializable(self, obj):
        """Convert numpy types to JSON-serializable types"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(val) for key, val in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        else:
            return obj
    
    def get_data(self) -> dict:
        """Return collected simulation data"""
        return self.data
    
    def set_pid_gains(self, kp: float = None, ki: float = None, kd: float = None):
        """
        Update PID gains
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
        """
        self.controller.set_gains(kp, ki, kd)
        gains = self.controller.get_gains()
        print(f"\nPID Gains updated:")
        print(f"  Kp = {gains.kp}")
        print(f"  Ki = {gains.ki}")
        print(f"  Kd = {gains.kd}")


if __name__ == "__main__":
    # ===== SATELLITE PARAMETERS =====
    # These represent a typical small satellite
    SATELLITE_INERTIA = 100.0  # kg·m² (moment of inertia)
    DAMPING_COEFF = 0.5  # N·m·s/rad (natural damping)
    DESIRED_ANGLE = 0.0  # radians (0 degrees - nadir-pointing)
    
    # ===== SIMULATION PARAMETERS =====
    SIMULATION_DURATION = 150.0  # seconds (2.5 minutes)
    
    # Create and run simulation
    sim = SatelliteSimulation(
        inertia=SATELLITE_INERTIA,
        damping=DAMPING_COEFF,
        desired_angle=DESIRED_ANGLE
    )
    
    # Run simulation
    stats = sim.run(duration=SIMULATION_DURATION)
    
    # Print summary
    sim.print_summary(stats)
    
    # Save results
    sim.save_data("satellite_simulation.json")
