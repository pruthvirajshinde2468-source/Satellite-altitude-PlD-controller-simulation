"""
Satellite Dynamics Model

Implements the satellite attitude dynamics equations:
I × α = τ_control - τ_disturbance - damping

Where:
- I: Moment of inertia (kg·m²)
- α: Angular acceleration (rad/s²)
- τ_control: Control torque from thrusters/reaction wheels (N·m)
- τ_disturbance: External disturbance torques (N·m)
- damping: Natural damping in the system (N·m·s/rad)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class SatelliteState:
    """Current state of the satellite"""
    angle: float = 0.0  # Current angle θ (radians)
    angular_velocity: float = 0.0  # Angular velocity dθ/dt (rad/s)
    angular_acceleration: float = 0.0  # Angular acceleration d²θ/dt² (rad/s²)


class SatelliteDynamics:
    """Satellite attitude dynamics model"""
    
    def __init__(self, inertia: float, damping_coeff: float):
        """
        Initialize satellite dynamics
        
        Args:
            inertia: Moment of inertia (kg·m²)
            damping_coeff: Damping coefficient (N·m·s/rad)
        """
        self.inertia = inertia  # I
        self.damping = damping_coeff  # b (damping coefficient)
        self.state = SatelliteState()
        
    def get_state(self) -> SatelliteState:
        """Return current satellite state"""
        return self.state
    
    def update(self, control_torque: float, disturbance_torque: float, dt: float):
        """
        Update satellite dynamics for time step dt
        
        Dynamics equation:
        d²θ/dt² = (τ_control - τ_disturbance - b × dθ/dt) / I
        
        Args:
            control_torque: Control torque applied by thrusters/reaction wheels (N·m)
            disturbance_torque: External disturbance torque (N·m)
            dt: Time step (seconds)
        """
        # Calculate net torque
        # τ_net = τ_control - τ_disturbance - damping × ω
        net_torque = control_torque - disturbance_torque - (self.damping * self.state.angular_velocity)
        
        # Calculate angular acceleration: α = τ_net / I
        angular_acceleration = net_torque / self.inertia
        self.state.angular_acceleration = angular_acceleration
        
        # Update angular velocity: ω = ω + α × dt
        self.state.angular_velocity += angular_acceleration * dt
        
        # Update angle: θ = θ + ω × dt
        self.state.angle += self.state.angular_velocity * dt
        
    def reset(self, initial_angle: float = 0.0, initial_velocity: float = 0.0):
        """Reset satellite to initial conditions"""
        self.state.angle = initial_angle
        self.state.angular_velocity = initial_velocity
        self.state.angular_acceleration = 0.0


class DisturbanceModel:
    """
    Models external disturbances acting on the satellite:
    - Solar pressure torque
    - Gravity gradient torque
    - Magnetic torques
    """
    
    def __init__(self, max_solar_torque: float = 0.001,
                 max_gravity_torque: float = 0.0015,
                 max_magnetic_torque: float = 0.001):
        """
        Initialize disturbance model
        
        Args:
            max_solar_torque: Maximum solar pressure torque (N·m)
            max_gravity_torque: Maximum gravity gradient torque (N·m)
            max_magnetic_torque: Maximum magnetic field torque (N·m)
        """
        self.max_solar = max_solar_torque
        self.max_gravity = max_gravity_torque
        self.max_magnetic = max_magnetic_torque
        self.time = 0.0
        
    def get_total_disturbance(self, time: float) -> float:
        """
        Calculate total disturbance torque at given time
        
        Uses sinusoidal perturbations to simulate time-varying disturbances
        
        Args:
            time: Current simulation time (seconds)
            
        Returns:
            Total disturbance torque (N·m)
        """
        # Solar pressure torque: varies with orbital position
        solar_torque = self.max_solar * np.sin(2 * np.pi * time / 100.0)
        
        # Gravity gradient torque: periodic component
        gravity_torque = self.max_gravity * np.sin(3 * np.pi * time / 100.0)
        
        # Magnetic field torque: higher frequency disturbance
        magnetic_torque = self.max_magnetic * np.sin(5 * np.pi * time / 100.0)
        
        return solar_torque + gravity_torque + magnetic_torque
    
    def get_disturbance_components(self, time: float) -> dict:
        """
        Get individual disturbance components for analysis
        
        Args:
            time: Current simulation time (seconds)
            
        Returns:
            Dictionary with solar, gravity, and magnetic torques
        """
        solar = self.max_solar * np.sin(2 * np.pi * time / 100.0)
        gravity = self.max_gravity * np.sin(3 * np.pi * time / 100.0)
        magnetic = self.max_magnetic * np.sin(5 * np.pi * time / 100.0)
        
        return {
            'solar': solar,
            'gravity': gravity,
            'magnetic': magnetic,
            'total': solar + gravity + magnetic
        }
