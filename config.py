"""
Simulation configuration dataclasses.

Centralises all tunable parameters so callers never pass bare magic numbers.
"""

from dataclasses import dataclass


@dataclass
class SatelliteConfig:
    inertia: float = 100.0   # kg·m²
    damping: float = 0.5     # N·m·s/rad


@dataclass
class PIDConfig:
    kp: float = 0.8
    ki: float = 0.05
    kd: float = 1.2
    integral_limit: float = 1.0          # Clamp on raw ∫e dt  (rad·s)
    output_limit: float = 0.1            # Actuator saturation  (N·m)
    derivative_filter_tau: float = 0.05  # Derivative LPF time constant (s)


@dataclass
class DisturbanceConfig:
    max_solar_torque: float = 0.001      # N·m
    max_gravity_torque: float = 0.0015  # N·m
    max_magnetic_torque: float = 0.001  # N·m
    solar_period: float = 100.0         # s
    gravity_period: float = 66.7        # s  (≡ original 3π/100 frequency)
    magnetic_period: float = 40.0       # s  (≡ original 5π/100 frequency)


@dataclass
class NoiseConfig:
    # Realistic sensor specs:
    #   star tracker ≈ 5 arcsec  → ~2.4e-5 rad  (we use 1e-4 for demo)
    #   rate gyro    ≈ 0.01 °/s → ~1.7e-4 rad/s (we use 5e-5 for demo)
    angle_std: float = 0.0001     # rad
    velocity_std: float = 0.00005 # rad/s
    enabled: bool = False


@dataclass
class SimulationConfig:
    dt: float = 0.01              # s
    duration: float = 150.0       # s
    desired_angle: float = 0.0    # rad
    integrator: str = 'rk4'       # 'rk4' or 'euler'
    use_noise: bool = False
