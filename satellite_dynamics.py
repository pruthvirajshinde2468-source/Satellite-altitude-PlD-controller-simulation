"""
Satellite Dynamics Model

Single-axis rigid-body attitude dynamics:
    I·α = τ_control - τ_disturbance - b·ω

Improvements vs. original:
  - 4th-order Runge-Kutta (RK4) integration replaces forward Euler
  - Angle wrapped to [-π, π] to prevent unbounded accumulation
  - SensorModel adds realistic Gaussian measurement noise
  - DisturbanceModel uses configurable periods (defaults match original)
  - ReactionWheel models actuator momentum and saturation
"""

import numpy as np
from dataclasses import dataclass


# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class SatelliteState:
    angle: float = 0.0                # θ        (rad)
    angular_velocity: float = 0.0    # ω = dθ/dt  (rad/s)
    angular_acceleration: float = 0.0  # α = dω/dt  (rad/s²)


# ──────────────────────────────────────────────────────────────────────────────
class SatelliteDynamics:
    """
    Rigid-body single-axis attitude plant.

    EOM:  I·α = τ_control - τ_disturbance - b·ω
    """

    def __init__(self, inertia: float, damping_coeff: float):
        self.inertia = inertia
        self.damping = damping_coeff
        self.state = SatelliteState()

    # ── integration ──────────────────────────────────────────────────
    def _alpha(self, omega: float, tau_c: float, tau_d: float) -> float:
        """Angular acceleration given current ω and torques."""
        return (tau_c - tau_d - self.damping * omega) / self.inertia

    def update(self, control_torque: float, disturbance_torque: float, dt: float,
               method: str = 'rk4'):
        """
        Advance the state by dt.

        Args:
            control_torque:     Applied torque from actuator (N·m)
            disturbance_torque: Environmental disturbance torque (N·m)
            dt:                 Time step (s)
            method:             'rk4' (default) or 'euler'
        """
        if method == 'rk4':
            self._rk4_step(control_torque, disturbance_torque, dt)
        else:
            self._euler_step(control_torque, disturbance_torque, dt)

        # Wrap to [-π, π]
        self.state.angle = (self.state.angle + np.pi) % (2.0 * np.pi) - np.pi

    def _rk4_step(self, tau_c: float, tau_d: float, dt: float):
        """4th-order Runge-Kutta integration for [θ, ω]."""
        θ = self.state.angle
        ω = self.state.angular_velocity

        # Stage derivatives  (dθ/dt = ω,  dω/dt = α)
        k1ω = self._alpha(ω, tau_c, tau_d)
        k1θ = ω

        k2ω = self._alpha(ω + 0.5 * dt * k1ω, tau_c, tau_d)
        k2θ = ω + 0.5 * dt * k1ω

        k3ω = self._alpha(ω + 0.5 * dt * k2ω, tau_c, tau_d)
        k3θ = ω + 0.5 * dt * k2ω

        k4ω = self._alpha(ω + dt * k3ω, tau_c, tau_d)
        k4θ = ω + dt * k3ω

        self.state.angle              = θ + (dt / 6.0) * (k1θ + 2*k2θ + 2*k3θ + k4θ)
        self.state.angular_velocity   = ω + (dt / 6.0) * (k1ω + 2*k2ω + 2*k3ω + k4ω)
        self.state.angular_acceleration = (k1ω + 2*k2ω + 2*k3ω + k4ω) / 6.0

    def _euler_step(self, tau_c: float, tau_d: float, dt: float):
        """Forward-Euler integration (kept for comparison)."""
        alpha = self._alpha(self.state.angular_velocity, tau_c, tau_d)
        self.state.angular_acceleration = alpha
        self.state.angular_velocity    += alpha * dt
        self.state.angle               += self.state.angular_velocity * dt

    def reset(self, initial_angle: float = 0.0, initial_velocity: float = 0.0):
        self.state.angle               = initial_angle
        self.state.angular_velocity    = initial_velocity
        self.state.angular_acceleration = 0.0

    def get_state(self) -> SatelliteState:
        return self.state


# ──────────────────────────────────────────────────────────────────────────────
class SensorModel:
    """
    Gaussian measurement noise on angle and angular-velocity readings.

    Typical hardware noise floors:
        Star tracker : 1–5 arcsec  ≈ 5e-6 – 2.4e-5 rad
        Rate gyro    : 0.01 deg/s  ≈ 1.7e-4 rad/s
    """

    def __init__(self, angle_std: float = 0.0001, velocity_std: float = 0.00005,
                 enabled: bool = True, seed: int = None):
        self.angle_std    = angle_std
        self.velocity_std = velocity_std
        self.enabled      = enabled
        self._rng = np.random.default_rng(seed)

    def measure(self, true_angle: float, true_velocity: float):
        """Return (noisy_angle, noisy_velocity)."""
        if not self.enabled:
            return true_angle, true_velocity
        return (true_angle    + self._rng.normal(0.0, self.angle_std),
                true_velocity + self._rng.normal(0.0, self.velocity_std))


# ──────────────────────────────────────────────────────────────────────────────
class DisturbanceModel:
    """
    Three independent sinusoidal disturbance torques:
      solar pressure, gravity gradient, magnetic field.

    Default periods match the original formulation:
        solar    : 2π t / 100   → period = 100 s
        gravity  : 3π t / 100   → period =  66.7 s
        magnetic : 5π t / 100   → period =  40 s
    """

    def __init__(self, max_solar_torque: float = 0.001,
                 max_gravity_torque: float = 0.0015,
                 max_magnetic_torque: float = 0.001,
                 solar_period: float = 100.0,
                 gravity_period: float = 66.7,
                 magnetic_period: float = 40.0):
        self.max_solar    = max_solar_torque
        self.max_gravity  = max_gravity_torque
        self.max_magnetic = max_magnetic_torque
        self.solar_period    = solar_period
        self.gravity_period  = gravity_period
        self.magnetic_period = magnetic_period

    def _solar(self, t):
        return self.max_solar * np.sin(2 * np.pi * t / self.solar_period)

    def _gravity(self, t):
        return self.max_gravity * np.sin(2 * np.pi * t / self.gravity_period)

    def _magnetic(self, t):
        return self.max_magnetic * np.sin(2 * np.pi * t / self.magnetic_period)

    def get_total_disturbance(self, time: float) -> float:
        return self._solar(time) + self._gravity(time) + self._magnetic(time)

    def get_disturbance_components(self, time: float) -> dict:
        s = self._solar(time)
        g = self._gravity(time)
        m = self._magnetic(time)
        return {'solar': s, 'gravity': g, 'magnetic': m, 'total': s + g + m}


# ──────────────────────────────────────────────────────────────────────────────
class ReactionWheel:
    """
    Single-axis reaction-wheel actuator model.

    The wheel stores angular momentum H.  Applying a torque τ to the satellite
    spins the wheel in the opposite direction:  dH/dt = τ.

    Limits modelled:
      max_torque   — motor peak torque  (N·m)
      max_momentum — wheel saturation   (N·m·s = kg·m²/s)
    """

    def __init__(self, wheel_inertia: float = 0.01,
                 max_torque: float = 0.1,
                 max_momentum: float = 1.0):
        self.wheel_inertia = wheel_inertia  # kg·m²
        self.max_torque    = max_torque     # N·m
        self.max_momentum  = max_momentum   # N·m·s
        self.momentum = 0.0  # H_rw (N·m·s)
        self.speed    = 0.0  # ω_rw (rad/s)

    def apply_torque(self, commanded: float, dt: float) -> float:
        """
        Clamp commanded torque to motor and momentum limits.

        Returns the actual torque delivered to the satellite.
        """
        # Motor torque limit
        actual = max(-self.max_torque, min(self.max_torque, commanded))

        # Momentum saturation: would adding this torque overflow the wheel?
        new_momentum = self.momentum + actual * dt
        if new_momentum > self.max_momentum:
            actual = (self.max_momentum - self.momentum) / dt
        elif new_momentum < -self.max_momentum:
            actual = (-self.max_momentum - self.momentum) / dt

        self.momentum += actual * dt
        self.speed     = self.momentum / self.wheel_inertia
        return actual

    @property
    def saturation_fraction(self) -> float:
        """Fraction of max momentum in use (0–1).  >0.8 ≈ approaching saturation."""
        return abs(self.momentum) / self.max_momentum

    def reset(self):
        self.momentum = 0.0
        self.speed    = 0.0
