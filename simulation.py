"""
Satellite Attitude Control Simulation

Closed-loop integrator:
  SatelliteDynamics  ←  control torque from PIDController
  PIDController      ←  noisy measurement from SensorModel
  ReactionWheel      ←  torque command, enforces momentum limits
  DisturbanceModel   ←  environmental torques injected at the plant

New in this version:
  - RK4 integration (default, switchable to Euler)
  - Optional sensor measurement noise
  - Attitude manoeuvre scheduler (time-tagged setpoint changes)
  - Reaction-wheel momentum tracking and saturation reporting
  - IAE / ISE / ITAE / ITSE performance indices
"""

import numpy as np
import json
from datetime import datetime

from satellite_dynamics import SatelliteDynamics, DisturbanceModel, SensorModel, ReactionWheel
from pid_controller import PIDController
from analysis import compute_performance_indices


class SatelliteSimulation:
    """Complete single-axis satellite attitude control simulation."""

    def __init__(self, inertia: float = 100.0, damping: float = 0.5,
                 desired_angle: float = 0.0,
                 use_noise: bool = False,
                 integrator: str = 'rk4'):
        """
        Args:
            inertia:       Satellite moment of inertia  (kg·m²)
            damping:       Linear damping coefficient   (N·m·s/rad)
            desired_angle: Initial attitude setpoint    (rad)
            use_noise:     Enable sensor measurement noise
            integrator:    'rk4' (default) or 'euler'
        """
        self.satellite    = SatelliteDynamics(inertia=inertia, damping_coeff=damping)
        self.controller   = PIDController(kp=0.8, ki=0.05, kd=1.2,
                                          integral_limit=1.0, output_limit=0.1,
                                          derivative_filter_tau=0.01)
        self.disturbance  = DisturbanceModel()
        self.sensor       = SensorModel(enabled=use_noise)
        self.wheel        = ReactionWheel(wheel_inertia=0.01,
                                          max_torque=0.1,
                                          max_momentum=1.0)

        self.setpoint   = desired_angle
        self.time       = 0.0
        self.dt         = 0.01
        self.integrator = integrator

        # Manoeuvre schedule: sorted list of (trigger_time, new_setpoint_rad)
        self._manoeuvres:  list = []
        self._mnvr_cursor: int  = 0

        self.data: dict = {
            'time': [], 'angle': [], 'velocity': [], 'error': [],
            'control_torque': [], 'disturbance_torque': [],
            'p_term': [], 'i_term': [], 'd_term': [],
            'setpoint': [], 'wheel_momentum': [],
        }

    # ── manoeuvre scheduling ──────────────────────────────────────────
    def schedule_manoeuvre(self, trigger_time: float, new_angle_rad: float):
        """Register a setpoint change at simulation time trigger_time."""
        self._manoeuvres.append((trigger_time, new_angle_rad))
        self._manoeuvres.sort(key=lambda x: x[0])

    # ── single step ───────────────────────────────────────────────────
    def step(self) -> dict:
        """
        Execute one simulation time step.

        Returns:
            dict with all logged quantities for this step.
        """
        # Fire any pending manoeuvres
        while (self._mnvr_cursor < len(self._manoeuvres) and
               self.time >= self._manoeuvres[self._mnvr_cursor][0]):
            self.setpoint = self._manoeuvres[self._mnvr_cursor][1]
            self._mnvr_cursor += 1

        # True satellite state
        true_angle    = self.satellite.state.angle
        true_velocity = self.satellite.state.angular_velocity

        # Sensor measurement (adds noise when enabled)
        meas_angle, _ = self.sensor.measure(true_angle, true_velocity)

        error = self.setpoint - meas_angle

        # PID computes torque command from measured (noisy) angle
        torque_cmd = self.controller.compute(self.setpoint, meas_angle, self.dt)

        # Reaction wheel enforces motor and momentum limits
        control_torque = self.wheel.apply_torque(torque_cmd, self.dt)

        disturbance_torque = self.disturbance.get_total_disturbance(self.time)

        # Integrate dynamics
        self.satellite.update(control_torque, disturbance_torque, self.dt,
                               method=self.integrator)

        pid_terms = self.controller.get_terms()
        step_data = {
            'time':               self.time,
            'angle':              true_angle,
            'velocity':           true_velocity,
            'error':              error,
            'control_torque':     control_torque,
            'disturbance_torque': disturbance_torque,
            'p_term':             pid_terms['proportional'],
            'i_term':             pid_terms['integral'],
            'd_term':             pid_terms['derivative'],
            'setpoint':           self.setpoint,
            'wheel_momentum':     self.wheel.momentum,
        }

        self._log(step_data)
        self.time += self.dt
        return step_data

    def _log(self, step_data: dict):
        for key, value in step_data.items():
            self.data[key].append(value)

    # ── full run ──────────────────────────────────────────────────────
    def run(self, duration: float = 100.0) -> dict:
        """
        Run the simulation for duration seconds.

        Returns:
            Statistics dict (see _compute_statistics).
        """
        print(f"Satellite attitude simulation")
        print(f"  Setpoint : {np.degrees(self.setpoint):.1f}°")
        print(f"  Duration : {duration:.1f} s   dt={self.dt} s   integrator={self.integrator}")
        if self._manoeuvres:
            for t, a in self._manoeuvres:
                print(f"  Manoeuvre: t={t:.1f}s → {np.degrees(a):.1f}°")
        print("-" * 65)

        steps      = int(duration / self.dt)
        last_print = 0.0

        for _ in range(steps):
            sd = self.step()
            if self.time - last_print >= 10.0:
                print(f"  t={self.time:6.1f}s | angle={np.degrees(sd['angle']):8.3f}deg | "
                      f"err={np.degrees(sd['error']):8.3f}deg | "
                      f"torque={sd['control_torque']:7.5f} Nm | "
                      f"wheel={self.wheel.saturation_fraction:4.1%}")
                last_print = self.time

        print("-" * 65)
        print("Simulation complete.")
        return self._compute_statistics()

    # ── statistics ────────────────────────────────────────────────────
    def _compute_statistics(self) -> dict:
        angles = np.array(self.data['angle'])
        errors = np.array(self.data['error'])
        times  = np.array(self.data['time'])

        perf = compute_performance_indices(times, errors, self.dt)

        # Overshoot: max exceedance beyond the setpoint (in the same sign direction)
        peak = float(np.max(angles))
        overshoot = max(0.0, peak - self.setpoint) if peak > self.setpoint else 0.0

        return {
            'final_angle':         float(angles[-1]),
            'final_error':         float(errors[-1]),
            'max_error':           float(np.max(np.abs(errors))),
            'mean_error':          float(np.mean(np.abs(errors))),
            'steady_state_error':  float(np.mean(np.abs(errors[-int(max(len(errors)*0.1, 1)):])) ),
            'overshoot':           overshoot,
            'settling_time':       self._settling_time(),
            'total_steps':         len(angles),
            'simulation_time':     self.time,
            'IAE':                 perf['IAE'],
            'ISE':                 perf['ISE'],
            'ITAE':                perf['ITAE'],
            'ITSE':                perf['ITSE'],
            'max_wheel_momentum':  float(np.max(np.abs(self.data['wheel_momentum']))),
        }

    def _settling_time(self, tolerance: float = 0.05) -> float:
        """
        First time after which the error stays within ±tolerance for 100
        consecutive samples (1 second at dt=0.01).
        """
        errors = np.array(self.data['error'])
        times  = np.array(self.data['time'])
        for idx in np.where(np.abs(errors) <= tolerance)[0]:
            end = min(idx + 100, len(errors))
            if np.all(np.abs(errors[idx:end]) <= tolerance):
                return float(times[idx])
        return float(times[-1])

    # ── reporting ─────────────────────────────────────────────────────
    def print_summary(self, stats: dict):
        g = self.controller.get_gains()
        print("\n" + "=" * 65)
        print("SIMULATION SUMMARY")
        print("=" * 65)
        print(f"  PID Gains:  Kp={g.kp}  Ki={g.ki}  Kd={g.kd}")
        print(f"  Integrator: {self.integrator.upper()}")
        print("-" * 65)
        print(f"  Final Angle         : {np.degrees(stats['final_angle']):9.4f}°")
        print(f"  Final Error         : {np.degrees(stats['final_error']):9.4f}°")
        print(f"  Max Error           : {np.degrees(stats['max_error']):9.4f}°")
        print(f"  Steady-State Error  : {np.degrees(stats['steady_state_error']):9.4f}°")
        print(f"  Overshoot           : {np.degrees(stats['overshoot']):9.4f}°")
        print(f"  Settling Time       : {stats['settling_time']:9.2f} s")
        print(f"  Max Wheel Momentum  : {stats['max_wheel_momentum']:9.4f} N·m·s")
        print(f"\n  Performance Indices:")
        print(f"    IAE   = {stats['IAE']:.6f}")
        print(f"    ISE   = {stats['ISE']:.6f}")
        print(f"    ITAE  = {stats['ITAE']:.6f}")
        print(f"    ITSE  = {stats['ITSE']:.6f}")
        print("=" * 65)

    # ── persistence ───────────────────────────────────────────────────
    def save_data(self, filename: str = "simulation_results.json"):
        g = self.controller.get_gains()
        output = {
            'metadata': {
                'timestamp':           datetime.now().isoformat(),
                'simulation_duration': self.time,
                'time_step':           self.dt,
                'setpoint_degrees':    np.degrees(self.setpoint),
                'integrator':          self.integrator,
                'pid_gains':           {'kp': g.kp, 'ki': g.ki, 'kd': g.kd},
            },
            'data':       self.data,
            'statistics': self._compute_statistics(),
        }
        with open(filename, 'w') as f:
            json.dump(self._serializable(output), f, indent=2)
        print(f"Results saved to: {filename}")

    def _serializable(self, obj):
        if isinstance(obj, dict):
            return {k: self._serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._serializable(i) for i in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        return obj

    def get_data(self) -> dict:
        return self.data

    def set_pid_gains(self, kp: float = None, ki: float = None, kd: float = None):
        self.controller.set_gains(kp, ki, kd)
        g = self.controller.get_gains()
        print(f"PID Gains updated -> Kp={g.kp}  Ki={g.ki}  Kd={g.kd}")


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sim = SatelliteSimulation(inertia=100.0, damping=0.5, desired_angle=0.0,
                              integrator='rk4')

    # Example multi-step manoeuvre sequence
    sim.schedule_manoeuvre(50.0,  np.radians(20.0))
    sim.schedule_manoeuvre(100.0, np.radians(0.0))

    stats = sim.run(duration=150.0)
    sim.print_summary(stats)
    sim.save_data("satellite_simulation.json")
