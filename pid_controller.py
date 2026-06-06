"""
PID Controller Implementation

Discrete PID with:
  - Derivative low-pass filter  (reduces noise amplification)
  - Conditional-integration anti-windup  (prevents integral windup at saturation)
  - Output saturation  (actuator limits)

u(t) = Kp·e + Ki·∫e dt + Kd·de/dt

Bug fixes vs. original:
  - Raw integral accumulator (_integral, rad·s) is now separate from the
    scaled i_term (N·m), so each call to compute() does not corrupt units.
  - get_terms()['integral_accumulation'] now returns the true ∫e dt value.
  - reset() fully clears the derivative filter state.
"""

from dataclasses import dataclass


@dataclass
class PIDGains:
    kp: float = 1.0
    ki: float = 0.1
    kd: float = 0.5


class PIDController:
    """
    Single-axis discrete PID controller.

    derivative_filter_tau controls the first-order low-pass on the D term:
        filtered_{k} = α·filtered_{k-1} + (1-α)·raw_{k},   α = τ/(τ+dt)
    Set tau=0 to disable filtering entirely.
    """

    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.5,
                 integral_limit: float = 1.0, output_limit: float = 0.1,
                 derivative_filter_tau: float = 0.05):
        self.gains = PIDGains(kp=kp, ki=ki, kd=kd)
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.derivative_filter_tau = derivative_filter_tau

        # Internal state
        self._integral = 0.0         # raw ∫e dt  (rad·s)
        self._prev_error = 0.0
        self._filtered_deriv = 0.0   # low-pass filtered  de/dt

        # Exposed for logging / analysis
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0
        self.output = 0.0

    # ------------------------------------------------------------------
    def compute(self, setpoint: float, measured_value: float, dt: float) -> float:
        """
        One discrete PID step.

        Args:
            setpoint:       Desired angle (rad)
            measured_value: Measured angle (rad)
            dt:             Time step (s)

        Returns:
            Control torque command (N·m)
        """
        error = setpoint - measured_value

        # ── Proportional ──────────────────────────────────────────────
        self.p_term = self.gains.kp * error

        # ── Derivative with low-pass filter ───────────────────────────
        if dt > 0:
            raw_deriv = (error - self._prev_error) / dt
            if self.derivative_filter_tau > 0:
                alpha = self.derivative_filter_tau / (self.derivative_filter_tau + dt)
                self._filtered_deriv = (alpha * self._filtered_deriv
                                        + (1.0 - alpha) * raw_deriv)
            else:
                self._filtered_deriv = raw_deriv
        self.d_term = self.gains.kd * self._filtered_deriv
        self._prev_error = error

        # ── Integral with conditional anti-windup ─────────────────────
        # Skip accumulation when output is already saturated AND this error
        # would push it further into saturation (prevents windup).
        saturated_high = self.output >= self.output_limit and error > 0
        saturated_low  = self.output <= -self.output_limit and error < 0
        if not (saturated_high or saturated_low):
            self._integral += error * dt

        # Hard clamp on raw integral as a safety backstop
        if self._integral > self.integral_limit:
            self._integral = self.integral_limit
        elif self._integral < -self.integral_limit:
            self._integral = -self.integral_limit

        self.i_term = self.gains.ki * self._integral

        # ── Total output with saturation ──────────────────────────────
        raw_output = self.p_term + self.i_term + self.d_term
        if raw_output > self.output_limit:
            self.output = self.output_limit
        elif raw_output < -self.output_limit:
            self.output = -self.output_limit
        else:
            self.output = raw_output

        return self.output

    # ------------------------------------------------------------------
    def reset(self):
        """Clear all controller state."""
        self._integral = 0.0
        self._prev_error = 0.0
        self._filtered_deriv = 0.0
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0
        self.output = 0.0

    def set_gains(self, kp: float = None, ki: float = None, kd: float = None):
        if kp is not None:
            self.gains.kp = kp
        if ki is not None:
            self.gains.ki = ki
        if kd is not None:
            self.gains.kd = kd

    def get_gains(self) -> PIDGains:
        return self.gains

    def get_terms(self) -> dict:
        return {
            'proportional': self.p_term,
            'integral': self.i_term,
            'derivative': self.d_term,
            'total_output': self.output,
            'integral_accumulation': self._integral,  # true ∫e dt  (rad·s)
        }
