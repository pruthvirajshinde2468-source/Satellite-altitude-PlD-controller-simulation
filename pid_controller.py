"""
PID Controller Implementation

Implements a Proportional-Integral-Derivative controller for satellite attitude control:
u(t) = Kp × e(t) + Ki × ∫e(t)dt + Kd × de(t)/dt

Where:
- u(t): Control output (torque command)
- e(t): Error (setpoint - measured value)
- Kp: Proportional gain
- Ki: Integral gain
- Kd: Derivative gain
"""

from dataclasses import dataclass, field


@dataclass
class PIDGains:
    """PID controller gain parameters"""
    kp: float = 1.0   # Proportional gain
    ki: float = 0.1   # Integral gain
    kd: float = 0.5   # Derivative gain


@dataclass
class PIDState:
    """Internal state of PID controller"""
    integral_error: float = 0.0  # Accumulated integral term
    previous_error: float = 0.0  # Previous error for derivative calculation
    last_update_time: float = 0.0


class PIDController:
    """
    Satellite attitude PID controller
    
    Controls satellite orientation by computing control torques based on:
    1. Current error (P term)
    2. Accumulated error (I term)
    3. Rate of error change (D term)
    """
    
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.5,
                 integral_limit: float = 1.0, output_limit: float = 0.1):
        """
        Initialize PID controller
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            integral_limit: Anti-windup limit for integral term
            output_limit: Maximum control torque magnitude (saturation limit)
        """
        self.gains = PIDGains(kp=kp, ki=ki, kd=kd)
        self.state = PIDState()
        self.integral_limit = integral_limit  # Anti-windup
        self.output_limit = output_limit  # Actuator saturation
        
        # Track components for analysis
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0
        self.output = 0.0
        
    def compute(self, setpoint: float, measured_value: float, dt: float) -> float:
        """
        Compute PID control output
        
        Args:
            setpoint: Desired angle (radians)
            measured_value: Current measured angle (radians)
            dt: Time step (seconds)
            
        Returns:
            Control torque command (N·m)
        """
        # Calculate error
        error = setpoint - measured_value
        
        # ===== PROPORTIONAL TERM =====
        # P_term = Kp × e(t)
        # Provides immediate response proportional to current error
        self.p_term = self.gains.kp * error
        
        # ===== INTEGRAL TERM =====
        # I_term = Ki × ∫e(t)dt
        # Accumulates past errors to eliminate steady-state error
        self.i_term += error * dt
        
        # Anti-windup: Limit integral to prevent excessive accumulation
        # This prevents "integral windup" when actuator is saturated
        if abs(self.i_term) > self.integral_limit:
            self.i_term = self.integral_limit * (1 if self.i_term > 0 else -1)
        
        self.i_term = self.gains.ki * self.i_term
        
        # ===== DERIVATIVE TERM =====
        # D_term = Kd × de(t)/dt
        # Predicts future error based on rate of change
        # Damps oscillations and improves stability
        if dt > 0:
            error_rate = (error - self.state.previous_error) / dt
            self.d_term = self.gains.kd * error_rate
        else:
            self.d_term = 0.0
        
        # Store error for next iteration
        self.state.previous_error = error
        
        # ===== TOTAL CONTROL OUTPUT =====
        # u(t) = P_term + I_term + D_term
        self.output = self.p_term + self.i_term + self.d_term
        
        # Apply output saturation (actuator limits)
        if abs(self.output) > self.output_limit:
            self.output = self.output_limit * (1 if self.output > 0 else -1)
        
        return self.output
    
    def reset(self):
        """Reset controller state (after achieving setpoint or error condition)"""
        self.state.integral_error = 0.0
        self.state.previous_error = 0.0
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0
        self.output = 0.0
    
    def set_gains(self, kp: float = None, ki: float = None, kd: float = None):
        """
        Update PID gains (useful for tuning)
        
        Args:
            kp: New proportional gain
            ki: New integral gain
            kd: New derivative gain
        """
        if kp is not None:
            self.gains.kp = kp
        if ki is not None:
            self.gains.ki = ki
        if kd is not None:
            self.gains.kd = kd
    
    def get_gains(self) -> PIDGains:
        """Return current PID gains"""
        return self.gains
    
    def get_terms(self) -> dict:
        """
        Return individual PID terms for analysis
        
        Returns:
            Dictionary with P, I, D terms and total output
        """
        return {
            'proportional': self.p_term,
            'integral': self.i_term,
            'derivative': self.d_term,
            'total_output': self.output,
            'integral_accumulation': self.state.integral_error
        }
