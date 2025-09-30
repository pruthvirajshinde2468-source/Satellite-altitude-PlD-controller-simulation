# Satellite-altitude-PlD-controller-simulation
Maintain the satellite's orientation (attitude) at a desired angle
\
Objective: Maintain the satellite's orientation (attitude) at a desired angle
Challenge: External disturbances (solar pressure, gravity gradient, magnetic torques) constantly try to change the orientation
Solution: Use thrusters or reaction wheels to apply corrective torques

PID Controller Fundamentals:
PID stands for Proportional-Integral-Derivative controller. 
u(t) = Kp × e(t) + Ki × ∫e(t)dt + Kd × de(t)/dt

u(t) = Control output (torque in our case)
e(t) = Error (setpoint - current angle)
Kp, Ki, Kd = Tuning parameters

1] Proportional Term (P):
P_term = Kp × error
What it does: Provides immediate response proportional to current error.
Effect:
Larger Kp → Faster response, but can cause oscillations
Smaller Kp → Slower response, less oscillation
Problem: Always leaves some steady-state error

2] Integral Term (I):
I_term = Ki × ∫error dt
What it does: Accumulates past errors over time
Effect:
Eliminates steady-state error
Can cause overshoot and instability if too large
Challenge: "Integral windup" - integral keeps growing when actuator is saturated

3] Derivative Term (D):
D_term = Kd × d(error)/dt
What it does: Predicts future error based on current rate of change
Effect:
Damps oscillations
Improves stability
Reduces overshoot
Challenge: Sensitive to measurement noise

Satellite Dynamics:









