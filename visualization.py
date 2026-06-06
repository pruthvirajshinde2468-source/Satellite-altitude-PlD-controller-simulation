"""
Visualization and Analysis Plots

Fixes vs. original:
  - Setpoint reference line uses actual data, not hardcoded 0°
  - Handles time-varying setpoints from manoeuvre sequences

New plots:
  - Phase portrait (angle vs. angular velocity — state-space trajectory)
  - Reaction-wheel momentum over time
  - Disturbance component breakdown
  - Comprehensive 6-panel dashboard
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Consistent style across all plots
plt.rcParams.update({
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 1.8,
    'font.size': 11,
})


# ──────────────────────────────────────────────────────────────────────────────
def load_simulation_data(filename: str = "satellite_simulation.json") -> dict:
    with open(filename, 'r') as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
def _setpoint_series(data: dict) -> np.ndarray:
    """
    Return the setpoint time-series from data.
    Falls back to metadata scalar for files saved by the original code.
    """
    if 'setpoint' in data['data'] and data['data']['setpoint']:
        return np.degrees(np.array(data['data']['setpoint']))
    # Legacy: single scalar setpoint
    sp_deg = data['metadata'].get('setpoint_degrees', 0.0)
    n      = len(data['data']['time'])
    return np.full(n, sp_deg)


# ──────────────────────────────────────────────────────────────────────────────
def plot_satellite_attitude(data: dict, save_filename: str = None):
    """Angle response and attitude error over time."""
    times      = np.array(data['data']['time'])
    angles_deg = np.degrees(np.array(data['data']['angle']))
    errors_deg = np.degrees(np.array(data['data']['error']))
    setpoints  = _setpoint_series(data)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax1.plot(times, angles_deg, 'b-', label='Satellite Angle')
    ax1.plot(times, setpoints,  'r--', linewidth=1.2, label='Setpoint')
    ax1.set_ylabel('Angle (°)')
    ax1.set_title('Satellite Attitude Response')
    ax1.legend()

    ax2.plot(times, errors_deg, 'g-', label='Error')
    ax2.axhline(0, color='k', linewidth=0.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error (°)')
    ax2.set_title('Attitude Error')
    ax2.legend()

    plt.tight_layout()
    _save_and_show(save_filename)


# ──────────────────────────────────────────────────────────────────────────────
def plot_pid_terms(data: dict, save_filename: str = None):
    """Four-panel breakdown of individual PID terms."""
    times   = np.array(data['data']['time'])
    p_term  = np.array(data['data']['p_term'])
    i_term  = np.array(data['data']['i_term'])
    d_term  = np.array(data['data']['d_term'])
    control = np.array(data['data']['control_torque'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('PID Controller Term Breakdown', fontsize=13)
    (ax1, ax2), (ax3, ax4) = axes

    for ax, y, color, title, ylabel in [
        (ax1, p_term,  'b', 'Proportional  P = Kp·e',         'P term (N·m)'),
        (ax2, i_term,  'g', 'Integral  I = Ki·∫e dt',          'I term (N·m)'),
        (ax3, d_term,  'r', 'Derivative  D = Kd·de/dt (filtered)', 'D term (N·m)'),
    ]:
        ax.plot(times, y, color=color)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time (s)')

    ax4.plot(times, p_term,  'b--', alpha=0.7, label='P')
    ax4.plot(times, i_term,  'g--', alpha=0.7, label='I')
    ax4.plot(times, d_term,  'r--', alpha=0.7, label='D')
    ax4.plot(times, control, 'k-',  linewidth=2, label='Total (saturated)')
    ax4.axhline(0, color='k', linewidth=0.5)
    ax4.set_title('Combined Output  u = P + I + D')
    ax4.set_ylabel('Torque (N·m)')
    ax4.set_xlabel('Time (s)')
    ax4.legend()

    plt.tight_layout()
    _save_and_show(save_filename)


# ──────────────────────────────────────────────────────────────────────────────
def plot_torques(data: dict, save_filename: str = None):
    """Control torque vs. disturbance torque."""
    times      = np.array(data['data']['time'])
    control    = np.array(data['data']['control_torque'])
    disturb    = np.array(data['data']['disturbance_torque'])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, control, 'b-', label='Control Torque (PID)')
    ax.plot(times, disturb, 'r--', label='Disturbance Torque')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (N·m)')
    ax.set_title('Control vs. Disturbance Torques')
    ax.legend(fontsize=11)
    plt.tight_layout()
    _save_and_show(save_filename)


# ──────────────────────────────────────────────────────────────────────────────
def plot_velocity(data: dict, save_filename: str = None):
    """Angular velocity over time."""
    times    = np.array(data['data']['time'])
    velocity = np.array(data['data']['velocity'])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, velocity, color='purple')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_title('Satellite Angular Velocity')
    plt.tight_layout()
    _save_and_show(save_filename)


# ──────────────────────────────────────────────────────────────────────────────
def plot_phase_portrait(data: dict, save_filename: str = None):
    """
    Phase portrait: angular velocity vs. angle.

    The trajectory starts from the initial condition and converges to the
    setpoint (origin in error coordinates).  A stable controlled system
    shows a spiral converging inward.
    """
    angles_deg = np.degrees(np.array(data['data']['angle']))
    velocity   = np.array(data['data']['velocity'])
    setpoints  = _setpoint_series(data)

    # Convert to error coordinates
    error_deg  = setpoints - angles_deg

    n = len(error_deg)
    colors = plt.cm.viridis(np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=(9, 7))
    # Colour each segment by simulation time (dark=early, light=late)
    for i in range(n - 1):
        ax.plot(error_deg[i:i+2], velocity[i:i+2], color=colors[i], linewidth=0.8)

    ax.plot(error_deg[0],  velocity[0],  'go', markersize=9, label='Start')
    ax.plot(error_deg[-1], velocity[-1], 'r*', markersize=12, label='End')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.axvline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Angle Error (°)')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_title('Phase Portrait  (colour = time progression)')
    ax.legend()

    sm = plt.cm.ScalarMappable(cmap='viridis',
                                norm=plt.Normalize(0, data['statistics']['simulation_time']))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Time (s)')
    plt.tight_layout()
    _save_and_show(save_filename)


# ──────────────────────────────────────────────────────────────────────────────
def plot_wheel_momentum(data: dict, save_filename: str = None):
    """Reaction-wheel angular momentum over time (only if logged)."""
    if 'wheel_momentum' not in data['data']:
        print("wheel_momentum not in data — skipping wheel plot.")
        return

    times    = np.array(data['data']['time'])
    momentum = np.array(data['data']['wheel_momentum'])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, momentum, color='darkorange')
    ax.axhline(0, color='k', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angular Momentum (N·m·s)')
    ax.set_title('Reaction Wheel Stored Momentum')
    plt.tight_layout()
    _save_and_show(save_filename)


# ──────────────────────────────────────────────────────────────────────────────
def plot_comprehensive_analysis(data: dict, save_filename: str = None):
    """
    Six-panel dashboard:
      top-left   : attitude angle
      top-right  : attitude error
      mid-left   : control vs. disturbance torques
      mid-right  : PID terms
      bot-left   : angular velocity
      bot-right  : statistics text box
    """
    times      = np.array(data['data']['time'])
    angles_deg = np.degrees(np.array(data['data']['angle']))
    errors_deg = np.degrees(np.array(data['data']['error']))
    control    = np.array(data['data']['control_torque'])
    disturb    = np.array(data['data']['disturbance_torque'])
    p_term     = np.array(data['data']['p_term'])
    i_term     = np.array(data['data']['i_term'])
    d_term     = np.array(data['data']['d_term'])
    velocity   = np.array(data['data']['velocity'])
    setpoints  = _setpoint_series(data)
    stats      = data['statistics']

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Satellite Attitude Control — Comprehensive Dashboard', fontsize=14)
    gs  = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[2, 0])
    ax6 = fig.add_subplot(gs[2, 1])

    # Angle
    ax1.plot(times, angles_deg, 'b-', label='Angle')
    ax1.plot(times, setpoints,  'r--', linewidth=1, label='Setpoint')
    ax1.set_ylabel('Angle (°)')
    ax1.set_title('Attitude Angle')
    ax1.legend(fontsize=9)

    # Error
    ax2.plot(times, errors_deg, 'g-')
    ax2.axhline(0, color='k', linewidth=0.5)
    ax2.set_ylabel('Error (°)')
    ax2.set_title('Attitude Error')

    # Torques
    ax3.plot(times, control, 'b-', label='Control')
    ax3.plot(times, disturb, 'r--', label='Disturbance')
    ax3.axhline(0, color='k', linewidth=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Torque (N·m)')
    ax3.set_title('Control vs. Disturbance')
    ax3.legend(fontsize=9)

    # PID terms
    ax4.plot(times, p_term, 'b--', alpha=0.7, label='P')
    ax4.plot(times, i_term, 'g--', alpha=0.7, label='I')
    ax4.plot(times, d_term, 'r--', alpha=0.7, label='D')
    ax4.plot(times, control,'k-',  linewidth=1.5, label='Output')
    ax4.axhline(0, color='k', linewidth=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('N·m')
    ax4.set_title('PID Terms')
    ax4.legend(fontsize=9)

    # Angular velocity
    ax5.plot(times, velocity, color='purple')
    ax5.axhline(0, color='k', linewidth=0.5)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('rad/s')
    ax5.set_title('Angular Velocity')

    # Statistics panel
    ax6.axis('off')
    iae  = stats.get('IAE',  float('nan'))
    itae = stats.get('ITAE', float('nan'))
    ise  = stats.get('ISE',  float('nan'))
    txt = (
        f"PERFORMANCE SUMMARY\n\n"
        f"Final Angle    : {np.degrees(stats['final_angle']):9.4f}°\n"
        f"Final Error    : {np.degrees(stats['final_error']):9.4f}°\n"
        f"Max Error      : {np.degrees(stats['max_error']):9.4f}°\n"
        f"Steady-State   : {np.degrees(stats['steady_state_error']):9.4f}°\n"
        f"Overshoot      : {np.degrees(stats['overshoot']):9.4f}°\n"
        f"Settling Time  : {stats['settling_time']:9.2f} s\n\n"
        f"IAE    = {iae:.5f}\n"
        f"ISE    = {ise:.5f}\n"
        f"ITAE   = {itae:.5f}\n\n"
        f"Sim Duration   : {stats['simulation_time']:.1f} s\n"
        f"Steps          : {stats['total_steps']}"
    )
    ax6.text(0.05, 0.95, txt, transform=ax6.transAxes,
             fontsize=10, family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    _save_and_show(save_filename)


# ──────────────────────────────────────────────────────────────────────────────
def print_detailed_statistics(data: dict):
    stats = data['statistics']
    print("\n" + "=" * 70)
    print("DETAILED SIMULATION STATISTICS")
    print("=" * 70)
    print(f"\n  Attitude Control Performance:")
    print(f"    Final Angle         : {np.degrees(stats['final_angle']):10.4f}°")
    print(f"    Final Error         : {np.degrees(stats['final_error']):10.4f}°")
    print(f"    Max Error           : {np.degrees(stats['max_error']):10.4f}°")
    print(f"    Mean Error          : {np.degrees(stats['mean_error']):10.4f}°")
    print(f"    Steady-State Error  : {np.degrees(stats['steady_state_error']):10.4f}°")
    print(f"\n  Response Characteristics:")
    print(f"    Overshoot           : {np.degrees(stats['overshoot']):10.4f}°")
    print(f"    Settling Time       : {stats['settling_time']:10.2f} s")
    print(f"\n  Performance Indices:")
    for key in ('IAE', 'ISE', 'ITAE', 'ITSE'):
        val = stats.get(key, float('nan'))
        print(f"    {key:<6}           : {val:12.6f}")
    print(f"\n  Simulation Details:")
    print(f"    Duration            : {stats['simulation_time']:10.2f} s")
    print(f"    Steps               : {stats['total_steps']:10}")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
def _save_and_show(filename: str = None):
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading simulation data...")
    data = load_simulation_data("satellite_simulation.json")

    print_detailed_statistics(data)

    print("\nGenerating plots...")
    plot_satellite_attitude(data,       "plot_attitude.png")
    plot_pid_terms(data,                "plot_pid_terms.png")
    plot_torques(data,                  "plot_torques.png")
    plot_velocity(data,                 "plot_velocity.png")
    plot_phase_portrait(data,           "plot_phase_portrait.png")
    plot_wheel_momentum(data,           "plot_wheel_momentum.png")
    plot_comprehensive_analysis(data,   "plot_comprehensive.png")

    print("\nAll plots generated.")
