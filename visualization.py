"""
Visualization and Analysis Tools

Plots simulation results for analysis and validation of the PID controller
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


def load_simulation_data(filename: str = "satellite_simulation.json") -> dict:
    """Load simulation data from JSON file"""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data


def plot_satellite_attitude(data: dict, save_filename: str = None):
    """
    Plot satellite attitude over time
    
    Args:
        data: Simulation data dictionary
        save_filename: Optional filename to save plot
    """
    times = np.array(data['data']['time'])
    angles_rad = np.array(data['data']['angle'])
    angles_deg = np.degrees(angles_rad)
    errors_deg = np.degrees(np.array(data['data']['error']))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Satellite Angle
    ax1.plot(times, angles_deg, 'b-', linewidth=2, label='Satellite Angle')
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1, label='Setpoint (0°)')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Angle (degrees)')
    ax1.set_title('Satellite Attitude Control - Angle Response')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Error
    ax2.plot(times, errors_deg, 'g-', linewidth=2, label='Error')
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Error (degrees)')
    ax2.set_title('Attitude Control Error Over Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_filename:
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_filename}")
    
    plt.show()


def plot_pid_terms(data: dict, save_filename: str = None):
    """
    Plot individual PID controller terms
    
    Args:
        data: Simulation data dictionary
        save_filename: Optional filename to save plot
    """
    times = np.array(data['data']['time'])
    p_term = np.array(data['data']['p_term'])
    i_term = np.array(data['data']['i_term'])
    d_term = np.array(data['data']['d_term'])
    control = np.array(data['data']['control_torque'])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Proportional Term
    ax1.plot(times, p_term, 'b-', linewidth=1.5)
    ax1.set_ylabel('Proportional Term (N·m)')
    ax1.set_title('PID Proportional Term: P = Kp × error')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Plot 2: Integral Term
    ax2.plot(times, i_term, 'g-', linewidth=1.5)
    ax2.set_ylabel('Integral Term (N·m)')
    ax2.set_title('PID Integral Term: I = Ki × ∫error dt')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Plot 3: Derivative Term
    ax3.plot(times, d_term, 'r-', linewidth=1.5)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Derivative Term (N·m)')
    ax3.set_title('PID Derivative Term: D = Kd × de/dt')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    # Plot 4: Total Control Output
    ax4.plot(times, p_term, 'b--', linewidth=1, alpha=0.7, label='P')
    ax4.plot(times, i_term, 'g--', linewidth=1, alpha=0.7, label='I')
    ax4.plot(times, d_term, 'r--', linewidth=1, alpha=0.7, label='D')
    ax4.plot(times, control, 'k-', linewidth=2, label='Total Output')
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Torque (N·m)')
    ax4.set_title('Total Control Output = P + I + D')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax4.legend()
    
    plt.tight_layout()
    
    if save_filename:
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_filename}")
    
    plt.show()


def plot_torques(data: dict, save_filename: str = None):
    """
    Plot control torque vs disturbance torque
    
    Args:
        data: Simulation data dictionary
        save_filename: Optional filename to save plot
    """
    times = np.array(data['data']['time'])
    control = np.array(data['data']['control_torque'])
    disturbance = np.array(data['data']['disturbance_torque'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(times, control, 'b-', linewidth=1.5, label='Control Torque (PID)')
    ax.plot(times, disturbance, 'r--', linewidth=1.5, label='Disturbance Torque')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Torque (N·m)')
    ax.set_title('Control vs Disturbance Torques')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_filename:
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_filename}")
    
    plt.show()


def plot_velocity(data: dict, save_filename: str = None):
    """
    Plot angular velocity over time
    
    Args:
        data: Simulation data dictionary
        save_filename: Optional filename to save plot
    """
    times = np.array(data['data']['time'])
    velocity = np.array(data['data']['velocity'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(times, velocity, 'purple', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Angular Velocity (rad/s)')
    ax.set_title('Satellite Angular Velocity Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_filename:
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_filename}")
    
    plt.show()


def plot_comprehensive_analysis(data: dict, save_filename: str = None):
    """
    Plot comprehensive 4-subplot analysis
    
    Args:
        data: Simulation data dictionary
        save_filename: Optional filename to save plot
    """
    times = np.array(data['data']['time'])
    angles_deg = np.degrees(np.array(data['data']['angle']))
    errors_deg = np.degrees(np.array(data['data']['error']))
    control = np.array(data['data']['control_torque'])
    disturbance = np.array(data['data']['disturbance_torque'])
    
    fig = plt.figure(figsize=(15, 10))
    
    # Create 2x2 grid
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)
    
    # Plot 1: Angle
    ax1.plot(times, angles_deg, 'b-', linewidth=2)
    ax1.axhline(y=0, color='r', linestyle='--', linewidth=1)
    ax1.set_ylabel('Angle (°)')
    ax1.set_title('Satellite Attitude')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error
    ax2.plot(times, errors_deg, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax2.set_ylabel('Error (°)')
    ax2.set_title('Attitude Error')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Torques
    ax3.plot(times, control, 'b-', linewidth=1.5, label='Control')
    ax3.plot(times, disturbance, 'r--', linewidth=1.5, label='Disturbance')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Torque (N·m)')
    ax3.set_title('Control vs Disturbance')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistics box
    stats = data['statistics']
    ax4.axis('off')
    
    stats_text = f"""SIMULATION STATISTICS

Final Angle:           {np.degrees(stats['final_angle']):8.2f}°
Final Error:           {np.degrees(stats['final_error']):8.2f}°
Max Error:             {np.degrees(stats['max_error']):8.2f}°
Mean Error:            {np.degrees(stats['mean_error']):8.2f}°
Steady-State Error:    {np.degrees(stats['steady_state_error']):8.2f}°
Overshoot:             {np.degrees(stats['overshoot']):8.2f}°
Settling Time:         {stats['settling_time']:8.2f} s

Total Simulation Time: {stats['simulation_time']:8.2f} s
Steps Executed:        {stats['total_steps']}
"""
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_filename:
        plt.savefig(save_filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_filename}")
    
    plt.show()


def print_detailed_statistics(data: dict):
    """Print detailed statistics from simulation"""
    stats = data['statistics']
    
    print("\n" + "=" * 70)
    print("DETAILED SIMULATION STATISTICS")
    print("=" * 70)
    print(f"\nAttitude Control Performance:")
    print(f"  Final Angle:              {np.degrees(stats['final_angle']):10.4f}°")
    print(f"  Final Error:              {np.degrees(stats['final_error']):10.4f}°")
    print(f"  Max Error:                {np.degrees(stats['max_error']):10.4f}°")
    print(f"  Mean Error:               {np.degrees(stats['mean_error']):10.4f}°")
    print(f"  Steady-State Error:       {np.degrees(stats['steady_state_error']):10.4f}°")
    print(f"\nResponse Characteristics:")
    print(f"  Overshoot:                {np.degrees(stats['overshoot']):10.4f}°")
    print(f"  Settling Time:            {stats['settling_time']:10.4f} seconds")
    print(f"\nSimulation Details:")
    print(f"  Total Duration:           {stats['simulation_time']:10.4f} seconds")
    print(f"  Total Steps:              {stats['total_steps']:10}")
    print("=" * 70)


if __name__ == "__main__":
    # Load simulation data
    print("Loading simulation data...")
    data = load_simulation_data("satellite_simulation.json")
    
    # Print statistics
    print_detailed_statistics(data)
    
    # Create plots
    print("\nGenerating plots...")
    plot_satellite_attitude(data, "plot_attitude.png")
    plot_pid_terms(data, "plot_pid_terms.png")
    plot_torques(data, "plot_torques.png")
    plot_velocity(data, "plot_velocity.png")
    plot_comprehensive_analysis(data, "plot_comprehensive.png")
    
    print("\nAll plots generated successfully!")
