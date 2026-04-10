import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import os

def compute_distances(waypoints):
    """Calculates cumulative distances along the raw waypoints."""
    clean_wps = [waypoints[0]]
    for w in waypoints[1:]:
        if np.hypot(w[0] - clean_wps[-1][0], w[1] - clean_wps[-1][1]) > 1e-4:
            clean_wps.append(w)
            
    waypoints = np.array(clean_wps)
    
    diffs = np.diff(waypoints, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    
    cum_dist = np.insert(np.cumsum(distances), 0, 0)
    return cum_dist, waypoints

def generate_trajectory(waypoints, mode='min_energy'):
    cum_dist, waypoints = compute_distances(waypoints)
    
    if mode == 'min_time':
        avg_velocity = 20.0  
        num_samples = 150  
    elif mode == 'min_energy':
        avg_velocity = 5.0   
        num_samples = 400  
    else:
        raise ValueError("Mode must be 'min_time' or 'min_energy'")

    time_points = cum_dist / avg_velocity

    # 2. Fit a Cubic Spline to the waypoints, parameterized by time
    cs_x = CubicSpline(time_points, waypoints[:, 0], bc_type='clamped')
    cs_y = CubicSpline(time_points, waypoints[:, 1], bc_type='clamped')

    # 3. Generate high-resolution smooth trajectory arrays
    t_smooth = np.linspace(0, time_points[-1], num_samples)
    x_smooth = cs_x(t_smooth)
    y_smooth = cs_y(t_smooth)
    
    # Output is a list of (x, y, t) tuples as required
    trajectory = list(zip(x_smooth, y_smooth, t_smooth))
    
    # --- Calculate Derivatives for Plotting ---
    # 1st derivative of position = velocity
    vx = cs_x(t_smooth, 1)
    vy = cs_y(t_smooth, 1)
    speed = np.hypot(vx, vy)
    
    # 2nd derivative of position = acceleration
    ax = cs_x(t_smooth, 2)
    ay = cs_y(t_smooth, 2)
    acceleration = np.hypot(ax, ay)
    
    return trajectory, speed, acceleration, t_smooth

def plot_trajectories(t_time, speed_time, acc_time, t_energy, speed_energy, acc_energy):
    """Generates and saves the required trajectory_comparison.png"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left Subplot: Speed vs Time
    ax1.plot(t_time, speed_time, 'r-', label='Min-Time (Fast)', linewidth=2)
    ax1.plot(t_energy, speed_energy, 'b-', label='Min-Energy (Smooth)', linewidth=2)
    ax1.set_title("Speed vs. Time Profile")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Speed (units/s)")
    ax1.legend()
    ax1.grid(True, linestyle=':', alpha=0.7)
    
    # Right Subplot: Acceleration vs Time
    ax2.plot(t_time, acc_time, 'r-', label='Min-Time', linewidth=2)
    ax2.plot(t_energy, acc_energy, 'b-', label='Min-Energy', linewidth=2)
    ax2.set_title("Acceleration vs. Time Profile")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Acceleration (units/s²)")
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'trajectory_comparison.png')    
    plt.savefig(save_path)
    print(f"\nTrajectory comparison plot saved successfully to: {save_path}")
    plt.show()

if __name__ == "__main__":
    sample_waypoints = [(5, 50), (25, 65), (50, 85), (75, 65), (95, 50)]
    
    print("Generating Minimum-Time Trajectory...")
    traj_time, spd_time, acc_time, t_t = generate_trajectory(sample_waypoints, 'min_time')
    
    print("Generating Minimum-Energy Trajectory...")
    traj_energy, spd_energy, acc_energy, t_e = generate_trajectory(sample_waypoints, 'min_energy')
    
    print("\n--- FIRST 20 WAYPOINTS OF MIN-ENERGY TRAJECTORY ---")
    print(f"{'X-Coord':<12} | {'Y-Coord':<12} | {'Time (s)':<12}")
    print("-" * 43)    
    for i in range(20):
        x, y, t = traj_energy[i]
        print(f"{x:<12.2f} | {y:<12.2f} | {t:<12.2f}")
        
    print("\nGenerating comparison plots (this will open a window)...")
    plot_trajectories(t_t, spd_time, acc_time, t_e, spd_energy, acc_energy)