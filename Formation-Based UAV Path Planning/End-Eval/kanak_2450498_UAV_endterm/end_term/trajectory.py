import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import os
def generate_trajectories(path_waypoints):
        
        path = np.array(path_waypoints)
        x = path[:, 0]
        y = path[:, 1]
        
        t_original = np.linspace(0, 1, len(path))
        
        
        cs_x = CubicSpline(t_original, x)
        cs_y = CubicSpline(t_original, y)
        
        t_smooth = np.linspace(0, 1, 500)
        x_smooth = cs_x(t_smooth)
        y_smooth = cs_y(t_smooth)
        
        
        total_time_min_time = 10.0
        time_steps_min_time = t_smooth * total_time_min_time
        min_time_traj = np.vstack((x_smooth, y_smooth, time_steps_min_time)).T
        
        
        total_time_min_energy = 25.0
        time_steps_min_energy = t_smooth * total_time_min_energy
        min_energy_traj = np.vstack((x_smooth, y_smooth, time_steps_min_energy)).T
        
        return min_time_traj, min_energy_traj

if __name__ == "__main__":
       
        sample_path = [(5, 50), (30, 45), (50, 35), (75, 45), (95, 50)]
        mt, me = generate_trajectories(sample_path)
        
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(mt[:, 0], mt[:, 1], 'b-', label='Smooth Path')
        plt.scatter([p[0] for p in sample_path], [p[1] for p in sample_path], c='red', label='Waypoints')
        plt.title("Trajectory Smoothing")
        plt.legend()
        
        plt.subplot(1, 2, 2)

def save_trajectory_comparison(mt, me):
    """
    Saves the speed and acceleration comparison required by the project guide.
    mt: Min-Time Trajectory (x, y, t)
    me: Min-Energy Trajectory (x, y, t)
    """
    v_mt = np.sqrt(np.diff(mt[:, 0])**2 + np.diff(mt[:, 1])**2) / np.diff(mt[:, 2])
    v_me = np.sqrt(np.diff(me[:, 0])**2 + np.diff(me[:, 1])**2) / np.diff(me[:, 2])
    
    a_mt = np.diff(v_mt) / np.diff(mt[:-1, 2])
    a_me = np.diff(v_me) / np.diff(me[:-1, 2])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(mt[:-1, 2], v_mt, label='Min-Time (Fast)', color='tab:blue')
    ax1.plot(me[:-1, 2], v_me, label='Min-Energy (Slow)', color='tab:orange')
    ax1.set_title("Speed vs Time")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Speed (units/s)")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(mt[:-2, 2], a_mt, label='Min-Time', color='tab:blue')
    ax2.plot(me[:-2, 2], a_me, label='Min-Energy', color='tab:orange')
    ax2.set_title("Acceleration vs Time")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Acceleration (units/s²)")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if not os.path.exists('results'):
        os.makedirs('results')
        
    plt.savefig('results/trajectory_comparison.png')
    print("Saved results/trajectory_comparison.png successfully!")
    plt.show()


    """
        v_mt = np.sqrt(np.diff(mt[:, 0])**2 + np.diff(mt[:, 1])**2) / np.diff(mt[:, 2])
        v_me = np.sqrt(np.diff(me[:, 0])**2 + np.diff(me[:, 1])**2) / np.diff(me[:, 2])
        
        plt.plot(mt[:-1, 2], v_mt, label='Min-Time (Fast)')
        plt.plot(me[:-1, 2], v_me, label='Min-Energy (Slow)')
        plt.title("Speed vs Time Comparison")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (u/s)")
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        """