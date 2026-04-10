import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

from map_setup import START, GOAL, OBSTACLE_CENTER, BASE_OBSTACLE_RADIUS, INFLATED_RADIUS, GRID_SIZE, Formation
from path_planner import run_rrt_star
from trajectory import generate_trajectory, compute_distances

def calculate_energy_proxy(acceleration_array, time_array):
    """A simple metric to estimate energy usage based on squared accelerations."""
    dt = np.diff(time_array)
    sq_accel = acceleration_array[:-1]**2
    return np.sum(sq_accel * dt)

def main():
    print("Starting Full UAV Simulation...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)

    print("\n--- Running RRT* Path Planner ---")
    tree, raw_waypoints = run_rrt_star()
    print(f"Path found with {len(raw_waypoints)} waypoints.")
    
    print("\n--- Generating Trajectories ---")
    traj_time, spd_time, acc_time, t_time = generate_trajectory(raw_waypoints, 'min_time')
    traj_energy, spd_energy, acc_energy, t_energy = generate_trajectory(raw_waypoints, 'min_energy')
    
    cum_dist, clean_wps = compute_distances(raw_waypoints)
    total_dist = cum_dist[-1]
    
    energy_proxy_time = calculate_energy_proxy(acc_time, t_time)
    energy_proxy_energy = calculate_energy_proxy(acc_energy, t_energy)
    
    print("\n=== FLIGHT SUMMARY ===")
    print(f"Total Path Distance: {total_dist:.2f} units")
    print("\nMin-Time Trajectory (Red):")
    print(f"  Total Time: {t_time[-1]:.2f} seconds")
    print(f"  Energy Cost (proxy): {energy_proxy_time:.2f}")
    print("\nMin-Energy Trajectory (Blue):")
    print(f"  Total Time: {t_energy[-1]:.2f} seconds")
    print(f"  Energy Cost (proxy): {energy_proxy_energy:.2f}")
    print("======================\n")

    print("Generating Animation... (This may take 15-30 seconds to render and save)")
    form = Formation()
    
    # Create side-by-side plots for the two trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f"UAV Formation Flight Simulation ({form.N}-Drone 'W')", fontsize=16)
    
    for ax, title in zip([ax1, ax2], ["Minimum-Time Trajectory (Fast)", "Minimum-Energy Trajectory (Smooth)"]):
        ax.set_xlim(-5, GRID_SIZE + 5)
        ax.set_ylim(-5, GRID_SIZE + 5)
        ax.set_title(title)
        ax.set_aspect('equal')
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # Draw Environment
        core = plt.Circle(OBSTACLE_CENTER, BASE_OBSTACLE_RADIUS, color='black', fill=True)
        safe = plt.Circle(OBSTACLE_CENTER, INFLATED_RADIUS, color='red', fill=False, linestyle='--')
        ax.add_patch(core)
        ax.add_patch(safe)
        
        # Draw Start, Goal, and RRT* Path
        ax.plot(START[0], START[1], 'go', markersize=8)
        ax.plot(GOAL[0], GOAL[1], 'ro', markersize=8)
        path_x = [p[0] for p in clean_wps]
        path_y = [p[1] for p in clean_wps]
        ax.plot(path_x, path_y, 'gray', linestyle='--', alpha=0.6)

    scat_time = ax1.scatter([], [], c='red', s=40, zorder=5)
    scat_energy = ax2.scatter([], [], c='blue', s=40, zorder=5)
    
    # Initialize Formation Lines (to visualize the 'W')
    lines_time = [ax1.plot([], [], 'r-', alpha=0.6)[0] for _ in range(form.N - 1)]
    lines_energy = [ax2.plot([], [], 'b-', alpha=0.6)[0] for _ in range(form.N - 1)]
    
    txt_time = ax1.text(0.05, 0.95, '', transform=ax1.transAxes, fontsize=12, verticalalignment='top')
    txt_energy = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, fontsize=12, verticalalignment='top')
    
    max_frames = max(len(traj_time), len(traj_energy))
    
    def update(frame):
        idx_t = min(frame, len(traj_time) - 1) 
        xt, yt, time_t = traj_time[idx_t]
        pos_t = np.array([xt, yt]) + form.offsets
        scat_time.set_offsets(pos_t)
        
        for i in range(form.N - 1):
            lines_time[i].set_data([pos_t[i,0], pos_t[i+1,0]], [pos_t[i,1], pos_t[i+1,1]])
        txt_time.set_text(f"Time: {time_t:.2f}s")
        
        idx_e = min(frame, len(traj_energy) - 1)
        xe, ye, time_e = traj_energy[idx_e]
        pos_e = np.array([xe, ye]) + form.offsets
        scat_energy.set_offsets(pos_e)
        
        for i in range(form.N - 1):
            lines_energy[i].set_data([pos_e[i,0], pos_e[i+1,0]], [pos_e[i,1], pos_e[i+1,1]])
        txt_energy.set_text(f"Time: {time_e:.2f}s")
        
        return scat_time, scat_energy, txt_time, txt_energy, *lines_time, *lines_energy

    ani = animation.FuncAnimation(fig, update, frames=max_frames, interval=30, blit=True)
    
    gif_path = os.path.join(results_dir, 'formation_animation.gif')
    ani.save(gif_path, writer='pillow', fps=30)
    print(f"\nSUCCESS! Animation saved to: {gif_path}")    
    plt.show()

if __name__ == '__main__':
    main()