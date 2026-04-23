import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

from map_setup import start , goal, obs_center, obs_rad, GRID_SIZE
from path_planner import get_dijkstra_path
from trajectory import generate_trajectories
from formation import get_formation_offsets, get_drone_positions

if not os.path.exists('results'):
    os.makedirs('results')

def run_simulation():
    path = get_dijkstra_path()
    if path is None:
        print("No path found!")
        return

    min_time_traj, min_energy_traj = generate_trajectories(path)
    
    offsets = get_formation_offsets("V") # Or "A"
    num_drones = len(offsets)

    plt.figure(figsize=(8, 8))
    circle = plt.Circle(obs_center, obs_rad, color='r', alpha=0.3, label='Obstacle')
    plt.gca().add_patch(circle)
    path_np = np.array(path)
    plt.plot(path_np[:, 0], path_np[:, 1], 'b--', label='Planned Path')
    plt.scatter(*start, color='g', s=100, label='Start')
    plt.scatter(*goal, color='k', s=100, label='Goal')
    plt.title("Path Planning Results")
    plt.legend()
    plt.savefig(os.path.join('results', 'path_plot.png'))
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    circle = plt.Circle(obs_center, obs_rad, color='r', alpha=0.3)
    ax.add_patch(circle)
    
    drones, = ax.plot([], [], 'bo', markersize=4, label='UAV Formation')
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    def update(frame):
        centroid = min_time_traj[frame, :2]
        current_time = min_time_traj[frame, 2]
        
        all_drone_pos = get_drone_positions(centroid, offsets)
        
        drones.set_data(all_drone_pos[:, 0], all_drone_pos[:, 1])
        time_text.set_text(f'Time: {current_time:.2f}s')
        return drones, time_text

    ani = FuncAnimation(fig, update, frames=len(min_time_traj), interval=50, blit=True)
    
    print("Saving animation... this may take a moment.")
    ani.save(os.path.join('results', 'formation_animation.gif'), writer='pillow')
    
    plt.show()

    print("\n--- Simulation Summary ---")
    print(f"Min-Time Trajectory: {min_time_traj[-1, 2]:.2f} seconds")
    print(f"Min-Energy Trajectory: {min_energy_traj[-1, 2]:.2f} seconds")

if __name__ == "__main__":
    run_simulation()