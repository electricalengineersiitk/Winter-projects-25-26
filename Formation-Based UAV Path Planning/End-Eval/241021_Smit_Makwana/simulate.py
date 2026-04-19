import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from map_setup import START, GOAL, OBSTACLE_CENTER, OBSTACLE_RADIUS, GRID_SIZE
from path_planner import plan_path
from trajectory import generate_trajectories
from formation import apply_formation

def calculate_derivatives(trajectory):
    """Calculates speed and acceleration given a (x, y, t) trajectory."""
    times = trajectory[:, 2]
    dt = np.diff(times)
    
    dx = np.diff(trajectory[:, 0])
    dy = np.diff(trajectory[:, 1])
    vx = dx / dt
    vy = dy / dt
    speed = np.sqrt(vx**2 + vy**2)
    
    dvx = np.diff(vx)
    dvy = np.diff(vy)
    dt_accel = dt[1:] 
    ax = dvx / dt_accel
    ay = dvy / dt_accel
    acceleration = np.sqrt(ax**2 + ay**2)
    
    speed = np.append(speed, speed[-1])
    acceleration = np.pad(acceleration, (0, 2), 'edge') 
    
    return times, speed, acceleration

def plot_trajectory_comparison(min_time_traj, min_energy_traj):
    """Generates the side-by-side speed and acceleration plots."""
    t_time, speed_time, accel_time = calculate_derivatives(min_time_traj)
    t_energy, speed_energy, accel_energy = calculate_derivatives(min_energy_traj)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(t_time, speed_time, label='Min-Time', color='red')
    ax1.plot(t_energy, speed_energy, label='Min-Energy', color='blue')
    ax1.set_title('Speed Profile')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Speed')
    ax1.legend()

    ax2.plot(t_time, accel_time, label='Min-Time', color='red')
    ax2.plot(t_energy, accel_energy, label='Min-Energy', color='blue')
    ax2.set_title('Acceleration Profile')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Acceleration')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join('results', 'trajectory_comparison.png'))
    plt.close()

def create_animation(formation_min_time, formation_min_energy, waypoints):
    """Animates the formation flying and saves it as a GIF."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    circle = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color='red', alpha=0.5, label='Obstacle')
    ax.add_patch(circle)
    
    x_coords, y_coords = zip(*waypoints)
    ax.plot(x_coords, y_coords, 'k:', alpha=0.5, label='Centroid Path')
    ax.plot(START[0], START[1], 'go', label='Start')
    ax.plot(GOAL[0], GOAL[1], 'yo', label='Goal')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')

    num_drones = formation_min_time.shape[0]
    scat_time = ax.scatter([], [], c='red', s=50, label='Min-Time Formation')
    scat_energy = ax.scatter([], [], c='blue', s=50, label='Min-Energy Formation')
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.legend(loc='lower right')

    max_frames = max(formation_min_time.shape[1], formation_min_energy.shape[1])

    def init():
        scat_time.set_offsets(np.empty((0, 2)))
        scat_energy.set_offsets(np.empty((0, 2)))
        time_text.set_text('')
        return scat_time, scat_energy, time_text

    def update(frame):
        idx_time = min(frame, formation_min_time.shape[1] - 1)
        positions_time = formation_min_time[:, idx_time, 0:2]
        scat_time.set_offsets(positions_time)

        idx_energy = min(frame, formation_min_energy.shape[1] - 1)
        positions_energy = formation_min_energy[:, idx_energy, 0:2]
        scat_energy.set_offsets(positions_energy)

        current_t = formation_min_energy[0, idx_energy, 2]
        time_text.set_text(f'Time: {current_t:.1f}s')
        
        return scat_time, scat_energy, time_text

    ani = animation.FuncAnimation(fig, update, frames=max_frames,
                                  init_func=init, blit=True, interval=50)
    
    print("Saving animation... this might take a minute.")
    ani.save(os.path.join('results', 'formation_animation.gif'), writer='pillow', fps=20)
    plt.close()

def main():
    os.makedirs('results', exist_ok=True) 

    print("Phase 2: Running A* Path Planner...")
    waypoints = plan_path()
    if not waypoints:
        print("Error: No path found.")
        return

    print("Phase 3: Generating Trajectories...")
    min_time_traj, min_energy_traj = generate_trajectories(waypoints)

    print("Phase 4: Applying Formation Offsets...")
    formation_min_time = apply_formation(min_time_traj)
    formation_min_energy = apply_formation(min_energy_traj)

    print("Phase 5a: Producing Static Path Plot...")
    fig, ax = plt.subplots()
    circle = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color='red', alpha=0.5, label='Obstacle')
    ax.add_patch(circle)
    x_coords, y_coords = zip(*waypoints)
    ax.plot(x_coords, y_coords, 'b--', label='A* Planned Path')
    ax.plot(START[0], START[1], 'go', markersize=8, label='Start')
    ax.plot(GOAL[0], GOAL[1], 'yo', markersize=8, label='Goal')
    ax.set_title('Planned Path around Obstacle')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend()
    plt.savefig(os.path.join('results', 'path_plot.png'))
    plt.close()

    print("Phase 5b: Plotting Trajectory Comparisons...")
    plot_trajectory_comparison(min_time_traj, min_energy_traj)

    print("Phase 5c: Creating Formation Animation...")
    create_animation(formation_min_time, formation_min_energy, waypoints)

    print("\nSimulation Complete!")
    print("Check the 'results' folder for your outputs.")

    print("\n--- Summary Statistics ---")
    print(f"Min-Time Trajectory: Duration = {min_time_traj[-1, 2]:.2f}s")
    print(f"Min-Energy Trajectory: Duration = {min_energy_traj[-1, 2]:.2f}s")

if __name__ == "__main__":
    main()