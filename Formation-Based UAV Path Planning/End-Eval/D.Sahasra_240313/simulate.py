import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from map_setup import START, GOAL, OBSTACLE_CENTER, OBSTACLE_RADIUS, GRID_SIZE
from path_planner import plan_path
from trajectory import generate_trajectories
from formation import apply_formation


RESULT_DIR = "results"


def calculate_derivatives(trajectory):
    """Return time, speed, and acceleration from (x, y, t) trajectory."""
    
    time_vals = trajectory[:, 2]
    delta_t = np.diff(time_vals)

    delta_x = np.diff(trajectory[:, 0])
    delta_y = np.diff(trajectory[:, 1])

    vel_x = delta_x / delta_t
    vel_y = delta_y / delta_t

    speed = np.sqrt(vel_x ** 2 + vel_y ** 2)

    delta_vx = np.diff(vel_x)
    delta_vy = np.diff(vel_y)

    accel_x = delta_vx / delta_t[1:]
    accel_y = delta_vy / delta_t[1:]

    acceleration = np.sqrt(accel_x ** 2 + accel_y ** 2)

    speed = np.append(speed, speed[-1])
    acceleration = np.pad(acceleration, (0, 2), mode="edge")

    return time_vals, speed, acceleration


def plot_trajectory_comparison(min_time_traj, min_energy_traj):
    """Create speed and acceleration comparison plots."""

    t1, speed1, accel1 = calculate_derivatives(min_time_traj)
    t2, speed2, accel2 = calculate_derivatives(min_energy_traj)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_speed, ax_accel = axes

    ax_speed.plot(t1, speed1, color="red", label="Min-Time")
    ax_speed.plot(t2, speed2, color="blue", label="Min-Energy")
    ax_speed.set_title("Speed Profile")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_ylabel("Speed")
    ax_speed.legend()

    ax_accel.plot(t1, accel1, color="red", label="Min-Time")
    ax_accel.plot(t2, accel2, color="blue", label="Min-Energy")
    ax_accel.set_title("Acceleration Profile")
    ax_accel.set_xlabel("Time (s)")
    ax_accel.set_ylabel("Acceleration")
    ax_accel.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, "trajectory_comparison.png"))
    plt.close()


def create_animation(formation_min_time, formation_min_energy, waypoints):
    """Animate drone formations and save GIF."""

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)

    obstacle = plt.Circle(
        OBSTACLE_CENTER,
        OBSTACLE_RADIUS,
        color="red",
        alpha=0.5,
        label="Obstacle"
    )
    ax.add_patch(obstacle)

    path_x, path_y = zip(*waypoints)
    ax.plot(path_x, path_y, "k:", alpha=0.5, label="Centroid Path")

    ax.plot(*START, "go", label="Start")
    ax.plot(*GOAL, "yo", label="Goal")

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    scatter_time = ax.scatter([], [], c="red", s=50, label="Min-Time Formation")
    scatter_energy = ax.scatter([], [], c="blue", s=50, label="Min-Energy Formation")

    timer_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)
    ax.legend(loc="lower right")

    total_frames = max(
        formation_min_time.shape[1],
        formation_min_energy.shape[1]
    )

    def init():
        scatter_time.set_offsets(np.empty((0, 2)))
        scatter_energy.set_offsets(np.empty((0, 2)))
        timer_text.set_text("")
        return scatter_time, scatter_energy, timer_text

    def update(frame):
        idx1 = min(frame, formation_min_time.shape[1] - 1)
        idx2 = min(frame, formation_min_energy.shape[1] - 1)

        pos1 = formation_min_time[:, idx1, 0:2]
        pos2 = formation_min_energy[:, idx2, 0:2]

        scatter_time.set_offsets(pos1)
        scatter_energy.set_offsets(pos2)

        current_time = formation_min_energy[0, idx2, 2]
        timer_text.set_text(f"Time: {current_time:.1f}s")

        return scatter_time, scatter_energy, timer_text

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        init_func=init,
        blit=True,
        interval=50
    )

    print("Saving animation... this might take a minute.")
    ani.save(
        os.path.join(RESULT_DIR, "formation_animation.gif"),
        writer="pillow",
        fps=20
    )

    plt.close()


def create_static_plot(waypoints):
    """Generate planned path image."""

    fig, ax = plt.subplots()

    obstacle = plt.Circle(
        OBSTACLE_CENTER,
        OBSTACLE_RADIUS,
        color="red",
        alpha=0.5,
        label="Obstacle"
    )
    ax.add_patch(obstacle)

    x_vals, y_vals = zip(*waypoints)

    ax.plot(x_vals, y_vals, "b--", label="A* Planned Path")
    ax.plot(*START, "go", markersize=8, label="Start")
    ax.plot(*GOAL, "yo", markersize=8, label="Goal")

    ax.set_title("Planned Path around Obstacle")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    ax.legend()

    plt.savefig(os.path.join(RESULT_DIR, "path_plot.png"))
    plt.close()


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

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
    create_static_plot(waypoints)

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
