import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle

from map_setup import *
from path_planner import *
from trajectory import *
from formation import *

def create_results_folder():
    if not os.path.exists("results"):
        os.makedirs("results")


def save_path_plot(grid, path):
    plt.imshow(grid.T, origin='lower', cmap='gray_r')
    x, y = zip(*path)
    plt.plot(x, y, label="Path")
    plt.scatter(*START, label='Start')
    plt.scatter(*GOAL, label='Goal')
    plt.title("Planned Path")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig("results/path_plot.png")
    plt.close()


def save_trajectory_plot(traj_fast, traj_slow):
    speed_f, accel_f, t_f = compute_velocity_acceleration(traj_fast)
    speed_s, accel_s, t_s = compute_velocity_acceleration(traj_slow)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    axs[0].plot(t_f, speed_f, label='Min-time')
    axs[0].plot(t_s, speed_s, label='Min-energy')
    axs[0].set_title("Speed vs Time")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Speed")
    axs[0].legend()

    axs[1].plot(t_f, accel_f, label='Min-time')
    axs[1].plot(t_s, accel_s, label='Min-energy')
    axs[1].set_title("Acceleration vs Time")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Acceleration")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("results/trajectory_comparison.png")
    plt.close()


def animate_formation(drones_fast, drones_slow):
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    drone_sets = [drones_fast, drones_slow]
    titles = ["Min-time Formation", "Min-energy Formation"]

    scatters_list = []

    for i, ax in enumerate(axes):
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(titles[i])

        obstacle = Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS + SAFETY_MARGIN, color="red", alpha=0.3)
        ax.add_patch(obstacle)
        ax.scatter(*START, marker="*", color="green", s=80, label="Start")
        ax.scatter(*GOAL, marker="X", color="blue", s=80, label="Goal")
        ax.legend(loc="upper right")

        scatters = [ax.plot([], [], 'o')[0] for _ in drones_fast]
        scatters_list.append(scatters)

    def update(frame):
        for scatters, drones in zip(scatters_list, drone_sets):
            for i, traj in enumerate(drones):
                if frame < len(traj):
                    x, y, _ = traj[frame]
                    scatters[i].set_data([x], [y])
        return sum(scatters_list, [])

    ani = animation.FuncAnimation(fig, update, frames=100, blit=True)
    ani.save("results/formation_animation.gif", writer="pillow", fps=20)


def compute_metrics(traj):
    import numpy as np
    traj = np.array(traj)
    x, y, t = traj[:,0], traj[:,1], traj[:,2]

    dist = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    total_time = t[-1]

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)

    energy = np.sum(ax**2 + ay**2)

    return total_time, dist, energy


if __name__ == "__main__":
    create_results_folder()

    grid = create_map()
    path = astar(grid, START, GOAL)

    traj_fast, traj_slow = generate_trajectories(path)

    # Compute and print metrics
    time_f, dist_f, energy_f = compute_metrics(traj_fast)
    time_s, dist_s, energy_s = compute_metrics(traj_slow)
    print(f"Min-time trajectory: Total time = {time_f:.2f}s, Distance = {dist_f:.2f}, Energy = {energy_f:.2f}")
    print(f"Min-energy trajectory: Total time = {time_s:.2f}s, Distance = {dist_s:.2f}, Energy = {energy_s:.2f}")

    offsets = get_formation_offsets()
    drones_fast = apply_formation(traj_fast, offsets)
    drones_slow = apply_formation(traj_slow, offsets)

    save_path_plot(grid, path)
    save_trajectory_plot(traj_fast, traj_slow)
    animate_formation(drones_fast, drones_slow)

    print("Done. Check results folder.")