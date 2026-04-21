import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from map_setup    import START, GOAL, OBSTACLE_CENTER, OBSTACLE_RADIUS, GRID_SIZE
from path_planner import plan_path
from trajectory   import get_trajectories, compute_speed, compute_accel, energy_estimate
from formation    import drone_positions, N_DRONES

os.makedirs("results", exist_ok=True)

def save_path_plot(waypoints):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect('equal')
    ax.set_title("Planned Path — A* Around Obstacle")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.add_patch(plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color='red', alpha=0.55, label='Obstacle'))

    xs = [p[0] for p in waypoints]
    ys = [p[1] for p in waypoints]
    ax.plot(xs, ys, 'b-o', markersize=5, linewidth=2, label='A* path')
    ax.plot(*START, 'go', markersize=13, label='Start')
    ax.plot(*GOAL,  'b*', markersize=15, label='Goal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "path_plot.png"), dpi=120)
    plt.close()
    print("Saved results/path_plot.png")

def save_trajectory_comparison(fast, slow):
    t_f, x_f, y_f = fast
    t_s, x_s, y_s = slow

    spd_f = compute_speed(t_f, x_f, y_f)
    spd_s = compute_speed(t_s, x_s, y_s)
    acc_f = compute_accel(t_f, x_f, y_f)
    acc_s = compute_accel(t_s, x_s, y_s)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Trajectory Comparison: Min-Time vs Min-Energy", fontsize=13)

    axes[0].plot(t_f, spd_f, color='crimson',   label='Min-time')
    axes[0].plot(t_s, spd_s, color='steelblue', label='Min-energy')
    axes[0].set_title("Speed vs Time")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (units/s)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_f, acc_f, color='crimson',   label='Min-time')
    axes[1].plot(t_s, acc_s, color='steelblue', label='Min-energy')
    axes[1].set_title("Acceleration vs Time")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration (units/s²)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join("results", "trajectory_comparison.png"), dpi=120)
    plt.close()
    print("Saved results/trajectory_comparison.png")

def build_animation(fast, slow):
    t_f, x_f, y_f = fast
    t_s, x_s, y_s = slow

    n_frames = 120
    idx_f = np.linspace(0, len(t_f) - 1, n_frames).astype(int)
    idx_s = np.linspace(0, len(t_s) - 1, n_frames).astype(int)

    colors = plt.cm.tab10(np.linspace(0, 0.5, N_DRONES))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("UAV Formation Animation (V-Shape, 5 Drones)", fontsize=13)

    for ax, label in zip(axes, ["Min-Time Trajectory", "Min-Energy Trajectory"]):
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)
        ax.set_aspect('equal')
        ax.set_title(label)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.add_patch(plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color='red', alpha=0.45))
        ax.plot(*START, 'go', markersize=10)
        ax.plot(*GOAL,  'b*', markersize=12)
        ax.grid(True, alpha=0.2)

    drone_dots_f = [axes[0].plot([], [], 'o', color=colors[i], markersize=9)[0] for i in range(N_DRONES)]
    drone_dots_s = [axes[1].plot([], [], 'o', color=colors[i], markersize=9)[0] for i in range(N_DRONES)]
    trail_f      = axes[0].plot([], [], 'gray', alpha=0.3, linewidth=1)[0]
    trail_s      = axes[1].plot([], [], 'gray', alpha=0.3, linewidth=1)[0]

    def compute_heading(x_arr, y_arr, i):
        if i < len(x_arr) - 1:
            dx = x_arr[i+1] - x_arr[i]
            dy = y_arr[i+1] - y_arr[i]
            return np.arctan2(dy, dx)
        return 0.0

    def update(frame):
        i_f = idx_f[frame]
        i_s = idx_s[frame]

        cx_f, cy_f = x_f[i_f], y_f[i_f]
        cx_s, cy_s = x_s[i_s], y_s[i_s]

        ang_f = compute_heading(x_f, y_f, i_f)
        ang_s = compute_heading(x_s, y_s, i_s)

        pos_f = drone_positions(cx_f, cy_f, ang_f)
        pos_s = drone_positions(cx_s, cy_s, ang_s)

        for i in range(N_DRONES):
            drone_dots_f[i].set_data([pos_f[i, 0]], [pos_f[i, 1]])
            drone_dots_s[i].set_data([pos_s[i, 0]], [pos_s[i, 1]])

        trail_f.set_data(x_f[:i_f], y_f[:i_f])
        trail_s.set_data(x_s[:i_s], y_s[:i_s])

        return drone_dots_f + drone_dots_s + [trail_f, trail_s]

    ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50, blit=True)
    return fig, ani

def print_summary(fast, slow):
    t_f, x_f, y_f = fast
    t_s, x_s, y_s = slow

    dist_f = np.sum(np.sqrt(np.diff(x_f)**2 + np.diff(y_f)**2))
    dist_s = np.sum(np.sqrt(np.diff(x_s)**2 + np.diff(y_s)**2))

    e_f = energy_estimate(*fast)
    e_s = energy_estimate(*slow)

    print("\n===  Summary  ===")
    print(f"Min-time   — duration: {t_f[-1]:.1f}s   distance: {dist_f:.1f}   energy: {e_f:.1f}")
    print(f"Min-energy — duration: {t_s[-1]:.1f}s   distance: {dist_s:.1f}   energy: {e_s:.1f}")
    print(f"Time saved by min-time: {t_s[-1] - t_f[-1]:.1f}s")
    print(f"Energy saved by min-energy: {e_f - e_s:.1f} ({(e_f - e_s)/e_f*100:.1f}%)")
    print("=================\n")

def main():
    print("Planning path...")
    waypoints = plan_path()

    print("Generating trajectories...")
    fast, slow = get_trajectories(waypoints)

    print("Saving path plot...")
    save_path_plot(waypoints)

    print("Saving trajectory comparison...")
    save_trajectory_comparison(fast, slow)

    print_summary(fast, slow)

    print("Building animation...")
    fig, ani = build_animation(fast, slow)

    gif_path = os.path.join("results", "formation_animation.gif")
    print("Saving animation (this takes a moment)...")
    ani.save(gif_path, writer='pillow', fps=24)
    print(f"Saved {gif_path}")

    plt.show()

if __name__ == "__main__":
    main()
