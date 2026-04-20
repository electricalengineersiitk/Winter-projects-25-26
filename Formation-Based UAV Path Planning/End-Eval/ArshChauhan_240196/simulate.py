import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

from map_setup    import plot_map, OBS_X, OBS_Y, OBS_W, OBS_H
from path_planner import plan_path
from trajectory   import generate_trajectories
from formation    import N_DRONES, COLORS, compute_all_positions

os.makedirs("results", exist_ok=True)

path = plan_path()
xs_path = [p[0] for p in path]
ys_path = [p[1] for p in path]

traj_fast, traj_slow = generate_trajectories(path)

fig, ax = plt.subplots(figsize=(7, 7))
plot_map(ax, title="Planned Path (A*)")
ax.plot(xs_path, ys_path, color="orange", linewidth=1.5,
        label="A* path", zorder=5)
ax.legend()
plt.tight_layout()
plt.savefig("results/path_plot.png", dpi=100)
plt.close()
print("  Saved: results/path_plot.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(traj_fast["t"], traj_fast["speed"], label="Min-Time",   color="orange")
axes[0].plot(traj_slow["t"], traj_slow["speed"], label="Min-Energy", color="green")
axes[0].set_title("Speed vs Time")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Speed (units/s)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(traj_fast["t"], traj_fast["accel"], label="Min-Time",   color="orange")
axes[1].plot(traj_slow["t"], traj_slow["accel"], label="Min-Energy", color="green")
axes[1].set_title("Acceleration vs Time")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Accel (units/s^2)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

fig.suptitle("Trajectory Comparison: Min-Time vs Min-Energy")
plt.tight_layout()
plt.savefig("results/trajectory_comparison.png", dpi=100)
plt.close()
print("  Saved: results/trajectory_comparison.png")

def make_animation(traj, filename, title):
    positions = compute_all_positions(traj)
    T = len(traj["t"])

    fig, ax = plt.subplots(figsize=(7, 7))
    plot_map(ax, title=title)
    ax.plot(xs_path, ys_path, color="orange", linewidth=1,
            alpha=0.4, label="Planned path")

    drone_dots = []
    for i in range(N_DRONES):
        dot, = ax.plot([], [], "o", color=COLORS[i], markersize=8,
                       label=f"D{i}")
        drone_dots.append(dot)

    time_text = ax.text(2, 95, "", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)

    SKIP = max(1, T // 100)
    frames = list(range(0, T, SKIP))

    def init():
        for dot in drone_dots:
            dot.set_data([], [])
        time_text.set_text("")
        return drone_dots + [time_text]

    def update(frame):
        for i in range(N_DRONES):
            x = positions[frame, i, 0]
            y = positions[frame, i, 1]
            drone_dots[i].set_data([x], [y])
        time_text.set_text(f"t = {traj['t'][frame]:.1f}s")
        return drone_dots + [time_text]

    anim = animation.FuncAnimation(
        fig, update, frames=frames,
        init_func=init, blit=True, interval=80
    )
    anim.save(filename, writer=animation.PillowWriter(fps=12), dpi=80)
    plt.close()
    print(f"  Saved: {filename}")

make_animation(traj_fast, "results/formation_animation_mintime.gif",
               "V-Formation: Min-Time")
make_animation(traj_slow, "results/formation_animation_minenergy.gif",
               "V-Formation: Min-Energy")
print("\n--- RESULTS SUMMARY ---")
print(f"Path waypoints  : {len(path)}")
print(f"Path distance   : {traj_fast['total_dist']:.1f} units")
print(f"Min-Time  total : {traj_fast['total_time']:.1f} s")
print(f"Min-Energy total: {traj_slow['total_time']:.1f} s")
print(f"Min-Energy is {traj_slow['total_time']/traj_fast['total_time']:.1f}x slower but uses {(1 - traj_slow['energy']/traj_fast['energy'])*100:.0f}% less energy")
print("\nAll output files saved to results/")
