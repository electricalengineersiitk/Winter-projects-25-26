"""
simulate.py
Main entry point for the UAV formation simulation.

Run:
    python simulate.py

What this script does:
  1. Builds the map and plans a collision-free path with A*.
  2. Generates min-time and min-energy trajectories along that path.
  3. Animates both trajectories in formation side by side.
  4. Saves all required outputs to the results/ folder.
  5. Prints a summary of time, distance, and energy metrics.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive; saves files without needing a display
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# ── Local modules ────────────────────────────────────────────────────────────
from map_setup   import (MAP_WIDTH, MAP_HEIGHT, OBSTACLE_CENTER,
                         OBSTACLE_RADIUS, SAFETY_MARGIN, START, GOAL,
                         plot_map)
from path_planner import plan_path
from trajectory  import (generate_trajectories, resample_at_dt,
                          compute_energy, plot_trajectory_comparison, DT)
from formation   import (get_drone_positions, NUM_DRONES,
                          DRONE_COLORS, FORMATION_OFFSETS,
                          plot_formation_snapshot)

# ── Output folder ─────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def _results(filename):
    return os.path.join(RESULTS_DIR, filename)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1  —  Path planning
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print(" UAV Formation Simulation")
print("=" * 55)
print("\n[1/5] Running A* path planner ...")
waypoints = plan_path()

if not waypoints:
    print("ERROR: path planner returned no path. Exiting.")
    sys.exit(1)

path_arr = np.array(waypoints)
total_path_dist = float(np.sum(
    np.hypot(np.diff(path_arr[:, 0]), np.diff(path_arr[:, 1]))
))
print(f"      Waypoints found : {len(waypoints)}")
print(f"      Path length     : {total_path_dist:.2f} units")

# Save path plot
fig, ax = plt.subplots(figsize=(7, 7))
plot_map(path_waypoints=waypoints,
         title="A* Planned Path — UAV Formation Simulation", ax=ax)
fig.savefig(_results("path_plot.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("      Saved → results/path_plot.png")


# ─────────────────────────────────────────────────────────────────────────────
# Step 2  —  Trajectory generation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Generating trajectories ...")
traj_mt, traj_me = generate_trajectories(waypoints)

# Uniform-time resamples for animation
x_mt, y_mt, t_mt = resample_at_dt(traj_mt, DT)
x_me, y_me, t_me = resample_at_dt(traj_me, DT)

e_mt = compute_energy(traj_mt)
e_me = compute_energy(traj_me)

print(f"      Min-Time   : duration = {traj_mt['t'][-1]:.2f} s  |  "
      f"energy proxy = {e_mt:.1f}")
print(f"      Min-Energy : duration = {traj_me['t'][-1]:.2f} s  |  "
      f"energy proxy = {e_me:.1f}")

# Save trajectory comparison plot
plot_trajectory_comparison(traj_mt, traj_me,
                            save_path=_results("trajectory_comparison.png"))
plt.close("all")
print("      Saved → results/trajectory_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Step 3  —  Build per-drone position arrays
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Computing drone positions ...")
# Shape: (NUM_DRONES, N_frames, 2)
drone_pos_mt = get_drone_positions(x_mt, y_mt)
drone_pos_me = get_drone_positions(x_me, y_me)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4  —  Animation
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Building animation (this may take ~30 s) ...")

n_mt = len(t_mt)
n_me = len(t_me)
n_frames = max(n_mt, n_me)

fig_anim, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 6))
fig_anim.patch.set_facecolor("#1a1a2e")

def _style_ax(ax, title):
    ax.set_facecolor("#16213e")
    ax.set_xlim(0, MAP_WIDTH)
    ax.set_ylim(0, MAP_HEIGHT)
    ax.set_aspect("equal")
    ax.set_title(title, color="white", fontsize=11, pad=8)
    ax.tick_params(colors="grey")
    ax.spines[:].set_color("grey")
    # Obstacle
    margin_c = patches.Circle(OBSTACLE_CENTER,
                               OBSTACLE_RADIUS + SAFETY_MARGIN,
                               color="salmon", alpha=0.25)
    obs_c = patches.Circle(OBSTACLE_CENTER,
                            OBSTACLE_RADIUS,
                            color="firebrick", alpha=0.7)
    ax.add_patch(margin_c)
    ax.add_patch(obs_c)
    # Path trace (static)
    ax.plot(path_arr[:, 0], path_arr[:, 1],
            "--", color="white", alpha=0.25, linewidth=1)
    # Start / goal markers
    ax.scatter(*START, s=100, color="limegreen",  zorder=6)
    ax.scatter(*GOAL,  s=100, color="deepskyblue", zorder=6)

_style_ax(ax_l, "Min-Time Trajectory")
_style_ax(ax_r, "Min-Energy Trajectory")

# Drone scatter artists
scat_l = [ax_l.scatter([], [], s=90, color=DRONE_COLORS[i],
                        edgecolors="white", linewidths=0.5, zorder=7)
          for i in range(NUM_DRONES)]
scat_r = [ax_r.scatter([], [], s=90, color=DRONE_COLORS[i],
                        edgecolors="white", linewidths=0.5, zorder=7)
          for i in range(NUM_DRONES)]

# Centroid trail lines
trail_l, = ax_l.plot([], [], color="white", alpha=0.4, linewidth=1)
trail_r, = ax_r.plot([], [], color="white", alpha=0.4, linewidth=1)

# Time labels
time_l = ax_l.text(2, 96, "", color="white", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="#0f3460", alpha=0.8))
time_r = ax_r.text(2, 96, "", color="white", fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3",
                               facecolor="#0f3460", alpha=0.8))

# Legend patches
legend_handles = [
    patches.Patch(color=DRONE_COLORS[i], label=f"Drone {i}")
    for i in range(NUM_DRONES)
]
ax_l.legend(handles=legend_handles, loc="lower right",
            fontsize=7, framealpha=0.5, labelcolor="white",
            facecolor="#0f3460")
ax_r.legend(handles=legend_handles, loc="lower right",
            fontsize=7, framealpha=0.5, labelcolor="white",
            facecolor="#0f3460")

plt.tight_layout(pad=2.0)


def _update(frame):
    # Min-time side
    fi_mt = min(frame, n_mt - 1)
    for i in range(NUM_DRONES):
        scat_l[i].set_offsets(drone_pos_mt[i, fi_mt, :].reshape(1, 2))
    trail_l.set_data(x_mt[:fi_mt + 1], y_mt[:fi_mt + 1])
    time_l.set_text(f"t = {t_mt[fi_mt]:.1f} s")

    # Min-energy side
    fi_me = min(frame, n_me - 1)
    for i in range(NUM_DRONES):
        scat_r[i].set_offsets(drone_pos_me[i, fi_me, :].reshape(1, 2))
    trail_r.set_data(x_me[:fi_me + 1], y_me[:fi_me + 1])
    time_r.set_text(f"t = {t_me[fi_me]:.1f} s")

    return (*scat_l, *scat_r, trail_l, trail_r, time_l, time_r)


# Sub-sample frames so the GIF stays small and renders quickly
# (show every 4th frame → effective fps = 20/4 = 5 but looks smooth)
frame_step = 4
sampled_frames = list(range(0, n_frames, frame_step))

ani = animation.FuncAnimation(
    fig_anim,
    _update,
    frames=sampled_frames,
    interval=50,
    blit=True
)

gif_path = _results("formation_animation.gif")
writer = animation.PillowWriter(fps=15)
ani.save(gif_path, writer=writer)
plt.close(fig_anim)
print(f"      Saved → results/formation_animation.gif")


# ─────────────────────────────────────────────────────────────────────────────
# Step 5  —  Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Summary")
print("-" * 45)
print(f"  Path length              : {total_path_dist:.2f} units")
print(f"  Min-Time   total time    : {traj_mt['t'][-1]:.2f} s")
print(f"  Min-Energy total time    : {traj_me['t'][-1]:.2f} s")
print(f"  Min-Time   energy proxy  : {e_mt:.2f}")
print(f"  Min-Energy energy proxy  : {e_me:.2f}")
speedup  = traj_me["t"][-1] / traj_mt["t"][-1]
e_saving = (1 - e_me / e_mt) * 100
print(f"\n  Min-Energy is {speedup:.1f}× slower but uses"
      f" ~{e_saving:.0f}% less energy.")
print("\nAll results saved to results/")
print("=" * 55)
