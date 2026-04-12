"""
simulate.py
Main entry point. Runs the full UAV formation simulation:
  1. Plan path with Dijkstra's algorithm.
  2. Generate min-time and min-energy trajectories.
  3. Animate both trajectories side-by-side with the 'R' formation.
  4. Save all required outputs to results/.

Run:
    python simulate.py
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")                 # headless / non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

# ── Local imports ─────────────────────────────────────────────────────────────
from map_setup   import (MAP_WIDTH, MAP_HEIGHT, START, GOAL,
                          OBSTACLE_CENTER, OBSTACLE_RADIUS, plot_map)
from path_planner import dijkstra, simplify_path
from trajectory   import (generate_min_time, generate_min_energy,
                           estimate_energy)
from formation    import N_DRONES, get_drone_positions, FORMATION_OFFSETS

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Animation sub-sampling: keep every k-th frame for manageable GIF size ─────
ANIM_SUBSAMPLE = 6    # take every 6th trajectory sample


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 – Path planning
# ══════════════════════════════════════════════════════════════════════════════
def step1_plan():
    print("═" * 60)
    print("STEP 1 │ Dijkstra path planning …")
    t0 = time.time()
    raw_path = dijkstra()
    path     = simplify_path(raw_path)
    elapsed  = time.time() - t0

    path_arr = np.array(path)
    dist = np.sum(np.linalg.norm(np.diff(path_arr, axis=0), axis=1))
    print(f"  Raw waypoints   : {len(raw_path)}")
    print(f"  After simplify  : {len(path)}")
    print(f"  Path length     : {dist:.2f} units")
    print(f"  Planning time   : {elapsed*1000:.1f} ms")
    return path, dist


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 – Trajectory generation
# ══════════════════════════════════════════════════════════════════════════════
def step2_trajectories(path):
    print("═" * 60)
    print("STEP 2 │ Generating trajectories …")

    mt = generate_min_time(path)
    me = generate_min_energy(path)

    t_mt, x_mt, y_mt, vx_mt, vy_mt, ax_mt, ay_mt = mt
    t_me, x_me, y_me, vx_me, vy_me, ax_me, ay_me = me

    e_mt = estimate_energy(ax_mt, ay_mt, t_mt)
    e_me = estimate_energy(ax_me, ay_me, t_me)

    print(f"  Min-time   │ duration {t_mt[-1]:.2f} s │ energy proxy {e_mt:.4f}")
    print(f"  Min-energy │ duration {t_me[-1]:.2f} s │ energy proxy {e_me:.4f}")
    print(f"  Time saving (min-time vs min-energy) : "
          f"{t_me[-1]-t_mt[-1]:.2f} s  "
          f"({(t_me[-1]-t_mt[-1])/t_me[-1]*100:.1f}% faster)")
    print(f"  Energy saving (min-energy vs min-time): "
          f"{e_mt-e_me:.4f}  "
          f"({(e_mt-e_me)/e_mt*100:.1f}% less)")
    return mt, me, e_mt, e_me


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 – path_plot.png
# ══════════════════════════════════════════════════════════════════════════════
def save_path_plot(path):
    print("═" * 60)
    print("STEP 3 │ Saving path plot …")

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_map(ax, path=path, title="Dijkstra's Collision-Free Path  (100×100 grid)")

    # Annotate waypoints
    p = np.array(path)
    ax.scatter(p[1:-1, 0], p[1:-1, 1],
               s=40, color="orange", zorder=6, label="Waypoints")
    ax.legend(loc="upper left")

    out = os.path.join(RESULTS_DIR, "path_plot.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 – trajectory_comparison.png
# ══════════════════════════════════════════════════════════════════════════════
def save_trajectory_comparison(mt, me, e_mt, e_me):
    print("═" * 60)
    print("STEP 4 │ Saving trajectory comparison …")

    t_mt, _, _, vx_mt, vy_mt, ax_mt, ay_mt = mt
    t_me, _, _, vx_me, vy_me, ax_me, ay_me = me

    speed_mt = np.sqrt(vx_mt**2 + vy_mt**2)
    speed_me = np.sqrt(vx_me**2 + vy_me**2)
    accel_mt = np.sqrt(ax_mt**2 + ay_mt**2)
    accel_me = np.sqrt(ax_me**2 + ay_me**2)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Trajectory Comparison: Min-Time vs Min-Energy",
        fontsize=14, fontweight="bold"
    )

    # Speed
    axes[0].plot(t_mt, speed_mt, color="crimson",   lw=2, label="Min-Time")
    axes[0].plot(t_me, speed_me, color="steelblue", lw=2, label="Min-Energy")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (units/s)")
    axes[0].set_title("Speed Profile")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)
    axes[0].annotate(
        f"Min-Time total: {t_mt[-1]:.1f} s\nMin-Energy total: {t_me[-1]:.1f} s",
        xy=(0.98, 0.96), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8)
    )

    # Acceleration
    axes[1].plot(t_mt, accel_mt, color="crimson",   lw=2, label="Min-Time")
    axes[1].plot(t_me, accel_me, color="steelblue", lw=2, label="Min-Energy")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration magnitude (units/s²)")
    axes[1].set_title("Acceleration Profile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)
    axes[1].annotate(
        f"Energy proxy\nMin-Time:   {e_mt:.3f}\nMin-Energy: {e_me:.3f}",
        xy=(0.98, 0.96), xycoords="axes fraction",
        ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", alpha=0.8)
    )

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "trajectory_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Helper – draw one animation frame
# ══════════════════════════════════════════════════════════════════════════════
def _draw_frame(ax, traj_xy, frame_idx, drone_artists, trail_line, centroid_dot,
                title_text, n_total):
    """Update matplotlib artists for a single animation frame."""
    cx, cy = traj_xy[frame_idx]
    drone_pos = get_drone_positions(np.array([cx, cy]))   # (N, 2)

    # Update drone scatter positions
    drone_artists.set_offsets(drone_pos)

    # Update trail
    trail_line.set_data(traj_xy[:frame_idx + 1, 0], traj_xy[:frame_idx + 1, 1])

    # Update centroid marker
    centroid_dot.set_data([cx], [cy])

    # Update title
    ax.set_title(
        f"{title_text}  │  frame {frame_idx+1}/{n_total}",
        fontsize=10, fontweight="bold"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Animation – formation_animation.gif
# ══════════════════════════════════════════════════════════════════════════════
def save_animation(mt, me):
    print("═" * 60)
    print("STEP 5 │ Building animation …")

    # Sub-sample trajectories for a leaner GIF
    _, x_mt, y_mt, *_ = mt
    _, x_me, y_me, *_ = me

    xy_mt = np.column_stack([x_mt, y_mt])[::ANIM_SUBSAMPLE]
    xy_me = np.column_stack([x_me, y_me])[::ANIM_SUBSAMPLE]

    # Both trajectories run in the animation side-by-side.
    # Pad the shorter one so they finish together.
    n_mt, n_me = len(xy_mt), len(xy_me)
    n_frames   = max(n_mt, n_me)
    xy_mt = np.vstack([xy_mt, np.tile(xy_mt[-1], (n_frames - n_mt, 1))])
    xy_me = np.vstack([xy_me, np.tile(xy_me[-1], (n_frames - n_me, 1))])

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#1a1a2e")

    def _setup_ax(ax, title_base):
        ax.set_facecolor("#16213e")
        ax.set_xlim(0, MAP_WIDTH)
        ax.set_ylim(0, MAP_HEIGHT)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")
        ax.set_xlabel("X (units)", color="white")
        ax.set_ylabel("Y (units)", color="white")

        # Obstacle
        obs = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS,
                          color="tomato", alpha=0.5)
        ax.add_patch(obs)

        # Start / Goal markers
        ax.plot(*START, "go", markersize=10, zorder=6)
        ax.plot(*GOAL,  "b*", markersize=12, zorder=6)
        ax.text(START[0]+1, START[1]+3, "Start", color="lightgreen", fontsize=8)
        ax.text(GOAL[0]-10, GOAL[1]+3,  "Goal",  color="deepskyblue",fontsize=8)

        # Artists to be updated each frame
        trail,   = ax.plot([], [], "-", color="yellow", lw=1.0, alpha=0.6)
        c_dot,   = ax.plot([], [], "o", color="white",  ms=5,   zorder=7)
        drones   = ax.scatter([], [], s=60, c="cyan", edgecolors="white",
                               linewidths=0.5, zorder=8)
        ax.set_title(title_base, color="white", fontsize=10, fontweight="bold")
        return trail, c_dot, drones

    tr_mt, cd_mt, dr_mt = _setup_ax(ax_l, "Min-Time Trajectory")
    tr_me, cd_me, dr_me = _setup_ax(ax_r, "Min-Energy Trajectory")

    fig.text(0.5, 0.97,
             f"Formation-Based UAV Simulation  –  Letter 'R'  ({N_DRONES} drones)",
             ha="center", va="top", color="white", fontsize=12, fontweight="bold")

    def _init():
        tr_mt.set_data([], [])
        tr_me.set_data([], [])
        cd_mt.set_data([], [])
        cd_me.set_data([], [])
        dr_mt.set_offsets(np.empty((0, 2)))
        dr_me.set_offsets(np.empty((0, 2)))
        return tr_mt, tr_me, cd_mt, cd_me, dr_mt, dr_me

    def _update(fi):
        _draw_frame(ax_l, xy_mt, fi, dr_mt, tr_mt, cd_mt,
                    "Min-Time",   n_frames)
        _draw_frame(ax_r, xy_me, fi, dr_me, tr_me, cd_me,
                    "Min-Energy", n_frames)
        return tr_mt, tr_me, cd_mt, cd_me, dr_mt, dr_me

    anim = FuncAnimation(
        fig, _update,
        frames=n_frames,
        init_func=_init,
        interval=50,
        blit=True
    )

    out = os.path.join(RESULTS_DIR, "formation_animation.gif")
    print("  Writing GIF (this may take a minute) …")
    writer = PillowWriter(fps=20)
    anim.save(out, writer=writer, dpi=90)
    plt.close(fig)
    print(f"  Saved → {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary printout
# ══════════════════════════════════════════════════════════════════════════════
def print_summary(path_dist, mt, me, e_mt, e_me):
    t_mt = mt[0][-1]
    t_me = me[0][-1]
    print("\n" + "═" * 60)
    print("SIMULATION SUMMARY")
    print("═" * 60)
    print(f"  Formation shape       : Letter 'R'")
    print(f"  Number of UAVs        : {N_DRONES}")
    print(f"  Planning algorithm    : Dijkstra's")
    print(f"  Path length           : {path_dist:.2f} units")
    print()
    print(f"  ┌──────────────┬──────────────┬──────────────┐")
    print(f"  │              │  Min-Time    │  Min-Energy  │")
    print(f"  ├──────────────┼──────────────┼──────────────┤")
    print(f"  │ Duration (s) │  {t_mt:>9.2f}   │  {t_me:>9.2f}   │")
    print(f"  │ Energy proxy │  {e_mt:>9.4f}   │  {e_me:>9.4f}   │")
    print(f"  └──────────────┴──────────────┴──────────────┘")
    print()
    print(f"  Min-time is {(t_me-t_mt)/t_me*100:.1f}% faster.")
    print(f"  Min-energy uses {(e_mt-e_me)/e_mt*100:.1f}% less energy.")
    print("═" * 60)
    print("All results saved to:", RESULTS_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    total_start = time.time()

    path, path_dist = step1_plan()
    mt, me, e_mt, e_me = step2_trajectories(path)

    save_path_plot(path)
    save_trajectory_comparison(mt, me, e_mt, e_me)
    save_animation(mt, me)

    print_summary(path_dist, mt, me, e_mt, e_me)
    print(f"\nTotal wall-clock time: {time.time()-total_start:.1f} s")
