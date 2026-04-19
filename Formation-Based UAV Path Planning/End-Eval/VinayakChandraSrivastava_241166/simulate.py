"""
simulate.py — Main entry point.

Runs the full UAV formation simulation:
  1. Plans a collision-free path (A*)
  2. Generates min-time and min-energy trajectories
  3. Animates both trajectories side by side
  4. Saves all plots and the animation to results/
  5. Prints a summary to the terminal
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — works on all machines
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches

# ── Project imports ──────────────────────────────────────────────
from map_setup    import (START, GOAL, OBSTACLE_CENTER, OBSTACLE_RADIUS,
                           MAP_WIDTH, MAP_HEIGHT, visualise_map)
from path_planner import plan_path
from trajectory   import (generate_min_time, generate_min_energy,
                           plot_trajectory_comparison)
from formation    import (get_drone_positions, NUM_DRONES, DRONE_COLOURS,
                           FORMATION_OFFSETS)

# ── Output directory ─────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _traj_to_array(traj):
    """Convert list of (x, y, t) to numpy arrays x, y, t."""
    arr = np.array(traj)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def run_simulation():
    print("=" * 60)
    print("  UAV Formation Path Planning Simulation")
    print("  Formation: Letter 'A'  |  Algorithm: A*")
    print(f"  Drones: {NUM_DRONES}")
    print("=" * 60)

    # ── Step 1 — Path planning ───────────────────────────────────
    print("\n[1/5] Running A* path planner …")
    waypoints = plan_path()

    # ── Step 2 — Trajectory generation ──────────────────────────
    print("[2/5] Generating trajectories …")
    traj_mt, meta_mt = generate_min_time(waypoints)
    traj_me, meta_me = generate_min_energy(waypoints)

    # ── Step 3 — Save static plots ───────────────────────────────
    print("[3/5] Saving static plots …")
    path_plot_file  = os.path.join(RESULTS_DIR, "path_plot.png")
    traj_plot_file  = os.path.join(RESULTS_DIR, "trajectory_comparison.png")

    visualise_map(path=waypoints, save_path=path_plot_file)
    plot_trajectory_comparison(meta_mt, meta_me, save_path=traj_plot_file)

    # ── Step 4 — Build & save animation ─────────────────────────
    print("[4/5] Rendering animation (this may take a moment) …")
    anim_file = os.path.join(RESULTS_DIR, "formation_animation.gif")
    _make_animation(traj_mt, meta_mt, traj_me, meta_me, anim_file)

    # ── Step 5 — Print summary ───────────────────────────────────
    print("\n[5/5] Summary")
    print("-" * 45)
    _print_summary(meta_mt, meta_me)
    print("\nAll outputs saved to:", RESULTS_DIR)
    print("=" * 60)


def _make_animation(traj_mt, meta_mt, traj_me, meta_me, save_path):
    """
    Side-by-side animation of min-time (left) and min-energy (right).
    Both animations are normalised to the same frame count for easy comparison.
    """
    x_mt, y_mt, t_mt = _traj_to_array(traj_mt)
    x_me, y_me, t_me = _traj_to_array(traj_me)

    # Down-sample to ~300 frames for a manageable GIF
    N_FRAMES = 300
    idx_mt = np.round(np.linspace(0, len(x_mt) - 1, N_FRAMES)).astype(int)
    idx_me = np.round(np.linspace(0, len(x_me) - 1, N_FRAMES)).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="#1a1a2e")
    fig.suptitle("UAV Formation Simulation — Letter 'A'  |  A* Path Planning",
                 fontsize=13, fontweight="bold", color="white", y=0.98)

    trail_len = 40   # number of trail points shown per drone

    drone_patches = []   # list of lists: drone_patches[ax_idx][drone_idx]
    trail_lines   = []

    for ax_idx, (ax, label, colour_title) in enumerate(
        zip(axes,
            [f"Min-Time  (total {meta_mt['total_time']:.1f} s)",
             f"Min-Energy  (total {meta_me['total_time']:.1f} s)"],
            ["#e74c3c", "#2980b9"])
    ):
        ax.set_facecolor("#0d0d1a")
        ax.set_xlim(0, MAP_WIDTH)
        ax.set_ylim(0, MAP_HEIGHT)
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11, color=colour_title, fontweight="bold")
        ax.set_xlabel("X", color="white", fontsize=9)
        ax.set_ylabel("Y", color="white", fontsize=9)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        # Obstacle
        obs = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS,
                         color="#e74c3c", alpha=0.7, zorder=3)
        ax.add_patch(obs)

        # Start / Goal markers
        ax.plot(*START, "go", markersize=10, zorder=4, label="Start")
        ax.plot(*GOAL,  "b*", markersize=12, zorder=4, label="Goal")

        # Planned path (faint)
        if ax_idx == 0:
            px = x_mt; py = y_mt
        else:
            px = x_me; py = y_me
        ax.plot(px, py, color="white", linewidth=0.8, alpha=0.15, zorder=2)

        ax.legend(fontsize=8, facecolor="#2c2c2c", labelcolor="white",
                  loc="upper right", markerscale=0.8)
        ax.grid(True, linestyle=":", alpha=0.15, color="white")

        # Drone markers
        plist = []
        tlist = []
        for d in range(NUM_DRONES):
            c = DRONE_COLOURS[d % len(DRONE_COLOURS)]
            patch, = ax.plot([], [], "o", color=c, markersize=6, zorder=6)
            trail, = ax.plot([], [], "-", color=c, linewidth=0.8,
                             alpha=0.5, zorder=5)
            plist.append(patch)
            tlist.append(trail)
        drone_patches.append(plist)
        trail_lines.append(tlist)

    def _init():
        for ax_idx in range(2):
            for d in range(NUM_DRONES):
                drone_patches[ax_idx][d].set_data([], [])
                trail_lines[ax_idx][d].set_data([], [])
        return [p for pl in drone_patches for p in pl] + \
               [t for tl in trail_lines  for t in tl]

    def _update(frame):
        for ax_idx, (idx_arr, xs, ys) in enumerate(
            [(idx_mt, x_mt, y_mt), (idx_me, x_me, y_me)]
        ):
            i = idx_arr[frame]
            centroid = (xs[i], ys[i])
            positions = get_drone_positions(centroid)  # (N, 2)

            lo = max(0, i - trail_len)
            trail_xs = xs[lo:i+1]
            trail_ys = ys[lo:i+1]

            for d in range(NUM_DRONES):
                dx, dy = FORMATION_OFFSETS[d]
                # Per-drone trail offset
                t_xs = trail_xs + dx
                t_ys = trail_ys + dy
                drone_patches[ax_idx][d].set_data([positions[d, 0]],
                                                   [positions[d, 1]])
                trail_lines[ax_idx][d].set_data(t_xs, t_ys)

        return [p for pl in drone_patches for p in pl] + \
               [t for tl in trail_lines  for t in tl]

    ani = animation.FuncAnimation(
        fig, _update, frames=N_FRAMES,
        init_func=_init, interval=40, blit=True
    )

    writer = animation.PillowWriter(fps=25)
    ani.save(save_path, writer=writer, dpi=90)
    plt.close(fig)
    print(f"[simulate] Saved animation → {save_path}")


def _print_summary(meta_mt, meta_me):
    headers = ["Metric", "Min-Time", "Min-Energy", "Difference"]
    dt  = meta_mt["total_time"]
    de  = meta_me["total_time"]
    dist_mt = meta_mt["total_dist"]
    dist_me = meta_me["total_dist"]
    en_mt = meta_mt["energy_proxy"]
    en_me = meta_me["energy_proxy"]

    rows = [
        ("Total time (s)",   f"{dt:.2f}",    f"{de:.2f}",
         f"MT is {de/dt:.1f}× faster"),
        ("Path length (u)",  f"{dist_mt:.1f}", f"{dist_me:.1f}", "—"),
        ("Energy proxy",     f"{en_mt:.2f}", f"{en_me:.2f}",
         f"ME uses {(en_mt-en_me)/en_mt*100:.1f}% less energy" if en_mt > en_me
         else f"MT uses {(en_me-en_mt)/en_me*100:.1f}% less energy"),
    ]
    col_w = [20, 14, 14, 30]
    fmt = "  ".join(f"{{:<{w}}}" for w in col_w)
    print(fmt.format(*headers))
    print("  " + "-" * (sum(col_w) + 2 * (len(col_w) - 1)))
    for row in rows:
        print(fmt.format(*row))


# ── Entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    run_simulation()
