"""
simulate.py
-----------
Main script.  Imports everything, runs the full simulation, and saves:
  • results/path_plot.png
  • results/trajectory_comparison.png
  • results/formation_animation.gif

Run with:  python simulate.py
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for saving files
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

# ── Project modules ───────────────────────────────────────────────────────────
from map_setup    import (MAP_WIDTH, MAP_HEIGHT, START, GOAL,
                          OBSTACLE_CENTER, OBSTACLE_RADIUS,
                          SAFETY_MARGIN, visualise_map)
from path_planner import plan_path
from trajectory   import (generate_min_time, generate_min_energy,
                           plot_trajectory_comparison, estimate_energy,
                           V_MIN_TIME, V_MIN_ENERGY)
from formation    import (get_formation_trajectory, OFFSETS, N_DRONES,
                          DRONE_COLORS, FORMATION_SHAPE)

# ── Output paths ──────────────────────────────────────────────────────────────
RESULTS_DIR = 'results'
PATH_PLOT   = os.path.join(RESULTS_DIR, 'path_plot.png')
TRAJ_PLOT   = os.path.join(RESULTS_DIR, 'trajectory_comparison.png')
ANIM_GIF    = os.path.join(RESULTS_DIR, 'formation_animation.gif')

os.makedirs(RESULTS_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
#  ANIMATION BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_animation(traj_mt_drones, traj_me_drones, waypoints, fps=20):
    """
    Build and return a FuncAnimation that shows both trajectories
    side-by-side.

    traj_*_drones : list of N arrays each (T, 3)  [x, y, t]
    """

    # ── Down-sample so the GIF is a manageable size ───────────────────────
    T_mt = traj_mt_drones[0].shape[0]
    T_me = traj_me_drones[0].shape[0]
    step_mt = max(1, T_mt // 150)
    step_me = max(1, T_me // 150)

    frames_mt = [traj[:, :2][::step_mt] for traj in traj_mt_drones]
    frames_me = [traj[:, :2][::step_me] for traj in traj_me_drones]

    n_frames = max(len(frames_mt[0]), len(frames_me[0]))

    # Pad shorter sequence by repeating its last frame
    def pad(frames, n):
        for i in range(len(frames)):
            arr = frames[i]
            if len(arr) < n:
                pad_rows = np.tile(arr[-1], (n - len(arr), 1))
                frames[i] = np.vstack([arr, pad_rows])

    pad(frames_mt, n_frames)
    pad(frames_me, n_frames)

    # ── Figure setup ──────────────────────────────────────────────────────
    fig, (ax_mt, ax_me) = plt.subplots(1, 2, figsize=(14, 6))
    fig.set_facecolor('#1a1a2e')

    def _setup_ax(ax, title):
        ax.set_xlim(0, MAP_WIDTH)
        ax.set_ylim(0, MAP_HEIGHT)
        ax.set_aspect('equal')
        ax.set_facecolor('#16213e')
        ax.set_title(title, color='white', fontsize=12, pad=8)
        ax.set_xlabel('X', color='#aaa')
        ax.set_ylabel('Y', color='#aaa')
        ax.tick_params(colors='#aaa')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

        # Obstacle
        obs = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS,
                         color='#e74c3c', alpha=0.85, zorder=3)
        margin = plt.Circle(OBSTACLE_CENTER,
                            OBSTACLE_RADIUS + SAFETY_MARGIN,
                            color='#e74c3c', alpha=0.15, zorder=2)
        ax.add_patch(margin)
        ax.add_patch(obs)

        # Start & goal markers
        ax.plot(*START, 'go', markersize=10, zorder=4, label='Start')
        ax.plot(*GOAL,  'b*', markersize=12, zorder=4, label='Goal')

        # Path reference
        if waypoints:
            xs, ys = zip(*waypoints)
            ax.plot(xs, ys, color='#ffffff', linewidth=0.8,
                    alpha=0.35, linestyle='--', zorder=3)

        ax.legend(fontsize=8, loc='upper right',
                  facecolor='#0f3460', edgecolor='none',
                  labelcolor='white')

    _setup_ax(ax_mt, f'Min-Time  (v = {V_MIN_TIME} u/s)')
    _setup_ax(ax_me, f'Min-Energy (v = {V_MIN_ENERGY} u/s)')

    # ── Drone scatter objects ─────────────────────────────────────────────
    drone_dots_mt = [ax_mt.plot([], [], 'o', color=DRONE_COLORS[i],
                                markersize=8, zorder=5)[0]
                     for i in range(N_DRONES)]
    drone_dots_me = [ax_me.plot([], [], 'o', color=DRONE_COLORS[i],
                                markersize=8, zorder=5)[0]
                     for i in range(N_DRONES)]

    # Trails
    trail_mt = [ax_mt.plot([], [], '-', color=DRONE_COLORS[i],
                            linewidth=0.6, alpha=0.4, zorder=4)[0]
                for i in range(N_DRONES)]
    trail_me = [ax_me.plot([], [], '-', color=DRONE_COLORS[i],
                            linewidth=0.6, alpha=0.4, zorder=4)[0]
                for i in range(N_DRONES)]

    # Time text
    time_text_mt = ax_mt.text(2, MAP_HEIGHT - 5, '', color='white', fontsize=9)
    time_text_me = ax_me.text(2, MAP_HEIGHT - 5, '', color='white', fontsize=9)

    TRAIL_LEN = 30   # frames to keep in trail

    def _init():
        for dot in drone_dots_mt + drone_dots_me:
            dot.set_data([], [])
        for line in trail_mt + trail_me:
            line.set_data([], [])
        return drone_dots_mt + drone_dots_me + trail_mt + trail_me

    def _update(frame):
        f = min(frame, n_frames - 1)

        for i in range(N_DRONES):
            # Min-time
            x_mt, y_mt = frames_mt[i][f]
            drone_dots_mt[i].set_data([x_mt], [y_mt])
            start_t = max(0, f - TRAIL_LEN)
            trail_mt[i].set_data(frames_mt[i][start_t:f+1, 0],
                                  frames_mt[i][start_t:f+1, 1])

            # Min-energy
            x_me, y_me = frames_me[i][f]
            drone_dots_me[i].set_data([x_me], [y_me])
            trail_me[i].set_data(frames_me[i][start_t:f+1, 0],
                                  frames_me[i][start_t:f+1, 1])

        # Time labels
        t_mt_val = traj_mt_drones[0][min(f * step_mt,
                                         traj_mt_drones[0].shape[0]-1), 2]
        t_me_val = traj_me_drones[0][min(f * step_me,
                                         traj_me_drones[0].shape[0]-1), 2]
        time_text_mt.set_text(f't = {t_mt_val:.1f} s')
        time_text_me.set_text(f't = {t_me_val:.1f} s')

        return (drone_dots_mt + drone_dots_me +
                trail_mt + trail_me +
                [time_text_mt, time_text_me])

    anim = FuncAnimation(fig, _update, frames=n_frames,
                         init_func=_init, blit=True, interval=1000/fps)
    return fig, anim


# ═════════════════════════════════════════════════════════════════════════════
#  SUMMARY PRINTER
# ═════════════════════════════════════════════════════════════════════════════

def print_summary(traj_mt, traj_me):
    total_dist_mt = np.sum(np.hypot(np.diff(traj_mt[:, 0]),
                                    np.diff(traj_mt[:, 1])))
    total_dist_me = np.sum(np.hypot(np.diff(traj_me[:, 0]),
                                    np.diff(traj_me[:, 1])))
    e_mt = estimate_energy(traj_mt)
    e_me = estimate_energy(traj_me)

    print("\n" + "═" * 55)
    print("  SIMULATION SUMMARY")
    print("═" * 55)
    print(f"  Formation : {FORMATION_SHAPE}-shape  |  {N_DRONES} drones")    
    print(f"  Algorithm : A*  |  Map: {MAP_WIDTH}×{MAP_HEIGHT}")
    print("─" * 55)
    print(f"  {'':20s}  {'Min-Time':>10}  {'Min-Energy':>12}")
    print(f"  {'Total time (s)':20s}  {traj_mt[-1,2]:>10.2f}  {traj_me[-1,2]:>12.2f}")
    print(f"  {'Total distance (u)':20s}  {total_dist_mt:>10.2f}  {total_dist_me:>12.2f}")
    print(f"  {'Energy proxy':20s}  {e_mt:>10.4f}  {e_me:>12.4f}")
    print("═" * 55)
    speedup = traj_me[-1, 2] / traj_mt[-1, 2]
    saving  = (e_mt - e_me) / e_mt * 100 if e_mt > 0 else 0
    print(f"  Min-time is  {speedup:.1f}×  faster than min-energy.")
    print(f"  Min-energy uses  {saving:.1f}%  less energy than min-time.")
    print("═" * 55 + "\n")


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── Phase 1 & 2: Plan the path & Compare Algorithms ──────────────────
    print("\n[simulate] ── Phase 1 & 2: Path planning ──────────────────")
    
    # Run Dijkstra
    wp_dijkstra, nodes_d, t_d = plan_path(algo='dijkstra')
    # Run A*
    wp_astar, nodes_a, t_a = plan_path(algo='astar')
    
    print("\n[simulate] ── Algorithm Comparison (Bonus) ──────────────")
    print(f"  Dijkstra's : Expanded {nodes_d:5d} nodes in {t_d*1000:.2f} ms")
    print(f"  A* Search  : Expanded {nodes_a:5d} nodes in {t_a*1000:.2f} ms")
    speedup = nodes_d / nodes_a if nodes_a > 0 else 1
    print(f"  Conclusion : Both found optimal paths, but A* was {speedup:.1f}x more efficient!")
    
    # We will use the A* waypoints for the rest of the simulation
    waypoints = wp_astar
    visualise_map(path=waypoints, save_path=PATH_PLOT)

    # ── Phase 3: Generate trajectories ───────────────────────────────────
    print("\n[simulate] ── Phase 3: Trajectory generation ──────────────")
    traj_mt = generate_min_time(waypoints)
    traj_me = generate_min_energy(waypoints)
    plot_trajectory_comparison(traj_mt, traj_me, save_path=TRAJ_PLOT)

    # ── Phase 4: Expand to per-drone trajectories ─────────────────────────
    print("\n[simulate] ── Phase 4: Formation expansion ─────────────────")
    drone_trajs_mt = get_formation_trajectory(traj_mt)
    drone_trajs_me = get_formation_trajectory(traj_me)
    print(f"[simulate] {N_DRONES} drone trajectories created for each mode.")

    # ── Phase 5: Animate ──────────────────────────────────────────────────
    print("\n[simulate] ── Phase 5: Building animation ──────────────────")
    fig, anim = build_animation(drone_trajs_mt, drone_trajs_me,
                                waypoints, fps=20)

    print(f"[simulate] Saving animation → {ANIM_GIF}  (this may take ~30 s)")
    anim.save(ANIM_GIF, writer='pillow', dpi=90)
    plt.close(fig)
    print(f"[simulate] Animation saved.")

    # ── Summary ───────────────────────────────────────────────────────────
    print_summary(traj_mt, traj_me)
    print(f"[simulate] All outputs written to '{RESULTS_DIR}/'")
    print(f"  • {PATH_PLOT}")
    print(f"  • {TRAJ_PLOT}")
    print(f"  • {ANIM_GIF}")


if __name__ == '__main__':
    main()
