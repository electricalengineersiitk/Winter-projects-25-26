import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')          
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from map_setup    import (MAP_WIDTH, MAP_HEIGHT, OBSTACLE_CENTER, OBSTACLE_RADIUS, SAFETY_MARGIN, START, GOAL)
from path_planner import plan_path
from trajectory   import (generate_trajectories, plot_trajectory_comparison, compute_metrics)
from formation    import (FORMATION_OFFSETS, N_DRONES, FORMATION_NAME, get_drone_positions, expand_to_drones)

RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

print("  UAV Formation Path Planning — Simulation")
print(f"  Formation : Letter '{FORMATION_NAME}'   Drones : {N_DRONES}")

print("\n[1/5] Running A* path planner…")
waypoints = plan_path()

print("\n[2/5] Saving path plot…")

def save_path_plot(waypoints):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, MAP_WIDTH)
    ax.set_ylim(0, MAP_HEIGHT)
    ax.set_aspect('equal')
    ax.set_facecolor('#f5f6fa')
    fig.patch.set_facecolor('#ecf0f1')

    # Safety margin circle
    ax.add_patch(plt.Circle(OBSTACLE_CENTER,
                            OBSTACLE_RADIUS + SAFETY_MARGIN,
                            color='#f39c12', alpha=0.25,
                            label='Safety margin'))
    # Obstacle
    ax.add_patch(plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS,
                            color='#c0392b', alpha=0.85,
                            label='Obstacle'))

    # Straight-line reference
    ax.plot([START[0], GOAL[0]], [START[1], GOAL[1]],
            '--', color='#bdc3c7', linewidth=1.5,
            label='Direct (blocked)', zorder=2)

    # A* path
    px, py = zip(*waypoints)
    ax.plot(px, py, '-o', color='#8e44ad', linewidth=2.5,
            markersize=5, label='A* path', zorder=4)

    # Start / Goal
    ax.plot(*START, 's', color='#27ae60', markersize=14,
            zorder=6, label='Start')
    ax.plot(*GOAL,  '*', color='#2980b9', markersize=16,
            zorder=6, label='Goal')

    ax.set_title("UAV Formation — Planned Path (A*)\n"
                 f"Formation: Letter '{FORMATION_NAME}', {N_DRONES} drones",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('X (units)', fontsize=10)
    ax.set_ylabel('Y (units)', fontsize=10)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.annotate(f"Obstacle @ {OBSTACLE_CENTER}, r={OBSTACLE_RADIUS}",
                xy=OBSTACLE_CENTER,
                xytext=(OBSTACLE_CENTER[0] + 12, OBSTACLE_CENTER[1] + 12),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=8, color='gray')

    save = os.path.join(RESULTS_DIR, 'path_plot.png')
    fig.savefig(save, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f"   Saved → {save}")

save_path_plot(waypoints)


print("\n[3/5] Generating trajectories…")
traj_time, traj_energy = generate_trajectories(waypoints)

plot_trajectory_comparison(
    traj_time, traj_energy,
    save_path=os.path.join(RESULTS_DIR, 'trajectory_comparison.png')
)


print("\n[4/5] Expanding to drone formation trajectories…")
drone_trajs_time   = expand_to_drones(traj_time)
drone_trajs_energy = expand_to_drones(traj_energy)


print("\n[5/5] Building animation (this may take ~60 s)…")

# Sub-sample to keep GIF manageable (target = 200 frames per panel)
def subsample(traj_list, max_frames=200):
    T = len(traj_list[0]['t'])
    step = max(1, T // max_frames)
    out = []
    for d in traj_list:
        out.append({k: v[::step] if isinstance(v, np.ndarray) else v
                    for k, v in d.items()})
    return out

dtt = subsample(drone_trajs_time)
dte = subsample(drone_trajs_energy)

n_frames_t = len(dtt[0]['t'])
n_frames_e = len(dte[0]['t'])
n_frames   = max(n_frames_t, n_frames_e)

def pad(dlist, target):
    cur = len(dlist[0]['t'])
    if cur >= target:
        return dlist
    for d in dlist:
        for k, v in d.items():
            if isinstance(v, np.ndarray) and len(v) == cur:
                d[k] = np.concatenate([v, np.full(target - cur, v[-1])])
    return dlist

dtt = pad(dtt, n_frames)
dte = pad(dte, n_frames)

COLOURS = plt.cm.plasma(np.linspace(0.15, 0.9, N_DRONES))

fig_anim, axes_anim = plt.subplots(1, 2, figsize=(14, 7))
fig_anim.patch.set_facecolor('#0d0d1a')

def _setup_ax(ax, title):
    ax.set_xlim(-5, MAP_WIDTH + 5)
    ax.set_ylim(-5, MAP_HEIGHT + 5)
    ax.set_aspect('equal')
    ax.set_facecolor('#0d0d1a')
    ax.set_title(title, color='white', fontsize=11, fontweight='bold')
    ax.set_xlabel('X', color='#aaaacc')
    ax.set_ylabel('Y', color='#aaaacc')
    ax.tick_params(colors='#aaaacc')
    for sp in ax.spines.values():
        sp.set_edgecolor('#2a2a4a')

    # Obstacle
    ax.add_patch(plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS,
                            color='#c0392b', alpha=0.8, zorder=3))
    ax.add_patch(plt.Circle(OBSTACLE_CENTER,
                            OBSTACLE_RADIUS + SAFETY_MARGIN,
                            color='#f39c12', alpha=0.15, zorder=2))
    # Waypoint path
    px, py = zip(*waypoints)
    ax.plot(px, py, '-', color='#444466', linewidth=1.2,
            zorder=1, alpha=0.5)
    # Start / Goal markers
    ax.plot(*START, 's', color='#2ecc71', markersize=10, zorder=5)
    ax.plot(*GOAL,  '*', color='#3498db', markersize=13, zorder=5)

_setup_ax(axes_anim[0], f"Min-Time  (speed={18} u/s)")
_setup_ax(axes_anim[1], f"Min-Energy (speed={6} u/s)")

scatter_t = [axes_anim[0].scatter([], [], s=90, color=COLOURS[i],
                                   zorder=6, edgecolors='white',
                                   linewidths=0.6)
             for i in range(N_DRONES)]
scatter_e = [axes_anim[1].scatter([], [], s=90, color=COLOURS[i],
                                   zorder=6, edgecolors='white',
                                   linewidths=0.6)
             for i in range(N_DRONES)]

trail_len = 25
trails_t = [axes_anim[0].plot([], [], '-', color=COLOURS[i],
                               linewidth=0.8, alpha=0.45)[0]
            for i in range(N_DRONES)]
trails_e = [axes_anim[1].plot([], [], '-', color=COLOURS[i],
                               linewidth=0.8, alpha=0.45)[0]
            for i in range(N_DRONES)]

time_text_t = axes_anim[0].text(2, 97, '', color='#ddddff',
                                 fontsize=8, va='top')
time_text_e = axes_anim[1].text(2, 97, '', color='#ddddff',
                                 fontsize=8, va='top')

# Legend
handles = [mpatches.Patch(color=COLOURS[i], label=f'D{i}')
           for i in range(N_DRONES)]
axes_anim[1].legend(handles=handles, loc='upper right',
                    fontsize=6, ncol=3,
                    facecolor='#1a1a2e', edgecolor='#3a3a5a',
                    labelcolor='white')

plt.tight_layout(pad=2.0)


def _update(frame):
    idx = min(frame, n_frames - 1)

    # Min-time panel
    for i in range(N_DRONES):
        x = dtt[i]['x'][idx]
        y = dtt[i]['y'][idx]
        scatter_t[i].set_offsets([[x, y]])
        t0 = max(0, idx - trail_len)
        trails_t[i].set_data(dtt[i]['x'][t0:idx+1],
                             dtt[i]['y'][t0:idx+1])
    time_text_t.set_text(f't = {dtt[0]["t"][idx]:.1f} s')

    # Min-energy panel
    for i in range(N_DRONES):
        x = dte[i]['x'][idx]
        y = dte[i]['y'][idx]
        scatter_e[i].set_offsets([[x, y]])
        t0 = max(0, idx - trail_len)
        trails_e[i].set_data(dte[i]['x'][t0:idx+1],
                             dte[i]['y'][t0:idx+1])
    time_text_e.set_text(f't = {dte[0]["t"][idx]:.1f} s')

    return (scatter_t + scatter_e + trails_t + trails_e +
            [time_text_t, time_text_e])


ani = animation.FuncAnimation(
    fig_anim, _update,
    frames=n_frames,
    interval=50,
    blit=True
)

gif_path = os.path.join(RESULTS_DIR, 'formation_animation.gif')
ani.save(gif_path, writer='pillow', fps=20, dpi=80)
plt.close(fig_anim)
print(f"   Animation saved → {gif_path}")


metrics = compute_metrics(traj_time, traj_energy)
path_dist = np.sum(np.hypot(np.diff([w[0] for w in waypoints]),
                             np.diff([w[1] for w in waypoints])))

print("  SIMULATION SUMMARY")
print(f"  Formation shape   : Letter '{FORMATION_NAME}'  ({N_DRONES} drones)")
print(f"  Planning algo     : A*")
print(f"  Path distance     : {path_dist:.2f} units")
print(f"  Waypoints         : {len(waypoints)}")
print()
print(f"  Min-Time  trajectory")
print(f"    Total time      : {metrics['time_duration']:.2f} s")
print(f"    Max speed       : {metrics['time_max_speed']:.2f} units/s")
print(f"    Energy proxy    : {metrics['time_energy']:.2f}")
print()
print(f"  Min-Energy trajectory")
print(f"    Total time      : {metrics['energy_duration']:.2f} s")
print(f"    Max speed       : {metrics['energy_max_speed']:.2f} units/s")
print(f"    Energy proxy    : {metrics['energy_energy']:.2f}")
print()
speedup = metrics['energy_duration'] / metrics['time_duration']
esave   = (1 - metrics['energy_energy'] / metrics['time_energy']) * 100
print(f"  Min-Time is  {speedup:.1f}×  faster than Min-Energy")
print(f"  Min-Energy uses  {esave:.0f}%  less energy (proxy) than Min-Time")
print("\nAll results saved to the 'results/' folder.")