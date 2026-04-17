import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter

from map_setup    import START, GOAL, OBSTACLE_CENTER, OBSTACLE_RADIUS, SAFETY_MARGIN, GRID_SIZE
from path_planner import plan_path
from trajectory   import generate_trajectories, compute_metrics
from formation    import get_all_positions_along_traj, FORMATION_OFFSETS, N_DRONES
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DARK_BG  = "#0d1117"
MID_BG   = "#1a1f2e"
COLOR_MT = "#f0c040"   
COLOR_ME = "#52e08a"   

print("Planning path with A* …")
waypoints = plan_path()
print(f"  → {len(waypoints)} waypoints found.")



print("Generating trajectories …")
traj_mt, traj_me = generate_trajectories(waypoints, n_samples=600)

print("\n── Trajectory metrics ──")
t_mt, d_mt, e_mt = compute_metrics(traj_mt, "Min-Time  ")
t_me, d_me, e_me = compute_metrics(traj_me, "Min-Energy")
pos_mt = get_all_positions_along_traj(traj_mt)   
pos_me = get_all_positions_along_traj(traj_me)   

print("\nSaving path_plot.png …")

fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)

obs_fill   = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color="#e05252", alpha=0.85, zorder=3)
obs_margin = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS + SAFETY_MARGIN,
                         color="#e05252", alpha=0.2, zorder=2)
ax.add_patch(obs_margin)
ax.add_patch(obs_fill)
ax.text(OBSTACLE_CENTER[0], OBSTACLE_CENTER[1], "OBS",
        ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=4)

wp = np.array(waypoints)
ax.plot(wp[:, 0], wp[:, 1], "o--", color=COLOR_MT, markersize=4,
        linewidth=1.5, zorder=5, label="A* waypoints")
ax.plot(*traj_mt[:, :2].T, color=COLOR_MT, linewidth=2.0, zorder=6, alpha=0.7, label="Spline (min-time)")
ax.plot(*traj_me[:, :2].T, color=COLOR_ME, linewidth=2.0, zorder=6, alpha=0.7, label="Spline (min-energy)")

ax.scatter(*START, color="#52e08a", s=140, zorder=7)
ax.scatter(*GOAL,  color="#52b8e0", s=140, zorder=7)
ax.annotate("START", START, xytext=(3, 3), textcoords="offset points", color="#52e08a", fontsize=9)
ax.annotate("GOAL",  GOAL,  xytext=(3, 3), textcoords="offset points", color="#52b8e0", fontsize=9)

ax.set_xlim(0, GRID_SIZE); ax.set_ylim(0, GRID_SIZE)
ax.set_aspect("equal")
ax.set_title("Planned Path (A*)", color="white", fontsize=13, pad=10)
ax.set_xlabel("X (units)", color="white"); ax.set_ylabel("Y (units)", color="white")
ax.tick_params(colors="white")
for sp in ax.spines.values(): sp.set_edgecolor("#333")
ax.legend(facecolor=MID_BG, labelcolor="white", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "path_plot.png"), dpi=150, facecolor=DARK_BG)
plt.close()
print("  → Saved.")

print("Saving trajectory_comparison.png …")

def _speed_accel(traj):
    t  = traj[:, 2]
    x, y = traj[:, 0], traj[:, 1]
    dx = np.gradient(x, t); dy = np.gradient(y, t)
    speed = np.hypot(dx, dy)
    ddx = np.gradient(dx, t); ddy = np.gradient(dy, t)
    accel = np.hypot(ddx, ddy)
    return t, speed, accel

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(DARK_BG)
for ax in axes:
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")

for traj, label, color in [(traj_mt, "Min-Time", COLOR_MT), (traj_me, "Min-Energy", COLOR_ME)]:
    t, speed, accel = _speed_accel(traj)
    axes[0].plot(t, speed, label=label, color=color, linewidth=2)
    axes[1].plot(t, accel, label=label, color=color, linewidth=2)

for ax, title, ylabel in [
    (axes[0], "Speed vs Time",        "Speed (units/s)"),
    (axes[1], "Acceleration vs Time", "Acceleration (units/s²)"),
]:
    ax.set_title(title, color="white", fontsize=12)
    ax.set_xlabel("Time (s)", color="white")
    ax.set_ylabel(ylabel, color="white")
    ax.legend(facecolor=MID_BG, labelcolor="white")

plt.suptitle("Trajectory Comparison — Min-Time vs Min-Energy", color="white", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "trajectory_comparison.png"), dpi=150,
            facecolor=DARK_BG, bbox_inches="tight")
plt.close()
print("  → Saved.")
print("Building animation (this may take ~30 s) …")

STEP   = 6
p_mt   = pos_mt[::STEP]   
p_me   = pos_me[::STEP]
n_frames = min(len(p_mt), len(p_me))
p_mt   = p_mt[:n_frames]
p_me   = p_me[:n_frames]
t_mt_s = traj_mt[::STEP, 2][:n_frames]
t_me_s = traj_me[::STEP, 2][:n_frames]

COLORS_D = plt.cm.plasma(np.linspace(0.2, 0.9, N_DRONES))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5))
fig.patch.set_facecolor(DARK_BG)

def _setup_ax(ax, title):
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, GRID_SIZE); ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect("equal")
    ax.set_title(title, color="white", fontsize=11, pad=6)
    ax.set_xlabel("X", color="white"); ax.set_ylabel("Y", color="white")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")
    
    obs = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color="#e05252", alpha=0.75, zorder=2)
    ax.add_patch(obs)
    
    ax.plot(traj_mt[:, 0], traj_mt[:, 1], color="white", alpha=0.08, linewidth=1, zorder=1)
    
    ax.scatter(*START, color="#52e08a", s=80, zorder=5)
    ax.scatter(*GOAL,  color="#52b8e0", s=80, zorder=5)

_setup_ax(ax1, "Min-Time Trajectory")
_setup_ax(ax2, "Min-Energy Trajectory")


scatters_mt = [ax1.scatter([], [], color=COLORS_D[i], s=90, zorder=6) for i in range(N_DRONES)]
scatters_me = [ax2.scatter([], [], color=COLORS_D[i], s=90, zorder=6) for i in range(N_DRONES)]

trails_mt   = [ax1.plot([], [], color=COLORS_D[i], alpha=0.35, linewidth=1)[0] for i in range(N_DRONES)]
trails_me   = [ax2.plot([], [], color=COLORS_D[i], alpha=0.35, linewidth=1)[0] for i in range(N_DRONES)]

time_text1 = ax1.text(2, 96, "", color="white", fontsize=9, va="top")
time_text2 = ax2.text(2, 96, "", color="white", fontsize=9, va="top")

TRAIL_LEN = 40

def init():
    for sc in scatters_mt + scatters_me:
        sc.set_offsets(np.empty((0, 2)))
    for ln in trails_mt + trails_me:
        ln.set_data([], [])
    time_text1.set_text(""); time_text2.set_text("")
    return scatters_mt + scatters_me + trails_mt + trails_me + [time_text1, time_text2]

def update(frame):
    mt_f = p_mt[frame]   
    me_f = p_me[frame]

    for i in range(N_DRONES):
        scatters_mt[i].set_offsets(mt_f[i:i+1])
        scatters_me[i].set_offsets(me_f[i:i+1])

        lo = max(0, frame - TRAIL_LEN)
        trail_mt = p_mt[lo:frame+1, i, :]
        trail_me = p_me[lo:frame+1, i, :]
        trails_mt[i].set_data(trail_mt[:, 0], trail_mt[:, 1])
        trails_me[i].set_data(trail_me[:, 0], trail_me[:, 1])

    time_text1.set_text(f"t = {t_mt_s[frame]:.1f} s")
    time_text2.set_text(f"t = {t_me_s[frame]:.1f} s")
    return scatters_mt + scatters_me + trails_mt + trails_me + [time_text1, time_text2]

anim = FuncAnimation(fig, update, frames=n_frames, init_func=init,
                     interval=40, blit=True)

writer = PillowWriter(fps=25)
anim_path = os.path.join(RESULTS_DIR, "formation_animation.gif")
anim.save(anim_path, writer=writer, dpi=90)
plt.close()
print(f"  → Saved: {anim_path}")

print("\n" + "=" * 55)
print("  SIMULATION SUMMARY")
print("=" * 55)
print(f"  Algorithm          : A*")
print(f"  Formation          : 'A' shape  ({N_DRONES} UAVs)")
print(f"  Waypoints          : {len(waypoints)}")
print(f"  --- Min-Time ---")
print(f"    Total time       : {t_mt:.2f} s")
print(f"    Total distance   : {d_mt:.2f} units")
print(f"    Energy proxy     : {e_mt:.4f}")
print(f"  --- Min-Energy ---")
print(f"    Total time       : {t_me:.2f} s")
print(f"    Total distance   : {d_me:.2f} units")
print(f"    Energy proxy     : {e_me:.4f}")
speedup = t_me / t_mt if t_mt > 0 else float("nan")
energy_save = (e_mt - e_me) / e_mt * 100 if e_mt > 0 else float("nan")
print(f"\n  Min-Time is {speedup:.1f}× faster than Min-Energy.")
print(f"  Min-Energy uses ~{energy_save:.1f}% less energy (∫a² dt proxy).")
print("=" * 55)
print("\nAll results saved to:", RESULTS_DIR)
