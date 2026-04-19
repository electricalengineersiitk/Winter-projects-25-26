import os
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from map_setup import build_grid, GRID_SIZE, START, GOAL
from path_planner import astar
from trajectory import generate_trajectories, compute_speed_acceleration
from formation import assign_drones, N, FORMATION

os.makedirs("results", exist_ok=True)

print("Building grid...")
grid = build_grid()

print("Running A*...")
waypoints = astar(grid, START, GOAL)
if not waypoints:
    raise RuntimeError("A* failed")
print(f"  {len(waypoints)} waypoints found.")

print("Generating trajectories...")
traj_fast, traj_energy = generate_trajectories(waypoints)

def background_ax(ax):
    img = np.ones((GRID_SIZE, GRID_SIZE, 3))
    img[~grid] = [0.25, 0.25, 0.25]
    ax.imshow(img.transpose(1, 0, 2), origin="lower", extent=[0, GRID_SIZE, 0, GRID_SIZE])

fig_p, ax_p = plt.subplots(figsize=(7, 7))
background_ax(ax_p)
wx, wy = zip(*waypoints)
ax_p.plot(wx, wy, "b-", linewidth=1.8, label="A* path")
ax_p.plot(*START, "go", markersize=10, label="Start")
ax_p.plot(*GOAL, "r*", markersize=13, label="Goal")
ax_p.set_title("A* Path")
ax_p.set_xlabel("X")
ax_p.set_ylabel("Y")
ax_p.legend()
fig_p.tight_layout()
fig_p.savefig("results/path_plot.png", dpi=150)
plt.close(fig_p)
print("Saved -> results/path_plot.png")

t_sf, spd_f, t_af, acc_f = compute_speed_acceleration(traj_fast)
t_se, spd_e, t_ae, acc_e = compute_speed_acceleration(traj_energy)

fig_t, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(t_sf, spd_f, color="steelblue", linewidth=2, label="Min-Time")
axes[0].plot(t_se, spd_e, color="darkorange", linewidth=2, label="Min-Energy")
axes[0].set_title("Speed vs Time")
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Speed (units/s)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t_af, acc_f, color="steelblue", linewidth=2, label="Min-Time")
axes[1].plot(t_ae, acc_e, color="darkorange", linewidth=2, label="Min-Energy")
axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
axes[1].set_title("Acceleration vs Time")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Acceleration (units/s²)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle("Trajectory Comparison", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/trajectory_comparison.png", dpi=150)
plt.close(fig_t)
print("Saved -> results/trajectory_comparison.png")

print("Preparing animation...")
STRIDE = 5
frames = traj_fast[::STRIDE]

def heading(traj, i):
    if i + 1 < len(traj):
        dx = traj[i+1][0] - traj[i][0]
        dy = traj[i+1][1] - traj[i][1]
        return math.degrees(math.atan2(dy, dx))
    return 0.0

headings = [heading(traj_fast, i * STRIDE) for i in range(len(frames))]
COLORS = plt.cm.tab10(np.linspace(0, 0.9, N))

fig_a, ax_a = plt.subplots(figsize=(7, 7))
background_ax(ax_a)
xf, yf, tf = zip(*traj_fast)
ax_a.plot(xf, yf, "--", color="white", linewidth=0.6, alpha=0.3)
ax_a.plot(*START, "go", markersize=8)
ax_a.plot(*GOAL, "r*", markersize=10)

drone_dots = [ax_a.plot([], [], "o", color=COLORS[k], markersize=8, label=f"D{k+1}")[0] for k in range(N)]
centroid_dot = ax_a.plot([], [], "w+", markersize=10, markeredgewidth=2)[0]
time_text = ax_a.text(2, 95, "", color="white", fontsize=9)

ax_a.set_xlim(0, GRID_SIZE)
ax_a.set_ylim(0, GRID_SIZE)
ax_a.set_title(f"UAV Formation — {FORMATION} (N={N})")
ax_a.set_xlabel("X")
ax_a.set_ylabel("Y")
ax_a.legend(loc="upper right", fontsize=7, framealpha=0.4)

def init():
    for d in drone_dots:
        d.set_data([], [])
    centroid_dot.set_data([], [])
    time_text.set_text("")
    return drone_dots + [centroid_dot, time_text]

def update(frame_idx):
    cx, cy, t = frames[frame_idx]
    hdg = headings[frame_idx]
    positions = assign_drones(cx, cy, heading_deg=hdg)
    for k, (px, py) in enumerate(positions):
        drone_dots[k].set_data([px], [py])
    centroid_dot.set_data([cx], [cy])
    time_text.set_text(f"t = {t:.1f} s")
    return drone_dots + [centroid_dot, time_text]

anim = FuncAnimation(fig_a, update, frames=len(frames), init_func=init, blit=True, interval=80)
anim.save("results/formation_animation.gif", writer=PillowWriter(fps=12))
plt.close(fig_a)
print("Saved -> results/formation_animation.gif")

def total_energy(traj):
    energy = 0.0
    for i in range(1, len(traj)):
        dx = traj[i][0] - traj[i-1][0]
        dy = traj[i][1] - traj[i-1][1]
        dt = traj[i][2] - traj[i-1][2]
        if dt > 0:
            v2 = (dx**2 + dy**2) / dt**2
            energy += v2 * dt
    return energy

t_fast = traj_fast[-1][2]
t_en = traj_energy[-1][2]
e_fast = total_energy(traj_fast)
e_energy = total_energy(traj_energy)

print("\n" + "="*45)
print("  TRAJECTORY COMPARISON")
print("="*45)
print(f"  Min-time   -> time: {t_fast:.1f} s  | energy: {e_fast:.1f}")
print(f"  Min-energy -> time: {t_en:.1f} s  | energy: {e_energy:.1f}")
print(f"  Time saved  : {t_en - t_fast:.1f} s ({(t_en - t_fast)/t_en*100:.1f}% faster)")
print(f"  Energy saved: {e_fast - e_energy:.1f}   ({(e_fast - e_energy)/e_fast*100:.1f}% less energy)")
print("="*45)
print("\nAll outputs saved in results/")
