import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import os

from path_planner import astar
from map_setup import plot_map
from trajectory import generate_trajectory, compute_metrics, generate_min_energy_trajectory
from formation import apply_formation

import numpy as np
from scipy.interpolate import splprep, splev


def smooth_drones(drones, s=2):
    smooth_drones = []

    for drone in drones:
        drone = np.array(drone)
        x, y, t = drone[:,0], drone[:,1], drone[:,2]

        if len(x) < 4:
            smooth_drones.append(drone)
            continue

        try:
            tck, _ = splprep([x, y], s=s)
            u_new = np.linspace(0, 1, len(x))

            x_s, y_s = splev(u_new, tck)

            smooth_path = [(x_s[i], y_s[i], t[i]) for i in range(len(x))]
            smooth_drones.append(smooth_path)

        except:
            smooth_drones.append(drone)

    return smooth_drones

os.makedirs("results", exist_ok=True)

path = astar()
plot_map(path)

x1, y1, t1 = generate_trajectory(path, mode="time")
x2, y2, t2 = generate_min_energy_trajectory(path)

s1, a1, d1, T1, e1 = compute_metrics(x1, y1, t1)
s2, a2, d2, T2, e2 = compute_metrics(x2, y2, t2)

fig, axs = plt.subplots(1,2, figsize=(10,4))

axs[0].plot(t1, s1, label="Min-Time")
axs[0].plot(t2, s2, label="Min-Energy")
axs[0].set_title("Speed vs Time")
axs[0].legend()

axs[1].plot(t1, a1, label="Min-Time")
axs[1].plot(t2, a2, label="Min-Energy")
axs[1].set_title("Acceleration vs Time")
axs[1].legend()

plt.savefig("results/trajectory_comparison.png")
plt.close()

drones = apply_formation(x1, y1, t1)
drones = smooth_drones(drones, s=5)   

fig, ax = plt.subplots()
ax.set_xlim(0,100)
ax.set_ylim(0,100)

from map_setup import obstacle_center, obstacle_radius

circle = plt.Circle(obstacle_center, obstacle_radius, color='red')
ax.add_patch(circle)
ax.set_title("UAV Formation for Minimum Time")

points = [ax.plot([], [], 'bo')[0] for _ in drones]

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

def update(frame):
    for i, drone in enumerate(drones):
        x, y, _ = drone[frame]
        points[i].set_data([x], [y])
    return points

ani = FuncAnimation(fig, update, frames=len(drones[0]), interval=40)

ani.save("results/formation_animation_minimum_time.gif", writer='pillow')

drones = apply_formation(x2, y2, t2)
drones = smooth_drones(drones, s=5)   

fig, ax = plt.subplots()
ax.set_xlim(0,100)
ax.set_ylim(0,100)

from map_setup import obstacle_center, obstacle_radius

circle = plt.Circle(obstacle_center, obstacle_radius, color='red')
ax.add_patch(circle)
ax.set_title("UAV Formation for Minimum Energy")

points = [ax.plot([], [], 'bo')[0] for _ in drones]

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys())

def update(frame):
    for i, drone in enumerate(drones):
        x, y, _ = drone[frame]
        points[i].set_data([x], [y])
    return points

ani = FuncAnimation(fig, update, frames=len(drones[0]), interval=40)

ani.save("results/formation_animation_minimum_energy.gif", writer='pillow')

print("Min-Time -> Time:", T1, "Distance:", d1, "Energy:", e1)
print("Min-Energy -> Time:", T2, "Distance:", d2, "Energy:", e2)