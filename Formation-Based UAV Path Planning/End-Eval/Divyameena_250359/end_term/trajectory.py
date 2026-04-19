import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import os

from path_planner import plan_path

MIN_TIME_SPEED = 5.0
MIN_ENERGY_SPEED = 1.5
DT = 0.1


def smooth_trajectory(waypoints, speed):
    waypoints = np.array(waypoints, dtype=float)
    diffs = np.diff(waypoints, axis=0)
    chord = np.sqrt((diffs ** 2).sum(axis=1))
    s = np.concatenate([[0], np.cumsum(chord)])

    cs_x = CubicSpline(s, waypoints[:, 0])
    cs_y = CubicSpline(s, waypoints[:, 1])

    total_length = s[-1]
    total_time = total_length / speed
    t_samples = np.arange(0, total_time + DT, DT)
    s_samples = (t_samples / total_time) * total_length

    xs = cs_x(s_samples)
    ys = cs_y(s_samples)
    return list(zip(xs.tolist(), ys.tolist(), t_samples.tolist()))


def compute_speed_acceleration(traj):
    coords = np.array([(x, y) for x, y, t in traj])
    times  = np.array([t for x, y, t in traj])

    dx = np.diff(coords[:, 0])
    dy = np.diff(coords[:, 1])
    dt = np.diff(times)

    speed = np.sqrt(dx**2 + dy**2) / dt
    t_speed = (times[:-1] + times[1:]) / 2

    acc = np.diff(speed) / dt[:-1]
    t_acc = (t_speed[:-1] + t_speed[1:]) / 2

    return t_speed, speed, t_acc, acc


def generate_trajectories(waypoints=None):
    if waypoints is None:
        waypoints = plan_path()
    traj_fast   = smooth_trajectory(waypoints, speed=MIN_TIME_SPEED)
    traj_energy = smooth_trajectory(waypoints, speed=MIN_ENERGY_SPEED)
    print(f"Min-time   trajectory: {len(traj_fast)} points, total time ~ {traj_fast[-1][2]:.1f} s")
    print(f"Min-energy trajectory: {len(traj_energy)} points, total time ~ {traj_energy[-1][2]:.1f} s")
    return traj_fast, traj_energy


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    waypoints = plan_path()
    traj_fast, traj_energy = generate_trajectories(waypoints)

    t_sf, spd_f, t_af, acc_f = compute_speed_acceleration(traj_fast)
    t_se, spd_e, t_ae, acc_e = compute_speed_acceleration(traj_energy)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(t_sf, spd_f,   color="steelblue",  linewidth=2, label="Min-Time")
    axes[0].plot(t_se, spd_e,   color="darkorange",  linewidth=2, label="Min-Energy")
    axes[0].set_title("Speed vs Time")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (units/s)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_af, acc_f,   color="steelblue",  linewidth=2, label="Min-Time")
    axes[1].plot(t_ae, acc_e,   color="darkorange",  linewidth=2, label="Min-Energy")
    axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
    axes[1].set_title("Acceleration vs Time")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration (units/s²)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Trajectory Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/trajectory_comparison.png", dpi=150)
    plt.show()
    print("Saved -> results/trajectory_comparison.png")
