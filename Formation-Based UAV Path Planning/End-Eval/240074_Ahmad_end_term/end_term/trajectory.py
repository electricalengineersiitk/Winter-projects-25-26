import numpy as np
from scipy.interpolate import CubicSpline
from path_planner import plan_path

def build_trajectory(waypoints, speed):
    xs = np.array([p[0] for p in waypoints], dtype=float)
    ys = np.array([p[1] for p in waypoints], dtype=float)

    dists = np.concatenate([[0], np.cumsum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2))])

    total_dist = dists[-1]
    n_points   = max(300, int(total_dist / speed * 60))

    cs_x = CubicSpline(dists, xs)
    cs_y = CubicSpline(dists, ys)

    s_vals = np.linspace(0, total_dist, n_points)
    t_vals = s_vals / speed

    x_smooth = cs_x(s_vals)
    y_smooth = cs_y(s_vals)

    return t_vals, x_smooth, y_smooth

def get_trajectories(waypoints=None):
    if waypoints is None:
        waypoints = plan_path()

    t_fast, x_fast, y_fast = build_trajectory(waypoints, speed=8.0)
    t_slow, x_slow, y_slow = build_trajectory(waypoints, speed=3.0)

    return (t_fast, x_fast, y_fast), (t_slow, x_slow, y_slow)

def compute_speed(t, x, y):
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    return np.sqrt(vx**2 + vy**2)

def compute_accel(t, x, y):
    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    return np.sqrt(ax**2 + ay**2)

def energy_estimate(t, x, y):
    accel = compute_accel(t, x, y)
    dt = np.diff(t)
    return float(np.sum(accel[:-1]**2 * dt))

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    wp = plan_path()
    fast, slow = get_trajectories(wp)

    t_f, x_f, y_f = fast
    t_s, x_s, y_s = slow

    spd_f = compute_speed(t_f, x_f, y_f)
    spd_s = compute_speed(t_s, x_s, y_s)
    acc_f = compute_accel(t_f, x_f, y_f)
    acc_s = compute_accel(t_s, x_s, y_s)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Trajectory Comparison")

    axes[0].plot(t_f, spd_f, label='Min-time', color='crimson')
    axes[0].plot(t_s, spd_s, label='Min-energy', color='steelblue')
    axes[0].set_title("Speed vs Time")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (units/s)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_f, acc_f, label='Min-time', color='crimson')
    axes[1].plot(t_s, acc_s, label='Min-energy', color='steelblue')
    axes[1].set_title("Acceleration vs Time")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration (units/s²)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/trajectory_comparison.png", dpi=120)
    plt.show()

    print(f"Min-time  — duration: {t_f[-1]:.1f}s  energy: {energy_estimate(*fast):.1f}")
    print(f"Min-energy — duration: {t_s[-1]:.1f}s  energy: {energy_estimate(*slow):.1f}")
