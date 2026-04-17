import numpy as np
from scipy.interpolate import CubicSpline

def arc_length_param(waypoints):
    pts = np.array(waypoints, dtype=float)
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    cumulative  = np.concatenate([[0], np.cumsum(seg_lengths)])
    return cumulative, pts

def build_spline(waypoints):
    s, pts = arc_length_param(waypoints)
    
    _, unique_idx = np.unique(s, return_index=True)
    s   = s[unique_idx]
    pts = pts[unique_idx]

    cs_x = CubicSpline(s, pts[:, 0])
    cs_y = CubicSpline(s, pts[:, 1])
    return cs_x, cs_y, s[-1]

def _speed_profile_min_time(s_vals, total_length, v_max=20.0):
    return np.full_like(s_vals, v_max)


def _speed_profile_min_energy(s_vals, total_length, v_cruise=6.0, accel_frac=0.2):
    v = np.full_like(s_vals, v_cruise)
    accel_dist = total_length * accel_frac

    ramp_mask = s_vals < accel_dist
    v[ramp_mask] = v_cruise * (s_vals[ramp_mask] / accel_dist)

    decel_mask = s_vals > (total_length - accel_dist)
    v[decel_mask] = v_cruise * ((total_length - s_vals[decel_mask]) / accel_dist)

    v = np.clip(v, 0.5, v_cruise)
    return v

def _build_trajectory(cs_x, cs_y, total_length, speed_fn, n_samples=1000):
    s_eval  = np.linspace(0, total_length, n_samples)
    speeds  = speed_fn(s_eval, total_length)

    
    ds  = np.gradient(s_eval)
    dt  = ds / np.maximum(speeds, 1e-6)
    t   = np.cumsum(dt)
    t  -= t[0]                 

    x = cs_x(s_eval)
    y = cs_y(s_eval)

    return np.column_stack([x, y, t])

def generate_trajectories(waypoints, n_samples=1000):
    cs_x, cs_y, total_length = build_spline(waypoints)

    traj_min_time   = _build_trajectory(cs_x, cs_y, total_length,
                                         _speed_profile_min_time,   n_samples)
    traj_min_energy = _build_trajectory(cs_x, cs_y, total_length,
                                         _speed_profile_min_energy, n_samples)
    return traj_min_time, traj_min_energy


def compute_metrics(traj, label=""):
    t   = traj[:, 2]
    x, y = traj[:, 0], traj[:, 1]
    dx  = np.gradient(x, t)
    dy  = np.gradient(y, t)
    speed = np.hypot(dx, dy)

    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)
    accel = np.hypot(ddx, ddy)

    total_time   = t[-1]
    total_dist   = np.sum(np.hypot(np.diff(x), np.diff(y)))
    energy_proxy = np.trapz(accel ** 2, t)   

    print(f"[{label}]  time={total_time:.2f}s  dist={total_dist:.2f}  "
          f"energy_proxy={energy_proxy:.4f}  peak_speed={speed.max():.2f}")
    return total_time, total_dist, energy_proxy


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from path_planner import plan_path

    waypoints = plan_path()
    mt, me    = generate_trajectories(waypoints)

    compute_metrics(mt, "min-time")
    compute_metrics(me, "min-energy")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    for traj, label, color in [(mt, "Min-Time", "#f0c040"), (me, "Min-Energy", "#52e08a")]:
        t = traj[:, 2]
        x, y = traj[:, 0], traj[:, 1]
        dx = np.gradient(x, t);  dy = np.gradient(y, t)
        speed = np.hypot(dx, dy)
        ddx = np.gradient(dx, t); ddy = np.gradient(dy, t)
        accel = np.hypot(ddx, ddy)
        axes[0].plot(t, speed, label=label, color=color)
        axes[1].plot(t, accel, label=label, color=color)

    axes[0].set_title("Speed vs Time",        color="white"); axes[0].set_xlabel("t (s)", color="white"); axes[0].set_ylabel("Speed (u/s)", color="white"); axes[0].legend(facecolor="#1a1f2e", labelcolor="white")
    axes[1].set_title("Acceleration vs Time", color="white"); axes[1].set_xlabel("t (s)", color="white"); axes[1].set_ylabel("Accel (u/s²)", color="white"); axes[1].legend(facecolor="#1a1f2e", labelcolor="white")

    plt.tight_layout()
    plt.show()
