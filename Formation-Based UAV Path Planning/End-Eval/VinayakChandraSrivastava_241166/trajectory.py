"""
trajectory.py — Converts waypoints into smooth min-time and min-energy
trajectories using cubic splines (scipy.interpolate.CubicSpline).

Each trajectory is a list of (x, y, t) tuples at uniform time steps.
"""

import numpy as np
from scipy.interpolate import CubicSpline

# ── Trajectory parameters ────────────────────────────────────────
# Min-time: high cruise speed, small time step
MINTIME_SPEED  = 12.0   # units per second
MINTIME_DT     = 0.05   # seconds per sample

# Min-energy: low cruise speed, gentle velocity profile
MINENERGY_SPEED = 5.0   # units per second
MINENERGY_DT    = 0.05  # seconds per sample


def _arc_lengths(waypoints):
    """Cumulative arc-length parameter along the waypoint polyline."""
    pts = np.array(waypoints, dtype=float)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    return np.concatenate([[0.0], np.cumsum(seg_lens)])


def _build_spline(waypoints):
    """
    Build a CubicSpline parameterised by arc length.
    Returns (spline_x, spline_y, total_length).
    """
    s = _arc_lengths(waypoints)
    pts = np.array(waypoints, dtype=float)

    # Remove duplicate s values (shouldn't happen after smoothing, but just in case)
    _, unique_idx = np.unique(s, return_index=True)
    s   = s[unique_idx]
    pts = pts[unique_idx]

    cs_x = CubicSpline(s, pts[:, 0], bc_type="natural")
    cs_y = CubicSpline(s, pts[:, 1], bc_type="natural")
    return cs_x, cs_y, s[-1]


def generate_min_time(waypoints):
    """
    Minimum-time trajectory: constant high speed.

    Returns
    -------
    traj : list of (x, y, t)
    meta : dict with 'total_time', 'total_dist', 'energy_proxy'
    """
    cs_x, cs_y, total_len = _build_spline(waypoints)
    total_time = total_len / MINTIME_SPEED
    num_steps  = max(2, int(total_time / MINTIME_DT) + 1)
    t_arr      = np.linspace(0, total_time, num_steps)
    s_arr      = t_arr * MINTIME_SPEED          # uniform speed → linear s(t)
    s_arr      = np.clip(s_arr, 0, total_len)

    x_arr = cs_x(s_arr)
    y_arr = cs_y(s_arr)

    # Velocity & acceleration (for plots)
    vx = cs_x(s_arr, 1) * MINTIME_SPEED
    vy = cs_y(s_arr, 1) * MINTIME_SPEED
    speed = np.hypot(vx, vy)

    ax_ = np.gradient(vx, t_arr)
    ay_ = np.gradient(vy, t_arr)
    accel = np.hypot(ax_, ay_)

    # Energy proxy: sum of squared accelerations × dt
    energy = float(np.sum(accel ** 2) * MINTIME_DT)

    traj = list(zip(x_arr.tolist(), y_arr.tolist(), t_arr.tolist()))
    meta = {
        "label":       "Min-Time",
        "total_time":  total_time,
        "total_dist":  total_len,
        "energy_proxy": energy,
        "t":    t_arr,
        "speed": speed,
        "accel": accel,
    }
    return traj, meta


def generate_min_energy(waypoints):
    """
    Minimum-energy trajectory: lower speed with a trapezoidal (ramp-up /
    cruise / ramp-down) velocity profile for smooth, gentle motion.

    Returns
    -------
    traj : list of (x, y, t)
    meta : dict with 'total_time', 'total_dist', 'energy_proxy'
    """
    cs_x, cs_y, total_len = _build_spline(waypoints)

    # Build trapezoidal speed profile
    ramp_frac = 0.20   # 20 % of total distance to ramp up / down
    ramp_dist = ramp_frac * total_len
    cruise_spd = MINENERGY_SPEED

    # Sample the arc-length domain densely
    N = 2000
    s_dense = np.linspace(0, total_len, N)
    speed_profile = np.where(
        s_dense < ramp_dist,
        cruise_spd * (s_dense / ramp_dist),          # ramp up
        np.where(
            s_dense > total_len - ramp_dist,
            cruise_spd * ((total_len - s_dense) / ramp_dist),  # ramp down
            cruise_spd,                               # cruise
        ),
    )
    speed_profile = np.clip(speed_profile, 0.1, cruise_spd)  # avoid div/0

    # Convert s → t by integration: dt = ds / v
    ds = np.diff(s_dense)
    dt_arr = ds / speed_profile[:-1]
    t_dense = np.concatenate([[0.0], np.cumsum(dt_arr)])
    total_time = float(t_dense[-1])

    # Re-sample at uniform time steps
    t_uniform = np.arange(0, total_time, MINENERGY_DT)
    t_uniform = np.append(t_uniform, total_time)
    s_uniform = np.interp(t_uniform, t_dense, s_dense)

    x_arr = cs_x(s_uniform)
    y_arr = cs_y(s_uniform)

    # Velocity & acceleration
    vx = np.gradient(x_arr, t_uniform)
    vy = np.gradient(y_arr, t_uniform)
    speed = np.hypot(vx, vy)
    ax_ = np.gradient(vx, t_uniform)
    ay_ = np.gradient(vy, t_uniform)
    accel = np.hypot(ax_, ay_)

    energy = float(np.sum(accel ** 2) * MINENERGY_DT)

    traj = list(zip(x_arr.tolist(), y_arr.tolist(), t_uniform.tolist()))
    meta = {
        "label":        "Min-Energy",
        "total_time":   total_time,
        "total_dist":   total_len,
        "energy_proxy": energy,
        "t":     t_uniform,
        "speed": speed,
        "accel": accel,
    }
    return traj, meta


def plot_trajectory_comparison(meta_mt, meta_me, save_path=None):
    """
    Two-subplot comparison: speed vs time (left) and acceleration vs time (right).
    """
    import matplotlib.pyplot as plt
    import os

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Trajectory Comparison: Min-Time vs Min-Energy", fontsize=14, fontweight="bold")

    colours = {"Min-Time": "#e74c3c", "Min-Energy": "#2980b9"}

    for meta in (meta_mt, meta_me):
        lbl = meta["label"]
        c   = colours[lbl]
        axes[0].plot(meta["t"], meta["speed"], color=c, linewidth=2, label=lbl)
        axes[1].plot(meta["t"], meta["accel"], color=c, linewidth=2, label=lbl)

    axes[0].set_title("Speed vs Time")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (units/s)")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.5)

    axes[1].set_title("Acceleration vs Time")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration (units/s²)")
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[trajectory] Saved comparison plot → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Quick self-test ──────────────────────────────────────────────
if __name__ == "__main__":
    from path_planner import plan_path

    wp = plan_path()
    tmt, mmt = generate_min_time(wp)
    tme, mme = generate_min_energy(wp)

    print(f"\nMin-Time   : {mmt['total_time']:.2f} s | dist={mmt['total_dist']:.1f} | energy≈{mmt['energy_proxy']:.2f}")
    print(f"Min-Energy : {mme['total_time']:.2f} s | dist={mme['total_dist']:.1f} | energy≈{mme['energy_proxy']:.2f}")
    plot_trajectory_comparison(mmt, mme)
