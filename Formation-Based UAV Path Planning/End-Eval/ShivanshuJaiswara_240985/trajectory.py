"""
trajectory.py
Converts a list of (x, y) waypoints into two smooth trajectories:

  1. Minimum-time   — drone travels at maximum constant speed.
  2. Minimum-energy — drone uses a smoother, lower speed profile
                      with gradual acceleration / deceleration.

Both trajectories are parameterised by arc-length and then
re-sampled at uniform time steps using cubic splines
(scipy.interpolate.CubicSpline).
"""

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


# ── Trajectory parameters ────────────────────────────────────────────────────
MAX_SPEED    = 8.0   # units / second  (min-time)
CRUISE_SPEED = 3.0   # units / second  (min-energy cruise)
DT           = 0.05  # simulation time-step in seconds


def _arc_lengths(waypoints):
    """
    Compute cumulative arc-length along the waypoint polyline.
    Returns a 1-D array of length N where arc[0] = 0.
    """
    pts = np.array(waypoints, dtype=float)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    return arc, pts


def _build_spline(waypoints):
    """
    Fit a CubicSpline parameterised by arc-length.
    Returns (spline_x, spline_y, total_length).
    """
    arc, pts = _arc_lengths(waypoints)

    # Deduplicate arc-length values so spline has strictly increasing x
    _, unique_idx = np.unique(arc, return_index=True)
    arc = arc[unique_idx]
    pts = pts[unique_idx]

    spline_x = CubicSpline(arc, pts[:, 0])
    spline_y = CubicSpline(arc, pts[:, 1])
    total_len = arc[-1]
    return spline_x, spline_y, total_len


def _velocity_profile_mintime(arc_samples, total_len, max_speed):
    """Constant max-speed profile → time = arc / speed."""
    times = arc_samples / max_speed
    speeds = np.full(len(arc_samples), max_speed)
    return times, speeds


def _velocity_profile_minenergy(arc_samples, total_len, cruise_speed):
    """
    Trapezoidal speed profile:
      - Accelerate from 0 to cruise_speed over the first 20 % of the path.
      - Cruise at cruise_speed for the middle 60 %.
      - Decelerate back to ~0 over the last 20 %.

    Returns time stamps and speed values for each arc sample.
    """
    accel_dist  = 0.20 * total_len
    decel_start = 0.80 * total_len
    n = len(arc_samples)
    speeds = np.zeros(n)

    for i, s in enumerate(arc_samples):
        if s <= accel_dist:
            speeds[i] = cruise_speed * (s / accel_dist)
        elif s <= decel_start:
            speeds[i] = cruise_speed
        else:
            remaining = total_len - s
            decel_len = total_len - decel_start
            speeds[i] = cruise_speed * (remaining / max(decel_len, 1e-6))

    # Avoid division by zero at the very start
    speeds = np.clip(speeds, 0.05, None)

    # Integrate ds/v to get time
    ds = np.diff(arc_samples)
    avg_v = 0.5 * (speeds[:-1] + speeds[1:])
    dt_seg = ds / avg_v
    times = np.concatenate([[0.0], np.cumsum(dt_seg)])
    return times, speeds


def generate_trajectories(waypoints, n_arc_samples=500):
    """
    Build both trajectories from the raw waypoint list.

    Parameters
    ----------
    waypoints      : list of (x, y) tuples
    n_arc_samples  : resolution along the path

    Returns
    -------
    traj_mintime   : dict with keys 'x', 'y', 't', 'speed', 'accel', 'label'
    traj_minenergy : dict (same keys)
    """
    spline_x, spline_y, total_len = _build_spline(waypoints)

    arc_samples = np.linspace(0, total_len, n_arc_samples)

    # ── Min-time ─────────────────────────────────────────────────────────────
    times_mt, speeds_mt = _velocity_profile_mintime(
        arc_samples, total_len, MAX_SPEED)

    # ── Min-energy ───────────────────────────────────────────────────────────
    times_me, speeds_me = _velocity_profile_minenergy(
        arc_samples, total_len, CRUISE_SPEED)

    def build_traj(arc_s, times, speeds, label):
        x = spline_x(arc_s)
        y = spline_y(arc_s)
        # Numerical acceleration
        accel = np.gradient(speeds, times)
        return dict(x=x, y=y, t=times, speed=speeds, accel=accel, label=label)

    traj_mt = build_traj(arc_samples, times_mt, speeds_mt, "Min-Time")
    traj_me = build_traj(arc_samples, times_me, speeds_me, "Min-Energy")

    return traj_mt, traj_me


def resample_at_dt(traj, dt=DT):
    """
    Re-sample a trajectory at uniform time intervals dt.
    Useful for frame-by-frame animation.

    Returns arrays: x, y, t (all 1-D, uniformly spaced in time).
    """
    t_uniform = np.arange(traj["t"][0], traj["t"][-1], dt)
    x_uniform = np.interp(t_uniform, traj["t"], traj["x"])
    y_uniform = np.interp(t_uniform, traj["t"], traj["y"])
    return x_uniform, y_uniform, t_uniform


def compute_energy(traj):
    """
    Simple energy proxy: integral of v² dt  (proportional to kinetic energy).
    """
    v = traj["speed"]
    t = traj["t"]
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return float(trapz(v ** 2, t))


def plot_trajectory_comparison(traj_mt, traj_me, save_path=None):
    """
    Two-panel figure: speed vs time and acceleration vs time for both
    trajectories side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Trajectory Comparison: Min-Time vs Min-Energy", fontsize=13)

    # Panel 1 — speed
    ax = axes[0]
    ax.plot(traj_mt["t"], traj_mt["speed"], color="crimson",
            linewidth=2, label=traj_mt["label"])
    ax.plot(traj_me["t"], traj_me["speed"], color="steelblue",
            linewidth=2, label=traj_me["label"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (units/s)")
    ax.set_title("Speed Profile")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    # Panel 2 — acceleration
    ax = axes[1]
    ax.plot(traj_mt["t"], traj_mt["accel"], color="crimson",
            linewidth=2, label=traj_mt["label"])
    ax.plot(traj_me["t"], traj_me["accel"], color="steelblue",
            linewidth=2, label=traj_me["label"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Acceleration (units/s²)")
    ax.set_title("Acceleration Profile")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.show()
