"""
trajectory.py
-------------
Converts the raw waypoint list from the path planner into two smooth
time-stamped trajectories using cubic splines:

  • Minimum-time   — high constant speed, short flight duration
  • Minimum-energy — low speed with smooth, gentle acceleration profile

Output of each trajectory function:
    numpy array of shape (N, 3)  → columns: x, y, t
"""

import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import os

from path_planner import plan_path

# ── Velocity parameters ───────────────────────────────────────────────────────
V_MIN_TIME   = 15.0   # units per second  (fast)
V_MIN_ENERGY =  4.0   # units per second  (gentle)

# Number of interpolated points along the spline
N_POINTS = 500


# ── Internal helpers ──────────────────────────────────────────────────────────

def _arc_length_param(waypoints):
    """
    Assign a cumulative arc-length parameter to each waypoint.
    Returns array of shape (M,) — one scalar per waypoint.
    """
    pts = np.array(waypoints, dtype=float)
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
    s = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    return s, pts


def _build_spline(waypoints):
    """
    Fit a CubicSpline parameterised by arc length.
    Returns (cs_x, cs_y, total_length).
    """
    s, pts = _arc_length_param(waypoints)
    cs_x = CubicSpline(s, pts[:, 0])
    cs_y = CubicSpline(s, pts[:, 1])
    return cs_x, cs_y, s[-1]


def _sample_trajectory(cs_x, cs_y, total_length, speed, n=N_POINTS):
    """
    Sample (x, y, t) along the spline at a given constant speed.
    Returns numpy array of shape (n, 3).
    """
    s_vals = np.linspace(0, total_length, n)
    t_vals = s_vals / speed   # constant-speed time stamps

    x_vals = cs_x(s_vals)
    y_vals = cs_y(s_vals)

    return np.column_stack([x_vals, y_vals, t_vals])


def _compute_kinematics(traj):
    """
    Given a (N,3) trajectory array, compute speed and acceleration arrays.
    Returns (speed, accel) each of length N.
    """
    x, y, t = traj[:, 0], traj[:, 1], traj[:, 2]
    dt = np.diff(t)
    dx = np.diff(x)
    dy = np.diff(y)

    vx = dx / dt
    vy = dy / dt
    speed = np.hypot(vx, vy)

    dvx = np.diff(vx) / dt[:-1]
    dvy = np.diff(vy) / dt[:-1]
    accel = np.hypot(dvx, dvy)

    # Pad to length N so arrays align with traj rows
    speed = np.append(speed, speed[-1])
    accel = np.append(np.array([0.0]), accel)
    accel = np.append(accel, accel[-1])

    return speed, accel


# ── Public trajectory builders ────────────────────────────────────────────────

def generate_min_time(waypoints):
    """Return min-time trajectory as (N,3) array [x, y, t]."""
    cs_x, cs_y, total_length = _build_spline(waypoints)
    traj = _sample_trajectory(cs_x, cs_y, total_length, V_MIN_TIME)
    print(f"[trajectory] Min-time  : total time = {traj[-1, 2]:.2f} s  |  "
          f"distance = {total_length:.2f} units")
    return traj


def generate_min_energy(waypoints):
    """Return min-energy trajectory as (N,3) array [x, y, t]."""
    cs_x, cs_y, total_length = _build_spline(waypoints)
    traj = _sample_trajectory(cs_x, cs_y, total_length, V_MIN_ENERGY)
    print(f"[trajectory] Min-energy: total time = {traj[-1, 2]:.2f} s  |  "
          f"distance = {total_length:.2f} units")
    return traj


def estimate_energy(traj):
    """
    Simple energy proxy: sum of squared accelerations × dt.
    (Proportional to control effort.)
    """
    _, accel = _compute_kinematics(traj)
    dt = np.diff(traj[:, 2])
    energy = np.sum(accel[:-1] ** 2 * dt)
    return energy


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_trajectory_comparison(traj_mt, traj_me, save_path=None):
    """
    Side-by-side plots: speed vs time and acceleration vs time.
    """
    speed_mt, accel_mt = _compute_kinematics(traj_mt)
    speed_me, accel_me = _compute_kinematics(traj_me)

    t_mt = traj_mt[:, 2]
    t_me = traj_me[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Speed ──
    axes[0].plot(t_mt, speed_mt, color='#e74c3c', linewidth=2,
                 label=f'Min-time  (v={V_MIN_TIME} u/s)')
    axes[0].plot(t_me, speed_me, color='#2980b9', linewidth=2,
                 label=f'Min-energy (v={V_MIN_ENERGY} u/s)')
    axes[0].set_title('Speed vs Time')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Speed (units/s)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Acceleration ──
    axes[1].plot(t_mt, accel_mt, color='#e74c3c', linewidth=2,
                 label='Min-time')
    axes[1].plot(t_me, accel_me, color='#2980b9', linewidth=2,
                 label='Min-energy')
    axes[1].set_title('Acceleration vs Time')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Acceleration (units/s²)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle('Trajectory Comparison: Min-Time vs Min-Energy', fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[trajectory] Saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ── Stand-alone test ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    waypoints = plan_path()
    traj_mt = generate_min_time(waypoints)
    traj_me = generate_min_energy(waypoints)

    e_mt = estimate_energy(traj_mt)
    e_me = estimate_energy(traj_me)
    print(f"[trajectory] Energy proxy — min-time: {e_mt:.4f}  |  "
          f"min-energy: {e_me:.4f}")

    save = os.path.join('results', 'trajectory_comparison.png')
    plot_trajectory_comparison(traj_mt, traj_me, save_path=save)
