"""
trajectory.py
Converts a list of (x, y) waypoints into two smooth trajectories:
  • Minimum-time   – constant high speed; reaches goal as fast as possible.
  • Minimum-energy – raised-cosine velocity profile; gentle acceleration /
                     deceleration to minimise thrust effort.
Both use a CubicSpline fitted along the arc-length of the waypoints so the
spatial path is always the same; only the speed profile differs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ── Speed parameters ──────────────────────────────────────────────────────────
MIN_TIME_SPEED   = 14.0   # units / s   (constant cruise, max speed)
MIN_ENERGY_SPEED =  5.0   # units / s   (mean speed of raised-cosine profile)
SAMPLES_PER_UNIT = 3      # spatial resolution for dense sampling


# ── Internal helper: build arc-length parameterised spline ───────────────────
def _build_spline(waypoints):
    """
    Fit CubicSplines for x(s) and y(s) where s is the normalised arc length.
    Returns (cs_x, cs_y, total_length).
    """
    wp = np.asarray(waypoints, dtype=float)
    diffs      = np.diff(wp, axis=0)
    seg_lens   = np.linalg.norm(diffs, axis=1)
    arc        = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len  = arc[-1]
    s          = arc / total_len          # normalised ∈ [0, 1]

    # Handle duplicate s-values (shouldn't happen after simplification, but guard)
    _, unique_idx = np.unique(s, return_index=True)
    s  = s[unique_idx]
    wp = wp[unique_idx]

    cs_x = CubicSpline(s, wp[:, 0])
    cs_y = CubicSpline(s, wp[:, 1])
    return cs_x, cs_y, total_len


# ── Trajectory generators ─────────────────────────────────────────────────────
def generate_min_time(waypoints, speed=MIN_TIME_SPEED):
    """
    Minimum-time trajectory: constant cruise speed.
    Returns arrays (t, x, y, vx, vy, ax, ay).
    """
    cs_x, cs_y, L = _build_spline(waypoints)
    total_time = L / speed
    n          = max(300, int(L * SAMPLES_PER_UNIT))
    t          = np.linspace(0.0, total_time, n)
    s          = t / total_time                     # s ∈ [0,1]

    x  = cs_x(s)
    y  = cs_y(s)
    # Velocity (chain rule): dx/dt = (dx/ds)(ds/dt) = cs_x'(s)/total_time
    vx = cs_x(s, 1) / total_time
    vy = cs_y(s, 1) / total_time
    ax = cs_x(s, 2) / total_time ** 2
    ay = cs_y(s, 2) / total_time ** 2

    return t, x, y, vx, vy, ax, ay


def generate_min_energy(waypoints, mean_speed=MIN_ENERGY_SPEED):
    """
    Minimum-energy trajectory: raised-cosine speed profile.
    Speed ramps smoothly from 0 → peak → 0, keeping the same spatial path.
    Returns arrays (t, x, y, vx, vy, ax, ay).
    """
    cs_x, cs_y, L = _build_spline(waypoints)

    # Raised cosine: v(φ) = v_peak × (1 − cos(2πφ)) / 2,  φ ∈ [0,1]
    # Mean of this profile = v_peak / 2, so v_peak = 2 × mean_speed.
    v_peak = 2.0 * mean_speed
    total_time = L / mean_speed               # same relation as constant speed

    n   = max(300, int(L * SAMPLES_PER_UNIT))
    phi = np.linspace(0.0, 1.0, n)            # phase ∈ [0,1]
    t   = phi * total_time

    # Normalised arc length: integrate velocity profile
    # s(φ) = ∫₀^φ v(φ') dφ' / L  (normalised by total arc length)
    v_profile  = v_peak * (1.0 - np.cos(2.0 * np.pi * phi)) / 2.0
    s_raw      = np.cumsum(v_profile) * (total_time / n)  # ∫ v dt ≈ arc len
    s          = s_raw / s_raw[-1]            # normalise to [0, 1]
    s          = np.clip(s, 0.0, 1.0)

    x  = cs_x(s)
    y  = cs_y(s)

    # Velocity via ds/dt = v(φ(t)), scaled to spatial
    ds_dt = v_profile / L                     # d(s_norm)/dt
    vx    = cs_x(s, 1) * ds_dt
    vy    = cs_y(s, 1) * ds_dt

    # Acceleration (finite differences – cleaner than double chain rule)
    dt = t[1] - t[0]
    ax = np.gradient(vx, dt)
    ay = np.gradient(vy, dt)

    return t, x, y, vx, vy, ax, ay


# ── Energy metric ─────────────────────────────────────────────────────────────
def estimate_energy(ax, ay, t):
    """Energy ∝ ∫ |a|² dt  (proxy for thrust effort)."""
    a_sq = ax ** 2 + ay ** 2
    return np.trapezoid(a_sq, t) if hasattr(np, "trapezoid") else np.trapz(a_sq, t)


# ── Standalone test / plot ────────────────────────────────────────────────────
if __name__ == "__main__":
    from path_planner import dijkstra, simplify_path

    print("Planning path …")
    path = simplify_path(dijkstra())

    print("Generating trajectories …")
    mt = generate_min_time(path)
    me = generate_min_energy(path)

    t_mt, x_mt, y_mt, vx_mt, vy_mt, ax_mt, ay_mt = mt
    t_me, x_me, y_me, vx_me, vy_me, ax_me, ay_me = me

    speed_mt = np.sqrt(vx_mt**2 + vy_mt**2)
    speed_me = np.sqrt(vx_me**2 + vy_me**2)
    accel_mt = np.sqrt(ax_mt**2 + ay_mt**2)
    accel_me = np.sqrt(ax_me**2 + ay_me**2)

    e_mt = estimate_energy(ax_mt, ay_mt, t_mt)
    e_me = estimate_energy(ax_me, ay_me, t_me)

    print(f"  Min-time   total time : {t_mt[-1]:.2f} s  |  energy proxy : {e_mt:.4f}")
    print(f"  Min-energy total time : {t_me[-1]:.2f} s  |  energy proxy : {e_me:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Trajectory Comparison: Min-Time vs Min-Energy", fontsize=14, fontweight="bold")

    # Speed vs time
    axes[0].plot(t_mt, speed_mt, color="crimson",   linewidth=2, label="Min-Time")
    axes[0].plot(t_me, speed_me, color="steelblue", linewidth=2, label="Min-Energy")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Speed (units/s)")
    axes[0].set_title("Speed Profile")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # Acceleration magnitude vs time
    axes[1].plot(t_mt, accel_mt, color="crimson",   linewidth=2, label="Min-Time")
    axes[1].plot(t_me, accel_me, color="steelblue", linewidth=2, label="Min-Energy")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration magnitude (units/s²)")
    axes[1].set_title("Acceleration Profile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "results", "trajectory_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Trajectory plot saved → {out}")
    plt.show()
