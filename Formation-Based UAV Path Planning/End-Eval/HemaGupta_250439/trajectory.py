%%writefile /content/end_term/trajectory.py
import numpy as np
import sys
sys.path.insert(0, '/content/end_term')
from scipy.interpolate import CubicSpline

V_MAX_TIME   = 12.0
V_MAX_ENERGY =  4.0

def _arc_lengths(waypoints):
    pts   = np.array(waypoints, dtype=float)
    diffs = np.diff(pts, axis=0)
    seg   = np.hypot(diffs[:,0], diffs[:,1])
    return np.concatenate([[0.0], np.cumsum(seg)])

def _build_spline(waypoints):
    s    = _arc_lengths(waypoints)
    pts  = np.array(waypoints, dtype=float)
    cs_x = CubicSpline(s, pts[:,0])
    cs_y = CubicSpline(s, pts[:,1])
    return cs_x, cs_y, s[-1]

def generate_trajectory(waypoints, v_max, n_points=500):
    cs_x, cs_y, total_dist = _build_spline(waypoints)
    s_vals = np.linspace(0, total_dist, n_points)
    x_vals = cs_x(s_vals)
    y_vals = cs_y(s_vals)
    ds     = np.diff(s_vals)
    dt     = ds / v_max
    t_arr  = np.concatenate([[0.0], np.cumsum(dt)])
    speed  = np.hypot(cs_x(s_vals,1), cs_y(s_vals,1)) * v_max
    accel  = np.hypot(cs_x(s_vals,2), cs_y(s_vals,2)) * v_max**2
    try:
        energy = float(np.trapezoid(speed**2, t_arr))
    except AttributeError:
        energy = float(np.trapz(speed**2, t_arr))
    return {'x':x_vals,'y':y_vals,'t':t_arr,'speed':speed,
            'accel':accel,'total_time':t_arr[-1],
            'total_dist':total_dist,'energy_proxy':energy}

def make_trajectories(waypoints):
    return generate_trajectory(waypoints, V_MAX_TIME), \
           generate_trajectory(waypoints, V_MAX_ENERGY)
