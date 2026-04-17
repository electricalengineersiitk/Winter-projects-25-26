import numpy as np
from scipy.interpolate import CubicSpline

def generate_flight_paths(waypoints):
    pts = np.array(waypoints)
    # Filter identical points
    diff = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.insert(diff > 0.01, 0, True)]
    
    dist_total = np.cumsum(np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1)))
    dist_total = np.insert(dist_total, 0, 0)
    
    # Min-Time: higher speed trajectory
    t_time_max = dist_total / 18.0
    cs_time = CubicSpline(t_time_max, pts, bc_type='clamped')
    
    # Min-Energy: Smoother , slower trajectory
    t_energy_max = dist_total / 6.0
    cs_energy = CubicSpline(t_energy_max, pts, bc_type='clamped')
    
    return cs_time, t_time_max[-1], cs_energy, t_energy_max[-1]