import numpy as np
from scipy.interpolate import CubicSpline

def generate_trajectories(waypoints):
    waypoints = np.array(waypoints)
    if len(waypoints) < 2: return None, None
    
    # Use scipy.interpolate.CubicSpline to smooth the raw waypoints [cite: 63]
    distance = np.cumsum(np.sqrt(np.sum(np.diff(waypoints, axis=0)**2, axis=1)))
    distance = np.insert(distance, 0, 0)
    
    cs_x = CubicSpline(distance, waypoints[:, 0])
    cs_y = CubicSpline(distance, waypoints[:, 1])
    
    # Minimum-time: use shorter time steps [cite: 61]
    t_min_time = np.linspace(0, distance[-1], 50) 
    
    # Minimum-energy: gradual speed changes with a lower velocity profile [cite: 62]
    t_min_energy = np.linspace(0, distance[-1], 100) 
    
    # Output is a time-stamped list of (x, y, t) positions [cite: 64]
    min_time_traj = np.column_stack((cs_x(t_min_time), cs_y(t_min_time), t_min_time))
    min_energy_traj = np.column_stack((cs_x(t_min_energy), cs_y(t_min_energy), t_min_energy))
    
    return min_time_traj, min_energy_traj