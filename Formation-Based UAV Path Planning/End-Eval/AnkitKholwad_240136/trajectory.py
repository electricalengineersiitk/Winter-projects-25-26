import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

def generate_trajectories(path):
    path = np.array(path)
    t = np.linspace(0, 1, len(path))

    cs_x = CubicSpline(t, path[:, 0])
    cs_y = CubicSpline(t, path[:, 1])

    # Min-time: 50 points over 5 seconds (dt=0.1)
    t_fast = np.linspace(0, 1, 50)
    x_fast = cs_x(t_fast)
    y_fast = cs_y(t_fast)
    time_fast = t_fast * 5
    traj_fast = list(zip(x_fast, y_fast, time_fast))

    # Min-energy: 100 points over 10 seconds (dt=0.1)
    t_slow = np.linspace(0, 1, 100)
    t_smooth = 3*t_slow**2 - 2*t_slow**3
    x_slow = cs_x(t_smooth)
    y_slow = cs_y(t_smooth)
    time_slow = t_slow * 10
    traj_slow = list(zip(x_slow, y_slow, time_slow))

    return traj_fast, traj_slow


def compute_velocity_acceleration(traj):
    traj = np.array(traj)
    x, y, t = traj[:,0], traj[:,1], traj[:,2]

    vx = np.gradient(x, t)
    vy = np.gradient(y, t)
    speed = np.sqrt(vx**2 + vy**2)

    ax = np.gradient(vx, t)
    ay = np.gradient(vy, t)
    accel = np.sqrt(ax**2 + ay**2)

    return speed, accel, t