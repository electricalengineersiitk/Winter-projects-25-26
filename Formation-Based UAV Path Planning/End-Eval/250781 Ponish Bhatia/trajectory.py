import numpy as np
from scipy.interpolate import CubicSpline
import numpy as np

def compute_velocity(x, y, t):
    x = np.array(x)
    y = np.array(y)
    t = np.array(t)

    n = len(x)

    vx = np.zeros(n)
    vy = np.zeros(n)

    for i in range(n - 1):
        dt = t[i+1] - t[i]
        vx[i] = (x[i+1] - x[i]) / dt
        vy[i] = (y[i+1] - y[i]) / dt

    vx[-1] = vx[-2]
    vy[-1] = vy[-2]

    speed = np.sqrt(vx**2 + vy**2)

    print(speed)

    return vx, vy, speed

import numpy as np
from scipy.interpolate import splprep, splev

def generate_min_energy_trajectory(path, total_time=10):

    path = np.array(path)

    # ---------- Smooth the geometric path ----------
    tck, _ = splprep([path[:,0], path[:,1]], s=3)

    u_fine = np.linspace(0, 1, 500)
    x_s, y_s = splev(u_fine, tck)

    x_s = np.array(x_s)
    y_s = np.array(y_s)

    # ---------- First derivatives ----------
    dx = np.gradient(x_s)
    dy = np.gradient(y_s)

    # ---------- Second derivatives ----------
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # ---------- Curvature ----------
    curvature = np.abs(dx * ddy - dy * ddx) / ((dx**2 + dy**2)**1.5 + 1e-6)

    # ---------- Radius of curvature ----------
    radius = 1 / (curvature + 1e-6)

    # ---------- Velocity profile ----------
    # v^2 / r ≈ constant
    velocity = np.sqrt(radius)

    # normalize velocity
    velocity = velocity / np.max(velocity)

    # avoid extremely small speeds
    velocity = 0.2 + 0.8 * velocity

    # ---------- Arc length ----------
    ds = np.sqrt(np.diff(x_s)**2 + np.diff(y_s)**2)

    # ---------- Time computation ----------
    dt = ds / velocity[:-1]

    t = np.insert(np.cumsum(dt), 0, 0)

    # normalize to desired total time
    t = total_time * t / t[-1]

    return x_s, y_s, t

def generate_trajectory(path, total_time=10, mode="time"):
    path = np.array(path)

    segment_lengths = np.sqrt(np.sum(np.diff(path, axis=0)**2, axis=1))
    total_length = np.sum(segment_lengths)

    time_stamps = [0]
    for length in segment_lengths:
        time_stamps.append(time_stamps[-1] + (length / total_length) * total_time)

    time_stamps = np.array(time_stamps)

    # # Resample trajectory (increase resolution)
    # t_new = np.linspace(0, total_time, 300)

    # x = np.interp(t_new, time_stamps, path[:,0])
    # y = np.interp(t_new, time_stamps, path[:,1])

    x=path[:,0]
    y=path[:,1]

    return x, y, time_stamps

def compute_metrics(x, y, t):
    vx, vy, speed=compute_velocity(x,y,t)
    ax,ay,accel=compute_velocity(vx,vy,t)

    distance = np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2))
    total_time = t[-1]

    energy = np.sum(accel**2)  # proxy

    return speed, accel, distance, total_time, energy