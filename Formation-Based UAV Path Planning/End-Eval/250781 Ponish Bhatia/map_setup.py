import numpy as np
import matplotlib.pyplot as plt
import os

GRID_SIZE = 100
FORMATION_RADIUS = 3   

start = (5, 50)
goal = (95, 50)

obstacle_center = (50, 50)
obstacle_radius = 10

def is_obstacle(x, y):
    r = obstacle_radius + FORMATION_RADIUS
    return (x - obstacle_center[0])**2 + (y - obstacle_center[1])**2 <= r**2

from scipy.interpolate import splprep, splev

def plot_map(path=None, save_path="results/path_plot.png"): 

    path =path[::2]
    fig, ax = plt.subplots()

    circle = plt.Circle(obstacle_center, obstacle_radius, color='red')
    ax.add_patch(circle)

    ax.plot(*start, 'go', label='Start')
    ax.plot(*goal, 'bo', label='Goal')

    if path:
        path = np.array(path)

        if len(path) > 3:
            try:
                tck, _ = splprep([path[:,0], path[:,1]], s=3)

                u_new = np.linspace(0, 1, 300)
                x_smooth, y_smooth = splev(u_new, tck)

                ax.plot(x_smooth, y_smooth, 'k-', label='Smoothed Path')
            except: #if spline fails, was happening randomly fsr
                ax.plot(path[:,0], path[:,1], 'k-', label='Planned Path')
        else:
            ax.plot(path[:,0], path[:,1], 'k-', label='Planned Path')

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_title("Path Planning")
    ax.legend()

    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path)
    plt.close()