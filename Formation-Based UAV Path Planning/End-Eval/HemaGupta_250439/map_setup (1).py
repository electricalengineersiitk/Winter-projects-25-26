import numpy as np
import matplotlib.pyplot as plt

MAP_SIZE        = 100
START           = (5, 50)
GOAL            = (95, 50)
OBSTACLE_CENTER = (50, 50)
OBSTACLE_RADIUS = 10
SAFETY_MARGIN   = 5

def is_in_obstacle(x, y, margin=0):
    cx, cy = OBSTACLE_CENTER
    r = OBSTACLE_RADIUS + margin
    return (x - cx)**2 + (y - cy)**2 <= r**2

def build_grid():
    grid = np.ones((MAP_SIZE, MAP_SIZE), dtype=bool)
    for row in range(MAP_SIZE):
        for col in range(MAP_SIZE):
            if is_in_obstacle(col, row, margin=SAFETY_MARGIN):
                grid[row, col] = False
    return grid

def visualise_map(path=None, save_to=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    safety = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS + SAFETY_MARGIN,
                        color='pink', alpha=0.35, label='Safety margin')
    obs    = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS,
                        color='salmon', alpha=0.8, label='Obstacle')
    ax.add_patch(safety)
    ax.add_patch(obs)
    ax.plot(*START, 'go', markersize=10, zorder=5, label='Start')
    ax.plot(*GOAL, marker='*', color='orange', markersize=14, zorder=5, label='Goal')
    if path is not None:
        xs, ys = zip(*path)
        ax.plot(xs, ys, 'b-o', linewidth=2, markersize=4, label='A* path', zorder=4)
    ax.set_xlim(0, MAP_SIZE); ax.set_ylim(0, MAP_SIZE)
    ax.set_aspect('equal')
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title('Collision-Free Path Around the Obstacle')
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.4)
    if save_to:
        plt.savefig(save_to, dpi=150, bbox_inches='tight')
    plt.show()
    return fig, ax
