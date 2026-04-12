import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

GRID_SIZE = 100
START = (5, 50)
GOAL  = (95, 50)

OBSTACLE_CENTER = (50, 50)
OBSTACLE_RADIUS = 10
SAFETY_MARGIN   = 3

def is_blocked(x, y):
    cx, cy = OBSTACLE_CENTER
    return (x - cx)**2 + (y - cy)**2 <= (OBSTACLE_RADIUS + SAFETY_MARGIN)**2

def build_grid():
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if is_blocked(x, y):
                grid[x, y] = True
    return grid

def preview_map():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect('equal')
    ax.set_title("Map Setup")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    obstacle = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color='red', alpha=0.6, label='Obstacle')
    safety   = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS + SAFETY_MARGIN,
                          color='orange', alpha=0.2, label='Safety margin')
    ax.add_patch(obstacle)
    ax.add_patch(safety)

    ax.plot(*START, 'go', markersize=12, label='Start')
    ax.plot(*GOAL,  'b*', markersize=14, label='Goal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    preview_map()
