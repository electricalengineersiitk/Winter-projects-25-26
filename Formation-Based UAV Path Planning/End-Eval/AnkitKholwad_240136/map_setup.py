import numpy as np
import matplotlib.pyplot as plt

GRID_SIZE = 100

START = (5, 50)
GOAL = (95, 50)

OBSTACLE_CENTER = (50, 50)
OBSTACLE_RADIUS = 10
SAFETY_MARGIN = 2


def create_map():
    grid = np.zeros((GRID_SIZE, GRID_SIZE))

    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            distance = np.sqrt(
                (x - OBSTACLE_CENTER[0])**2 +
                (y - OBSTACLE_CENTER[1])**2
            )

            if distance <= (OBSTACLE_RADIUS + SAFETY_MARGIN):
                grid[x, y] = 1

    return grid


def visualize_map(grid):
    plt.imshow(grid.T, origin='lower', cmap='gray_r')
    plt.scatter(*START, label='Start')
    plt.scatter(*GOAL, label='Goal')
    plt.legend()
    plt.grid(True)
    plt.show()