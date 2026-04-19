%%writefile /content/end_term/map_setup.py
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
