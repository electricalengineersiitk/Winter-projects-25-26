import numpy as np
import os

GRID_SIZE = 100
START = (5, 50)
GOAL = (95, 50)
OBSTACLE_CENTER = (50, 50)
OBSTACLE_RADIUS = 10
SAFETY_MARGIN = 2


def build_grid():
    grid = np.ones((GRID_SIZE, GRID_SIZE), dtype=bool)
    cx, cy = OBSTACLE_CENTER
    r = OBSTACLE_RADIUS + SAFETY_MARGIN
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                grid[x, y] = False
    return grid


def is_free(grid, x, y):
    if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
        return grid[x, y]
    return False


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    os.makedirs("results", exist_ok=True)

    grid = build_grid()
    display = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    display[grid] = [1, 1, 1]
    display[~grid] = [0.2, 0.2, 0.2]

    sx, sy = START
    gx, gy = GOAL
    display[sx, sy] = [0, 0.8, 0]
    display[gx, gy] = [0.9, 0.1, 0.1]

    plt.figure(figsize=(6, 6))
    plt.imshow(display.transpose(1, 0, 2), origin="lower")
    plt.title("Map")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.tight_layout()
    plt.savefig("results/map_preview.png", dpi=120)
    plt.show()
    print("Grid built. Blocked cells:", int((~grid).sum()))

