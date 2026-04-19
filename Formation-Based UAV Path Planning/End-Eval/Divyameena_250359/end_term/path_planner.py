import heapq
import math
import numpy as np
import matplotlib.pyplot as plt
import os

from map_setup import build_grid, START, GOAL, GRID_SIZE

MOVES = [
    (1, 0), (-1, 0), (0, 1), (0, -1),
    (1, 1), (1, -1), (-1, 1), (-1, -1),
]


def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def get_neighbours(grid, node):
    x, y = node
    neighbours = []
    for dx, dy in MOVES:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and grid[nx, ny]:
            cost = math.hypot(dx, dy)
            neighbours.append(((nx, ny), cost))
    return neighbours


def astar(grid, start, goal):
    open_heap = []
    heapq.heappush(open_heap, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        for neighbour, move_cost in get_neighbours(grid, current):
            tentative_g = g_score[current] + move_cost
            if tentative_g < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g
                f = tentative_g + heuristic(neighbour, goal)
                heapq.heappush(open_heap, (f, neighbour))
    return []


def plan_path():
    grid = build_grid()
    path = astar(grid, START, GOAL)
    if not path:
        raise RuntimeError("A* could not find a path")
    print(f"Path found: {len(path)} waypoints")
    return path


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    grid = build_grid()
    path = astar(grid, START, GOAL)

    if not path:
        print("No path found!")
    else:
        print(f"Path found with {len(path)} waypoints.")
        img = np.ones((GRID_SIZE, GRID_SIZE, 3))
        img[~grid] = [0.2, 0.2, 0.2]
        xs, ys = zip(*path)
        plt.figure(figsize=(7, 7))
        plt.imshow(img.transpose(1, 0, 2), origin="lower", extent=[0, GRID_SIZE, 0, GRID_SIZE])
        plt.plot(xs, ys, "b-", linewidth=1.5, label="A* path")
        plt.plot(*START, "go", markersize=10, label="Start")
        plt.plot(*GOAL, "r*", markersize=12, label="Goal")
        plt.legend()
        plt.title("A* Path")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.savefig("results/path_plot.png", dpi=150)
        plt.show()
        print("Saved -> results/path_plot.png")
