import heapq
import numpy as np
import matplotlib.pyplot as plt

from map_setup import MAP_W, MAP_H, START, GOAL, SAFETY_MARGIN, point_in_obstacle, plot_map

def build_obstacle_grid():
    grid = np.zeros((MAP_H, MAP_W), dtype=int)
    for y in range(MAP_H):
        for x in range(MAP_W):
            if point_in_obstacle(x, y, margin=SAFETY_MARGIN):
                grid[y][x] = 1
    return grid

def heuristic(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

def astar(start, goal, grid):
    open_list = []
    heapq.heappush(open_list, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = current[0] + dx, current[1] + dy
                if not (0 <= nx < MAP_W and 0 <= ny < MAP_H):
                    continue
                if grid[ny][nx] == 1:
                    continue
                move_cost = (dx**2 + dy**2) ** 0.5
                new_g = g_score[current] + move_cost
                neighbour = (nx, ny)
                if new_g < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = new_g
                    f = new_g + heuristic(neighbour, goal)
                    heapq.heappush(open_list, (f, neighbour))

    return []

def plan_path():
    grid = build_obstacle_grid()
    start = (int(START[0]), int(START[1]))
    goal  = (int(GOAL[0]),  int(GOAL[1]))
    path  = astar(start, goal, grid)
    print(f"Path found with {len(path)} steps")
    return path

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)

    path = plan_path()
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_map(ax, title="A* Planned Path")
    ax.plot(xs, ys, "orange", linewidth=1.5, label="A* path", zorder=5)
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/path_plot.png", dpi=100)
    plt.show()
    print("Path plot saved to results/path_plot.png")
