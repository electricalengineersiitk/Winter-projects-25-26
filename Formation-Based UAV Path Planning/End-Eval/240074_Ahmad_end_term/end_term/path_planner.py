import heapq
import numpy as np
from map_setup import START, GOAL, GRID_SIZE, build_grid

def heuristic(a, b):
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def get_neighbors(x, y, grid):
    directions = [
        (1,0),(-1,0),(0,1),(0,-1),
        (1,1),(1,-1),(-1,1),(-1,-1)
    ]
    neighbors = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and not grid[nx, ny]:
            cost = 1.414 if dx != 0 and dy != 0 else 1.0
            neighbors.append((nx, ny, cost))
    return neighbors

def reconstruct(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def astar(grid=None):
    if grid is None:
        grid = build_grid()

    start = START
    goal  = GOAL

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct(came_from, current)

        x, y = current
        for nx, ny, cost in get_neighbors(x, y, grid):
            neighbor = (nx, ny)
            tentative_g = g_score[current] + cost

            if tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor]   = tentative_g
                f_score[neighbor]   = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None

def smooth_path(raw_path, step=5):
    waypoints = [raw_path[0]]
    for i in range(step, len(raw_path) - 1, step):
        waypoints.append(raw_path[i])
    waypoints.append(raw_path[-1])
    return waypoints

def plan_path():
    grid = build_grid()
    raw  = astar(grid)
    if raw is None:
        raise RuntimeError("A* could not find a path — check obstacle settings.")
    return smooth_path(raw, step=6)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from map_setup import OBSTACLE_CENTER, OBSTACLE_RADIUS

    waypoints = plan_path()
    xs = [p[0] for p in waypoints]
    ys = [p[1] for p in waypoints]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect('equal')
    ax.set_title("A* Planned Path")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    obstacle = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color='red', alpha=0.6, label='Obstacle')
    ax.add_patch(obstacle)

    ax.plot(xs, ys, 'b-o', markersize=5, linewidth=2, label='A* path')
    ax.plot(*START, 'go', markersize=12, label='Start')
    ax.plot(*GOAL,  'b*', markersize=14, label='Goal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/path_plot.png", dpi=120)
    plt.show()
    print(f"Path has {len(waypoints)} waypoints.")
