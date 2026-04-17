import heapq
import numpy as np
from map_setup import build_grid, START, GOAL, GRID_SIZE, SAFETY_MARGIN

def heuristic(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def astar(grid, start, goal):
    start = (int(start[0]), int(start[1]))
    goal  = (int(goal[0]),  int(goal[1]))
    H, W = grid.shape
    open_heap = []          
    heapq.heappush(open_heap, (0.0, 0.0, start))
    came_from = {}
    g_score   = {start: 0.0}
    neighbours = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for dx, dy in neighbours:
            nx, ny = current[0] + dx, current[1] + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if not grid[ny, nx]:          
                continue

            move_cost = np.hypot(dx, dy)
            tentative_g = g + move_cost

            neighbour = (nx, ny)
            if tentative_g < g_score.get(neighbour, float("inf")):
                came_from[neighbour] = current
                g_score[neighbour]   = tentative_g
                f = tentative_g + heuristic(neighbour, goal)
                heapq.heappush(open_heap, (f, tentative_g, neighbour))

    return []   


def smooth_path(path, window=5):
    if len(path) < window:
        return path
    path   = np.array(path, dtype=float)
    smooth = path.copy()
    for i in range(1, len(path) - 1):
        lo = max(0, i - window // 2)
        hi = min(len(path), i + window // 2 + 1)
        smooth[i] = path[lo:hi].mean(axis=0)
    smooth[0]  = path[0]
    smooth[-1] = path[-1]
    return [tuple(p) for p in smooth]


def plan_path():
    grid = build_grid(margin=SAFETY_MARGIN)
    raw  = astar(grid, START, GOAL)
    if not raw:
        raise RuntimeError("A* found no path — check obstacle / grid settings.")
    
    thinned = raw[::5] + [raw[-1]]
    return smooth_path(thinned, window=5)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from map_setup import visualise_map

    path = plan_path()
    print(f"Path has {len(path)} waypoints.")
    print("First 5:", path[:5])
    visualise_map(path=path)
