import heapq
import numpy as np
from map_setup import (
    build_obstacle_grid, world_to_grid, grid_to_world,
    START, GOAL, MAP_WIDTH, MAP_HEIGHT, GRID_RESOLUTION
)

def _euclidean(a, b):
    """Euclidean distance between two (col, row) grid cells."""
    return np.hypot(a[0] - b[0], a[1] - b[1])

_MOVES = [
    ( 1,  0, 1.0),   ( -1,  0, 1.0),
    ( 0,  1, 1.0),   (  0, -1, 1.0),
    ( 1,  1, 1.4142),( -1,  1, 1.4142),
    ( 1, -1, 1.4142),( -1, -1, 1.4142),
]

def _neighbours(col, row, grid):
    rows, cols = grid.shape
    for dc, dr, cost in _MOVES:
        nc, nr = col + dc, row + dr
        if 0 <= nc < cols and 0 <= nr < rows and not grid[nr, nc]:
            yield nc, nr, cost

def astar(grid, start_cell, goal_cell):
    open_heap = []          
    heapq.heappush(open_heap, (0.0, 0.0, start_cell[0], start_cell[1]))

    came_from = {}         
    g_score   = {start_cell: 0.0}
    closed    = set()

    while open_heap:
        f, g, col, row = heapq.heappop(open_heap)
        node = (col, row)

        if node in closed:
            continue
        closed.add(node)

        if node == goal_cell:
            path = []
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start_cell)
            path.reverse()
            return path

        for nc, nr, move_cost in _neighbours(col, row, grid):
            nb = (nc, nr)
            if nb in closed:
                continue
            tentative_g = g + move_cost
            if tentative_g < g_score.get(nb, float('inf')):
                g_score[nb] = tentative_g
                h = _euclidean(nb, goal_cell)
                heapq.heappush(open_heap, (tentative_g + h, tentative_g, nc, nr))
                came_from[nb] = node

    return []  


def _line_of_sight(c0, r0, c1, r1, grid):
    rows, cols = grid.shape
    dc = abs(c1 - c0); dr = abs(r1 - r0)
    sc = 1 if c0 < c1 else -1
    sr = 1 if r0 < r1 else -1
    err = dc - dr
    c, r = c0, r0
    while True:
        if not (0 <= c < cols and 0 <= r < rows):
            return False
        if grid[r, c]:
            return False
        if c == c1 and r == r1:
            return True
        e2 = 2 * err
        if e2 > -dr:
            err -= dr; c += sc
        if e2 < dc:
            err += dc; r += sr


def smooth_path(path, grid):
    if len(path) < 3:
        return path
    smoothed = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            c0, r0 = path[i]
            c1, r1 = path[j]
            if _line_of_sight(c0, r0, c1, r1, grid):
                break
            j -= 1
        smoothed.append(path[j])
        i = j
    return smoothed

def plan_path(start=None, goal=None):
    if start is None: start = START
    if goal  is None: goal  = GOAL

    grid = build_obstacle_grid()

    start_cell = world_to_grid(*start)
    goal_cell  = world_to_grid(*goal)

    if grid[start_cell[1], start_cell[0]]:
        raise ValueError("Start position is inside the obstacle!")
    if grid[goal_cell[1], goal_cell[0]]:
        raise ValueError("Goal position is inside the obstacle!")

    raw_path = astar(grid, start_cell, goal_cell)
    if not raw_path:
        raise RuntimeError("A* could not find a path. Check obstacle/safety margin.")

    pruned = smooth_path(raw_path, grid)

    waypoints = [grid_to_world(c, r) for c, r in pruned]

    waypoints[0]  = start
    waypoints[-1] = goal

    print(f"[path_planner] Raw path  : {len(raw_path)} cells")
    print(f"[path_planner] After pruning : {len(waypoints)} waypoints")
    return waypoints

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from map_setup import visualise_map

    wp = plan_path()
    print(f"First waypoint : {wp[0]}")
    print(f"Last  waypoint : {wp[-1]}")

    fig, ax = visualise_map(path=wp)
    plt.show()