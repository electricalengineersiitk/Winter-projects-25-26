%%writefile /content/end_term/path_planner.py
import heapq, math, sys
import numpy as np
sys.path.insert(0, '/content/end_term')
from map_setup import build_grid, START, GOAL

def _heuristic(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def _neighbours(node, grid):
    x, y = node
    rows, cols = grid.shape
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x+dx, y+dy
            if 0 <= nx < cols and 0 <= ny < rows and grid[ny, nx]:
                yield (nx, ny)

def astar(grid, start, goal):
    open_heap = []
    heapq.heappush(open_heap, (0.0, start))
    came_from = {}
    g_score   = {start: 0.0}
    f_score   = {start: _heuristic(start, goal)}
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
        for nb in _neighbours(current, grid):
            move_cost   = math.hypot(nb[0]-current[0], nb[1]-current[1])
            tentative_g = g_score[current] + move_cost
            if tentative_g < g_score.get(nb, float('inf')):
                came_from[nb] = current
                g_score[nb]   = tentative_g
                f_score[nb]   = tentative_g + _heuristic(nb, goal)
                heapq.heappush(open_heap, (f_score[nb], nb))
    return []

def thin_path(path, step=5):
    if not path:
        return path
    thinned = path[::step]
    if thinned[-1] != path[-1]:
        thinned.append(path[-1])
    return thinned

def plan_path(start=START, goal=GOAL):
    grid      = build_grid()
    raw_path  = astar(grid, start, goal)
    if not raw_path:
        raise RuntimeError("A* found no path!")
    waypoints = thin_path(raw_path, step=5)
    print(f"Raw: {len(raw_path)}  Thinned: {len(waypoints)}")
    return waypoints
