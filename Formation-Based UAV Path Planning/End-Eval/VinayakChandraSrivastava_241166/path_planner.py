"""
path_planner.py — Implements A* search on a 2-D grid to find a
collision-free waypoint path from START to GOAL.
"""

import heapq
import math
import numpy as np

from map_setup import (
    START, GOAL, MAP_WIDTH, MAP_HEIGHT,
    build_grid, SAFETY_MARGIN
)


# ── 8-connected neighbours ───────────────────────────────────────
_MOVES = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


def _heuristic(a, b):
    """Euclidean distance heuristic."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def astar(grid, start, goal):
    """
    A* on a boolean grid (True = free).

    Parameters
    ----------
    grid  : np.ndarray[bool], shape (height, width)
    start : (col, row)  i.e. (x, y)
    goal  : (col, row)

    Returns
    -------
    list of (x, y) waypoints from start to goal, or [] on failure.
    """
    rows, cols = grid.shape
    # Convert (x,y) → (col, row) for indexing
    sx, sy = int(round(start[0])), int(round(start[1]))
    gx, gy = int(round(goal[0])),  int(round(goal[1]))

    open_heap = []          # (f, g, node)
    heapq.heappush(open_heap, (0.0, 0.0, (sx, sy)))

    came_from = {}
    g_score = {(sx, sy): 0.0}
    visited  = set()

    while open_heap:
        f, g, current = heapq.heappop(open_heap)

        if current in visited:
            continue
        visited.add(current)

        cx, cy = current
        if cx == gx and cy == gy:
            # Reconstruct path
            path = []
            node = current
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append((sx, sy))
            path.reverse()
            return path

        for dr, dc in _MOVES:
            nx, ny = cx + dc, cy + dr
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            if not grid[ny, nx]:          # blocked
                continue
            neighbour = (nx, ny)
            step = math.hypot(dc, dr)
            tentative_g = g + step
            if tentative_g < g_score.get(neighbour, math.inf):
                g_score[neighbour] = tentative_g
                came_from[neighbour] = current
                f_new = tentative_g + _heuristic(neighbour, (gx, gy))
                heapq.heappush(open_heap, (f_new, tentative_g, neighbour))

    return []   # no path found


def smooth_path(path, window=5):
    """
    Apply a simple sliding-window average to reduce jaggedness
    while keeping start and goal fixed.
    """
    if len(path) <= 2:
        return path
    arr = np.array(path, dtype=float)
    smoothed = arr.copy()
    half = window // 2
    for i in range(1, len(arr) - 1):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        smoothed[i] = arr[lo:hi].mean(axis=0)
    smoothed[0]  = arr[0]
    smoothed[-1] = arr[-1]
    return [(float(p[0]), float(p[1])) for p in smoothed]


def plan_path():
    """Build the grid and run A*. Returns smoothed waypoint list."""
    grid = build_grid(margin=SAFETY_MARGIN)
    raw  = astar(grid, START, GOAL)
    if not raw:
        raise RuntimeError("A* failed — no path found. Check obstacle/margin settings.")
    waypoints = smooth_path(raw, window=7)
    print(f"[path_planner] A* found {len(raw)} raw nodes → {len(waypoints)} waypoints after smoothing.")
    return waypoints


# ── Quick self-test ──────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from map_setup import visualise_map, OBSTACLE_CENTER, OBSTACLE_RADIUS

    wp = plan_path()
    print(f"First waypoint : {wp[0]}")
    print(f"Last  waypoint : {wp[-1]}")
    visualise_map(path=wp)
