"""
path_planner.py
Implements A* on the grid built by map_setup.py.

A* combines the actual cost from the start (g) with a heuristic
estimate to the goal (h) so that it expands the most promising
node first.  Euclidean distance is used as the heuristic because
the grid allows diagonal moves; it is admissible (never over-estimates).
"""

import heapq
import numpy as np
from map_setup import build_grid, START, GOAL, MAP_WIDTH, MAP_HEIGHT


# 8-connected neighbour offsets with their movement costs
_MOVES = [
    (-1,  0, 1.0),   # up
    ( 1,  0, 1.0),   # down
    ( 0, -1, 1.0),   # left
    ( 0,  1, 1.0),   # right
    (-1, -1, 1.4142), # diagonals
    (-1,  1, 1.4142),
    ( 1, -1, 1.4142),
    ( 1,  1, 1.4142),
]


def _heuristic(r, c, goal_r, goal_c):
    """Euclidean distance from (r,c) to the goal cell."""
    return np.hypot(r - goal_r, c - goal_c)


def _xy_to_rc(x, y):
    """Convert (x, y) map coordinates → (row, col) grid indices."""
    return int(round(y)), int(round(x))


def _rc_to_xy(r, c):
    """Convert (row, col) grid indices → (x, y) map coordinates."""
    return float(c), float(r)


def astar(grid=None):
    """
    Run A* on the occupancy grid from map_setup.

    Returns
    -------
    List of (x, y) waypoints from START to GOAL, or [] if no path found.
    """
    if grid is None:
        grid = build_grid()

    start_rc = _xy_to_rc(*START)
    goal_rc  = _xy_to_rc(*GOAL)

    # Priority queue entries: (f_score, g_score, row, col)
    open_heap = []
    heapq.heappush(open_heap, (0.0, 0.0, *start_rc))

    came_from = {}          # child → parent
    g_score   = {start_rc: 0.0}
    closed    = set()

    gr, gc = goal_rc

    while open_heap:
        f, g, r, c = heapq.heappop(open_heap)

        if (r, c) in closed:
            continue
        closed.add((r, c))

        if (r, c) == goal_rc:
            return _reconstruct(came_from, (r, c))

        for dr, dc, cost in _MOVES:
            nr, nc = r + dr, c + dc

            # Boundary check
            if not (0 <= nr < MAP_HEIGHT and 0 <= nc < MAP_WIDTH):
                continue

            # Obstacle check
            if not grid[nr, nc]:
                continue

            if (nr, nc) in closed:
                continue

            tentative_g = g + cost

            if tentative_g < g_score.get((nr, nc), float("inf")):
                g_score[(nr, nc)] = tentative_g
                came_from[(nr, nc)] = (r, c)
                h = _heuristic(nr, nc, gr, gc)
                heapq.heappush(open_heap, (tentative_g + h, tentative_g, nr, nc))

    print("A* found no path — check obstacle inflation settings.")
    return []


def _reconstruct(came_from, current):
    """Trace back from goal to start and return the (x, y) waypoint list."""
    path_rc = [current]
    while current in came_from:
        current = came_from[current]
        path_rc.append(current)
    path_rc.reverse()

    waypoints = [_rc_to_xy(r, c) for r, c in path_rc]
    return waypoints


def simplify_path(waypoints, tolerance=1.5):
    """
    Remove collinear intermediate points to reduce waypoint count.
    Uses the Ramer–Douglas–Peucker idea: keep a point if it is
    farther than `tolerance` from the line joining its neighbours.
    """
    if len(waypoints) <= 2:
        return waypoints

    keep = [True] * len(waypoints)

    def rdp(start_i, end_i):
        if end_i - start_i < 2:
            return
        xs, ys = waypoints[start_i]
        xe, ye = waypoints[end_i]
        dx, dy = xe - xs, ye - ys
        seg_len = np.hypot(dx, dy)
        if seg_len == 0:
            for i in range(start_i + 1, end_i):
                keep[i] = False
            return
        max_dist = 0.0
        max_idx  = start_i + 1
        for i in range(start_i + 1, end_i):
            xi, yi = waypoints[i]
            # perpendicular distance from point to line
            dist = abs(dy * xi - dx * yi + xe * ys - ye * xs) / seg_len
            if dist > max_dist:
                max_dist = dist
                max_idx  = i
        if max_dist < tolerance:
            for i in range(start_i + 1, end_i):
                keep[i] = False
        else:
            rdp(start_i, max_idx)
            rdp(max_idx, end_i)

    rdp(0, len(waypoints) - 1)
    return [wp for wp, k in zip(waypoints, keep) if k]


def plan_path():
    """
    Public interface used by other modules.
    Returns a simplified list of (x, y) waypoints.
    """
    grid     = build_grid()
    raw_path = astar(grid)
    if not raw_path:
        return []
    simplified = simplify_path(raw_path)
    return simplified
