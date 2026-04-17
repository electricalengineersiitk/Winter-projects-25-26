"""
path_planner.py
Implements Dijkstra's algorithm on an 8-connected grid to find a
collision-free waypoint path from START to GOAL around the obstacle.
"""

import heapq
import os
import numpy as np
import matplotlib.pyplot as plt

from map_setup import (
    MAP_WIDTH, MAP_HEIGHT,
    START, GOAL,
    OBSTACLE_CENTER, OBSTACLE_RADIUS,
    get_obstacle_mask, plot_map,
)

# Safety margin (cells) added around the obstacle to keep drones away from edges
SAFETY_MARGIN = 5

# 8-directional movement: (dx, dy)
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1)]


def dijkstra(start=START, goal=GOAL, safety_margin=SAFETY_MARGIN):
    """
    Run Dijkstra's shortest-path algorithm on a 100×100 grid.

    Parameters
    ----------
    start, goal   : (x, y) integer grid coordinates
    safety_margin : extra cells inflated around the obstacle

    Returns
    -------
    path : list of (x, y) tuples from start → goal
    """
    blocked = get_obstacle_mask(safety_margin=safety_margin)

    # dist[y, x] = best cost found so far
    dist = np.full((MAP_HEIGHT, MAP_WIDTH), np.inf, dtype=np.float64)
    sx, sy = start
    gx, gy = goal
    dist[sy, sx] = 0.0

    # prev[(x, y)] = (px, py) for path reconstruction
    prev = {}

    # Min-heap: (cost, x, y)
    heap = [(0.0, sx, sy)]

    while heap:
        cost, x, y = heapq.heappop(heap)

        # Goal reached
        if (x, y) == (gx, gy):
            break

        # Skip stale entries
        if cost > dist[y, x]:
            continue

        for dx, dy in _DIRS:
            nx, ny = x + dx, y + dy

            # Bounds check
            if not (0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT):
                continue

            # Obstacle check (inflated)
            if blocked[ny, nx]:
                continue

            # Diagonal moves cost √2; cardinal moves cost 1
            step_cost = np.sqrt(2) if (dx != 0 and dy != 0) else 1.0
            new_cost = cost + step_cost

            if new_cost < dist[ny, nx]:
                dist[ny, nx] = new_cost
                prev[(nx, ny)] = (x, y)
                heapq.heappush(heap, (new_cost, nx, ny))

    # ── Reconstruct path ──────────────────────────────────────────────────────
    if (gx, gy) not in prev and (gx, gy) != (sx, sy):
        raise RuntimeError("Dijkstra could not find a path to the goal.")

    path = []
    cur = (gx, gy)
    while cur != (sx, sy):
        path.append(cur)
        cur = prev[cur]
    path.append((sx, sy))
    path.reverse()

    return path


def simplify_path(path, tolerance=2.0):
    """
    Ramer-Douglas-Peucker line simplification to reduce waypoint count.
    Keeps the path shape while removing collinear intermediate points.
    """
    if len(path) <= 2:
        return path

    def _rdp(pts, eps):
        if len(pts) <= 2:
            return pts
        p1, p2 = np.array(pts[0]), np.array(pts[-1])
        seg = p2 - p1
        seg_len = np.linalg.norm(seg)
        if seg_len == 0:
            dists = [np.linalg.norm(np.array(p) - p1) for p in pts]
        else:
            dists = [abs(np.cross(seg, np.array(p) - p1)) / seg_len
                     for p in pts]
        idx = int(np.argmax(dists))
        if dists[idx] > eps:
            left  = _rdp(pts[:idx + 1], eps)
            right = _rdp(pts[idx:],     eps)
            return left[:-1] + right
        return [pts[0], pts[-1]]

    return _rdp(path, tolerance)


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running Dijkstra …")
    raw_path = dijkstra()
    path = simplify_path(raw_path)

    print(f"  Raw waypoints   : {len(raw_path)}")
    print(f"  Simplified      : {len(path)}")
    path_arr = np.array(path)
    dist = np.sum(np.linalg.norm(np.diff(path_arr, axis=0), axis=1))
    print(f"  Total distance  : {dist:.2f} units")

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_map(ax, path=path, title="Dijkstra's Shortest Path")

    out = os.path.join(os.path.dirname(__file__), "results", "path_plot.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Path plot saved → {out}")
    plt.show()
