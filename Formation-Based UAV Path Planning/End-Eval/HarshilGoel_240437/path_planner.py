"""
path_planner.py
---------------
Implements A* and Dijkstra's algorithm to compare performance.
Finds a collision-free waypoint path from START to GOAL.
"""

import heapq
import math
import time
from map_setup import build_grid, START, GOAL, MAP_WIDTH, MAP_HEIGHT, visualise_map

def _heuristic(a, b):
    """Euclidean distance heuristic for A*."""
    return math.hypot(b[0] - a[0], b[1] - a[1])

def astar(grid, start, goal):
    """A* search using a distance heuristic."""
    open_heap = []
    heapq.heappush(open_heap, (0.0, 0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    nodes_expanded = 0

    neighbours = [(-1,-1),(-1,0),(-1,1), (0,-1), (0,1), (1,-1),(1,0),(1,1)]

    while open_heap:
        _, g, current = heapq.heappop(open_heap)
        nodes_expanded += 1

        if current == goal:
            return _reconstruct(came_from, current), nodes_expanded

        if g > g_score.get(current, math.inf):
            continue

        cx, cy = current
        for dx, dy in neighbours:
            nx, ny = cx + dx, cy + dy

            if not (0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT): continue
            if not grid[ny, nx]: continue

            step = math.hypot(dx, dy)
            tentative_g = g_score[current] + step

            if tentative_g < g_score.get((nx, ny), math.inf):
                g_score[(nx, ny)] = tentative_g
                came_from[(nx, ny)] = current
                f = tentative_g + _heuristic((nx, ny), goal)
                heapq.heappush(open_heap, (f, tentative_g, (nx, ny)))

    return [], nodes_expanded

def dijkstra(grid, start, goal):
    """Dijkstra's search (no heuristic)."""
    open_heap = []
    heapq.heappush(open_heap, (0.0, start))
    came_from = {}
    g_score = {start: 0.0}
    nodes_expanded = 0

    neighbours = [(-1,-1),(-1,0),(-1,1), (0,-1), (0,1), (1,-1),(1,0),(1,1)]

    while open_heap:
        g, current = heapq.heappop(open_heap)
        nodes_expanded += 1

        if current == goal:
            return _reconstruct(came_from, current), nodes_expanded

        if g > g_score.get(current, math.inf):
            continue

        cx, cy = current
        for dx, dy in neighbours:
            nx, ny = cx + dx, cy + dy

            if not (0 <= nx < MAP_WIDTH and 0 <= ny < MAP_HEIGHT): continue
            if not grid[ny, nx]: continue

            step = math.hypot(dx, dy)
            tentative_g = g_score[current] + step

            if tentative_g < g_score.get((nx, ny), math.inf):
                g_score[(nx, ny)] = tentative_g
                came_from[(nx, ny)] = current
                heapq.heappush(open_heap, (tentative_g, (nx, ny)))

    return [], nodes_expanded

def _reconstruct(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def simplify_path(path, step=5):
    if len(path) <= 2: return path
    simplified = path[::step]
    if simplified[-1] != path[-1]:
        simplified.append(path[-1])
    return simplified

def plan_path(algo='astar'):
    """Run the chosen algorithm and return (waypoints, nodes_expanded, time_taken)."""
    grid = build_grid()
    
    t0 = time.time()
    if algo == 'astar':
        raw_path, nodes = astar(grid, START, GOAL)
    else:
        raw_path, nodes = dijkstra(grid, START, GOAL)
    t_elapsed = time.time() - t0

    if not raw_path:
        raise RuntimeError(f"{algo.upper()} found no valid path.")

    waypoints = simplify_path(raw_path, step=4)
    return waypoints, nodes, t_elapsed