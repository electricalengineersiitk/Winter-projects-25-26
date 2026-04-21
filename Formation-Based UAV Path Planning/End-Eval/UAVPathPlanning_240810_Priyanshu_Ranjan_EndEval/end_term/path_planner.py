from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from map_setup import MapConfig, default_map, is_blocked

GridPoint = Tuple[int, int]
Waypoint = Tuple[float, float]


@dataclass
class AStarResult:
    waypoints: List[Waypoint]
    visited_nodes: int


_NEIGHBORS = [
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
]


def _heuristic(a: GridPoint, b: GridPoint) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _in_bounds(point: GridPoint, config: MapConfig) -> bool:
    return 0 <= point[0] <= config.width and 0 <= point[1] <= config.height


def _segment_clear(a: np.ndarray, b: np.ndarray, config: MapConfig) -> bool:
    center = np.asarray(config.obstacle_center, dtype=float)
    segment = b - a
    length_sq = float(np.dot(segment, segment))
    if length_sq == 0.0:
        return not is_blocked(tuple(a), config)
    projection = float(np.dot(center - a, segment) / length_sq)
    projection = min(1.0, max(0.0, projection))
    closest = a + projection * segment
    distance = float(np.linalg.norm(closest - center))
    return distance > config.inflated_radius


def _prune_path(path: List[Waypoint], config: MapConfig) -> List[Waypoint]:
    if len(path) <= 2:
        return path

    pruned = [path[0]]
    index = 0
    while index < len(path) - 1:
        next_index = len(path) - 1
        for candidate_index in range(len(path) - 1, index, -1):
            if _segment_clear(np.asarray(path[index], dtype=float), np.asarray(path[candidate_index], dtype=float), config):
                next_index = candidate_index
                break
        pruned.append(path[next_index])
        index = next_index
    return pruned


def reconstruct_path(came_from: Dict[GridPoint, GridPoint], current: GridPoint) -> List[Waypoint]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return [(float(x), float(y)) for x, y in path]


def astar_path(config: MapConfig | None = None) -> AStarResult:
    config = config or default_map()
    start = (int(round(config.start[0])), int(round(config.start[1])))
    goal = (int(round(config.goal[0])), int(round(config.goal[1])))

    if is_blocked(start, config):
        raise ValueError("Start point lies inside the inflated obstacle.")
    if is_blocked(goal, config):
        raise ValueError("Goal point lies inside the inflated obstacle.")

    open_heap: List[Tuple[float, float, GridPoint]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start))
    came_from: Dict[GridPoint, GridPoint] = {}
    g_score: Dict[GridPoint, float] = {start: 0.0}
    visited = set()
    visited_nodes = 0

    while open_heap:
        _, current_cost, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)
        visited_nodes += 1

        if current == goal:
            raw_path = reconstruct_path(came_from, current)
            waypoints = _prune_path(raw_path, config)
            return AStarResult(waypoints=waypoints, visited_nodes=visited_nodes)

        for dx, dy, step_cost in _NEIGHBORS:
            neighbor = (current[0] + dx, current[1] + dy)
            if not _in_bounds(neighbor, config):
                continue
            if is_blocked(neighbor, config):
                continue

            # Prevent diagonal corner cutting through the inflated obstacle.
            if dx != 0 and dy != 0:
                horizontal = (current[0] + dx, current[1])
                vertical = (current[0], current[1] + dy)
                if is_blocked(horizontal, config) or is_blocked(vertical, config):
                    continue

            tentative_g = current_cost + step_cost
            if tentative_g >= g_score.get(neighbor, float("inf")):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            priority = tentative_g + _heuristic(neighbor, goal)
            heapq.heappush(open_heap, (priority, tentative_g, neighbor))

    raise RuntimeError("A* failed to find a path through the map.")
