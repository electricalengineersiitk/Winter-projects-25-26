import heapq
import numpy as np
from map_setup import start, goal, is_obstacle, GRID_SIZE

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def get_neighbors(node):
    x, y = node
    dirs = [(-1,0),(1,0),(0,-1),(0,1),
            (-1,-1),(1,1),(-1,1),(1,-1)]
    neighbors = []

    for dx, dy in dirs:
        nx, ny = x+dx, y+dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            if not is_obstacle(nx, ny):
                neighbors.append((nx, ny))
    return neighbors

def astar():
    open_set = [(0, start)]
    came_from = {}
    g = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for nb in get_neighbors(current):
            temp = g[current] + heuristic(current, nb)

            if nb not in g or temp < g[nb]:
                g[nb] = temp
                f = temp + heuristic(nb, goal)
                heapq.heappush(open_set, (f, nb))
                came_from[nb] = current

    return []   