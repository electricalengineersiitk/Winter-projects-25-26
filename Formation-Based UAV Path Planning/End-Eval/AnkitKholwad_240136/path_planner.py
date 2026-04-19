import numpy as np
import heapq
import matplotlib.pyplot as plt
from map_setup import *

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def get_neighbors(node, grid):
    directions = [(-1,0),(1,0),(0,-1),(0,1),
                  (-1,-1),(-1,1),(1,-1),(1,1)]

    neighbors = []
    for dx, dy in directions:
        x, y = node[0] + dx, node[1] + dy

        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            if grid[x, y] == 0:
                neighbors.append((x, y))

    return neighbors


def astar(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current, grid):
            tentative_g = g_score[current] + heuristic(current, neighbor)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    return None


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def plot_path(grid, path):
    plt.imshow(grid.T, origin='lower', cmap='gray_r')
    x, y = zip(*path)
    plt.plot(x, y, label="Path")
    plt.scatter(*START)
    plt.scatter(*GOAL)
    plt.legend()
    plt.show()