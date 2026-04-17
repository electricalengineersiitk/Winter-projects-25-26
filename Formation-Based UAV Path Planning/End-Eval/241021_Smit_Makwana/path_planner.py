import heapq
import numpy as np
from map_setup import START, GOAL, is_collision, GRID_SIZE

def heuristic(a, b):
    # Heuristic: Euclidean distance to the goal 
    return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def plan_path():
    # Implement chosen algorithm (A*) to find a waypoint path [cite: 55]
    open_set = []
    heapq.heappush(open_set, (0, START))
    came_from = {}
    g_score = {START: 0}
    
    # Each cell connects to its 8 neighbours 
    neighbors = [(0,1),(0,-1),(1,0),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == GOAL:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(START)
            return path[::-1] # Output is a list of (x, y) waypoints [cite: 56]

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Boundary check
            if not (0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE):
                continue
            # Collision check
            if is_collision(neighbor[0], neighbor[1]):
                continue
                
            tentative_g_score = g_score[current] + np.sqrt(dx**2 + dy**2)
            
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, GOAL)
                heapq.heappush(open_set, (f_score, neighbor))
    return []