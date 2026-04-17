from formation import uav_group
import numpy as np
import math
from map_setup import MISSION_START, MISSION_GOAL, MAP_LIMIT, PLAN_L, PLAN_R, PLAN_B, PLAN_T

class PathNode:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.parent = None
        self.cost = 0.0

def calculate_dist(n1, n2):
    return math.hypot(n1.x - n2.x, n1.y - n2.y)

def check_collision(n1, n2):
    for t in np.linspace(0, 1, 15):
        px = n1.x + t * (n2.x - n1.x)
        py = n1.y + t * (n2.y - n1.y)
        
        # Check ALL drones (formation safety) [cite: 87]
        for offset in uav_group.drone_offsets:
            drone_x = px + offset[0]
            drone_y = py + offset[1]
            
            if (PLAN_L <= drone_x <= PLAN_R) and (PLAN_B <= drone_y <= PLAN_T):
                return False
    return True

def run_mission_planner(iterations=1800):
    start_node = PathNode(*MISSION_START)
    goal_node = PathNode(*MISSION_GOAL)
    nodes_list = [start_node]
    step_dist = 6.0
    
    for _ in range(iterations):
        if np.random.rand() < 0.1:
            random_pt = goal_node
        else:
            random_pt = PathNode(np.random.uniform(0, MAP_LIMIT), np.random.uniform(0, MAP_LIMIT))
            
        nearest = nodes_list[np.argmin([calculate_dist(n, random_pt) for n in nodes_list])]
        angle = math.atan2(random_pt.y - nearest.y, random_pt.x - nearest.x)
        new_node = PathNode(nearest.x + step_dist * math.cos(angle), nearest.y + step_dist * math.sin(angle))
        
        if check_collision(nearest, new_node):
            new_node.parent = nearest
            new_node.cost = nearest.cost + calculate_dist(nearest, new_node)
            nodes_list.append(new_node)
            
            if calculate_dist(new_node, goal_node) < step_dist:
                goal_node.parent = new_node
                goal_node.cost = new_node.cost + calculate_dist(new_node, goal_node)
                nodes_list.append(goal_node)
                return nodes_list, goal_node
    return nodes_list, None

def extract_waypoints(goal_node):
    path = []
    current = goal_node
    while current:
        path.append((current.x, current.y))
        current = current.parent
    return path[::-1]