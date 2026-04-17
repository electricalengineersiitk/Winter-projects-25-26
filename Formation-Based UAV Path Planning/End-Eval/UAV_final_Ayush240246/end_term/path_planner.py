import numpy as np
import math
import matplotlib.pyplot as plt
import os

from map_setup import START, GOAL, OBSTACLE_CENTER, INFLATED_RADIUS, GRID_SIZE

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0.0

def is_collision_free(node_a, node_b):
    dx = node_b.x - node_a.x
    dy = node_b.y - node_a.y
    fx = node_a.x - OBSTACLE_CENTER[0]
    fy = node_a.y - OBSTACLE_CENTER[1]
    
    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = (fx**2 + fy**2) - INFLATED_RADIUS**2

    if a < 1e-7:
        return True 
        
    discriminant = b**2 - 4*a*c
    if discriminant < 0:
        return True 
        
    discriminant = math.sqrt(discriminant)
    t1 = (-b - discriminant) / (2*a)
    t2 = (-b + discriminant) / (2*a)
    
    if (0 <= t1 <= 1) or (0 <= t2 <= 1):
        return False 
    return True

def get_random_point():
    """Samples a random point, with a 10% bias to sample the exact goal to pull the tree forward."""
    if np.random.rand() < 0.10:
        return Node(GOAL[0], GOAL[1])
    return Node(np.random.uniform(0, GRID_SIZE), np.random.uniform(0, GRID_SIZE))

def get_distance(node_a, node_b):
    return math.hypot(node_a.x - node_b.x, node_a.y - node_b.y)

def find_nearest(tree, random_node):
    """Finds the closest existing node in the tree to the randomly sampled point."""
    distances = [get_distance(node, random_node) for node in tree]
    nearest_idx = np.argmin(distances)
    return tree[nearest_idx]

def steer(from_node, to_node, step_size):
    """Creates a new node by moving from the nearest node toward the random node by step_size."""
    dist = get_distance(from_node, to_node)
    if dist <= step_size:
        return Node(to_node.x, to_node.y)
    
    theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
    new_x = from_node.x + step_size * math.cos(theta)
    new_y = from_node.y + step_size * math.sin(theta)
    return Node(new_x, new_y)

def run_rrt_star():
    print("Initializing RRT* Search...")
    start_node = Node(START[0], START[1])
    goal_node = Node(GOAL[0], GOAL[1])
    tree = [start_node]
    
    max_iter = 1500     
    step_size = 5.0      
    search_radius = 15.0 

    for i in range(max_iter):
        rand_node = get_random_point()
        nearest_node = find_nearest(tree, rand_node)
        new_node = steer(nearest_node, rand_node, step_size)
        
        if is_collision_free(nearest_node, new_node):
            neighbors = [node for node in tree if get_distance(node, new_node) <= search_radius]
            
            new_node.parent = nearest_node
            new_node.cost = nearest_node.cost + get_distance(nearest_node, new_node)
            
            for near_node in neighbors:
                potential_cost = near_node.cost + get_distance(near_node, new_node)
                if potential_cost < new_node.cost and is_collision_free(near_node, new_node):
                    new_node.parent = near_node
                    new_node.cost = potential_cost
            
            tree.append(new_node)
            
            for near_node in neighbors:
                rewired_cost = new_node.cost + get_distance(new_node, near_node)
                if rewired_cost < near_node.cost and is_collision_free(new_node, near_node):
                    near_node.parent = new_node
                    near_node.cost = rewired_cost
    
    distances_to_goal = [get_distance(node, goal_node) for node in tree]
    closest_to_goal_idx = np.argmin(distances_to_goal)
    final_node = tree[closest_to_goal_idx]
    
    path = []
    current = final_node
    while current is not None:
        path.append((current.x, current.y))
        current = current.parent
    path.reverse()
    
    if get_distance(final_node, goal_node) <= step_size:
        path.append(GOAL)
        
    print(f"Path found with {len(path)} waypoints. Total Cost: {final_node.cost:.2f}")
    return tree, path

def plot_rrt_star(tree, path):
    """Visualizes the map, the massive RRT* tree, and the final optimized path."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.plot(START[0], START[1], 'go', markersize=10, zorder=5)
    ax.plot(GOAL[0], GOAL[1], 'ro', markersize=10, zorder=5)
    core_obstacle = plt.Circle(OBSTACLE_CENTER, 10, color='black', fill=True, zorder=3)
    safe_boundary = plt.Circle(OBSTACLE_CENTER, INFLATED_RADIUS, color='red', fill=False, linestyle='--', linewidth=2, zorder=3)
    ax.add_patch(core_obstacle)
    ax.add_patch(safe_boundary)
    
    # Draw the RRT* Tree Branches
    for node in tree:
        if node.parent is not None:
            ax.plot([node.x, node.parent.x], [node.y, node.parent.y], color='lightgray', linewidth=0.5, zorder=1)
            
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax.plot(path_x, path_y, 'b-', linewidth=3, label='RRT* Shortest Path', zorder=4)
    ax.plot(path_x, path_y, 'bo', markersize=4, zorder=4)

    ax.set_xlim(-5, GRID_SIZE + 5)
    ax.set_ylim(-5, GRID_SIZE + 5)
    ax.set_title("Phase 2: RRT* Path Planning")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'rrt_path.png')
    # -----------------------------------

    plt.savefig(save_path)
    print(f"Path visualization saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    rrt_tree, final_waypoints = run_rrt_star()
    
    print("\n--- EXTRACTED RAW WAYPOINTS ---")
    print(final_waypoints)
    print("-------------------------------\n")
    
    plot_rrt_star(rrt_tree, final_waypoints)