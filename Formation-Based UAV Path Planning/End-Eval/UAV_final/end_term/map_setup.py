import numpy as np
import matplotlib.pyplot as plt
import os

# I didnt make formation.py seperately the formation class is integrated in this map setup file only
class Formation:
    def __init__(self):
        # The drones are made in a W shape
        self.offsets = np.array([
            [-3.0, 3.0],   
            [-2.25, 0.0],  
            [-1.5, -3.0],  
            [0.0, 0.0],    
            [1.5, -3.0],
            [2.25, 0.0],  
            [3.0, 3.0] 
        ])
        self.N = len(self.offsets)
        distances = np.linalg.norm(self.offsets, axis=1)
        self.max_spread = np.max(distances)
form = Formation()

# These are for RRT* Algo 
GRID_SIZE = 101          
START = (0, 0)             # Start coordinate
GOAL = (100, 100)          # Goal coordinate
OBSTACLE_CENTER = (50, 50) # Physical obstacle location
BASE_OBSTACLE_RADIUS = 10  # Physical obstacle size

# Base Radius (10) + Formation Spread+ Safety Buffer( just for safety)
INFLATED_RADIUS = BASE_OBSTACLE_RADIUS + form.max_spread + 2.0

def create_discrete_grid():
    """Generates a visual 2D array of the environment."""
    x = np.arange(GRID_SIZE)
    y = np.arange(GRID_SIZE)
    xx, yy = np.meshgrid(x, y)
    
    # Distance from center
    distances_squared = (xx - OBSTACLE_CENTER[0])**2 + (yy - OBSTACLE_CENTER[1])**2    
    grid = (distances_squared <= INFLATED_RADIUS**2).astype(int)
    return grid.T

def plot_environment(grid):
    """Renders the map, obstacle, safe zone, and starting formation."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid.T, cmap='Greys', origin='lower', alpha=0.2)
    
    # Start and Goal points
    ax.plot(START[0], START[1], 'go', markersize=10, label='Start (0, 0)')
    ax.plot(GOAL[0], GOAL[1], 'ro', markersize=10, label='Goal (100, 100)')
    
    core_obstacle = plt.Circle(OBSTACLE_CENTER, BASE_OBSTACLE_RADIUS, color='black', fill=True, label='Core Obstacle')
    safe_boundary = plt.Circle(OBSTACLE_CENTER, INFLATED_RADIUS, color='red', fill=False, linestyle='--', linewidth=2, label=f'Safe Boundary (r={INFLATED_RADIUS:.1f})')
    ax.add_patch(core_obstacle)
    ax.add_patch(safe_boundary)
    
    # Draw the 'W' Formation at the start position
    start_drones = np.array(START) + form.offsets
    ax.plot(start_drones[:, 0], start_drones[:, 1], 'bo', markersize=6, label=f'N={form.N} Drones')
    for i in range(form.N - 1):
        ax.plot([start_drones[i, 0], start_drones[i+1, 0]], 
                [start_drones[i, 1], start_drones[i+1, 1]], 'b-', alpha=0.6)
                
    ax.set_xlim(-5, GRID_SIZE + 5)
    ax.set_ylim(-5, GRID_SIZE + 5)
    ax.set_title("Phase 1: Dynamic Environment Setup")
    ax.legend(loc='upper left')
    ax.grid(True, linestyle=':', alpha=0.5)
        
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'map_plot.png')
    
    plt.savefig(save_path)
    print(f"Map visualization successfully saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    env_grid = create_discrete_grid()
    plot_environment(env_grid)