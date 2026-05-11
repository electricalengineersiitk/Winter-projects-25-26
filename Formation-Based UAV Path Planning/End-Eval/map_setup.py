import numpy as np

# Suggested map: 100 x 100 unit grid [cite: 53]
GRID_SIZE = 100
START = (5, 50) # [cite: 53]
GOAL = (95, 50) # [cite: 53]

# Obstacle: centred around (50, 50), radius 10 [cite: 53]
OBSTACLE_CENTER = (50, 50)
OBSTACLE_RADIUS = 10

def is_collision(x, y, safety_margin=2):
    # Mark obstacle cells (and a safety margin around them) as blocked 
    dist = np.sqrt((x - OBSTACLE_CENTER[0])**2 + (y - OBSTACLE_CENTER[1])**2)
    return dist <= (OBSTACLE_RADIUS + safety_margin)