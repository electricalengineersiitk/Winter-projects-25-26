import numpy as np
import matplotlib.pyplot as plt


GRID_SIZE = 100

start = (5, 50)
goal = (95, 50)

obs_center = (50, 50)
obs_rad = 10

def is_collision(x, y, margin=1):
    """
    Checks if a point (x, y) collides with the obstacle.
    
    """
    distance = np.sqrt((x - obs_center[0])**2 + (y - obs_center[1])**2)
    return distance <= (obs_rad + margin)

if __name__ == "__main__":
  
    fig, ax = plt.subplots(figsize=(6,6))
    
   
    circle = plt.Circle(obs_center, obs_rad, color='r', label='Obstacle')
    ax.add_patch(circle)
    
   
    ax.scatter(*start, color='g', s=100, label='Start')
    ax.scatter(*goal, color='b', s=100, label='Goal')
    

    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_title("UAV Simulation Map Setup")
    ax.legend()
    ax.grid(True)
print(f"Map initialized. Start: {start}, Goal: {goal}") 
plt.show()