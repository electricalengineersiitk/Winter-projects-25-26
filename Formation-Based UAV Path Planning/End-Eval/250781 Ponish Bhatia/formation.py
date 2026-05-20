import numpy as np
from map_setup import is_obstacle

FORMATION_RADIUS = 3  

def get_offsets():
    return np.array([
        (-2,0), (-1,-1), (0,0), (1,-1), (2,0)
    ])

def apply_formation(x, y, t):
    offsets = get_offsets()
    drones = []

    for offset in offsets:
        dx, dy = offset
        path = []

        for xi, yi, ti in zip(x, y, t):
            px, py = xi + dx, yi + dy

            # collision check
            # if is_obstacle(px, py):
            #     raise ValueError("Drone collision detected!")

            path.append((px, py, ti))

        drones.append(path)

    return drones