import numpy as np

def get_formation_offsets():
    return np.array([
        (-2, 0),
        (-1, -1),
        (0, 0),
        (1, -1),
        (2, 0)
    ])


def apply_formation(trajectory, offsets):
    trajectory = np.array(trajectory)
    drones = []

    for dx, dy in offsets:
        x = trajectory[:,0] + dx
        y = trajectory[:,1] + dy
        t = trajectory[:,2]
        drones.append(list(zip(x, y, t)))

    return drones