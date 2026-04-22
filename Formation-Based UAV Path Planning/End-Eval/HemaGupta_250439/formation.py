%%writefile /content/end_term/formation.py
import numpy as np

V_OFFSETS = np.array([
    [-4,  4],
    [-2,  2],
    [ 0,  0],
    [ 2,  2],
    [ 4,  4],
], dtype=float)

N_DRONES     = len(V_OFFSETS)
DRONE_COLORS = ['#e63946','#f4a261','#2a9d8f','#457b9d','#a8dadc']

def get_drone_positions(centroid_x, centroid_y):
    cx       = np.atleast_1d(centroid_x)
    cy       = np.atleast_1d(centroid_y)
    centroid = np.column_stack([cx, cy])
    return centroid[:, np.newaxis, :] + V_OFFSETS[np.newaxis, :, :]
