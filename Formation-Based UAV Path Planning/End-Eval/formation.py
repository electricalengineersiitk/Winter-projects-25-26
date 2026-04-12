import numpy as np

class DroneFormation:
    def __init__(self):
        # Unique 5-drone 'A' shape offsets
        self.drone_offsets = np.array([
            [0, 3], [-1.5, 0], [1.5, 0], [-2.5, -2], [2.5, -2]
        ])
        self.radius_buffer = np.max(np.linalg.norm(self.drone_offsets, axis=1))

uav_group = DroneFormation()