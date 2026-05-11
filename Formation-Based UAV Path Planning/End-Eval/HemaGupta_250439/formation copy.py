import numpy as np

def get_formation_offsets():
    # Offsets for a 5-drone 'V' shape 
    return np.array([
        [-2, 0],
        [-1, -1],
        [0, 0],
        [1, -1],
        [2, 0]
    ])

def apply_formation(centroid_trajectory):
    offsets = get_formation_offsets()
    drones_paths = []
    
    # Each drone's position = centroid position + its fixed offset from the centroid 
    for offset in offsets:
        drone_path = []
        for point in centroid_trajectory:
            x, y, t = point
            drone_path.append([x + offset[0], y + offset[1], t])
        drones_paths.append(drone_path)
        
    return np.array(drones_paths)