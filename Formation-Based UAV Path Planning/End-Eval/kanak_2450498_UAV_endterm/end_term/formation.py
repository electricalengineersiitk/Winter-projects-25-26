import numpy as np
import matplotlib.pyplot as plt

def get_formation_offsets(shape_name="V"):
   
    if shape_name == "V":
        # A 5-drone 'V' shape 
        # Centroid is at (0,0). Drones are spread out around it.
        offsets = np.array([
            [0, 0],    # Drone 0 (Leader/Centroid)
            [-2, 2],   # Drone 1 (Back Left)
            [-4, 4],   # Drone 2 (Far Back Left)
            [-2, -2],  # Drone 3 (Back Right)
            [-4, -4]   # Drone 4 (Far Back Right)
        ])
    elif shape_name == "A":
        # Example 'A' shape offsets
        offsets = np.array([
            [0, 0],    # Top
            [-2, -1],  # Middle left
            [-4, -2],  # Bottom left
            [-2, 1],   # Middle right
            [-4, 2],   # Bottom right
            [-2, 0]    # Middle bar
        ])
    else:
        # default shape: triangle
        offsets = np.array([[0,0], [-2, 2], [-2, -2]])

    return offsets

def get_drone_positions(centroid_pos, offsets):
   
    return centroid_pos + offsets

if __name__ == "__main__":
    test_centroid = np.array([50, 50])
    v_offsets = get_formation_offsets("V")
    drone_coords = get_drone_positions(test_centroid, v_offsets)

    plt.figure(figsize=(5,5))
    plt.scatter(drone_coords[:, 0], drone_coords[:, 1], c='blue', s=100, label='Drones')
    plt.scatter(test_centroid[0], test_centroid[1], c='red', marker='x', label='Centroid')
    
    plt.plot(drone_coords[[2,1,0,3,4], 0], drone_coords[[2,1,0,3,4], 1], 'k--')
    
    plt.title("Phase 4: Formation Shape Verification")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()