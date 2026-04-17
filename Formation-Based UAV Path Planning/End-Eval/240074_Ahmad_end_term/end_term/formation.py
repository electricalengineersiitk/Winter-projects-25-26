import numpy as np
import matplotlib.pyplot as plt

N_DRONES = 5

FORMATION_OFFSETS = np.array([
    [-4,  4],
    [-2,  2],
    [ 0,  0],
    [ 2,  2],
    [ 4,  4],
], dtype=float)

def get_offsets():
    return FORMATION_OFFSETS

def drone_positions(centroid_x, centroid_y, heading_angle=0.0):
    cos_a = np.cos(heading_angle)
    sin_a = np.sin(heading_angle)

    R = np.array([[cos_a, -sin_a],
                  [sin_a,  cos_a]])

    rotated = (R @ FORMATION_OFFSETS.T).T

    positions = rotated + np.array([centroid_x, centroid_y])
    return positions

def preview_formation():
    pos = drone_positions(50, 50, heading_angle=0)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title("V-Formation (5 Drones)")
    ax.set_aspect('equal')

    for i, (x, y) in enumerate(pos):
        ax.plot(x, y, 'o', markersize=14, color='royalblue')
        ax.text(x, y, f'D{i+1}', ha='center', va='center', color='white', fontsize=8, fontweight='bold')

    ax.plot(pos[:, 0], pos[:, 1], 'k--', alpha=0.4)
    ax.plot(50, 50, 'r+', markersize=14, markeredgewidth=2, label='Centroid')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    preview_formation()
