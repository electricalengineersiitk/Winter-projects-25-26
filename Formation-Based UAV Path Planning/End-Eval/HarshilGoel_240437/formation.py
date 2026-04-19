"""
formation.py
------------
Defines the UAV formation shape and assigns each drone a fixed offset.
Supports multiple letter shapes.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ── Formation definition ──────────────────────────────────────────────────────

# Dictionary of different letter formations (raw offsets)
SHAPES = {
    'V': np.array([[-6, 5], [-4, 3], [-2, 1.5], [0, 0], [2, 1.5], [4, 3], [6, 5]]),
    'T': np.array([[-6, 6], [-3, 6], [0, 6], [3, 6], [6, 6], [0, 3], [0, 0]]),
    'I': np.array([[0, -6], [0, -4], [0, -2], [0, 0], [0, 2], [0, 4], [0, 6]]),
    'U': np.array([[-4, 6], [-4, 3], [-4, 0], [0, -2], [4, 0], [4, 3], [4, 6]]),
    'X': np.array([[-4, 4], [-2, 2], [0, 0], [2, -2], [4, -4], [-4, -4], [4, 4], [-2, -2], [2, 2]])
}

# CHANGE THIS to select a different letter!
FORMATION_SHAPE = 'T'   

if FORMATION_SHAPE not in SHAPES:
    raise ValueError(f"Shape '{FORMATION_SHAPE}' not found in SHAPES dictionary.")

_RAW_OFFSETS = SHAPES[FORMATION_SHAPE].astype(float)

# Centre the offsets so the centroid is exactly at (0,0)
OFFSETS = _RAW_OFFSETS - _RAW_OFFSETS.mean(axis=0)
N_DRONES = len(OFFSETS)

# CRITICAL FIX: Calculate the maximum distance any drone sits from the centroid
FORMATION_RADIUS = np.max(np.hypot(OFFSETS[:, 0], OFFSETS[:, 1]))

DRONE_COLORS = [
    '#e74c3c', '#e67e22', '#f1c40f', '#2ecc71',
    '#1abc9c', '#3498db', '#9b59b6', '#e84393', '#fdcb6e'
]

# ── Core functions ────────────────────────────────────────────────────────────

def get_drone_positions(centroid_xy):
    centroid_xy = np.asarray(centroid_xy, dtype=float)
    return centroid_xy + OFFSETS

def get_formation_trajectory(centroid_traj):
    drone_trajs = []
    for offset in OFFSETS:
        drone_xy = centroid_traj[:, :2] + offset
        drone_t  = centroid_traj[:, 2:3]
        drone_trajs.append(np.hstack([drone_xy, drone_t]))
    return drone_trajs

# ── Visualisation ─────────────────────────────────────────────────────────────

def visualise_formation(save_path=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Draw formation radius
    circle = plt.Circle((0, 0), FORMATION_RADIUS, color='gray', alpha=0.1, label='Formation Bound')
    ax.add_patch(circle)

    for i, (ox, oy) in enumerate(OFFSETS):
        color = DRONE_COLORS[i % len(DRONE_COLORS)]
        ax.scatter(ox, oy, s=200, color=color, zorder=5, label=f'D{i} ({ox:.1f}, {oy:.1f})')
        ax.annotate(f'D{i}', (ox, oy), textcoords='offset points', xytext=(5, 5), fontsize=9)

    ax.scatter(0, 0, s=80, color='black', marker='+', zorder=6, label='Centroid (0,0)')

    ax.set_title(f"Formation Shape: '{FORMATION_SHAPE}' — {N_DRONES} Drones\nRadius: {FORMATION_RADIUS:.2f} units")
    ax.set_xlabel('X offset (units)')
    ax.set_ylabel('Y offset (units)')
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[formation] Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)

if __name__ == '__main__':
    visualise_formation()