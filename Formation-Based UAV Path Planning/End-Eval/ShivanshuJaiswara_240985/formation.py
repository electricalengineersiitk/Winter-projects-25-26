"""
formation.py
Defines the UAV formation shape and computes per-drone positions
at each time step along a trajectory.

Formation chosen: 'V'-shape with 5 drones.
The centroid follows the planned trajectory; each drone keeps
a fixed offset from that centroid throughout the entire flight.

Offset table (x_offset, y_offset):
  Drone 0 → tip of the V         ( 0.0,  0.0) ← centroid / lead drone
  Drone 1 → left inner wing      (-1.5, -1.5)
  Drone 2 → right inner wing     ( 1.5, -1.5)
  Drone 3 → left outer wing      (-3.0, -3.0)
  Drone 4 → right outer wing     ( 3.0, -3.0)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── Formation definition ─────────────────────────────────────────────────────
NUM_DRONES = 5

# Fixed (dx, dy) offsets from the centroid for each drone
FORMATION_OFFSETS = np.array([
    [ 0.0,  0.0],   # lead drone (centroid)
    [-1.5, -1.5],   # left  inner
    [ 1.5, -1.5],   # right inner
    [-3.0, -3.0],   # left  outer
    [ 3.0, -3.0],   # right outer
], dtype=float)

DRONE_COLORS = ["gold", "deepskyblue", "limegreen", "tomato", "mediumpurple"]


def get_drone_positions(centroid_x, centroid_y):
    """
    Given the centroid position (scalars or arrays), return a
    (NUM_DRONES, 2) array of drone positions at that instant.

    Works with scalar inputs (single frame) and 1-D array inputs
    (entire trajectory at once).
    """
    cx = np.asarray(centroid_x)
    cy = np.asarray(centroid_y)

    if cx.ndim == 0:
        # Single frame
        positions = np.empty((NUM_DRONES, 2))
        for i, (dx, dy) in enumerate(FORMATION_OFFSETS):
            positions[i, 0] = cx + dx
            positions[i, 1] = cy + dy
        return positions
    else:
        # Full trajectory: shape → (NUM_DRONES, N, 2)
        n = len(cx)
        positions = np.empty((NUM_DRONES, n, 2))
        for i, (dx, dy) in enumerate(FORMATION_OFFSETS):
            positions[i, :, 0] = cx + dx
            positions[i, :, 1] = cy + dy
        return positions


def check_formation_maintained(drone_positions_over_time, tol=1e-6):
    """
    Sanity check: verify that the pairwise distances between drones
    stay constant across all frames (within floating-point tolerance).

    Parameters
    ----------
    drone_positions_over_time : (NUM_DRONES, N, 2) array
    tol                       : allowed deviation

    Returns True if formation is maintained.
    """
    n_drones, n_frames, _ = drone_positions_over_time.shape
    ref_frame = drone_positions_over_time[:, 0, :]   # shape (N_drones, 2)

    for f in range(1, n_frames):
        frame = drone_positions_over_time[:, f, :]
        diff  = frame - ref_frame                    # (N_drones, 2)
        # offsets must stay identical to frame 0 offsets
        if np.max(np.abs(diff - (ref_frame - drone_positions_over_time[:, 0, :]))) > tol:
            return False
    return True


def plot_formation_snapshot(centroid_x, centroid_y, title="Formation Snapshot", ax=None):
    """
    Draw a single snapshot of the formation at a given centroid location.
    """
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    positions = get_drone_positions(centroid_x, centroid_y)

    for i, (x, y) in enumerate(positions):
        ax.scatter(x, y, s=160, color=DRONE_COLORS[i], zorder=5,
                   edgecolors="k", linewidths=0.6,
                   label=f"Drone {i}")

    # Draw connecting lines of the V
    # Spine: 3→1→0→2→4
    spine_order = [3, 1, 0, 2, 4]
    xs = [positions[k, 0] for k in spine_order]
    ys = [positions[k, 1] for k in spine_order]
    ax.plot(xs, ys, "k--", linewidth=1.2, alpha=0.6)

    ax.scatter(centroid_x, centroid_y, s=60, color="k",
               zorder=6, marker="+", label="Centroid")

    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()
