"""
formation.py
Defines the 'R' letter formation using 10 UAVs.

How formation flying works
──────────────────────────
1. The centroid of all drones follows the planned trajectory.
2. Each drone's world position = centroid_position + its fixed offset.
3. Offsets are constant → shape is rigid throughout the flight.

The 'R' skeleton is constructed from 10 key points that trace:
  • A vertical stroke (5 drones)
  • The upper closed bowl (3 drones + junction point)
  • The diagonal descending leg (1 drone)
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── Number of UAVs ─────────────────────────────────────────────────────────────
N_DRONES = 10

# ── Scale factor (units) ───────────────────────────────────────────────────────
# Controls how large the 'R' appears in the simulation.
_SCALE = 4.0

# ── Raw skeleton points for letter 'R' (local coords, before centring) ────────
#
#   y↑
#   4 ●───●           ← top of vertical + top of bowl
#   3 |   ●           ← upper bowl right
#   2 ●───●           ← mid junction + lower bowl right
#   1 |   ●           ← lower leg (diagonal tip)
#   0 ●               ← bottom of vertical
#     0 1 2  x→
#
_RAW = np.array([
    [0.0, 0.0],   # 0  bottom of vertical stroke
    [0.0, 1.0],   # 1  lower vertical
    [0.0, 2.0],   # 2  mid vertical (where bowl closes & leg diverges)
    [0.0, 3.0],   # 3  upper vertical
    [0.0, 4.0],   # 4  top-left corner
    [1.0, 4.0],   # 5  top-right of bowl
    [2.0, 3.5],   # 6  upper-right bowl
    [2.0, 2.5],   # 7  lower-right bowl
    [1.0, 2.0],   # 8  bowl closes back to vertical
    [2.0, 1.0],   # 9  diagonal leg tip (bottom-right)
], dtype=float)

# Centre & scale
_centroid = _RAW.mean(axis=0)
FORMATION_OFFSETS = (_RAW - _centroid) * _SCALE   # shape (N_DRONES, 2)


# ── API ───────────────────────────────────────────────────────────────────────
def get_offsets() -> np.ndarray:
    """Return (N_DRONES × 2) array of (dx, dy) offsets from the centroid."""
    return FORMATION_OFFSETS.copy()


def get_drone_positions(centroid_xy: np.ndarray) -> np.ndarray:
    """
    Compute world positions for all drones given the centroid position.

    Parameters
    ----------
    centroid_xy : array-like, shape (2,) or (T, 2)
        Centroid position(s).

    Returns
    -------
    positions : np.ndarray
        If centroid_xy is (2,)   → shape (N_DRONES, 2)
        If centroid_xy is (T, 2) → shape (T, N_DRONES, 2)
    """
    c = np.asarray(centroid_xy, dtype=float)
    offsets = get_offsets()          # (N, 2)

    if c.ndim == 1:
        return c[np.newaxis, :] + offsets   # (N, 2)
    else:
        # c: (T, 2) → broadcast to (T, N, 2)
        return c[:, np.newaxis, :] + offsets[np.newaxis, :, :]


# ── Visualisation helper ──────────────────────────────────────────────────────
def plot_formation(ax=None, centroid=(0.0, 0.0), label_drones=True):
    """Draw the formation at a given centroid position."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 6))

    pos = get_drone_positions(np.array(centroid))   # (N, 2)

    # Draw connecting lines (stroke order of the letter R)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),      # vertical stroke
        (4, 5), (5, 6), (6, 7), (7, 8),      # bowl
        (2, 8),                                # bowl closes
        (8, 9),                                # diagonal leg
    ]
    for i, j in edges:
        ax.plot([pos[i, 0], pos[j, 0]],
                [pos[i, 1], pos[j, 1]],
                "b-", linewidth=1.5, alpha=0.6)

    # Draw drones
    ax.scatter(pos[:, 0], pos[:, 1],
               s=120, c="dodgerblue", zorder=5, edgecolors="k")

    if label_drones:
        for i, (x, y) in enumerate(pos):
            ax.annotate(f"D{i}", (x, y),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=7, color="navy")

    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Letter 'R' Formation  ({N_DRONES} UAVs)", fontsize=12, fontweight="bold")
    ax.set_xlabel("X (units)")
    ax.set_ylabel("Y (units)")
    return ax


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(5, 7))
    plot_formation(ax, centroid=(0.0, 0.0))

    print("Formation offsets (dx, dy) for each drone:")
    for i, (dx, dy) in enumerate(FORMATION_OFFSETS):
        print(f"  Drone {i:2d}: ({dx:+.2f}, {dy:+.2f})")

    out = os.path.join(os.path.dirname(__file__), "results", "formation_shape.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFormation shape saved → {out}")
    plt.show()
