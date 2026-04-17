"""
formation.py — Defines the UAV formation shape (letter 'A') and assigns
each drone a fixed offset from the centroid.

Formation flying works by adding constant per-drone offsets to the
centroid trajectory at every time step.
"""

import numpy as np
import matplotlib.pyplot as plt


# ── Formation definition — letter 'A' with 9 drones ─────────────
#
#  Offsets are in (dx, dy) relative to the formation centroid.
#  Scale in units that make sense on the 100×100 map.
#
#        *           dy = +4   (apex)
#       * *          dy = +2
#      *   *         dy =  0   (crossbar anchors)
#     * * * *        dy =  0   (crossbar — 4 drones)
#    *       *       dy = -3   (base)
#
#  9 drones total.

FORMATION_OFFSETS = np.array([
    # Apex
    ( 0.0,  4.0),
    # Upper legs
    (-1.5,  2.0),
    ( 1.5,  2.0),
    # Crossbar
    (-1.5,  0.0),
    ( 0.0,  0.0),
    ( 1.5,  0.0),
    # Lower legs
    (-3.0, -3.0),
    ( 0.0, -3.0),   # extra centre drone at base for symmetry
    ( 3.0, -3.0),
], dtype=float)

NUM_DRONES = len(FORMATION_OFFSETS)

# Colour for each drone in the animation
DRONE_COLOURS = [
    "#e74c3c", "#e67e22", "#f1c40f",
    "#2ecc71", "#1abc9c", "#3498db",
    "#9b59b6", "#e91e63", "#00bcd4",
]


def get_drone_positions(centroid_xy):
    """
    Given centroid position (x, y), return array of shape (N, 2)
    with each drone's world position.
    """
    cx, cy = centroid_xy
    return FORMATION_OFFSETS + np.array([cx, cy])


def visualise_formation(centroid=(50, 50), save_path=None):
    """Plot the formation shape centred at `centroid`."""
    positions = get_drone_positions(centroid)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#1a1a2e")

    for i, (px, py) in enumerate(positions):
        ax.plot(px, py, "o", color=DRONE_COLOURS[i % len(DRONE_COLOURS)],
                markersize=14, zorder=5)
        ax.annotate(f"D{i}", (px, py), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color="white")

    # Draw letter 'A' outline for reference
    outline_order = [6, 0, 8,    # left-base → apex → right-base
                     None,        # break
                     3, 5]        # crossbar
    xs, ys = [], []
    for idx in outline_order:
        if idx is None:
            ax.plot(xs, ys, "--", color="white", alpha=0.3, linewidth=1)
            xs, ys = [], []
        else:
            xs.append(positions[idx, 0])
            ys.append(positions[idx, 1])
    if xs:
        ax.plot(xs, ys, "--", color="white", alpha=0.3, linewidth=1)

    ax.plot(*centroid, "+", color="yellow", markersize=16, markeredgewidth=2,
            label="Centroid", zorder=6)

    margin = 8
    ax.set_xlim(centroid[0] - margin, centroid[0] + margin)
    ax.set_ylim(centroid[1] - margin, centroid[1] + margin)
    ax.set_aspect("equal")
    ax.set_title(f"Formation Shape — Letter 'A'  ({NUM_DRONES} drones)",
                 fontsize=13, fontweight="bold", color="white")
    ax.set_xlabel("X offset (units)", color="white")
    ax.set_ylabel("Y offset (units)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    ax.legend(facecolor="#2c2c2c", labelcolor="white")
    ax.grid(True, linestyle=":", alpha=0.3, color="white")

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[formation] Saved formation plot → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Quick self-test ──────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Number of drones : {NUM_DRONES}")
    print("Offsets (dx, dy):")
    for i, off in enumerate(FORMATION_OFFSETS):
        print(f"  Drone {i:02d}: ({off[0]:+.1f}, {off[1]:+.1f})")
    visualise_formation()
