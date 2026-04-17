"""
map_setup.py
Defines the 2D grid environment: map dimensions, obstacle, start and goal.
All other scripts import constants from here.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── Map dimensions ────────────────────────────────────────────────────────────
MAP_WIDTH  = 100
MAP_HEIGHT = 100

# ── Key positions ─────────────────────────────────────────────────────────────
START = (5, 50)
GOAL  = (95, 50)

# ── Obstacle: single circular obstacle centred between start and goal ─────────
OBSTACLE_CENTER = (50, 50)
OBSTACLE_RADIUS = 12


# ── Helper: build boolean obstacle mask ───────────────────────────────────────
def get_obstacle_mask(safety_margin: int = 0) -> np.ndarray:
    """
    Returns a (MAP_HEIGHT × MAP_WIDTH) boolean array.
    True  → cell is inside obstacle (+ optional safety margin).
    False → cell is free.
    """
    Y, X = np.mgrid[0:MAP_HEIGHT, 0:MAP_WIDTH]
    cx, cy = OBSTACLE_CENTER
    r = OBSTACLE_RADIUS + safety_margin
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2


# ── Visualisation helper ──────────────────────────────────────────────────────
def plot_map(ax=None, path=None, title="UAV Path Planning Map"):
    """Draw the map. Optionally overlay a planned path."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    # Obstacle
    circle = plt.Circle(
        OBSTACLE_CENTER, OBSTACLE_RADIUS,
        color="tomato", alpha=0.55, label="Obstacle"
    )
    ax.add_patch(circle)
    ax.plot(*OBSTACLE_CENTER, "r+", markersize=10)

    # Start / Goal
    ax.plot(*START, "go", markersize=12, label="Start", zorder=5)
    ax.plot(*GOAL,  "b*", markersize=14, label="Goal",  zorder=5)

    # Optional path overlay
    if path is not None:
        p = np.asarray(path)
        ax.plot(p[:, 0], p[:, 1], "k--", linewidth=2,
                label="Planned Path", zorder=4)

    ax.set_xlim(0, MAP_WIDTH)
    ax.set_ylim(0, MAP_HEIGHT)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("X (units)")
    ax.set_ylabel("Y (units)")
    ax.legend(loc="upper left")
    return ax


# ── Quick standalone test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_map(ax)
    out = os.path.join(os.path.dirname(__file__), "results", "map_preview.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"Map preview saved → {out}")
    plt.show()
