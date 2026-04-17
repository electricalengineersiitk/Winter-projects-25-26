"""
map_setup.py — Defines the 2D grid map, obstacle, start, and goal.
All other scripts import from here.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Map dimensions ──────────────────────────────────────────────
MAP_WIDTH  = 100
MAP_HEIGHT = 100

# ── Key coordinates ─────────────────────────────────────────────
START = (5, 50)
GOAL  = (95, 50)

# ── Obstacle (circular) ─────────────────────────────────────────
OBSTACLE_CENTER = (50, 50)
OBSTACLE_RADIUS = 10

# Safety margin added around the obstacle during planning
SAFETY_MARGIN = 3


def is_in_obstacle(x, y, margin=0):
    """Return True if (x, y) falls inside the obstacle + optional margin."""
    cx, cy = OBSTACLE_CENTER
    r = OBSTACLE_RADIUS + margin
    return (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2


def build_grid(margin=SAFETY_MARGIN):
    """
    Build a boolean occupancy grid.
    True  = cell is FREE
    False = cell is BLOCKED (obstacle or safety margin)
    """
    grid = np.ones((MAP_HEIGHT, MAP_WIDTH), dtype=bool)
    for row in range(MAP_HEIGHT):
        for col in range(MAP_WIDTH):
            if is_in_obstacle(col, row, margin):
                grid[row, col] = False
    return grid


def visualise_map(path=None, save_path=None):
    """
    Plot the map with obstacle, start, goal, and optional path.
    If save_path is given the figure is saved there; otherwise it is shown.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, MAP_WIDTH)
    ax.set_ylim(0, MAP_HEIGHT)
    ax.set_aspect("equal")
    ax.set_facecolor("#f5f5f5")

    # Obstacle
    obs_patch = plt.Circle(
        OBSTACLE_CENTER, OBSTACLE_RADIUS, color="#e74c3c", alpha=0.85, label="Obstacle"
    )
    safety_patch = plt.Circle(
        OBSTACLE_CENTER,
        OBSTACLE_RADIUS + SAFETY_MARGIN,
        color="#e74c3c",
        alpha=0.25,
        linestyle="--",
        fill=True,
        label=f"Safety margin (+{SAFETY_MARGIN})",
    )
    ax.add_patch(safety_patch)
    ax.add_patch(obs_patch)

    # Start & Goal
    ax.plot(*START, "go", markersize=12, label="Start", zorder=5)
    ax.plot(*GOAL,  "b*", markersize=14, label="Goal",  zorder=5)

    # Optional path
    if path is not None:
        xs, ys = zip(*path)
        ax.plot(xs, ys, "k--", linewidth=1.5, label="Planned path", zorder=4)
        ax.plot(xs, ys, "ko", markersize=3, zorder=4)

    ax.set_title("UAV Formation — Map & Obstacle", fontsize=14, fontweight="bold")
    ax.set_xlabel("X (units)")
    ax.set_ylabel("Y (units)")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.4)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[map_setup] Saved map plot → {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Quick self-test ──────────────────────────────────────────────
if __name__ == "__main__":
    grid = build_grid()
    blocked = int((~grid).sum())
    print(f"Map size   : {MAP_WIDTH} x {MAP_HEIGHT}")
    print(f"Start      : {START}")
    print(f"Goal       : {GOAL}")
    print(f"Obstacle   : centre={OBSTACLE_CENTER}, radius={OBSTACLE_RADIUS}")
    print(f"Blocked cells (with margin): {blocked}")
    visualise_map()
