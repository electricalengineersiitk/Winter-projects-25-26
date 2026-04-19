"""
map_setup.py
Defines the 2D grid environment: map dimensions, obstacle, start, and goal.
All other modules import constants from here.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Map dimensions ──────────────────────────────────────────────────────────
MAP_WIDTH  = 100   # x range: 0 → 100
MAP_HEIGHT = 100   # y range: 0 → 100

# ── Key coordinates ──────────────────────────────────────────────────────────
START = (5,  50)
GOAL  = (95, 50)

# ── Circular obstacle ────────────────────────────────────────────────────────
OBSTACLE_CENTER = (50, 50)
OBSTACLE_RADIUS = 10

# Safety margin added around the obstacle when blocking grid cells
SAFETY_MARGIN = 3   # units; total clearance = OBSTACLE_RADIUS + SAFETY_MARGIN


def build_grid():
    """
    Returns a 2-D boolean numpy array (MAP_HEIGHT x MAP_WIDTH).
    True  → cell is free
    False → cell is blocked (inside obstacle + safety margin)
    """
    grid = np.ones((MAP_HEIGHT, MAP_WIDTH), dtype=bool)

    ox, oy = OBSTACLE_CENTER
    blocked_r = OBSTACLE_RADIUS + SAFETY_MARGIN

    for row in range(MAP_HEIGHT):
        for col in range(MAP_WIDTH):
            dist = np.hypot(col - ox, row - oy)
            if dist <= blocked_r:
                grid[row, col] = False

    return grid


def plot_map(path_waypoints=None, title="Environment Map", ax=None):
    """
    Visualise the map.  Optionally overlay a planned path.

    Parameters
    ----------
    path_waypoints : list of (x, y) tuples, optional
    title          : str
    ax             : matplotlib Axes (created if None)
    """
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))

    # Draw obstacle (true radius in red, safety margin in light red)
    margin_patch = plt.Circle(
        OBSTACLE_CENTER,
        OBSTACLE_RADIUS + SAFETY_MARGIN,
        color="salmon", alpha=0.35, label="Safety margin"
    )
    obs_patch = plt.Circle(
        OBSTACLE_CENTER,
        OBSTACLE_RADIUS,
        color="firebrick", alpha=0.8, label="Obstacle"
    )
    ax.add_patch(margin_patch)
    ax.add_patch(obs_patch)

    # Start and goal
    ax.scatter(*START, s=120, color="green",  zorder=5, label="Start")
    ax.scatter(*GOAL,  s=120, color="royalblue", zorder=5, label="Goal")
    ax.annotate("Start", START, textcoords="offset points", xytext=(6, 4),
                fontsize=9, color="green")
    ax.annotate("Goal",  GOAL,  textcoords="offset points", xytext=(6, 4),
                fontsize=9, color="royalblue")

    # Planned path
    if path_waypoints is not None and len(path_waypoints) > 1:
        xs, ys = zip(*path_waypoints)
        ax.plot(xs, ys, "o-", color="darkorange", linewidth=1.8,
                markersize=3, label="Planned path", zorder=4)

    ax.set_xlim(0, MAP_WIDTH)
    ax.set_ylim(0, MAP_HEIGHT)
    ax.set_aspect("equal")
    ax.set_xlabel("X (units)")
    ax.set_ylabel("Y (units)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)

    if show:
        plt.tight_layout()
        plt.show()
