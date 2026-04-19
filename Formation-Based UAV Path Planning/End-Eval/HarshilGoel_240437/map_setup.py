"""
map_setup.py
------------
Defines the 2D grid map, obstacle, start, and goal positions.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# Import the radius of our formation to prevent wing-clipping
from formation import FORMATION_RADIUS

# ── Map dimensions ──────────────────────────────────────────────────────────
MAP_WIDTH  = 100   
MAP_HEIGHT = 100   

START = (5, 50)
GOAL  = (95, 50)

OBSTACLE_CENTER = (50, 50)
OBSTACLE_RADIUS = 10

BASE_MARGIN = 3
SAFETY_MARGIN = BASE_MARGIN + int(np.ceil(FORMATION_RADIUS))

def build_grid():
    grid = np.ones((MAP_HEIGHT, MAP_WIDTH), dtype=bool) 
    ox, oy = OBSTACLE_CENTER
    r = OBSTACLE_RADIUS + SAFETY_MARGIN

    for y in range(MAP_HEIGHT):
        for x in range(MAP_WIDTH):
            if (x - ox) ** 2 + (y - oy) ** 2 <= r ** 2:
                grid[y, x] = False 
    return grid


def is_in_obstacle(x, y, margin=0):
    """Return True if point (x, y) is inside the obstacle (+ optional extra margin)."""
    ox, oy = OBSTACLE_CENTER
    r = OBSTACLE_RADIUS + margin
    return (x - ox) ** 2 + (y - oy) ** 2 <= r ** 2


def visualise_map(path=None, save_path=None):
    """
    Draw the map.  Optionally overlay a waypoint path.
    If save_path is given the figure is written to that file instead of shown.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    # Grid background
    ax.set_xlim(0, MAP_WIDTH)
    ax.set_ylim(0, MAP_HEIGHT)
    ax.set_aspect('equal')
    ax.set_facecolor('#f0f4f8')

    # Obstacle
    obs = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS, color='#e74c3c',
                     alpha=0.85, label='Obstacle')
    margin_circle = plt.Circle(OBSTACLE_CENTER, OBSTACLE_RADIUS + SAFETY_MARGIN,
                               color='#e74c3c', alpha=0.2, linestyle='--',
                               fill=True, label='Safety margin')
    ax.add_patch(margin_circle)
    ax.add_patch(obs)

    # Start & Goal
    ax.plot(*START, 'go', markersize=12, label=f'Start {START}', zorder=5)
    ax.plot(*GOAL,  'b*', markersize=14, label=f'Goal {GOAL}',  zorder=5)

    # Planned path (if provided)
    if path is not None:
        xs, ys = zip(*path)
        ax.plot(xs, ys, 'k--', linewidth=1.5, label='Planned path', zorder=4)
        ax.plot(xs, ys, 'ko',  markersize=3,  zorder=4)

    ax.set_title('2-D Map — Obstacle, Start, Goal & Planned Path', fontsize=13)
    ax.set_xlabel('X (units)')
    ax.set_ylabel('Y (units)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f"[map_setup] Saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)


if __name__ == '__main__':
    grid = build_grid()
    free  = grid.sum()
    total = MAP_WIDTH * MAP_HEIGHT
    print(f"Grid size : {MAP_WIDTH} × {MAP_HEIGHT} = {total} cells")
    print(f"Free cells: {free}  |  Blocked: {total - free}")
    print(f"Start     : {START}")
    print(f"Goal      : {GOAL}")
    print(f"Obstacle  : centre={OBSTACLE_CENTER}, radius={OBSTACLE_RADIUS}, "
          f"safety margin={SAFETY_MARGIN}")
    visualise_map()
